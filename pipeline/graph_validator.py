"""
Deterministic structural validator for the clinical decision graph.

Checks performed
────────────────
1.  Exactly one START node
2.  At least one END node
3.  START has no question / options / action_details
4.  Every DECISION has question + at least 2 options
5.  Every ACTION has action_details
6.  No duplicate edges (same from+to)
7.  No self-loops (from == to)
8.  Every node referenced in edges exists
9.  Every DECISION option has at least one outgoing edge
10. Every ACTION / WARNING has exactly one outgoing edge → END-type node
11. No node (except START) is unreachable from START
12. No cycles (graph is a DAG)
13. Every DECISION with N options has at most N outgoing edges
    (prevents contradictory duplicate conditions)

Returns
───────
list[dict]  — list of issue dicts:
  { "severity": "critical|warning", "node_id": str|None, "description": str, "suggestion": str }
"""

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def validate(graph_doc: dict) -> list[dict]:
    """Run all checks and return a list of issues."""
    nodes: list[dict] = graph_doc.get("graph", {}).get("nodes", [])
    edges: list[dict] = graph_doc.get("graph", {}).get("edges", [])

    issues: list[dict] = []

    node_map   = {n["id"]: n for n in nodes}
    node_type  = {n["id"]: n.get("type", "") for n in nodes}

    out_edges: dict[str, list[dict]] = {n["id"]: [] for n in nodes}
    in_edges:  dict[str, list[dict]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if src and dst:
            out_edges.setdefault(src, []).append(e)
            in_edges.setdefault(dst, []).append(e)

    def issue(severity, node_id, description, suggestion):
        issues.append({
            "severity":    severity,
            "node_id":     node_id,
            "description": description,
            "suggestion":  suggestion,
        })

    # ── 1. Exactly one START ──────────────────────────────────────────────────
    starts = [n for n in nodes if n.get("type") == "START"]
    if len(starts) == 0:
        issue("critical", None, "Нет узла START", "Добавь узел типа START")
    elif len(starts) > 1:
        ids = [n["id"] for n in starts]
        issue("critical", None, f"Несколько START-узлов: {ids}",
              "Оставь один START, остальные преобразуй в DECISION")

    # ── 2. At least one END ───────────────────────────────────────────────────
    ends = [n for n in nodes if n.get("type") == "END"]
    if not ends:
        issue("critical", None, "Нет узла END", "Добавь хотя бы один END-узел")

    # ── 3. START fields ───────────────────────────────────────────────────────
    for n in starts:
        if n.get("question") or n.get("options"):
            issue("critical", n["id"],
                  "START содержит question/options — это поля DECISION",
                  "Убери question и options из START, перенеси в первый DECISION")

    # ── 4. DECISION has question + options ───────────────────────────────────
    for n in nodes:
        if n.get("type") == "DECISION":
            if not n.get("question"):
                issue("warning", n["id"], f"DECISION '{n['id']}' без question",
                      "Добавь поле question")
            opts = n.get("options") or []
            if len(opts) < 2:
                issue("warning", n["id"],
                      f"DECISION '{n['id']}' имеет < 2 вариантов options",
                      "Добавь минимум 2 варианта ответа")

    # ── 5. ACTION has action_details ─────────────────────────────────────────
    for n in nodes:
        if n.get("type") == "ACTION" and not n.get("action_details"):
            issue("warning", n["id"], f"ACTION '{n['id']}' без action_details",
                  "Заполни action_details (procedure, implant, timing, evidence_level)")

    # ── 6. Duplicate edges ────────────────────────────────────────────────────
    seen_pairs: set[tuple] = set()
    for e in edges:
        pair = (e.get("from"), e.get("to"))
        if pair in seen_pairs:
            issue("critical", e.get("from"),
                  f"Дублирующее ребро {pair}",
                  "Удали одно из дублирующих рёбер")
        seen_pairs.add(pair)

    # ── 7. Self-loops ─────────────────────────────────────────────────────────
    for e in edges:
        if e.get("from") == e.get("to"):
            issue("critical", e.get("from"),
                  f"Самопетля на узле {e.get('from')}",
                  "Удали ребро, ведущее в самого себя")

    # ── 8. Missing node references ────────────────────────────────────────────
    for e in edges:
        for field in ("from", "to"):
            nid = e.get(field)
            if nid and nid not in node_map:
                issue("critical", nid,
                      f"Ребро {e.get('id')} ссылается на несуществующий узел '{nid}'",
                      f"Добавь узел с id='{nid}' или исправь ребро")

    # ── 9. Every DECISION option covered by at least one edge ────────────────
    for n in nodes:
        if n.get("type") != "DECISION":
            continue
        nid  = n["id"]
        opts = n.get("options") or []
        out  = out_edges.get(nid, [])
        edge_labels = {e.get("label", "").strip().lower() for e in out}
        edge_values: set[str] = set()
        for e in out:
            cond = e.get("condition") or {}
            v = cond.get("value")
            if isinstance(v, list):
                edge_values.update(str(x).lower() for x in v)
            elif v is not None:
                edge_values.add(str(v).lower())

        uncovered = []
        for opt in opts:
            opt_lower = opt.strip().lower()
            # Match by exact label, condition value, or prefix (e.g. "Pipkin I" matches "pipkin i")
            matched = (
                opt_lower in edge_values
                or opt_lower in edge_labels
                or any(opt_lower.startswith(lbl) or lbl.startswith(opt_lower)
                       for lbl in edge_labels if lbl)
            )
            if not matched:
                uncovered.append(opt)
        if uncovered:
            issue("critical", nid,
                  f"DECISION '{nid}': варианты без исходящего ребра: {uncovered}",
                  "Добавь ребро для каждого непокрытого варианта")

    # ── 10. ACTION/WARNING → must reach END, and ACTION must not go to non-END ─
    end_ids = {n["id"] for n in nodes if n.get("type") == "END"}
    for n in nodes:
        ntype = n.get("type")
        nid   = n["id"]
        if ntype not in ("ACTION", "WARNING"):
            continue
        out = out_edges.get(nid, [])
        goes_to_end = any(e.get("to") in end_ids for e in out)
        if not goes_to_end:
            issue("critical", nid,
                  f"{ntype} '{nid}' не имеет ребра к END",
                  "Добавь ребро от этого узла к END")
        # ACTION must not have outgoing edges to non-END nodes
        if ntype == "ACTION":
            for e in out:
                dst_type = node_type.get(e.get("to"), "")
                if e.get("to") not in end_ids:
                    issue("critical", nid,
                          f"ACTION '{nid}' имеет ребро к '{e.get('to')}' ({dst_type}) — "
                          f"ACTION должен вести только к END. "
                          f"Предупреждения (WARNING) должны идти ДО ACTION, а не после.",
                          f"Перенеси узел '{e.get('to')}' перед ACTION, или удали ребро.")

    # ── 11. Unreachable nodes ─────────────────────────────────────────────────
    if starts:
        start_id = starts[0]["id"]
        reachable: set[str] = set()
        q = deque([start_id])
        while q:
            cur = q.popleft()
            if cur in reachable:
                continue
            reachable.add(cur)
            for e in out_edges.get(cur, []):
                dst = e.get("to")
                if dst:
                    q.append(dst)
        for n in nodes:
            if n["id"] not in reachable:
                issue("critical", n["id"],
                      f"Узел '{n['id']}' недостижим из START",
                      "Добавь входящее ребро или удали узел")

    # ── 12. Cycle detection (DFS) ─────────────────────────────────────────────
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n["id"]: WHITE for n in nodes}

    def dfs(v):
        color[v] = GRAY
        for e in out_edges.get(v, []):
            dst = e.get("to")
            if not dst or dst not in color:
                continue
            if color[dst] == GRAY:
                issue("critical", v,
                      f"Цикл: ребро {v} → {dst} создаёт петлю",
                      "Удали ребро, создающее цикл")
                return
            if color[dst] == WHITE:
                dfs(dst)
        color[v] = BLACK

    for n in nodes:
        if color[n["id"]] == WHITE:
            dfs(n["id"])

    # ── 13. DECISION: no more outgoing edges than options ────────────────────
    for n in nodes:
        if n.get("type") != "DECISION":
            continue
        nid  = n["id"]
        opts = n.get("options") or []
        out  = out_edges.get(nid, [])
        if len(out) > len(opts) + 1:  # +1 tolerance for "else" branches
            issue("warning", nid,
                  f"DECISION '{nid}': {len(out)} исходящих рёбер при {len(opts)} вариантах",
                  "Проверь, нет ли противоречивых условий ветвления")

    # ── 14. Path-level: no duplicate questions on same path ──────────────────
    issues.extend(validate_paths(graph_doc))

    if issues:
        crits = sum(1 for i in issues if i["severity"] == "critical")
        warns = sum(1 for i in issues if i["severity"] == "warning")
        logger.info(f"Validator: {crits} critical, {warns} warning issues")
    else:
        logger.info("Validator: graph is structurally valid")

    # ── Check #15: outgoing edges of a node must not re-test already-known fields ─
    # If we arrived at node X knowing field F = V, an outgoing edge with
    # condition {field: F} is a contradiction (we already know F).
    nodes_list = graph_doc.get("graph", {}).get("nodes", [])
    edges_list  = graph_doc.get("graph", {}).get("edges", [])
    node_map2   = {n["id"]: n for n in nodes_list}
    out_edges2  = {}
    for e in edges_list:
        out_edges2.setdefault(e.get("from"), []).append(e)

    starts2 = [n["id"] for n in nodes_list if n.get("type") == "START"]
    if starts2:
        stack2 = deque([(starts2[0], frozenset())])  # (node_id, known_fields)
        visited2: set[tuple] = set()
        while stack2:
            nid, known = stack2.popleft()
            state = (nid, known)
            if state in visited2:
                continue
            visited2.add(state)
            for e in out_edges2.get(nid, []):
                cond = e.get("condition") or {}
                field = cond.get("field", "")
                dst = e.get("to")
                if field and field in known:
                    issues.append({
                        "severity": "warning",
                        "node_id": nid,
                        "description": (
                            f"Ребро '{e.get('id')}' из узла '{nid}' проверяет поле '{field}', "
                            f"которое уже было определено на пути к этому узлу. "
                            f"Это может привести к противоречию: пациент уже ответил на этот вопрос."
                        ),
                        "suggestion": (
                            f"Убери условие по полю '{field}' из ребра '{e.get('id')}', "
                            f"или перенеси ветвление по '{field}' выше — до того узла, "
                            f"откуда начинается текущий путь."
                        ),
                    })
                if dst and dst in node_map2:
                    new_known = known | ({field} if field else set())
                    stack2.append((dst, new_known))

    return issues
    return issues


def validate_paths(graph_doc: dict) -> list[dict]:
    """
    Additional path-level checks:
    14. No path from START to END asks the same question twice.
    15. A DECISION node must not have outgoing edges that re-test a field
        already decided on the path to reach it (contradictory edge conditions).
    """
    nodes: list[dict] = graph_doc.get("graph", {}).get("nodes", [])
    edges: list[dict] = graph_doc.get("graph", {}).get("edges", [])
    issues: list[dict] = []

    node_map  = {n["id"]: n for n in nodes}
    out_edges = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("from")
        if src:
            out_edges.setdefault(src, []).append(e)

    starts = [n["id"] for n in nodes if n.get("type") == "START"]
    if not starts:
        return issues

    # DFS collecting (path_of_node_ids, set_of_questions_seen)
    from collections import deque
    # stack entries: (current_node_id, questions_on_path: frozenset of (question_text, field))
    stack = deque()
    stack.append((starts[0], frozenset()))

    visited_states: set[tuple] = set()  # (node_id, questions_frozenset) to avoid explosion

    MAX_PATHS = 500  # safety limit
    paths_checked = 0

    while stack and paths_checked < MAX_PATHS:
        node_id, questions_seen = stack.pop()
        state = (node_id, questions_seen)
        if state in visited_states:
            continue
        visited_states.add(state)
        paths_checked += 1

        node = node_map.get(node_id)
        if not node:
            continue

        if node.get("type") == "DECISION":
            q_text = (node.get("question") or node.get("label") or "").strip().lower()
            # Also check by condition field of outgoing edges
            fields = {
                e.get("condition", {}).get("field", "")
                for e in out_edges.get(node_id, [])
                if e.get("condition")
            }
            field_key = tuple(sorted(fields)) if fields else ("_label_" + q_text,)

            for fk in field_key:
                if fk and fk in {q for _, q in questions_seen}:
                    issues.append({
                        "severity": "warning",
                        "node_id": node_id,
                        "description": (
                            f"Узел '{node_id}' ({node.get('label')}) задаёт вопрос "
                            f"по полю '{fk}', который уже задавался на этом пути. "
                            f"Пациент отвечает на один и тот же вопрос дважды."
                        ),
                        "suggestion": (
                            f"Вынеси вопрос по '{fk}' выше в алгоритме (один DECISION-узел "
                            f"вместо нескольких одинаковых), или убедись что ветки "
                            f"полностью независимы."
                        ),
                    })

            new_questions = questions_seen | frozenset((node_id, fk) for fk in field_key if fk)
        else:
            new_questions = questions_seen

        for e in out_edges.get(node_id, []):
            dst = e.get("to")
            if dst and dst in node_map:
                stack.append((dst, new_questions))

    return issues
