"""Stage 5b: Fix graph based on Stage 5a validation report.

Iterative repair loop:
  validate → fix (LLM) → _post_process (deterministic) → re-validate
  Up to MAX_FIX_LOOPS LLM calls.
"""
import json
import logging
from copy import deepcopy
from .base import BasePipelineStage, PipelineError
from .graph_validator import validate_graph_structure as validate

logger = logging.getLogger(__name__)
MAX_FIX_LOOPS = 2

_PROMPT = """\
Исправь граф принятия решений на основе отчёта о проблемах.

ТЕКУЩИЙ ГРАФ:
Узлы:
{nodes_json}

Рёбра:
{edges_json}

ПРОБЛЕМЫ:
{issues_text}

ИСХОДНЫЙ ТЕКСТ (справочно):
{source_text}

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
УЗЛЫ:
• START: ровно один; question=null, options=[], action_details=null
• DECISION: question + options ≥2; action_details=null
• ACTION: question=null, options=[]; action_details обязателен
• WARNING: идёт ДО ACTION НЕ после
• END: question=null, options=[], action_details=null

РЁБРА:
• Каждый вариант DECISION.options → ровно одно ребро (label == вариант)
• ACTION → END (condition=null, единственное исходящее)
• Нет дублей, нет самопетель, нет циклов, все узлы достижимы

ПАРАМЕТРЫ:
• Параметр, нужный в нескольких ветках → один DECISION-узел в начале
• НЕЛЬЗЯ: ребро проверяет поле, уже определённое на пути к этому узлу
• НЕЛЬЗЯ: один DECISION-узел доступен из двух веток с разным значением одного параметра

РАЗВОРАЧИВАНИЕ ВЕТОК — ОБЯЗАТЕЛЬНО:
• ACTION должен описывать ОДНУ конкретную операцию, не группу ("Лечение Pipkin I-IV" — ОШИБКА)
• Нельзя один ACTION-узел использовать для двух разных нозологий (разные incoming из разных веток)
• Нестабильный перелом без разделения по возрасту — ОШИБКА, добавь DECISION(возраст)
ты можешь дополнять уже существующие графы (пример: добавить недостающие типы переломов, варианты выборов)
• Если ACTION схлопывает несколько нозологий — замени его на DECISION + отдельные ACTION:
  "Лечение Pipkin" → DECISION(подтип)[I,II,III,IV] → ACTION(I), ACTION(II), ACTION(III), ACTION(IV)
• Нельзя один ACTION-узел использовать для двух разных нозологий (разные incoming из разных веток)

МИНИМУМ ACTION-узлов для полного покрытия:
  Pipkin: ≥5, Garden: ≥4, Чрезвертельные: ≥3. Итого ≥12.
Верни СТРОГО JSON:
{{
  "nodes": [...],
  "edges": [...],
}}"""


def _format_issues(structural: list[dict], clinical: list[dict]) -> str:
    lines = []
    if structural:
        lines.append("СТРУКТУРНЫЕ:")
        for i, iss in enumerate(structural, 1):
            nid = f" [{iss['node_id']}]" if iss.get("node_id") else ""
            lines.append(f"  {i}. [{iss['severity'].upper()}]{nid} {iss['description']}")
            lines.append(f"     → {iss['suggestion']}")
    clin = [c for c in clinical if c.get("severity") in ("critical", "warning")]
    if clin:
        lines.append("\nКЛИНИЧЕСКИЕ:")
        for i, c in enumerate(clin, 1):
            lines.append(f"  {i}. [{c['severity'].upper()}] {c['description']}")
            lines.append(f"     → {c.get('suggestion', '')}")
    missing = [c for c in clinical if c.get("importance") in ("high", "medium")]
    if missing:
        lines.append("\nНЕПОКРЫТЫЕ СЦЕНАРИИ:")
        for i, m in enumerate(missing, 1):
            lines.append(f"  {i}. [{m.get('importance','').upper()}] {m.get('description','')}")
    return "\n".join(lines) if lines else "Проблем не выявлено."



def _remove_unreachable(data: dict, accumulated_changelog: list | None = None) -> dict:
    """Remove nodes that are not reachable from START (deterministic, no LLM)."""
    from collections import deque
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    starts = [n["id"] for n in nodes if n.get("type") == "START"]
    if not starts:
        return data

    # BFS from START
    out_map: dict[str, list[str]] = {}
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if src and dst:
            out_map.setdefault(src, []).append(dst)

    reachable: set[str] = set()
    queue = deque(starts)
    while queue:
        cur = queue.popleft()
        if cur in reachable:
            continue
        reachable.add(cur)
        for dst in out_map.get(cur, []):
            queue.append(dst)

    node_ids = {n["id"] for n in nodes}
    unreachable = node_ids - reachable
    if not unreachable:
        return data

    for uid in unreachable:
        logger.warning(f"post_process: removing unreachable node '{uid}'")

    data["nodes"] = [n for n in nodes if n["id"] not in unreachable]
    data["edges"] = [
        e for e in edges
        if e.get("from") not in unreachable and e.get("to") not in unreachable
    ]
    return data

def _post_process(data: dict) -> dict:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_type = {n["id"]: n.get("type") for n in nodes}

    # Fix START fields
    for n in nodes:
        if n.get("type") == "START":
            n["question"] = None
            n["options"] = []
            n["action_details"] = None

    # Remove duplicate/self-loop edges
    seen: set[tuple] = set()
    clean = []
    for e in edges:
        pair = (e.get("from"), e.get("to"))
        if pair[0] == pair[1]:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        clean.append(e)
    data["edges"] = clean

    # ── Fix duplicate labels from same DECISION node ──────────────────────────
    # If two edges leave the same node with the same label → they were meant for
    # different branches but got merged. Keep only the first, log a warning.
    from collections import defaultdict
    label_map: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in data["edges"]:
        src = e.get("from", "")
        lbl = (e.get("label") or "").strip().lower()
        if lbl and node_type.get(src) == "DECISION":
            label_map[src][lbl].append(e)

    edges_to_remove: set[str] = set()
    for src, lbl_edges in label_map.items():
        for lbl, elist in lbl_edges.items():
            if len(elist) > 1:
                logger.warning(
                    f"_post_process: DECISION '{src}' has {len(elist)} edges with "
                    f"label '{lbl}' — keeping first, removing duplicates"
                )
                for dup in elist[1:]:
                    edges_to_remove.add(dup.get("id", ""))

    if edges_to_remove:
        data["edges"] = [e for e in data["edges"] if e.get("id") not in edges_to_remove]

    # ── Fix DECISION options with no outgoing edge (dead-end options) ─────────
    # For every DECISION that has an option with no matching outgoing edge,
    # insert a placeholder ACTION node + edge so the graph stays traversable.
    out_map: dict[str, list[dict]] = defaultdict(list)
    for e in data["edges"]:
        out_map[e.get("from", "")].append(e)

    counter_fix = sum(1 for e in data["edges"]
                      if e.get("id", "").startswith("edge_post_fix_")) + 1
    counter_node = sum(1 for n in nodes
                       if n.get("id", "").startswith("node_missing_")) + 1

    end_nodes = [n["id"] for n in nodes if n.get("type") == "END"]
    # Ensure END exists before we reference it
    if not end_nodes:
        end_id = "node_end"
        nodes.append({
            "id": end_id, "type": "END", "label": "Завершение",
            "question": None, "options": [], "action_details": None,
        })
        node_type[end_id] = "END"
        end_nodes = [end_id]
    global_end = end_nodes[0]

    for n in nodes:
        if n.get("type") != "DECISION":
            continue
        nid = n["id"]
        options = n.get("options") or []
        out_edges = out_map.get(nid, [])
        out_labels = {(e.get("label") or "").strip().lower() for e in out_edges}
        out_values = set()
        for e in out_edges:
            cond = e.get("condition") or {}
            v = cond.get("value")
            if v:
                out_values.add(str(v).strip().lower())

        for opt in options:
            opt_lower = opt.strip().lower()
            if opt_lower in out_labels or opt_lower in out_values:
                continue
            # Option has no outgoing edge — create placeholder ACTION + edges
            node_id = f"node_missing_{counter_node:03d}"
            edge_to_id = f"edge_post_fix_{counter_fix:03d}"
            edge_end_id = f"edge_post_fix_{counter_fix + 1:03d}"
            counter_node += 1
            counter_fix += 2

            logger.warning(
                f"_post_process: DECISION '{nid}' option '{opt}' has no outgoing edge "
                f"— inserting placeholder ACTION '{node_id}'"
            )
            nodes.append({
                "id": node_id,
                "type": "ACTION",
                "label": f"[Требует уточнения] {opt}",
                "question": None,
                "options": [],
                "action_details": {
                    "procedure": f"Требует уточнения для варианта: {opt}",
                    "implant": None,
                    "timing": None,
                    "evidence_level": None,
                    "contraindications": [],
                    "notes": "Автоматически добавлено — необходимо уточнить тактику",
                },
            })
            node_type[node_id] = "ACTION"
            data["edges"].append({
                "id": edge_to_id,
                "from": nid, "to": node_id,
                "label": opt, "condition": None,
            })
            data["edges"].append({
                "id": edge_end_id,
                "from": node_id, "to": global_end,
                "label": None, "condition": None,
            })
            out_map[nid].append(data["edges"][-2])

    # Ensure END exists (re-check after possible additions above)
    end_nodes = [n["id"] for n in nodes if n.get("type") == "END"]
    global_end = end_nodes[0]

    # Add → END for ACTION/WARNING nodes that lack it
    has_end_edge = {e["from"] for e in data["edges"] if node_type.get(e.get("to")) == "END"}
    counter_end = sum(1 for e in data["edges"]
                      if e.get("id", "").startswith("edge_post_fix_")) + 1

    for n in nodes:
        nid, ntype = n["id"], n.get("type")
        if ntype in ("ACTION", "WARNING") and nid not in has_end_edge:
            eid = f"edge_post_fix_{counter_end:03d}"
            counter_end += 1
            data["edges"].append({
                "id": eid, "from": nid, "to": global_end,
                "label": None, "condition": None,
            })
            has_end_edge.add(nid)

    data["nodes"] = nodes
    return data


class Stage5bFix(BasePipelineStage):
    stage_name = "stage5b_fix"

    def build_prompt(
        self,
        graph: dict,
        structural_issues: list[dict],
        clinical_issues: list[dict],
        missing_scenarios: list[dict],
        source_text: str,
    ) -> str:
        g = graph if "nodes" in graph else graph.get("graph", graph)
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])
        all_clinical = clinical_issues + [
            {"severity": "info", **ms} for ms in missing_scenarios
            if ms.get("importance") in ("high", "medium")
        ]
        return _PROMPT.format(
            nodes_json=json.dumps(nodes, ensure_ascii=False, indent=2)[:5000],
            edges_json=json.dumps(edges, ensure_ascii=False, indent=2)[:3500],
            issues_text=_format_issues(structural_issues, all_clinical),
            source_text=source_text[:2000],
            algorithm_json=json.dumps(
                self._algorithm.get("algorithm", self._algorithm) if self._algorithm else {},
                ensure_ascii=False,
            )[:3000],
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "nodes" not in data or "edges" not in data:
            raise PipelineError("Stage5b: missing 'nodes' or 'edges'")
        return data

    def run(
        self,
        graph: dict,
        validation: dict,
        source_text: str,
        algorithm: dict | None = None,
    ) -> dict:
        self._algorithm = algorithm  # store for build_prompt access
        clinical_issues = validation.get("issues", [])
        missing_scenarios = validation.get("missing_scenarios", [])

        # Normalise graph to flat {nodes, edges}
        current = deepcopy(graph)
        if "graph" in current:
            current = current["graph"]

        for loop in range(MAX_FIX_LOOPS):
            graph_doc = {"graph": current}
            structural = validate(graph_doc)
            crits = [i for i in structural if i["severity"] == "critical"]
            clin_act = [i for i in clinical_issues if i["severity"] in ("critical", "warning")]
            imp_miss = [m for m in missing_scenarios if m.get("importance") in ("high", "medium")]

            if not (crits or clin_act or imp_miss):
                logger.info(f"Stage 5b loop {loop + 1}: no issues — done")
                break

            logger.info(
                f"Stage 5b loop {loop + 1}/{MAX_FIX_LOOPS}: "
                f"{len(crits)} structural, {len(clin_act)} clinical"
            )
            prompt = self.build_prompt(
                current, structural, clinical_issues, missing_scenarios, source_text
            )
            fixed = self._execute_with_retry(prompt)
            fixed = _post_process(fixed)
            current = {"nodes": fixed["nodes"], "edges": fixed["edges"]}

        # Deterministic: remove unreachable nodes
        current = _remove_unreachable(current)

        # Final deterministic pass
        final = _post_process({"nodes": current["nodes"], "edges": current["edges"]})

        return {
            "nodes": final["nodes"],
            "edges": final["edges"],
        }