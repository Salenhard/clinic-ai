"""
Stage 7: Graph repair — validate → fix loop.

Flow:
  1. Run structural validator  →  list of issues
  2. Combine with Stage 6 clinical issues
  3. Send to LLM with explicit fix instructions
  4. Run deterministic _post_process on LLM response
  5. Re-validate; if critical issues remain and attempts left → repeat (max 2 loops)

Output schema is IDENTICAL to Stage 5:
  { "metadata": {...}, "graph": { "nodes": [...], "edges": [...] } }
"changelog" is attached at top level for metrics only.
"""
import json
import logging
from copy import deepcopy
from typing import Optional

from .base import BasePipelineStage, PipelineError
from .graph_validator import validate

logger = logging.getLogger(__name__)

MAX_FIX_LOOPS = 2   # max LLM repair attempts


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Ты — эксперт по клиническим алгоритмам. Исправь граф принятия решений.

═══════════════════════════════════════════
ТЕКУЩИЙ ГРАФ:
═══════════════════════════════════════════
УЗЛЫ:
{nodes_json}

РЁБРА:
{edges_json}

═══════════════════════════════════════════
ПРОБЛЕМЫ, КОТОРЫЕ НУЖНО ИСПРАВИТЬ:
═══════════════════════════════════════════
{all_issues_text}

═══════════════════════════════════════════
ИСХОДНЫЙ ТЕКСТ (справочно):
═══════════════════════════════════════════
{source_text}

═══════════════════════════════════════════
ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ГРАФА:
═══════════════════════════════════════════
УЗЛЫ:
• START: ровно один; question=null, options=[], action_details=null
• DECISION: имеет question + минимум 2 options; action_details=null
• ACTION: question=null, options=[]; action_details обязателен
• WARNING: question=null, options=[]
• END: question=null, options=[], action_details=null; может быть несколько

РЁБРА:
• START → первый DECISION (одно безусловное ребро)
• Каждый вариант из DECISION.options → ровно одно исходящее ребро
  (метка ребра или condition.value должны точно соответствовать варианту)
• После каждого ACTION → ребро к END (condition=null)
• После каждого WARNING → ребро к END (condition=null)
• WARNING-узел должен идти ДО ACTION, а не после:
  ПРАВИЛЬНО: DECISION → WARNING → ACTION → END
  НЕПРАВИЛЬНО: DECISION → ACTION → WARNING → END
• Нет дублирующих рёбер (одинаковые from+to)
• Нет самопетель
• Нет циклов
• Все узлы достижимы из START

УСЛОВИЯ (condition):
• Если DECISION ветвится по двум параметрам (например, тип перелома И возраст),
  используй вложенные DECISION-узлы — не пытайся объединить два условия в одном ребре
• НЕ добавляй ребро с условием по полю F из узла N, если поле F уже было
  определено на пути к узлу N (противоречивое условие)
• НЕЛЬЗЯ направлять две разные ветки возраста в один и тот же DECISION-узел
  типа перелома — создай отдельный узел для каждой возрастной группы
• Условия рёбер из одного DECISION взаимоисключающие

Сохрани id существующих узлов/рёбер без изменений.
Новые узлы: node_fix_001, node_fix_002, ...  
Новые рёбра: edge_fix_001, edge_fix_002, ...

Верни СТРОГО JSON (только nodes + edges + changelog, без metadata):
{{
  "nodes": [
    {{
      "id": "...",
      "type": "START|DECISION|ACTION|WARNING|END",
      "label": "...",
      "question": "текст или null",
      "options": [],
      "action_details": {{
        "procedure": "...", "implant": "...", "timing": "...",
        "evidence_level": "...", "contraindications": [], "notes": "..."
      }}
    }}
  ],
  "edges": [
    {{
      "id": "...", "from": "...", "to": "...",
      "label": "...",
      "condition": {{"field": "...", "operator": "...", "value": "..."}}
    }}
  ],
  "changelog": [
    {{"action": "modified|added|removed", "element": "node|edge",
      "id": "...", "reason": "..."}}
  ]
}}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_all_issues(structural: list[dict], clinical: list[dict]) -> str:
    lines = []
    if structural:
        lines.append("СТРУКТУРНЫЕ проблемы:")
        for i, iss in enumerate(structural, 1):
            nid = f" [узел: {iss['node_id']}]" if iss.get("node_id") else ""
            lines.append(f"  {i}. [{iss['severity'].upper()}]{nid} {iss['description']}")
            lines.append(f"     → {iss['suggestion']}")
    else:
        lines.append("Структурных проблем нет.")

    actionable_clinical = [
        c for c in clinical if c.get("severity") in ("critical", "warning")
    ]
    if actionable_clinical:
        lines.append("\nКЛИНИЧЕСКИЕ проблемы (из Stage 6):")
        for i, iss in enumerate(actionable_clinical, 1):
            nid = f" [узел: {iss.get('node_id')}]" if iss.get("node_id") else ""
            lines.append(f"  {i}. [{iss['severity'].upper()}]{nid} {iss['description']}")
            lines.append(f"     → {iss.get('suggestion', 'не указано')}")

    missing = [c for c in clinical if c.get("importance") in ("high", "medium")]
    if missing:
        lines.append("\nНЕПОКРЫТЫЕ СЦЕНАРИИ (добавить):")
        for i, ms in enumerate(missing, 1):
            lines.append(f"  {i}. [{ms.get('importance','').upper()}] {ms.get('description','')}")

    return "\n".join(lines) if lines else "Проблем не выявлено."


def _post_process(data: dict, node_type: dict) -> dict:
    """
    Deterministic fixes applied after every LLM response.
    - Remove duplicate edges
    - Remove self-loops
    - Ensure every ACTION/WARNING has edge to END
    - Ensure START has no question/options
    """
    nodes: list = data.get("nodes", [])
    edges: list = data.get("edges", [])

    # Rebuild node_type map from current nodes
    nt = {n["id"]: n.get("type", "") for n in nodes}

    # Fix START fields
    for n in nodes:
        if n.get("type") == "START":
            n["question"] = None
            n["options"] = []
            n["action_details"] = None

    # Remove duplicate/self-loop edges
    seen: set[tuple] = set()
    clean_edges = []
    for e in edges:
        pair = (e.get("from"), e.get("to"))
        if pair[0] == pair[1]:
            logger.debug(f"post_process: self-loop {e.get('id')} removed")
            continue
        if pair in seen:
            logger.debug(f"post_process: duplicate edge {pair} removed")
            continue
        seen.add(pair)
        clean_edges.append(e)
    data["edges"] = clean_edges

    # Ensure END node exists
    end_nodes = [n["id"] for n in nodes if n.get("type") == "END"]
    if not end_nodes:
        end_id = "node_end"
        nodes.append({
            "id": end_id, "type": "END", "label": "Завершение",
            "question": None, "options": [], "action_details": None,
        })
        nt[end_id] = "END"
        end_nodes = [end_id]
        logger.warning("post_process: no END node — created node_end")

    global_end = end_nodes[0]
    sources_to_end = {e["from"] for e in data["edges"] if nt.get(e.get("to")) == "END"}

    counter = sum(
        1 for e in data["edges"] if e.get("id", "").startswith("edge_post_fix_")
    ) + 1

    for n in nodes:
        nid, ntype = n["id"], n.get("type")
        if ntype in ("ACTION", "WARNING") and nid not in sources_to_end:
            new_id = f"edge_post_fix_{counter:03d}"
            counter += 1
            data["edges"].append({
                "id": new_id, "from": nid, "to": global_end,
                "label": "Завершение ветки", "condition": None,
            })
            sources_to_end.add(nid)
            data.setdefault("changelog", []).append({
                "action": "added", "element": "edge", "id": new_id,
                "reason": f"Автоисправление: {ntype} '{nid}' не имел ребра к END",
            })
            logger.warning(f"post_process: added END edge {new_id} from {nid}")

    data["nodes"] = nodes
    return data


# ── Stage class ───────────────────────────────────────────────────────────────

class Stage7Fix(BasePipelineStage):
    stage_name = "stage7_fix"

    def build_prompt(
        self,
        graph: dict,
        structural_issues: list[dict],
        clinical_issues: list[dict],
        missing_scenarios: list[dict],
        source_text: str,
    ) -> str:
        nodes = graph.get("graph", {}).get("nodes", [])
        edges = graph.get("graph", {}).get("edges", [])

        all_clinical = clinical_issues + [
            {"severity": "info", **ms} for ms in missing_scenarios
            if ms.get("importance") in ("high", "medium")
        ]

        return _PROMPT_TEMPLATE.format(
            nodes_json=json.dumps(nodes, ensure_ascii=False, indent=2)[:6000],
            edges_json=json.dumps(edges, ensure_ascii=False, indent=2)[:4000],
            all_issues_text=_format_all_issues(structural_issues, all_clinical),
            source_text=source_text[:3000],
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "nodes" not in data or "edges" not in data:
            raise PipelineError("Stage7: response must contain 'nodes' and 'edges'")
        return data

    def _assemble(self, original_graph: dict, fixed_data: dict, all_changelog: list) -> dict:
        """Build final document in Stage-5 schema."""
        result = deepcopy(original_graph)
        result["graph"]["nodes"] = fixed_data["nodes"]
        result["graph"]["edges"] = fixed_data["edges"]
        result["metadata"]["version"] = self._bump_version(
            result["metadata"].get("version", "1.0")
        )
        result["changelog"] = all_changelog
        return result

    @staticmethod
    def _bump_version(v: str) -> str:
        try:
            major, minor = v.split(".")
            return f"{major}.{int(minor) + 1}"
        except Exception:
            return v + ".1"

    def run(self, graph: dict, verification: dict, source_text: str) -> dict:
        """
        Validate → fix loop (up to MAX_FIX_LOOPS LLM calls).
        Always returns a document in Stage-5 schema.
        """
        clinical_issues  = verification.get("issues", [])
        missing_scenarios = verification.get("missing_scenarios", [])

        # Work on a mutable copy
        current = deepcopy(graph)
        node_type_map = {n["id"]: n.get("type") for n in current["graph"]["nodes"]}
        accumulated_changelog: list[dict] = []

        for loop in range(MAX_FIX_LOOPS):
            structural_issues = validate(current)
            critical_structural = [i for i in structural_issues if i["severity"] == "critical"]
            actionable_clinical = [i for i in clinical_issues if i["severity"] in ("critical", "warning")]
            important_missing   = [m for m in missing_scenarios if m.get("importance") in ("high", "medium")]

            has_work = bool(critical_structural or actionable_clinical or important_missing)

            if not has_work:
                logger.info(f"Stage 7 loop {loop + 1}: no issues — done")
                break

            logger.info(
                f"Stage 7 loop {loop + 1}/{MAX_FIX_LOOPS}: "
                f"{len(critical_structural)} structural, "
                f"{len(actionable_clinical)} clinical issues"
            )

            prompt = self.build_prompt(
                current, structural_issues, clinical_issues, missing_scenarios, source_text
            )
            fixed = self._execute_with_retry(prompt)

            # Deterministic post-processing
            fixed = _post_process(fixed, node_type_map)

            # Accumulate changelog
            accumulated_changelog.extend(fixed.get("changelog", []))

            # Update current graph
            current["graph"]["nodes"] = fixed["nodes"]
            current["graph"]["edges"] = fixed["edges"]
            node_type_map = {n["id"]: n.get("type") for n in fixed["nodes"]}

            # After last loop, run post-process only (no more LLM calls)
            if loop == MAX_FIX_LOOPS - 1:
                remaining = validate(current)
                remaining_critical = [i for i in remaining if i["severity"] == "critical"]
                if remaining_critical:
                    logger.warning(
                        f"Stage 7: {len(remaining_critical)} critical issues remain after "
                        f"{MAX_FIX_LOOPS} fix loops"
                    )
                    for iss in remaining_critical:
                        logger.warning(f"  [{iss['severity'].upper()}] {iss['description']}")
        else:
            # Loop exhausted without clean validation
            logger.warning("Stage 7: fix loops exhausted")

        # Final pass: always run post_process to catch anything LLM missed
        dummy = {
            "nodes":     current["graph"]["nodes"],
            "edges":     current["graph"]["edges"],
            "changelog": [],
        }
        dummy = _post_process(dummy, node_type_map)
        accumulated_changelog.extend(dummy.get("changelog", []))
        current["graph"]["nodes"] = dummy["nodes"]
        current["graph"]["edges"] = dummy["edges"]

        return self._assemble(graph, {"nodes": current["graph"]["nodes"],
                                       "edges": current["graph"]["edges"]},
                              accumulated_changelog)
