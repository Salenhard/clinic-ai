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
• WARNING: идёт ДО ACTION (DECISION → WARNING → ACTION → END), НЕ после
• END: question=null, options=[], action_details=null

РЁБРА:
• Каждый вариант DECISION.options → ровно одно ребро (label == вариант)
• ACTION → END (condition=null, единственное исходящее)
• Нет дублей, нет самопетель, нет циклов, все узлы достижимы

ПАРАМЕТРЫ:
• Параметр, нужный в нескольких ветках → один DECISION-узел в начале
• НЕЛЬЗЯ: ребро проверяет поле, уже определённое на пути к этому узлу
• НЕЛЬЗЯ: один DECISION-узел доступен из двух веток с разным значением одного параметра

Новые id: node_fix_NNN / edge_fix_NNN
Сохрани существующие id без изменений.

Верни СТРОГО JSON:
{{
  "nodes": [...],
  "edges": [...],
  "changelog": [
    {{"action": "added|modified|removed", "element": "node|edge", "id": "...", "reason": "..."}}
  ]
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

    # Ensure END exists
    end_nodes = [n["id"] for n in nodes if n.get("type") == "END"]
    if not end_nodes:
        end_id = "node_end"
        nodes.append({
            "id": end_id, "type": "END", "label": "Завершение",
            "question": None, "options": [], "action_details": None,
        })
        node_type[end_id] = "END"
        end_nodes = [end_id]

    global_end = end_nodes[0]
    has_end_edge = {e["from"] for e in data["edges"] if node_type.get(e.get("to")) == "END"}
    counter = sum(1 for e in data["edges"]
                  if e.get("id", "").startswith("edge_post_fix_")) + 1

    for n in nodes:
        nid, ntype = n["id"], n.get("type")
        if ntype in ("ACTION", "WARNING") and nid not in has_end_edge:
            eid = f"edge_post_fix_{counter:03d}"
            counter += 1
            data["edges"].append({
                "id": eid, "from": nid, "to": global_end,
                "label": "Завершение ветки", "condition": None,
            })
            has_end_edge.add(nid)
            data.setdefault("changelog", []).append({
                "action": "added", "element": "edge", "id": eid,
                "reason": f"Автоисправление: {ntype} '{nid}' не имел ребра к END",
            })

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
    ) -> dict:
        clinical_issues = validation.get("issues", [])
        missing_scenarios = validation.get("missing_scenarios", [])

        # Normalise graph to flat {nodes, edges}
        current = deepcopy(graph)
        if "graph" in current:
            current = current["graph"]

        accumulated_changelog: list[dict] = []

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
            accumulated_changelog.extend(fixed.get("changelog", []))
            current = {"nodes": fixed["nodes"], "edges": fixed["edges"]}

        # Final deterministic pass
        final = _post_process({"nodes": current["nodes"], "edges": current["edges"], "changelog": []})
        accumulated_changelog.extend(final.get("changelog", []))

        return {
            "nodes": final["nodes"],
            "edges": final["edges"],
            "changelog": accumulated_changelog,
        }
