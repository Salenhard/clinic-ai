"""
Stage 7: Graph repair based on Stage 6 verification report.

Output schema is IDENTICAL to Stage 5:
  {
    "metadata": { ... },
    "graph": { "nodes": [...], "edges": [...] }
  }

"changelog" is attached at the top level for metrics only.
"""
import json
import logging
from copy import deepcopy

from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Ты — эксперт по клиническим алгоритмам. Исправь граф принятия клинических \
решений на основе результатов экспертной проверки.

═══════════════════════════════════════════
ТЕКУЩИЙ ГРАФ:
═══════════════════════════════════════════
УЗЛЫ:
{nodes_json}

РЁБРА:
{edges_json}

═══════════════════════════════════════════
КЛИНИЧЕСКИЕ ПРОБЛЕМЫ (Stage 6):
═══════════════════════════════════════════
Оценка точности: {accuracy_score}

Проблемы для исправления:
{issues_text}

Непокрытые сценарии для добавления:
{missing_text}

═══════════════════════════════════════════
ИСХОДНЫЙ ТЕКСТ (справочно):
═══════════════════════════════════════════
{source_text}

═══════════════════════════════════════════
ОБЯЗАТЕЛЬНЫЕ СТРУКТУРНЫЕ ПРАВИЛА:
═══════════════════════════════════════════

УЗЛЫ:
1. START — ровно один, question=null, options=[], action_details=null
2. DECISION — имеет question и options; action_details=null
3. ACTION — question=null, options=[], action_details обязателен
4. WARNING — question=null, options=[], может иметь action_details или null
5. END — question=null, options=[], action_details=null; может быть несколько

РЁБРА:
6. START → первый DECISION (одно ребро, condition=null)
7. Каждый вариант из DECISION.options → ровно одно исходящее ребро
8. После каждого ACTION → ребро к END (condition=null)
9. После каждого WARNING → ребро к END (condition=null)
10. Нет дублирующих рёбер (одинаковые from+to)
11. Нет самопетель (from == to)
12. Нет циклов
13. Все узлы достижимы из START
14. Условия рёбер из одного DECISION-узла взаимоисключающие

ИСПРАВЛЕНИЯ:
- Исправь все clinical проблемы из отчёта Stage 6
- Добавь недостающие сценарии (важность high/medium)
- Сохрани id существующих узлов/рёбер
- Новые узлы: node_fix_001, node_fix_002, ...
- Новые рёбра: edge_fix_001, edge_fix_002, ...

Верни СТРОГО JSON — только nodes, edges, changelog (без metadata):
{{
  "nodes": [
    {{
      "id": "node_001",
      "type": "START | DECISION | ACTION | WARNING | END",
      "label": "...",
      "question": "текст вопроса или null",
      "options": ["вариант А", "вариант Б"],
      "action_details": {{
        "procedure": "...",
        "implant": "...",
        "timing": "...",
        "evidence_level": "...",
        "contraindications": [],
        "notes": "..."
      }}
    }}
  ],
  "edges": [
    {{
      "id": "edge_001",
      "from": "node_001",
      "to": "node_002",
      "label": "текст условия",
      "condition": {{
        "field": "fracture_type",
        "operator": "==",
        "value": "Garden I-II"
      }}
    }}
  ],
  "changelog": [
    {{
      "action": "modified | added | removed",
      "element": "node | edge",
      "id": "...",
      "reason": "краткое объяснение"
    }}
  ]
}}
"""


def _format_issues(issues: list) -> str:
    actionable = [i for i in issues if i.get("severity") in ("critical", "warning")]
    if not actionable:
        return "  (нет критических или важных проблем)"
    lines = []
    for i, iss in enumerate(actionable, 1):
        sev = iss.get("severity", "").upper()
        node = f" [узел: {iss['node_id']}]" if iss.get("node_id") else ""
        lines.append(
            f"  {i}. [{sev}]{node} {iss.get('description', '')}\n"
            f"     → {iss.get('suggestion', 'не указано')}"
        )
    return "\n".join(lines)


def _format_missing(scenarios: list) -> str:
    important = [m for m in scenarios if m.get("importance") in ("high", "medium")]
    if not important:
        return "  (все важные сценарии покрыты)"
    lines = []
    for i, ms in enumerate(important, 1):
        imp = ms.get("importance", "").upper()
        lines.append(f"  {i}. [{imp}] {ms.get('description', '')}")
    return "\n".join(lines)


def _post_process(data: dict) -> dict:
    """
    Lightweight deterministic fixes applied AFTER LLM response
    to catch common structural violations.
    """
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    node_map = {n["id"]: n for n in nodes}
    node_types = {n["id"]: n.get("type") for n in nodes}

    # ── Fix 1: START must not have question/options ───────────────────────────
    for node in nodes:
        if node.get("type") == "START":
            node["question"] = None
            node["options"] = []
            node["action_details"] = None

    # ── Fix 2: Remove duplicate edges (same from+to) ──────────────────────────
    seen_pairs: set[tuple] = set()
    clean_edges = []
    for edge in edges:
        pair = (edge.get("from"), edge.get("to"))
        if pair[0] == pair[1]:
            logger.warning(f"Stage7 post-process: self-loop edge {edge.get('id')} removed")
            continue
        if pair in seen_pairs:
            logger.warning(f"Stage7 post-process: duplicate edge {pair} removed")
            continue
        seen_pairs.add(pair)
        clean_edges.append(edge)
    data["edges"] = clean_edges
    edges = clean_edges

    # ── Fix 3: Ensure every ACTION/WARNING has an edge to an END node ─────────
    end_nodes = [n["id"] for n in nodes if n.get("type") == "END"]
    if not end_nodes:
        # Create a global END node if none exists
        end_id = "node_end"
        nodes.append({
            "id": end_id,
            "type": "END",
            "label": "Завершение",
            "question": None,
            "options": [],
            "action_details": None,
        })
        node_types[end_id] = "END"
        end_nodes = [end_id]
        logger.warning("Stage7 post-process: no END node found, created node_end")

    global_end = end_nodes[0]
    sources_to_end = {e["from"] for e in edges if node_types.get(e["to"]) == "END"}

    fix_edge_counter = 1
    for node in nodes:
        nid = node["id"]
        ntype = node.get("type")
        if ntype in ("ACTION", "WARNING") and nid not in sources_to_end:
            new_edge_id = f"edge_post_fix_{fix_edge_counter:03d}"
            fix_edge_counter += 1
            data["edges"].append({
                "id": new_edge_id,
                "from": nid,
                "to": global_end,
                "label": "Завершение ветки",
                "condition": None,
            })
            sources_to_end.add(nid)
            logger.warning(
                f"Stage7 post-process: added missing END edge {new_edge_id} "
                f"from {nid} ({ntype})"
            )
            data.setdefault("changelog", []).append({
                "action": "added",
                "element": "edge",
                "id": new_edge_id,
                "reason": f"Автоисправление: {ntype}-узел {nid} не имел ребра к END",
            })

    data["nodes"] = nodes
    return data


class Stage7Fix(BasePipelineStage):
    stage_name = "stage7_fix"

    def build_prompt(self, graph: dict, verification: dict, source_text: str) -> str:
        nodes = graph.get("graph", {}).get("nodes", [])
        edges = graph.get("graph", {}).get("edges", [])
        return _PROMPT_TEMPLATE.format(
            nodes_json=json.dumps(nodes, ensure_ascii=False, indent=2)[:6000],
            edges_json=json.dumps(edges, ensure_ascii=False, indent=2)[:4000],
            accuracy_score=verification.get("clinical_accuracy_score", "N/A"),
            issues_text=_format_issues(verification.get("issues", [])),
            missing_text=_format_missing(verification.get("missing_scenarios", [])),
            source_text=source_text[:3000],
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)

        if "nodes" not in data or "edges" not in data:
            raise PipelineError("Stage7: response must contain 'nodes' and 'edges'")

        # Apply deterministic post-processing
        data = _post_process(data)

        changelog = data.get("changelog", [])
        added    = sum(1 for c in changelog if c.get("action") == "added")
        modified = sum(1 for c in changelog if c.get("action") == "modified")
        removed  = sum(1 for c in changelog if c.get("action") == "removed")
        logger.info(
            f"Stage 7: {len(data['nodes'])} nodes, {len(data['edges'])} edges | "
            f"changelog: +{added} ~{modified} -{removed}"
        )
        return data

    def _has_actionable_issues(self, verification: dict) -> bool:
        issues  = verification.get("issues", [])
        missing = verification.get("missing_scenarios", [])
        return (
            any(i.get("severity") in ("critical", "warning") for i in issues)
            or any(m.get("importance") in ("high", "medium") for m in missing)
        )

    def run(self, graph: dict, verification: dict, source_text: str) -> dict:
        """
        Returns a graph document in the SAME schema as Stage 5 output:
          { "metadata": {...}, "graph": { "nodes": [...], "edges": [...] } }
        "changelog" is appended at the top level for metrics.
        """
        if not self._has_actionable_issues(verification):
            logger.info("Stage 7: no actionable issues — running structural check only")
            # Still run post-processing on the existing graph
            dummy = {
                "nodes": graph["graph"]["nodes"],
                "edges": graph["graph"]["edges"],
                "changelog": [],
            }
            fixed = _post_process(dummy)
            result = deepcopy(graph)
            result["graph"]["nodes"] = fixed["nodes"]
            result["graph"]["edges"] = fixed["edges"]
            result["changelog"] = fixed.get("changelog", [])
            return result

        issues  = [i for i in verification.get("issues", []) if i.get("severity") in ("critical", "warning")]
        missing = [m for m in verification.get("missing_scenarios", []) if m.get("importance") in ("high", "medium")]
        logger.info(f"Stage 7: fixing {len(issues)} issues, {len(missing)} missing scenarios")

        prompt = self.build_prompt(graph, verification, source_text)
        fixed  = self._execute_with_retry(prompt)

        # Assemble final document in Stage 5 schema
        result = deepcopy(graph)
        result["graph"]["nodes"] = fixed["nodes"]
        result["graph"]["edges"] = fixed["edges"]
        result["metadata"]["version"] = self._bump_version(
            result["metadata"].get("version", "1.0")
        )
        result["changelog"] = fixed.get("changelog", [])
        return result

    @staticmethod
    def _bump_version(version: str) -> str:
        try:
            major, minor = version.split(".")
            return f"{major}.{int(minor) + 1}"
        except Exception:
            return version + ".1"
