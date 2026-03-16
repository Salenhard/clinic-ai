import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_PROMPT = """\
Преобразуй структурированный алгоритм и данные о методах лечения в ГРАФ принятия решений.

АЛГОРИТМ (ветки и логика):
{algorithm_json}

МЕТОДЫ ОСТЕОСИНТЕЗА:
{osteosynthesis_json}

МЕТОДЫ ЭНДОПРОТЕЗИРОВАНИЯ:
{arthroplasty_json}

ЛЕЧЕНИЕ ПО ТИПАМ ПЕРЕЛОМОВ:
{fracture_json}

ПРОТИВОПОКАЗАНИЯ:
{contraindications_json}

ПРАВИЛА УЗЛОВ:
• START  — ровно один; question=null, options=[], action_details=null
• DECISION — question + options (≥2); action_details=null
• ACTION — question=null, options=[]; action_details ОБЯЗАТЕЛЕН с полями:
  procedure, implant, timing, notes
• WARNING — question=null, options=[]; идёт ДО ACTION (не после)
• END — question=null, options=[], action_details=null
• Сначало должен определятся возвраст, а затем тип перелома, далее идут остальные дополнительные вопросы
ПРАВИЛА РЁБЕР:
• START → первый DECISION (condition=null, label=null)
• Каждый вариант из DECISION.options → РОВНО ОДНО исходящее ребро
  label ребра ДОЛЖЕН точно совпадать с текстом варианта из options
• Если вариант из options не имеет ребра — это ОШИБКА (тупик)
• ACTION → END (condition=null, единственное исходящее ребро)
• WARNING → ACTION → END
• одинаковые from+to, нет самопетель, нет циклов
• Все узлы достижимы из START

Верни СТРОГО JSON (только nodes + edges):
{{
  "nodes": [
    {{
      "id": "start_001",
      "type": "START | DECISION | ACTION | WARNING | END",
      "label": "...",
      "question": "текст или null",
      "options": [],
      "action_details": null
    }}
  ],
  "edges": [
    {{
      "id": "edge_001",
      "from": "...",
      "to": "...",
      "label": "...",
      "condition": {{"field": "...", "operator": "==", "value": "..."}}
    }}
  ]
}}"""


class Stage4Graph(BasePipelineStage):
    stage_name = "stage4_graph"

    def build_prompt(
        self,
        algorithm: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> str:
        def _j(d, key, n=2500):
            return json.dumps(d.get(key, []), ensure_ascii=False)[:n]

        return _PROMPT.format(
            algorithm_json=json.dumps(algorithm.get("algorithm", {}), ensure_ascii=False)[:4000],
            osteosynthesis_json=_j(osteosynthesis, "osteosynthesis_methods"),
            arthroplasty_json=_j(arthroplasty, "arthroplasty_methods"),
            fracture_json=_j(fracture_treatments, "fracture_treatments"),
            contraindications_json=_j(factors, "contraindications"),
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "nodes" not in data or "edges" not in data:
            raise PipelineError("Stage4: missing 'nodes' or 'edges'")

        # Enforce START constraints
        starts = [n for n in data["nodes"] if n.get("type") == "START"]
        if not starts:
            data["nodes"].insert(0, {
                "id": "start_001", "type": "START", "label": "Начало консультации",
                "question": None, "options": [], "action_details": None,
            })
        for n in data["nodes"]:
            if n.get("type") == "START":
                n["question"] = None
                n["options"] = []
                n["action_details"] = None

        # Remove self-loops and duplicates
        seen: set[tuple] = set()
        clean = []
        for e in data["edges"]:
            pair = (e.get("from"), e.get("to"))
            if pair[0] == pair[1]:
                logger.warning(f"Stage4: self-loop {e.get('id')} removed")
                continue
            if pair in seen:
                logger.warning(f"Stage4: duplicate {pair} removed")
                continue
            seen.add(pair)
            clean.append(e)
        data["edges"] = clean

        node_type = {n["id"]: n.get("type") for n in data["nodes"]}
        n_nodes = len(data["nodes"])
        n_actions = sum(1 for n in data["nodes"] if n.get("type") == "ACTION")

        # Warn if too few ACTION nodes (likely collapsed branches)
        if n_actions < 8:
            logger.warning(
                f"Stage4: only {n_actions} ACTION nodes — graph likely has collapsed "
                f"branches. Expected ≥ 12 for full Pipkin/Garden/Чрезвертельные coverage."
            )

        # Detect ACTION nodes shared between multiple incoming sources (nozology collapse)
        incoming: dict[str, list[str]] = {}
        for e in data["edges"]:
            dst = e.get("to")
            src = e.get("from")
            if dst and src:
                incoming.setdefault(dst, []).append(src)
        for nid, srcs in incoming.items():
            if len(srcs) > 1 and node_type.get(nid) == "ACTION":
                logger.warning(
                    f"Stage4: ACTION '{nid}' reached from {len(srcs)} sources {srcs} — "
                    f"possible nozology collapse (same ACTION for different fracture types)."
                )

        # Check DECISION options coverage
        out_edges: dict[str, list[dict]] = {}
        for e in data["edges"]:
            out_edges.setdefault(e.get("from"), []).append(e)

        for n in data["nodes"]:
            if n.get("type") != "DECISION":
                continue
            opts = n.get("options") or []
            out = out_edges.get(n["id"], [])
            edge_labels = {(e.get("label") or "").strip().lower() for e in out}
            edge_values: set[str] = set()
            for e in out:
                cond = e.get("condition") or {}
                v = cond.get("value")
                if v:
                    edge_values.add(str(v).lower())
            uncovered = [
                o for o in opts
                if o.strip().lower() not in edge_labels
                and o.strip().lower() not in edge_values
            ]
            if uncovered:
                logger.warning(
                    f"Stage4: DECISION '{n['id']}' has uncovered options: {uncovered} — "
                    f"these will create dead ends."
                )

        logger.info(f"Stage 4: {n_nodes} nodes ({n_actions} actions), {len(data['edges'])} edges")
        return data

    def run(
        self,
        algorithm: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> dict:
        prompt = self.build_prompt(algorithm, osteosynthesis, arthroplasty,
                                    fracture_treatments, factors)
        return self._execute_with_retry(prompt)
