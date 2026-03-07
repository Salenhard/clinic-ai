"""Stage 4: Generate the formal graph JSON from the algorithm branches.

This stage converts the structured algorithm (Stage 3) + all extracted data
into the canonical node/edge graph format.
"""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_PROMPT = """\
Преобразуй структурированный алгоритм и данные о методах лечения в ГРАФ принятия решений.

═══════════════════════════════════════════
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
═══════════════════════════════════════════

ПРАВИЛА УЗЛОВ:
• START  — ровно один; question=null, options=[], action_details=null
• DECISION — question + options (≥2); action_details=null
• ACTION — question=null, options=[]; action_details обязателен
• WARNING — question=null, options=[]; используй для предупреждений о запрещённых имплантатах
• END — question=null, options=[], action_details=null; может быть несколько

ПРАВИЛА РЁБЕР:
• START → первый DECISION (condition=null)
• Каждый вариант DECISION.options → ровно одно ребро; label == вариант из options
• ACTION → END (condition=null)
• WARNING → ACTION → END (предупреждение ДО действия, не после)
• Нет дублей (одинаковые from+to), нет самопетель, нет циклов

ПРАВИЛО ПАРАМЕТРОВ:
• Параметр, нужный в нескольких ветках → один DECISION-узел в начале общей ветки
• Параметр, нужный только в одной ветке → DECISION-узел внутри этой ветки
• НЕЛЬЗЯ: один DECISION-узел достигается из двух веток с разным значением того же параметра
• НЕЛЬЗЯ: ребро проверяет поле, уже определённое на пути к этому узлу

ACTION.action_details:
{{
  "procedure": "...", "implant": "...", "timing": "...",
  "evidence_level": "...", "contraindications": [], "notes": "..."
}}

Верни СТРОГО JSON (без metadata — только graph):
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

        node_types = {n["id"]: n.get("type") for n in data["nodes"]}
        n_nodes = len(data["nodes"])
        n_actions = sum(1 for n in data["nodes"] if n.get("type") == "ACTION")
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
