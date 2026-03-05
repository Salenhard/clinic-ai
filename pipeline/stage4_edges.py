"""Stage 4: Build graph edges between nodes."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по построению клинических алгоритмов принятия решений.

Для построенных узлов графа определи ВСЕ рёбра (переходы между узлами).

УЗЛЫ ГРАФА:
{nodes_json}

Правила построения рёбер:
1. Каждый DECISION-узел должен иметь исходящее ребро для КАЖДОГО варианта ответа
2. ACTION и WARNING узлы ведут к следующему шагу или END
3. START ведёт к первому DECISION-узлу
4. Граф должен быть СВЯЗНЫМ — недостижимых узлов быть не должно
5. НЕТ ЦИКЛОВ — это дерево решений, не конечный автомат
6. Для каждого ребра укажи понятное человеку условие перехода

Верни СТРОГО JSON без пояснений, без markdown-блоков:
{{
  "edges": [
    {{
      "id": "edge_001",
      "from": "node_001",
      "to": "node_002",
      "label": "текст условия перехода (например: 'Возраст < 60 лет')",
      "condition": {{
        "field": "age | fracture_type | activity_level | cognitive_status | bone_quality | ...",
        "operator": "> | < | >= | <= | == | in | yes | no",
        "value": "60 | Garden_III | активный | ..."
      }}
    }}
  ]
}}"""


class Stage4Edges(BasePipelineStage):
    stage_name = "stage4_edges"

    def build_prompt(self, nodes: dict) -> str:
        return PROMPT_TEMPLATE.format(
            nodes_json=json.dumps(nodes, ensure_ascii=False, indent=2)[:8000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "edges" not in data:
            raise PipelineError("Stage4: missing 'edges' key in response")
        logger.info(f"Stage 4: built {len(data['edges'])} edges")
        return data

    def run(self, nodes: dict) -> dict:
        prompt = self.build_prompt(nodes)
        return self._execute_with_retry(prompt)
