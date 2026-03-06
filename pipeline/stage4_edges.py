"""Stage 4: Build graph edges between nodes."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по построению клинических алгоритмов принятия решений.

Для построенных узлов графа определи ВСЕ рёбра (переходы между узлами).

УЗЛЫ ГРАФА:
{nodes_json}

═══════════════════════════════════════════
СТРОГИЕ ПРАВИЛА ПОСТРОЕНИЯ РЁБЕР:
═══════════════════════════════════════════

1. START → первый DECISION-узел (одно ребро, без условий)

2. DECISION → следующий узел:
   • Каждый вариант из options ОБЯЗАН иметь РОВНО ОДНО исходящее ребро
   • Условие ребра должно точно соответствовать варианту из options
   • Если вариант предполагает вложенное ветвление → ведёт в другой DECISION
   • Если вариант однозначно определяет лечение → ведёт в ACTION
   • Все условия ВЗАИМОИСКЛЮЧАЮЩИЕ — одна ситуация не может вести в два разных ACTION

3. ACTION → END (обязательное ребро без условий после каждого ACTION)

4. WARNING → END (обязательное ребро без условий после каждого WARNING)

5. END — не имеет исходящих рёбер

6. ЗАПРЕЩЕНО:
   • Дублирующие рёбра (два ребра из одного узла в один и тот же узел)
   • Рёбра из END
   • Циклы любого вида
   • Висячие узлы (узел без входящих рёбер, кроме START)
   • ACTION или WARNING без исходящего ребра к END

7. Условие ребра (condition):
   • field: имя параметра (fracture_type, age, activity_level, cognitive_status, ...)
   • operator: == | != | > | < | >= | <= | in | not_in | exists | yes | no
   • value: конкретное значение или список
   • Если ребро безусловное (START→DECISION, ACTION→END): condition = null

ПРИМЕР правильной структуры для одной ветки:
  START → DECISION(тип перелома) → DECISION(возраст) → ACTION(лечение) → END

Верни СТРОГО JSON без пояснений, без markdown:
{{
  "edges": [
    {{
      "id": "edge_001",
      "from": "node_001",
      "to": "node_002",
      "label": "текст условия перехода",
      "condition": {{
        "field": "fracture_type",
        "operator": "==",
        "value": "Garden I-II"
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

        # Deduplicate edges by (from, to) pair — keep first occurrence
        seen: set[tuple] = set()
        unique_edges = []
        for edge in data["edges"]:
            key = (edge.get("from"), edge.get("to"))
            if key in seen:
                logger.warning(f"Stage4: duplicate edge {key} removed")
                continue
            if edge.get("from") == edge.get("to"):
                logger.warning(f"Stage4: self-loop edge {edge.get('id')} removed")
                continue
            seen.add(key)
            unique_edges.append(edge)

        data["edges"] = unique_edges
        logger.info(f"Stage 4: built {len(data['edges'])} edges")
        return data

    def run(self, nodes: dict) -> dict:
        return self._execute_with_retry(self.build_prompt(nodes))
