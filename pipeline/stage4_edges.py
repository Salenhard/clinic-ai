"""Stage 4: Build graph edges between nodes."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по построению клинических алгоритмов принятия решений.

Для построенных узлов определи ВСЕ рёбра (переходы между узлами).

УЗЛЫ ГРАФА:
{nodes_json}

═══════════════════════════════════════════
СТРОГИЕ ПРАВИЛА ПОСТРОЕНИЯ РЁБЕР:
═══════════════════════════════════════════

1. START → первый DECISION (одно безусловное ребро, condition=null)

2. DECISION → следующий узел:
   • Каждый вариант из options → РОВНО ОДНО исходящее ребро
   • label ребра ДОЛЖЕН совпадать с текстом варианта из options
   • Условия из одного DECISION взаимоисключающие

3. ACTION → END (обязательное безусловное ребро, condition=null)

4. WARNING → END (обязательное безусловное ребро, condition=null)

5. ЗАПРЕЩЕНО:
   • Два ребра с одинаковыми from+to
   • Самопетли (from == to)
   • Циклы
   • Узлы без входящих рёбер (кроме START)

═══════════════════════════════════════════
КРИТИЧЕСКИ ВАЖНОЕ ПРАВИЛО — ВОПРОС ЗАДАЁТСЯ ОДИН РАЗ:
═══════════════════════════════════════════

Если в графе есть несколько DECISION-узлов с одинаковым вопросом
(например, два узла "Возраст пациента"), это означает что вопрос
должен был быть вынесен в начало. При построении рёбер УБЕДИСЬ что:

• Пациент НЕ проходит через один и тот же тип вопроса дважды на одном пути
• Если узлы с одинаковым вопросом существуют в разных ветках — это допустимо
  ТОЛЬКО если ветки полностью независимы и не пересекаются
• Проверь каждый путь от START до END: на нём не должно быть двух узлов
  с одинаковым полем condition.field

ПРИМЕР ДОПУСТИМОЙ СТРУКТУРЫ:
  START → DECISION(тип перелома)
    ├→ "Pipkin II" → DECISION(возраст) → ACTION
    └→ "31A2"     → DECISION(возраст) → ACTION
  (два узла "возраст" в разных ветках — OK, пути не пересекаются)

ПРИМЕР НЕДОПУСТИМОЙ СТРУКТУРЫ:
  START → DECISION(возраст) → DECISION(тип перелома)
    └→ "Pipkin II" → DECISION(возраст) ← ПОВТОРНЫЙ ВОПРОС НА ОДНОМ ПУТИ

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

        seen: set[tuple] = set()
        unique_edges = []
        for edge in data["edges"]:
            key = (edge.get("from"), edge.get("to"))
            if key[0] == key[1]:
                logger.warning(f"Stage4: self-loop edge {edge.get('id')} removed")
                continue
            if key in seen:
                logger.warning(f"Stage4: duplicate edge {key} removed")
                continue
            seen.add(key)
            unique_edges.append(edge)

        data["edges"] = unique_edges
        logger.info(f"Stage 4: built {len(data['edges'])} edges")
        return data

    def run(self, nodes: dict) -> dict:
        return self._execute_with_retry(self.build_prompt(nodes))
