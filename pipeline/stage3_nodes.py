"""Stage 3: Build graph nodes from rules."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по построению клинических алгоритмов принятия решений.

На основе извлечённых правил построй УЗЛЫ графа консультации.

ПРАВИЛА:
{rules_json}

Типы узлов:
- START: единственная точка входа в алгоритм
- DECISION: вопрос с ветвлением (да/нет или множественный выбор)
- ACTION: рекомендуемое лечение/вмешательство
- WARNING: предупреждение или противопоказание
- END: завершение ветки консультации

Важные требования:
1. Должен быть РОВНО ОДИН узел START
2. Каждая ветка должна завершаться узлом END
3. Узлы DECISION группируют логически связанные условия
4. Узлы ACTION соответствуют конкретным терапевтическим решениям
5. Не дублируй узлы — одно и то же действие = один узел ACTION

Верни СТРОГО JSON без пояснений, без markdown-блоков:
{{
  "nodes": [
    {{
      "id": "node_001",
      "type": "START | DECISION | ACTION | WARNING | END",
      "label": "Человекочитаемое короткое название",
      "question": "Текст вопроса (только для DECISION-узлов)",
      "options": ["Вариант А", "Вариант Б", "Вариант В"],
      "source_rules": ["rule_001", "rule_002"],
      "action_details": {{
        "procedure": "название процедуры (только для ACTION)",
        "implant": "тип имплантата или null",
        "timing": "сроки/нагрузка или null",
        "evidence_level": "уровень доказательности или null",
        "contraindications": [],
        "notes": "клинические примечания"
      }}
    }}
  ]
}}"""


class Stage3Nodes(BasePipelineStage):
    stage_name = "stage3_nodes"

    def build_prompt(self, rules: dict) -> str:
        return PROMPT_TEMPLATE.format(
            rules_json=json.dumps(rules, ensure_ascii=False, indent=2)[:8000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "nodes" not in data:
            raise PipelineError("Stage3: missing 'nodes' key in response")
        
        # Validate node types
        valid_types = {"START", "DECISION", "ACTION", "WARNING", "END"}
        for node in data["nodes"]:
            if node.get("type") not in valid_types:
                logger.warning(f"Node {node.get('id')} has invalid type: {node.get('type')}")
                node["type"] = "ACTION"  # fallback
        
        logger.info(f"Stage 3: built {len(data['nodes'])} nodes")
        return data

    def run(self, rules: dict) -> dict:
        prompt = self.build_prompt(rules)
        return self._execute_with_retry(prompt)
