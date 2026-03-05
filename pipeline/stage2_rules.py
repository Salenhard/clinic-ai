"""Stage 2: Extract IF-THEN clinical decision rules."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — медицинский эксперт по клиническим рекомендациям.

На основе выявленных сущностей и исходного текста извлеки ВСЕ IF-THEN правила клинической логики.

ВЫЯВЛЕННЫЕ СУЩНОСТИ:
{entities_json}

ИСХОДНЫЙ ТЕКСТ:
{text}

Для каждого правила определи:
- Условия (могут быть вложенными): возраст, тип перелома, активность пациента и т.д.
- Действие: конкретный метод лечения или вмешательство
- Уровень доказательности (если указан в тексте)

Верни СТРОГО JSON без пояснений, без markdown-блоков:
{{
  "rules": [
    {{
      "id": "rule_001",
      "description": "краткое описание правила",
      "conditions": [
        {{
          "field": "имя параметра (age, fracture_type, activity_level, cognitive_status, ...)",
          "operator": "> | < | >= | <= | == | != | in | not_in",
          "value": "значение или список значений",
          "logic": "AND | OR"
        }}
      ],
      "action": "рекомендуемое лечение/вмешательство",
      "action_type": "surgical | conservative | diagnostic | monitoring",
      "evidence_level": "1A | 1B | 2A | 2B | 3 | 4 | 5 | не указан",
      "priority": "absolute | preferred | alternative | contraindicated",
      "source_text": "цитата из текста"
    }}
  ]
}}"""


class Stage2Rules(BasePipelineStage):
    stage_name = "stage2_rules"

    def build_prompt(self, text: str, entities: dict) -> str:
        return PROMPT_TEMPLATE.format(
            entities_json=json.dumps(entities, ensure_ascii=False, indent=2)[:3000],
            text=text[:10000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "rules" not in data:
            raise PipelineError("Stage2: missing 'rules' key in response")
        logger.info(f"Stage 2: extracted {len(data['rules'])} rules")
        return data

    def run(self, text: str, entities: dict) -> dict:
        prompt = self.build_prompt(text, entities)
        return self._execute_with_retry(prompt)
