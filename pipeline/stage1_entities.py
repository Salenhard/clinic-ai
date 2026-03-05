"""Stage 1: Extract clinical entities from text."""
import json
import logging
from typing import Optional
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — медицинский эксперт по клиническим рекомендациям.

Из следующего текста клинических рекомендаций извлеки ВСЕ клинические сущности:
- нозологии и типы переломов/заболеваний
- классификации (Garden, Pauwels, AO/OTA, Pipkin и др.)
- методы лечения и операции
- критерии выбора тактики (возраст, активность, когнитивный статус и др.)
- осложнения и противопоказания
- уровни доказательности

{section_filter}

ТЕКСТ:
{text}

Верни СТРОГО JSON без пояснений, без markdown-блоков:
{{
  "entities": [
    {{
      "id": "ent_001",
      "type": "nosology | classification | treatment | criterion | complication | contraindication | evidence_level",
      "name": "название сущности",
      "description": "краткое описание",
      "values": ["возможные значения, если есть"],
      "source_fragment": "цитата из текста"
    }}
  ]
}}"""


class Stage1Entities(BasePipelineStage):
    stage_name = "stage1_entities"

    def build_prompt(self, text: str, section: Optional[str] = None) -> str:
        section_filter = ""
        if section:
            section_filter = f"Сфокусируйся на разделе: «{section}»\n"
        return PROMPT_TEMPLATE.format(
            text=text[:12000],  # limit context
            section_filter=section_filter
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "entities" not in data:
            raise PipelineError("Stage1: missing 'entities' key in response")
        logger.info(f"Stage 1: extracted {len(data['entities'])} entities")
        return data

    def run(self, text: str, section: Optional[str] = None) -> dict:
        prompt = self.build_prompt(text, section)
        return self._execute_with_retry(prompt)
