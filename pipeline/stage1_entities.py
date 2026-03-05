"""Stage 1: Extract clinical entities from text (chunk-aware)."""
import json
import logging
from typing import List, Optional

from .base import BasePipelineStage, PipelineError
from .chunker import Chunk

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты — медицинский эксперт по клиническим рекомендациям. "
    "Всегда отвечай строго в формате JSON без пояснений и без markdown."
)

_PROMPT_TEMPLATE = """\
Из следующего фрагмента клинических рекомендаций извлеки ВСЕ клинические сущности:
- нозологии и типы переломов/заболеваний
- классификации (Garden, Pauwels, AO/OTA, Pipkin и др.)
- методы лечения и операции
- критерии выбора тактики (возраст, активность, когнитивный статус и др.)
- осложнения и противопоказания
- уровни доказательности

{section_filter}{chunk_header}ТЕКСТ:
{text}

Верни СТРОГО JSON:
{{
  "entities": [
    {{
      "id": "ent_001",
      "type": "nosology | classification | treatment | criterion | complication | contraindication | evidence_level",
      "name": "название",
      "description": "краткое описание",
      "values": ["возможные значения"],
      "source_fragment": "цитата из текста"
    }}
  ]
}}"""


def _merge_entities(results: list[dict]) -> dict:
    """Deduplicate entities from multiple chunks by name+type."""
    seen: dict[tuple, dict] = {}
    for r in results:
        for ent in r.get("entities", []):
            key = (ent.get("type", ""), ent.get("name", "").lower().strip())
            if key not in seen:
                seen[key] = ent
    # Re-number IDs
    merged = list(seen.values())
    for i, e in enumerate(merged):
        e["id"] = f"ent_{i+1:03d}"
    logger.info(f"Stage 1 merge: {len(merged)} unique entities")
    return {"entities": merged}


class Stage1Entities(BasePipelineStage):
    stage_name = "stage1_entities"

    def build_prompt(self, text: str, chunk_index: int = 0, total_chunks: int = 1,
                     section: Optional[str] = None) -> str:
        section_filter = f"Сфокусируйся на разделе: «{section}»\n" if section else ""
        chunk_header = (
            f"[Фрагмент {chunk_index + 1} из {total_chunks}]\n"
            if total_chunks > 1 else ""
        )
        return _PROMPT_TEMPLATE.format(
            section_filter=section_filter,
            chunk_header=chunk_header,
            text=text,
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "entities" not in data:
            raise PipelineError("Stage1: missing 'entities' key")
        logger.info(f"Stage 1: found {len(data['entities'])} entities in chunk")
        return data

    def run(
        self,
        chunks: List[Chunk],
        section: Optional[str] = None,
    ) -> dict:
        def build_prompt(text, idx, total):
            return self.build_prompt(text, idx, total, section)

        return self._execute_over_chunks(chunks, build_prompt, _merge_entities)
