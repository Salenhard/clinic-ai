"""Stage 1a: Extract osteosynthesis (bone fixation) methods from the guidelines."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .chunker import Chunk
from typing import List

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты — эксперт-аналитик клинических рекомендаций. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений, комментариев и markdown-разметки."
)

_PROMPT = """\
Задача: извлечь все методы ОСТЕОСИНТЕЗА (фиксации костных отломков) \
из фрагмента клинических рекомендаций.

## ВХОДНЫЕ ДАННЫЕ
Чанк {chunk_idx} из {total_chunks}:
{text}

## ПРАВИЛА ИЗВЛЕЧЕНИЯ
- Извлекай только методы фиксации костных отломков — не эндопротезирование.
- Используй только данные из текста. Не добавляй методы, которых нет в этом чанке.
- Если метод упомянут для нескольких групп пациентов — создай одну запись,
  перечислив все группы в поле age_range.
- Значения полей берутся из текста дословно, не нормализуй их.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "osteosynthesis_methods": [
    {{
      "name": "полное название метода, как указано в тексте",
      "fixation_type": "динамическая | статическая | комбинированная",
      "age_range": "возрастная группа или 'универсально', как указано в тексте",
      "fracture_types": ["тип перелома 1", "тип перелома 2"],
      "implant": "название имплантата или инструментария, или null",
      "key_features": ["ключевая техническая характеристика 1"]
    }}
  ]
}}"""


class Stage1aOsteosynthesis(BasePipelineStage):
    stage_name = "stage1a_osteosynthesis"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        safe_text = text[:10000].replace("{" , "{{").replace("}", "}}")
        return _PROMPT.format(text=safe_text, chunk_idx=chunk_idx + 1, total_chunks=total)

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "osteosynthesis_methods" not in data:
            raise PipelineError("Stage1a: missing 'osteosynthesis_methods'")
        logger.info(f"Stage 1a: extracted {len(data['osteosynthesis_methods'])} osteosynthesis methods")
        return data

    @staticmethod
    def _merge(results: list[dict]) -> dict:
        seen, merged = set(), []
        for r in results:
            for m in r.get("osteosynthesis_methods", []):
                key = m.get("name", "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(m)
        return {"osteosynthesis_methods": merged}

    def run(self, chunks: List[Chunk]) -> dict:
        return self._execute_over_chunks(
            chunks,
            lambda text, idx, total: self.build_prompt(text, idx, total),
            self._merge,
        )
