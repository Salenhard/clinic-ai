"""Stage 1c: Extract recommended methods per fracture type."""
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
Задача: для каждого типа перелома, упомянутого в тексте, извлечь рекомендованную тактику лечения.

## ВХОДНЫЕ ДАННЫЕ
Чанк {chunk_idx} из {total_chunks}:
{text}

## ПРАВИЛА ИЗВЛЕЧЕНИЯ
- Создавай запись только для типов переломов, явно упомянутых в этом чанке.
- Название типа перелома и системы классификации бери дословно из текста.
- Если тактика различается по какому-либо признаку пациента (возраст, активность и др.),
  заполни поле patient_modifications: ключи — значения этого признака из текста,
  значения — соответствующие методы лечения.
- Не добавляй пороговые значения или названия методов, которых нет в тексте.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "fracture_treatments": [
    {{
      "fracture_type": "название типа перелома, как в тексте",
      "classification": "система классификации, как в тексте",
      "primary_treatment": "основной рекомендуемый метод",
      "alternative": "альтернативный метод или null",
      "patient_modifications": {{
        "значение признака 1": "метод лечения для этой группы",
        "значение признака 2": "метод лечения для этой группы"
      }},
      "key_notes": ["важное клиническое примечание из текста"]
    }}
  ]
}}"""


class Stage1cFractureSpecific(BasePipelineStage):
    stage_name = "stage1c_fracture_specific"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        return _PROMPT.format(text=text[:10000], chunk_idx=chunk_idx + 1, total_chunks=total)

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "fracture_treatments" not in data:
            raise PipelineError("Stage1c: missing 'fracture_treatments'")
        logger.info(f"Stage 1c: extracted {len(data['fracture_treatments'])} fracture treatments")
        return data

    @staticmethod
    def _merge(results: list[dict]) -> dict:
        seen, merged = set(), []
        for r in results:
            for m in r.get("fracture_treatments", []):
                key = m.get("fracture_type", "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(m)
                elif key in seen:
                    for existing in merged:
                        if existing.get("fracture_type", "").strip().lower() == key:
                            existing.setdefault("patient_modifications", {}).update(
                                m.get("patient_modifications", {})
                            )
                            break
        return {"fracture_treatments": merged}

    def run(self, chunks: List[Chunk]) -> dict:
        return self._execute_over_chunks(
            chunks,
            lambda text, idx, total: self.build_prompt(text, idx, total),
            self._merge,
        )
