"""Stage 1c: Extract recommended methods per fracture type."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .chunker import Chunk
from typing import List

logger = logging.getLogger(__name__)

_PROMPT = """\
Проанализируй разделы документа, посвящённые конкретным типам переломов.
Для каждого типа перелома извлеки рекомендации по лечению.

ТЕКСТ (чанк {chunk_idx}/{total_chunks}):
{text}

Типы переломов для поиска: Pipkin I, Pipkin II, Pipkin III, Pipkin IV,
Garden I-II, Garden III-IV, 31A1.3, 31A2, подвертельные, другие.

Для каждого типа укажи:
- fracture_type      : название типа перелома
- classification     : система классификации (Pipkin/Garden/AO)
- primary_treatment  : рекомендуемый метод (название)
- alternative        : альтернативный метод если есть
- age_modifications  : объект {{age_group: method}} если тактика меняется по возрасту
- evidence_level     : уровень доказательности если указан
- key_notes          : важные клинические примечания

Верни СТРОГО JSON:
{{
  "fracture_treatments": [
    {{
      "fracture_type": "...",
      "classification": "...",
      "primary_treatment": "...",
      "alternative": null,
      "age_modifications": {{}},
      "evidence_level": null,
      "key_notes": []
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
                    # Merge age_modifications from duplicate entries
                    for existing in merged:
                        if existing.get("fracture_type", "").strip().lower() == key:
                            existing["age_modifications"].update(
                                m.get("age_modifications", {})
                            )
                            break
        return {"fracture_treatments": merged}

    def run(self, chunks: List[Chunk]) -> dict:
        return self._execute_over_chunks(
            chunks,
            lambda text, idx, total: self.build_prompt(text, idx, total),
            self._merge,
        )
