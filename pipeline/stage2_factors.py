"""Stage 2: Extract all decision factors — patient factors, fracture classifications,
technical criteria, and contraindications."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .chunker import Chunk
from typing import List

logger = logging.getLogger(__name__)

_PROMPT = """\
Проанализируй текст клинических рекомендаций. Извлеки ВСЕ факторы и критерии принятия решений.

ТЕКСТ (чанк {chunk_idx}/{total_chunks}):
{text}

Извлеки четыре блока:

1. patient_factors — факторы пациента, влияющие на выбор метода:
   Поля: name, possible_values (список), effect_on_treatment (описание), contraindications (список)
   Примеры факторов: возраст, функциональная активность, когнитивный статус, остеопороз

2. fracture_classifications — классификации переломов из документа:
   Поля: system (Garden/Pipkin/AO), type_label, description, stability, risks (список), recommended_method

3. decision_criteria — технические и анатомические критерии:
   Поля: name, method, value_or_rule (измеримый критерий: "TAD < 25 мм" и т.д.), importance

4. contraindications — противопоказания к методам:
   Поля: method_name, absolute (список), relative (список), risk_factors (список)

Верни СТРОГО JSON:
{{
  "patient_factors": [
    {{
      "name": "...",
      "possible_values": [],
      "effect_on_treatment": "...",
      "contraindications": []
    }}
  ],
  "fracture_classifications": [
    {{
      "system": "...",
      "type_label": "...",
      "description": "...",
      "stability": "стабильный | нестабильный | не определено",
      "risks": [],
      "recommended_method": "..."
    }}
  ],
  "decision_criteria": [
    {{
      "name": "...",
      "method": "...",
      "value_or_rule": "...",
      "importance": "обязательно | рекомендуется"
    }}
  ],
  "contraindications": [
    {{
      "method_name": "...",
      "absolute": [],
      "relative": [],
      "risk_factors": []
    }}
  ]
}}"""


class Stage2Factors(BasePipelineStage):
    stage_name = "stage2_factors"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        return _PROMPT.format(text=text[:10000], chunk_idx=chunk_idx + 1, total_chunks=total)

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        for key in ("patient_factors", "fracture_classifications",
                    "decision_criteria", "contraindications"):
            if key not in data:
                data[key] = []
        logger.info(
            f"Stage 2: patient_factors={len(data['patient_factors'])}, "
            f"classifications={len(data['fracture_classifications'])}, "
            f"criteria={len(data['decision_criteria'])}, "
            f"contraindications={len(data['contraindications'])}"
        )
        return data

    @staticmethod
    def _merge(results: list[dict]) -> dict:
        merged: dict = {
            "patient_factors": [],
            "fracture_classifications": [],
            "decision_criteria": [],
            "contraindications": [],
        }
        seen: dict = {k: set() for k in merged}
        key_fields = {
            "patient_factors": "name",
            "fracture_classifications": "type_label",
            "decision_criteria": "name",
            "contraindications": "method_name",
        }
        for r in results:
            for section, kf in key_fields.items():
                for item in r.get(section, []):
                    k = item.get(kf, "").strip().lower()
                    if k and k not in seen[section]:
                        seen[section].add(k)
                        merged[section].append(item)
        return merged

    def run(self, chunks: List[Chunk]) -> dict:
        return self._execute_over_chunks(
            chunks,
            lambda text, idx, total: self.build_prompt(text, idx, total),
            self._merge,
        )
