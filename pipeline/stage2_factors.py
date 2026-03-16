"""Stage 2: Extract all decision factors — patient factors, fracture classifications,
technical criteria, and contraindications."""
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
Задача: извлечь все факторы и критерии принятия клинических решений.

## ВХОДНЫЕ ДАННЫЕ
Чанк {chunk_idx} из {total_chunks}:
{text}

## ЧТО ИЗВЛЕКАТЬ

### 1. patient_factors
Характеристики пациента, которые влияют на выбор метода лечения.
Название фактора, возможные значения и описание влияния — дословно из текста.

### 2. fracture_classifications
Все упомянутые классификации переломов с их типами.
Названия систем классификации и типов — дословно из текста.

### 3. decision_criteria
Технические или анатомические пороговые значения и правила,
определяющие выбор импланта или техники операции.
Значения критериев — дословно из текста (числа, единицы, знаки).

## НЕЛЬЗЯ
- Выдумывать факторы, классификации или критерии, отсутствующие в тексте.
- Добавлять пороги или значения, не указанные явно.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "patient_factors": [
    {{
      "name": "название фактора",
      "possible_values": ["значение 1, как в тексте", "значение 2"],
      "effect_on_treatment": "как влияет на выбор метода, как описано в тексте",
      "contraindications": ["метод, противопоказанный при определённом значении"]
    }}
  ],
  "fracture_classifications": [
    {{
      "system": "название системы классификации",
      "type_label": "обозначение типа, как в тексте",
      "description": "краткое описание из текста",
      "stability": "стабильный | нестабильный | не определено",
      "recommended_method": "рекомендованный метод, как в тексте"
    }}
  ],
  "decision_criteria": [
    {{
      "name": "название критерия",
      "method": "метод, к которому относится критерий",
      "value_or_rule": "измеримое пороговое значение или правило, как в тексте",
      "importance": "обязательно | рекомендуется"
    }}
  ],
  "contraindications": []
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
