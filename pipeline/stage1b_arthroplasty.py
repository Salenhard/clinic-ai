"""Stage 1b: Extract arthroplasty (joint replacement) methods from the guidelines."""
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
Задача: извлечь все методы ЭНДОПРОТЕЗИРОВАНИЯ из фрагмента клинических рекомендаций.

## ВХОДНЫЕ ДАННЫЕ
Чанк {chunk_idx} из {total_chunks}:
{text}

## ПРАВИЛА ИЗВЛЕЧЕНИЯ
- Извлекай только методы имплантации эндопротезов — не остеосинтез.
- Используй только данные из текста. Не добавляй методы, которых нет в этом чанке.
- Показания для каждого метода (возраст, активность, когнитивный статус и др.)
  извлекай дословно из текста.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "arthroplasty_methods": [
    {{
      "name": "полное название метода, как указано в тексте",
      "fixation_type": "тип фиксации компонентов, как указано в тексте",
      "indications": {{
        "age": "возрастная группа или null, как указано в тексте",
        "activity": "уровень активности или null, как указано в тексте",
        "cognitive": "когнитивный статус или null, как указано в тексте",
        "other": "прочие показания или null"
      }},
      "special_features": ["конструктивная особенность 1"],
      "advantages": ["преимущество, упомянутое в тексте"],
      "limitations": ["ограничение, упомянутое в тексте"],
      "fracture_types": ["тип перелома, при котором применяется"]
    }}
  ]
}}"""


class Stage1bArthroplasty(BasePipelineStage):
    stage_name = "stage1b_arthroplasty"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        safe_text = text[:10000].replace("{" , "{{").replace("}", "}}")
        return _PROMPT.format(text=safe_text, chunk_idx=chunk_idx + 1, total_chunks=total)

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "arthroplasty_methods" not in data:
            raise PipelineError("Stage1b: missing 'arthroplasty_methods'")
        logger.info(f"Stage 1b: extracted {len(data['arthroplasty_methods'])} arthroplasty methods")
        return data

    @staticmethod
    def _merge(results: list[dict]) -> dict:
        seen, merged = set(), []
        for r in results:
            for m in r.get("arthroplasty_methods", []):
                key = m.get("name", "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(m)
        return {"arthroplasty_methods": merged}

    def run(self, chunks: List[Chunk]) -> dict:
        return self._execute_over_chunks(
            chunks,
            lambda text, idx, total: self.build_prompt(text, idx, total),
            self._merge,
        )
