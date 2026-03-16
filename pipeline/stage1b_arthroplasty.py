"""Stage 1b: Extract arthroplasty (joint replacement) methods from the guidelines."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .chunker import Chunk
from typing import List

logger = logging.getLogger(__name__)

_PROMPT = """\
Проанализируй текст клинических рекомендаций. Создай список всех видов ЭНДОПРОТЕЗИРОВАНИЯ тазобедренного сустава.

ТЕКСТ (чанк {chunk_idx}/{total_chunks}):
{text}

Для каждого метода укажи:
- name             : полное название (ТЭТС, гемиэндопротезирование и т.д.)
- fixation_type    : "цементная" | "бесцементная" | "гибридная" | список
- indications      : показания — возраст, активность, когнитивный статус
- special_features : особые конструктивные особенности (двойная мобильность и т.д.)
- advantages       : преимущества
- limitations      : ограничения
- fracture_types   : при каких типах переломов применяется

Верни СТРОГО JSON:
{{
  "arthroplasty_methods": [
    {{
      "name": "...",
      "fixation_type": "...",
      "indications": {{}},
      "special_features": [],
      "advantages": [],
      "limitations": [],
      "fracture_types": []
    }}
  ]
}}"""


class Stage1bArthroplasty(BasePipelineStage):
    stage_name = "stage1b_arthroplasty"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        return _PROMPT.format(text=text[:10000], chunk_idx=chunk_idx + 1, total_chunks=total)

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
