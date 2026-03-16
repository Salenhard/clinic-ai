"""Stage 1a: Extract osteosynthesis (bone fixation) methods from the guidelines."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .chunker import Chunk
from typing import List

logger = logging.getLogger(__name__)

_SYSTEM = "Ты — анализатор текста. Отвечай ТОЛЬКО валидным JSON без пояснений и markdown."

_PROMPT = """\
Проанализируй текст клинических рекомендаций по переломам проксимального отдела бедренной кости.
Создай структурированный список всех методов ОСТЕОСИНТЕЗА (фиксации костных отломков).

ТЕКСТ (чанк {chunk_idx}/{total_chunks}):
{text}

Для каждого метода укажи поля:
- name          : полное название метода
- fixation_type : "динамическая" | "статическая" | "комбинированная"
- age_range     : "< 60" | ">= 60" | "универсально" | список если несколько
- fracture_types: список типов переломов
- implant       : название имплантата/инструментария
- key_features  : список ключевых технических характеристик

Верни СТРОГО JSON:
{{
  "osteosynthesis_methods": [
    {{
      "name": "...",
      "fixation_type": "...",
      "age_range": "...",
      "fracture_types": [],
      "implant": "...",
      "key_features": [],
    }}
  ]
}}"""


class Stage1aOsteosynthesis(BasePipelineStage):
    stage_name = "stage1a_osteosynthesis"

    def build_prompt(self, text: str, chunk_idx: int = 0, total: int = 1) -> str:
        return _PROMPT.format(text=text[:10000], chunk_idx=chunk_idx + 1, total_chunks=total)

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
