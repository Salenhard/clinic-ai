"""Stage 2: Extract IF-THEN clinical decision rules (chunk-aware)."""
import json
import logging
from typing import List, Optional

from .base import BasePipelineStage, PipelineError
from .chunker import Chunk

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
На основе выявленных клинических сущностей и фрагмента текста извлеки все IF-THEN правила.

СУЩНОСТИ:
{entities_json}

{chunk_header}ТЕКСТ:
{text}

Для каждого правила:
- Условия (могут быть вложенными): возраст, тип перелома, активность и т.д.
- Действие: конкретный метод лечения
- Уровень доказательности (если указан)

Верни СТРОГО JSON:
{{
  "rules": [
    {{
      "id": "rule_001",
      "description": "краткое описание",
      "conditions": [
        {{
          "field": "age | fracture_type | activity_level | cognitive_status | ...",
          "operator": "> | < | >= | <= | == | != | in | not_in",
          "value": "значение",
          "logic": "AND | OR"
        }}
      ],
      "action": "рекомендуемое лечение",
      "action_type": "surgical | conservative | diagnostic | monitoring",
      "evidence_level": "1A | 1B | 2A | 2B | 3 | 4 | не указан",
      "priority": "absolute | preferred | alternative | contraindicated",
      "source_text": "цитата"
    }}
  ]
}}"""


def _merge_rules(results: list[dict]) -> dict:
    """Deduplicate rules by action+conditions signature."""
    seen: dict[str, dict] = {}
    for r in results:
        for rule in r.get("rules", []):
            key = rule.get("action", "").lower().strip()
            conds = tuple(
                (c.get("field",""), c.get("operator",""), str(c.get("value","")))
                for c in rule.get("conditions", [])
            )
            full_key = f"{key}|{conds}"
            if full_key not in seen:
                seen[full_key] = rule
    merged = list(seen.values())
    for i, r in enumerate(merged):
        r["id"] = f"rule_{i+1:03d}"
    logger.info(f"Stage 2 merge: {len(merged)} unique rules")
    return {"rules": merged}


class Stage2Rules(BasePipelineStage):
    stage_name = "stage2_rules"

    def build_prompt(self, text: str, entities: dict,
                     chunk_index: int = 0, total_chunks: int = 1) -> str:
        chunk_header = (
            f"[Фрагмент {chunk_index + 1} из {total_chunks}]\n"
            if total_chunks > 1 else ""
        )
        return _PROMPT_TEMPLATE.format(
            entities_json=json.dumps(entities, ensure_ascii=False, indent=2)[:3000],
            chunk_header=chunk_header,
            text=text,
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "rules" not in data:
            raise PipelineError("Stage2: missing 'rules' key")
        logger.info(f"Stage 2: found {len(data['rules'])} rules in chunk")
        return data

    def run(self, chunks: List[Chunk], entities: dict) -> dict:
        def build_prompt(text, idx, total):
            return self.build_prompt(text, entities, idx, total)

        return self._execute_over_chunks(chunks, build_prompt, _merge_rules)
