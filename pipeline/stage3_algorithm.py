"""Stage 3: Structure the decision algorithm — identify primary/secondary branches
and define IF-THEN-ELSE logic for each split point."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_PROMPT = """\
На основе извлечённых данных построй логическую структуру алгоритма принятия решений
о методе лечения перелома проксимального отдела бедренной кости.

═══════════════════════════════════════════
МЕТОДЫ ОСТЕОСИНТЕЗА:
{osteosynthesis_json}

МЕТОДЫ ЭНДОПРОТЕЗИРОВАНИЯ:
{arthroplasty_json}

ЛЕЧЕНИЕ ПО ТИПАМ ПЕРЕЛОМОВ:
{fracture_json}

ФАКТОРЫ ПАЦИЕНТА:
{factors_json}
═══════════════════════════════════════════

Задача: определи РАЗВИЛКИ алгоритма и их IF-THEN-ELSE правила.

Для каждой развилки укажи:
- id           : уникальный идентификатор (branch_001 и т.д.)
- question     : вопрос врачу (текст)
- field        : программный идентификатор поля (fracture_type / age / stability и т.д.)
- level        : "primary" (первая развилка) | "secondary" | "tertiary"
- parent_branch: id родительской развилки и значение условия, при котором мы сюда попадаем
- options      : список вариантов ответа (точные строки)
- terminals    : объект {{вариант: название_метода_лечения}} для вариантов, ведущих сразу к действию
- sub_branches : список id дочерних развилок

ПРАВИЛА:
• Параметр, используемый в НЕСКОЛЬКИХ ветках → спрашивается ОДИН РАЗ как primary/secondary
• Параметр, нужный только в одной ветке → локальная secondary/tertiary развилка
• Каждый вариант options должен вести либо к terminal, либо к sub_branch
• Нет циклов, нет недостижимых путей

Верни СТРОГО JSON:
{{
  "algorithm": {{
    "entry_question": "текст первого вопроса",
    "branches": [
      {{
        "id": "branch_001",
        "question": "...",
        "field": "...",
        "level": "primary",
        "parent_branch": null,
        "options": [],
        "terminals": {{}},
        "sub_branches": []
      }}
    ]
  }}
}}"""


class Stage3Algorithm(BasePipelineStage):
    stage_name = "stage3_algorithm"

    def build_prompt(
        self,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> str:
        def _trim(d, key, n=3000):
            return json.dumps(d.get(key, []), ensure_ascii=False)[:n]

        return _PROMPT.format(
            osteosynthesis_json=_trim(osteosynthesis, "osteosynthesis_methods"),
            arthroplasty_json=_trim(arthroplasty, "arthroplasty_methods"),
            fracture_json=_trim(fracture_treatments, "fracture_treatments"),
            factors_json=_trim(factors, "patient_factors"),
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "algorithm" not in data:
            raise PipelineError("Stage3: missing 'algorithm'")
        branches = data["algorithm"].get("branches", [])
        logger.info(
            f"Stage 3: {len(branches)} branches, "
            f"entry='{data['algorithm'].get('entry_question', '')[:60]}'"
        )
        return data

    def run(
        self,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> dict:
        prompt = self.build_prompt(osteosynthesis, arthroplasty, fracture_treatments, factors)
        return self._execute_with_retry(prompt)
