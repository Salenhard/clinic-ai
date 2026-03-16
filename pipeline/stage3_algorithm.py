"""Stage 3: Structure the decision algorithm — identify primary/secondary branches
and define IF-THEN-ELSE logic for each split point."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты — эксперт-аналитик клинических рекомендаций. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений, комментариев и markdown-разметки."
)

_PROMPT = """\
Задача: построить логическую структуру алгоритма принятия клинических решений \
на основе извлечённых данных.

## ВХОДНЫЕ ДАННЫЕ

### Методы остеосинтеза:
{osteosynthesis_json}

### Методы эндопротезирования:
{arthroplasty_json}

### Лечение по типам переломов:
{fracture_json}

### Факторы пациента:
{factors_json}

### Фрагмент исходного текста (справочно):
{source_text}

## ПРАВИЛА ПОСТРОЕНИЯ АЛГОРИТМА

### Правило 1 — Полное покрытие вариантов
Каждый вариант в options ОБЯЗАН приводить к результату — либо через terminals,
либо через дочернюю ветку в sub_branches. Вариант без терминала и без
дочерней ветки — ошибка структуры.

### Правило 2 — Независимость поддеревьев
Каждый тип перелома получает своё независимое поддерево.
Нельзя использовать одну ветку как общую для двух разных типов переломов.
Если один и тот же фактор (например, возраст) влияет на тактику в нескольких
типах — для каждого типа создаётся отдельная ветка с этим фактором.

### Правило 3 — Порядок факторов внутри поддерева
Порядок факторов определяется данными:
- Первым ставь тот фактор, который сильнее всего разветвляет тактику
  (после него наибольшее число вариантов приходит к терминалу).
- Фактор ставится только там, где он реально влияет на тактику — не выноси
  его выше, если он нужен лишь в части веток.
- Порядок факторов определяется логикой документа, а не общими предположениями.

### Правило 4 — Точность данных
Названия типов переломов, факторов, вариантов и методов лечения берутся
дословно из входных данных. Не придумывай термины, отсутствующие в данных.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "algorithm": {{
    "entry_question": "текст первого вопроса врачу",
    "branches": [
      {{
        "id": "branch_001",
        "question": "вопрос врачу",
        "field": "название поля (fracture_type / age / stability / location / status / ...)",
        "level": "primary | secondary | tertiary",
        "parent_branch": null,
        "options": ["вариант 1", "вариант 2"],
        "terminals": {{"вариант без дочерней ветки": "метод лечения"}},
        "sub_branches": ["branch_002"]
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
        source_text: str = "",
    ) -> str:
        def _trim(d, key, n=3000):
            return json.dumps(d.get(key, []), ensure_ascii=False)[:n]

        return _PROMPT.format(
            osteosynthesis_json=_trim(osteosynthesis, "osteosynthesis_methods"),
            arthroplasty_json=_trim(arthroplasty, "arthroplasty_methods"),
            fracture_json=_trim(fracture_treatments, "fracture_treatments"),
            factors_json=_trim(factors, "patient_factors"),
            source_text=source_text[:8000] if source_text else "не предоставлен",
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

        # Validate branch structure: every option must go somewhere
        issues = []
        branch_ids = {b["id"] for b in branches}
        for b in branches:
            opts = b.get("options", [])
            terminals = b.get("terminals", {})
            sub_branches = b.get("sub_branches", [])
            for sb_id in sub_branches:
                if sb_id not in branch_ids:
                    issues.append(f"branch '{b['id']}': sub_branch '{sb_id}' not found")
            uncovered = [o for o in opts if o not in terminals and len(sub_branches) == 0]
            if uncovered:
                issues.append(
                    f"branch '{b['id']}': options {uncovered} have no terminal and no sub_branch"
                )
        if issues:
            for iss in issues:
                logger.warning(f"Stage 3 branch validation: {iss}")
        return data

    def run(
        self,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
        source_text: str = "",
    ) -> dict:
        prompt = self.build_prompt(
            osteosynthesis, arthroplasty, fracture_treatments, factors, source_text
        )
        return self._execute_with_retry(prompt)
