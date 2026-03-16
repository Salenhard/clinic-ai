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

### Список типов переломов для обязательного включения:
{fracture_types_list}

### Методы остеосинтеза:
{osteosynthesis_json}

### Методы эндопротезирования:
{arthroplasty_json}

### Лечение по типам переломов (подробно):
{fracture_json}

### Факторы пациента:
{factors_json}

### Фрагмент исходного текста (справочно):
{source_text}

## ПРАВИЛА ПОСТРОЕНИЯ АЛГОРИТМА

### Правило 0 — Обязательное включение всех типов переломов
Каждый тип перелома из раздела "Список типов переломов" ОБЯЗАН присутствовать
как вариант в алгоритме. Пропустить тип перелома — критическая ошибка.
Перед возвратом ответа проверь: количество типов в алгоритме ==
количеству типов в списке.

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

### Правило 5 — Ключи terminals должны точно совпадать с options (КРИТИЧНО)
Ключ в terminals — это ТОЧНАЯ КОПИЯ одного из вариантов из options этой же ветки.
Нельзя создавать ключ с дополнительными словами или уточнениями.

ОШИБКА:  options=["Pipkin II"],  terminals={{"Pipkin II (моложе 60 лет)": "..."}}
ПРАВИЛЬНО: options=["Pipkin II"], sub_branches=["branch_pipkin_ii_age"]
           И отдельная ветка branch_pipkin_ii_age с options=["моложе 60 лет", "старше 60 лет"]

Проверь перед возвратом: каждый ключ в terminals[branch] ∈ options[branch].
Если ключ не совпадает ни с одним вариантом из options — это ошибка,
нужна дочерняя ветка вместо терминала.

### Правило 6 — Один вариант → либо terminal, либо sub_branch, не оба
Для каждого варианта в options ровно одно из двух:
- Он есть в terminals (прямой исход, нет дальнейших вопросов).
- Для него есть дочерняя ветка в sub_branches (нужен ещё один вопрос).
Один и тот же вариант не может быть одновременно в terminals и sub_branches.

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
        "terminals": {{"вариант 1": "метод лечения — только если вариант не требует доп. вопросов"}},
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
        def _trim(d, key, n=5000):
            raw = json.dumps(d.get(key, []), ensure_ascii=False)[:n]
            # Escape braces so str.format() doesn't mistake JSON for placeholders
            return raw.replace("{", "{{").replace("}", "}}")

        def _safe(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        # Build explicit numbered list of fracture types for Rule 0 check
        fracture_types = [
            t.get("fracture_type", "")
            for t in fracture_treatments.get("fracture_treatments", [])
            if t.get("fracture_type")
        ]
        fracture_types_list = "\n".join(
            f"  {i+1}. {ft}" for i, ft in enumerate(fracture_types)
        ) or "  (нет данных)"

        return _PROMPT.format(
            fracture_types_list=_safe(fracture_types_list),
            osteosynthesis_json=_trim(osteosynthesis, "osteosynthesis_methods"),
            arthroplasty_json=_trim(arthroplasty, "arthroplasty_methods"),
            fracture_json=_trim(fracture_treatments, "fracture_treatments"),
            factors_json=_trim(factors, "patient_factors"),
            source_text=_safe(source_text[:8000]) if source_text else "не предоставлен",
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

        # --- Check 1: every option must go somewhere ---
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

        # --- Check 3: terminal keys must exactly match options ---
        terminal_mismatches = []
        empty_option_branches = []
        for b in branches:
            opts = b.get("options", [])
            terms = b.get("terminals", {})
            subs = b.get("sub_branches", [])

            # Sub-branches with terminals but no options
            if not opts and (terms or subs) and b.get("level") != "primary":
                empty_option_branches.append(
                    f"branch '{b['id']}' (parent={b.get('parent_branch')}): "
                    f"has terminals/sub_branches but empty options — "
                    f"terminals should live in the parent branch"
                )

            opts_set = {o.strip().lower() for o in opts}
            for key in terms.keys():
                if opts_set and key.strip().lower() not in opts_set:
                    terminal_mismatches.append(
                        f"branch '{b['id']}': terminal key '{key}' "
                        f"not in options {opts}"
                    )

        all_violations = terminal_mismatches + empty_option_branches
        if all_violations:
            for v in all_violations:
                logger.warning(f"Stage 3 structure violation: {v}")
            raise PipelineError(
                f"Stage3: {len(all_violations)} structure violation(s) "
                f"(Rules 5/6 — terminal keys must match options exactly, "
                f"sub-branches must have their own options). "
                f"First: {all_violations[0]}"
            )
        # _expected_fracture_types is set by run() before calling _execute_with_retry
        expected = getattr(self, "_expected_fracture_types", [])
        if expected:
            # Collect all option values across all branches
            all_options: set[str] = set()
            for b in branches:
                for opt in b.get("options", []):
                    all_options.add(opt.strip().lower())
                for term_key in b.get("terminals", {}).keys():
                    all_options.add(term_key.strip().lower())

            missing_types = [
                ft for ft in expected
                if not any(
                    ft.strip().lower() in opt or opt in ft.strip().lower()
                    for opt in all_options
                )
            ]
            if missing_types:
                logger.warning(
                    f"Stage 3 coverage: {len(missing_types)} fracture type(s) from Stage 1c "
                    f"not found in algorithm: {missing_types}"
                )
                # Raise so _execute_with_retry can retry with the same prompt
                raise PipelineError(
                    f"Stage3: algorithm missing fracture types: {missing_types}. "
                    f"Retry expected."
                )
            else:
                logger.info(
                    f"Stage 3 coverage: all {len(expected)} fracture types present in algorithm"
                )

        return data

    def validate_structure(self, data: dict) -> None:
        """Validate already-parsed algorithm data. Raises PipelineError if invalid.
        Used to validate cached results without re-running the LLM.
        """
        if "algorithm" not in data:
            raise PipelineError("Stage3 cache: missing 'algorithm'")

        branches = data["algorithm"].get("branches", [])
        branch_ids = {b["id"] for b in branches}
        violations = []

        for b in branches:
            opts = b.get("options", [])
            terms = b.get("terminals", {})
            subs = b.get("sub_branches", [])

            # Empty options with content (sub-branch tunnel anti-pattern)
            if not opts and (terms or subs) and b.get("level") != "primary":
                violations.append(
                    f"branch '{b['id']}': has terminals/sub_branches but empty options"
                )

            # Terminal keys must exactly match options
            opts_set = {o.strip().lower() for o in opts}
            for key in terms.keys():
                if opts_set and key.strip().lower() not in opts_set:
                    violations.append(
                        f"branch '{b['id']}': terminal key '{key}' not in options {opts}"
                    )

            # Sub-branch ids must exist
            for sb_id in subs:
                if sb_id not in branch_ids:
                    violations.append(
                        f"branch '{b['id']}': sub_branch '{sb_id}' not found"
                    )

        if violations:
            raise PipelineError(
                f"Stage3 cache invalid: {len(violations)} violation(s). "
                f"First: {violations[0]}"
            )

    def run(
        self,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
        source_text: str = "",
    ) -> dict:
        # Store expected fracture types for coverage check in parse_response
        self._expected_fracture_types = [
            t.get("fracture_type", "")
            for t in fracture_treatments.get("fracture_treatments", [])
            if t.get("fracture_type")
        ]
        prompt = self.build_prompt(
            osteosynthesis, arthroplasty, fracture_treatments, factors, source_text
        )
        return self._execute_with_retry(prompt)
