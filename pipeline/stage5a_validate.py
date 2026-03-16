"""Stage 5a: Validate graph completeness against the source document."""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .graph_validator import validate_graph_structure as validate

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты — эксперт-аналитик клинических рекомендаций. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений, комментариев и markdown-разметки."
)

_PROMPT = """\
Задача: проверить, полностью ли граф принятия решений покрывает клинические сценарии \
из исходного документа.

## ВХОДНЫЕ ДАННЫЕ

### Граф:
{graph_json}

### Эталон — лечение по типам переломов (из Stage 1c):
{fracture_json}

### Методы остеосинтеза (из Stage 1a):
{osteosynthesis_json}

### Методы эндопротезирования (из Stage 1b):
{arthroplasty_json}

### Фрагмент исходного текста:
{source_text}

## КАК ПРОВЕРЯТЬ

### Шаг 1 — Подсчёт ожидаемых маршрутов
Для каждого типа перелома в fracture_json определи все уникальные комбинации
значений факторов пациента (patient_modifications). Каждая комбинация = отдельный
маршрут, который должен иметь свой ACTION-узел в графе.
Ожидаемое число ACTION-узлов = сумма таких комбинаций по всем типам переломов.

### Шаг 2 — Верификация каждого маршрута
Для каждого ожидаемого маршрута (тип перелома + факторы пациента):
1. Проследи путь START → ... → END в графе.
2. Проверь, совпадает ли конечный ACTION с рекомендацией в fracture_json.
3. Если путь не существует или приводит к неверному ACTION — это issue.

### Шаг 3 — Флагировать ТОЛЬКО доказанные проблемы
Проблема флагируется как issue, только если ты можешь указать конкретный маршрут
(список узлов START → ... → END), который:
а) не существует в графе, или
б) существует, но конечный ACTION не соответствует рекомендации.
Нельзя флагировать "потенциальные" или "возможные" проблемы без конкретного маршрута.

### Шаг 4 — Схлопывание нозологий
ACTION-узел считается схлопнутым, только если в граф ведут рёбра из двух разных
нозологий (разных типов переломов или разных значений факторов). Это проверяется
по входящим рёбрам узла, а не по названию операции. Одинаковое название операции
для разных нозологий — НЕ схлопывание, если входящие рёбра из разных поддеревьев
ведут в разные ACTION-узлы.

## ВЫХОДНОЙ ФОРМАТ (строго JSON, без пояснений)
{{
  "completeness_score": 0.0,
  "covered_fracture_types": [],
  "missing_fracture_types": [],
  "covered_methods": [],
  "missing_methods": [],
  "age_handling_correct": true,
  "issues": [
    {{
      "severity": "critical | warning | info",
      "node_id": "id проблемного узла или null",
      "path": ["start_001", "dec_fracture_type", "...", "act_X"],
      "description": "точное описание проблемы с указанием конкретного маршрута",
      "suggestion": "конкретное исправление"
    }}
  ],
  "missing_scenarios": [
    {{
      "fracture_type": "тип перелома",
      "patient_params": {{"фактор": "значение"}},
      "expected_action": "ожидаемый метод лечения",
      "importance": "high | medium | low"
    }}
  ],
  "strengths": [],
  "overall_comment": "..."
}}"""


class Stage5aValidate(BasePipelineStage):
    stage_name = "stage5a_validate"

    def build_prompt(
        self,
        graph: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        source_text: str,
    ) -> str:
        def _safe(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        g = graph if "nodes" in graph else graph.get("graph", graph)
        return _PROMPT.format(
            graph_json=_safe(json.dumps(g, ensure_ascii=False)[:12000]),
            osteosynthesis_json=_safe(json.dumps(
                osteosynthesis.get("osteosynthesis_methods", []), ensure_ascii=False)[:2000]),
            arthroplasty_json=_safe(json.dumps(
                arthroplasty.get("arthroplasty_methods", []), ensure_ascii=False)[:2000]),
            fracture_json=_safe(json.dumps(
                fracture_treatments.get("fracture_treatments", []), ensure_ascii=False)[:3000]),
            source_text=_safe(source_text[:2000]),
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "completeness_score" not in data:
            raise PipelineError("Stage5a: missing 'completeness_score'")
        logger.info(
            f"Stage 5a: completeness={data.get('completeness_score')}, "
            f"missing_fractures={data.get('missing_fracture_types', [])}, "
            f"issues={len(data.get('issues', []))}"
        )
        return data

    def run(
        self,
        graph: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        source_text: str,
    ) -> dict:
        # Structural validation (deterministic, no LLM)
        graph_doc = graph if "graph" in graph else {"graph": graph}
        structural_issues = validate(graph_doc)

        # Clinical completeness (LLM)
        prompt = self.build_prompt(
            graph, osteosynthesis, arthroplasty, fracture_treatments, source_text
        )
        result = self._execute_with_retry(prompt)

        # Merge structural issues into result
        result["structural_issues"] = structural_issues
        result["structural_critical"] = sum(
            1 for i in structural_issues if i["severity"] == "critical"
        )
        result["structural_warnings"] = sum(
            1 for i in structural_issues if i["severity"] == "warning"
        )
        return result
