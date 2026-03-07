"""Stage 5a: Validate graph completeness against the source document.

Checks:
- All fracture types covered
- All age categories handled
- All treatment methods represented
- Technical requirements met
Returns a completeness report + structural validation results.
"""
import json
import logging
from .base import BasePipelineStage, PipelineError
from .graph_validator import validate_graph_structure as validate

logger = logging.getLogger(__name__)

_PROMPT = """\
Проверь построенный граф принятия решений на ПОЛНОТУ соответствия исходному документу.

ФИНАЛЬНЫЙ ГРАФ:
{graph_json}

ИЗВЛЕЧЁННЫЕ МЕТОДЫ ОСТЕОСИНТЕЗА:
{osteosynthesis_json}

ИЗВЛЕЧЁННЫЕ МЕТОДЫ ЭНДОПРОТЕЗИРОВАНИЯ:
{arthroplasty_json}

ЛЕЧЕНИЕ ПО ТИПАМ ПЕРЕЛОМОВ (эталон):
{fracture_json}

ИСХОДНЫЙ ТЕКСТ (фрагмент):
{source_text}

Проверь:
1. Все ли типы переломов (Pipkin I-IV, Garden I-IV, 31A1.3, 31A2 и др.) представлены в графе?
2. Все ли возрастные категории (<60, >=60, >=70) корректно обработаны?
3. Все ли методы лечения из документа представлены в ACTION-узлах?
4. Есть ли клинически важные ситуации, не покрытые графом?
5. Корректны ли уровни доказательности в action_details?

Верни СТРОГО JSON:
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
      "node_id": null,
      "description": "...",
      "suggestion": "..."
    }}
  ],
  "missing_scenarios": [
    {{
      "description": "...",
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
        g = graph if "nodes" in graph else graph.get("graph", graph)
        return _PROMPT.format(
            graph_json=json.dumps(g, ensure_ascii=False)[:6000],
            osteosynthesis_json=json.dumps(
                osteosynthesis.get("osteosynthesis_methods", []), ensure_ascii=False)[:2000],
            arthroplasty_json=json.dumps(
                arthroplasty.get("arthroplasty_methods", []), ensure_ascii=False)[:2000],
            fracture_json=json.dumps(
                fracture_treatments.get("fracture_treatments", []), ensure_ascii=False)[:2500],
            source_text=source_text[:2000],
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
        # Structural validation (deterministic)
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
