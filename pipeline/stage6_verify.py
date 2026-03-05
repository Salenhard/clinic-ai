"""Stage 6: Clinical validation and semantic verification."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — опытный клинический эксперт-рецензент.

Проверь граф принятия клинических решений на медицинскую корректность.

ГРАФ:
{graph_json}

ИСХОДНЫЙ ТЕКСТ РЕКОМЕНДАЦИЙ:
{text}

Проверь:
1. Все ли классификации переломов/заболеваний представлены корректно?
2. Соответствуют ли рекомендации по возрасту и другим критериям тексту рекомендаций?
3. Есть ли противоречия между ветками алгоритма?
4. Какие клинически важные ситуации НЕ покрыты графом?
5. Корректны ли уровни доказательности?

Верни СТРОГО JSON без пояснений, без markdown-блоков:
{{
  "clinical_accuracy_score": 0.85,
  "classification_accuracy": true,
  "age_criteria_correct": true,
  "issues": [
    {{
      "severity": "critical | warning | info",
      "node_id": "node_XXX или null",
      "description": "описание проблемы",
      "suggestion": "как исправить"
    }}
  ],
  "missing_scenarios": [
    {{
      "description": "описание непокрытого сценария",
      "importance": "high | medium | low"
    }}
  ],
  "strengths": ["что сделано хорошо"],
  "overall_comment": "общий комментарий эксперта"
}}"""


class Stage6Verify(BasePipelineStage):
    stage_name = "stage6_verify"

    def build_prompt(self, graph: dict, text: str) -> str:
        return PROMPT_TEMPLATE.format(
            graph_json=json.dumps(graph, ensure_ascii=False, indent=2)[:8000],
            text=text[:3000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "clinical_accuracy_score" not in data:
            raise PipelineError("Stage6: missing 'clinical_accuracy_score'")
        logger.info(
            f"Stage 6: accuracy={data.get('clinical_accuracy_score')}, "
            f"issues={len(data.get('issues', []))}"
        )
        return data

    def run(self, graph: dict, text: str) -> dict:
        prompt = self.build_prompt(graph, text)
        return self._execute_with_retry(prompt)
