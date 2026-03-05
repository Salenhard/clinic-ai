"""Stage 5: Final graph assembly and enrichment."""
import json
import logging
from datetime import datetime, timezone
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — медицинский эксперт по клиническим рекомендациям.

Собери финальный граф принятия клинических решений. Для каждого ACTION-узла добавь или дополни:
- evidence_level: уровень доказательности из текста (если не заполнен)
- contraindications: список противопоказаний
- notes: клинические примечания и нюансы применения

УЗЛЫ:
{nodes_json}

РЁБРА:
{edges_json}

ИСХОДНЫЙ ТЕКСТ (для справки):
{text_fragment}

Верни СТРОГО JSON без пояснений, без markdown-блоков.
Структура должна содержать enriched_nodes — массив всех узлов с дополненными action_details:
{{
  "enriched_nodes": [
    {{
      "id": "node_001",
      "type": "...",
      "label": "...",
      "question": "...",
      "options": [],
      "action_details": {{
        "procedure": "...",
        "implant": "...",
        "timing": "...",
        "evidence_level": "...",
        "contraindications": ["..."],
        "notes": "..."
      }}
    }}
  ]
}}"""


class Stage5Assembly(BasePipelineStage):
    stage_name = "stage5_assembly"

    def build_prompt(self, nodes: dict, edges: dict, text: str) -> str:
        return PROMPT_TEMPLATE.format(
            nodes_json=json.dumps(nodes, ensure_ascii=False, indent=2)[:5000],
            edges_json=json.dumps(edges, ensure_ascii=False, indent=2)[:3000],
            text_fragment=text[:2000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "enriched_nodes" not in data:
            raise PipelineError("Stage5: missing 'enriched_nodes' key")
        logger.info(f"Stage 5: enriched {len(data['enriched_nodes'])} nodes")
        return data

    def assemble_final_graph(
        self,
        enriched_nodes: list,
        edges: list,
        source_document: str,
        topic: str
    ) -> dict:
        """Build the final output JSON structure."""
        return {
            "metadata": {
                "source_document": source_document,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "topic": topic
            },
            "graph": {
                "nodes": enriched_nodes,
                "edges": edges
            }
        }

    def run(self, nodes: dict, edges: dict, text: str) -> dict:
        prompt = self.build_prompt(nodes, edges, text)
        return self._execute_with_retry(prompt)
