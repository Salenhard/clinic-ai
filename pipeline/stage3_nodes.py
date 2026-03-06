"""Stage 3: Build graph nodes from rules."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Ты — эксперт по построению клинических алгоритмов принятия решений.

На основе извлечённых правил построй УЗЛЫ графа консультации.

ПРАВИЛА:
{rules_json}

═══════════════════════════════════════════
ТИПЫ УЗЛОВ И СТРОГИЕ ПРАВИЛА ДЛЯ КАЖДОГО:
═══════════════════════════════════════════

START — точка входа:
  • Ровно ОДИН в графе
  • НЕТ question, НЕТ options, НЕТ action_details — все поля null
  • label: "Начало консультации"
  • Единственная его роль — указать на первый DECISION-узел

DECISION — вопрос с ветвлением:
  • Содержит question (текст вопроса) и options (список вариантов ответа)
  • Каждый вариант из options ОБЯЗАН иметь исходящее ребро в Stage 4
  • action_details = null

ACTION — рекомендуемое лечение:
  • question = null, options = []
  • action_details ОБЯЗАТЕЛЕН: procedure, implant, timing, evidence_level, contraindications[], notes
  • После ACTION всегда следует END (через ребро)

WARNING — предупреждение:
  • question = null, options = []
  • После WARNING всегда следует END (через ребро)

END — завершение ветки:
  • question = null, options = [], action_details = null
  • Может быть несколько END-узлов (по одному на каждую терминальную ветку)

═══════════════════════════════════════════
ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
═══════════════════════════════════════════
1. НЕ добавляй question/options в START — только в DECISION
2. Каждая ветка алгоритма ДОЛЖНА завершаться узлом END
3. ACTION и WARNING — листовые узлы (из них исходит только ребро к END)
4. Не дублируй одно и то же лечение — одна процедура = один ACTION-узел
5. Условия ветвления (возраст, тип перелома) — это атрибуты рёбер, не узлов

Верни СТРОГО JSON без пояснений, без markdown:
{{
  "nodes": [
    {{
      "id": "node_001",
      "type": "START | DECISION | ACTION | WARNING | END",
      "label": "...",
      "question": "текст вопроса или null",
      "options": ["вариант А", "вариант Б"],
      "source_rules": ["rule_001"],
      "action_details": {{
        "procedure": "...",
        "implant": "...",
        "timing": "...",
        "evidence_level": "...",
        "contraindications": [],
        "notes": "..."
      }}
    }}
  ]
}}"""


class Stage3Nodes(BasePipelineStage):
    stage_name = "stage3_nodes"

    def build_prompt(self, rules: dict) -> str:
        return PROMPT_TEMPLATE.format(
            rules_json=json.dumps(rules, ensure_ascii=False, indent=2)[:8000]
        )

    def parse_response(self, response_text: str) -> dict:
        cleaned = self._clean_json(response_text)
        data = json.loads(cleaned)
        if "nodes" not in data:
            raise PipelineError("Stage3: missing 'nodes' key in response")

        valid_types = {"START", "DECISION", "ACTION", "WARNING", "END"}
        start_count = 0
        for node in data["nodes"]:
            t = node.get("type")
            if t not in valid_types:
                logger.warning(f"Node {node.get('id')} invalid type '{t}' → ACTION")
                node["type"] = "ACTION"
            if t == "START":
                start_count += 1
                # Enforce: START must not have question/options
                node["question"] = None
                node["options"] = []
                node["action_details"] = None
            if t in ("ACTION", "WARNING") and node.get("question"):
                node["question"] = None
                node["options"] = []

        if start_count == 0:
            logger.warning("Stage3: no START node found, adding one")
            data["nodes"].insert(0, {
                "id": "node_start",
                "type": "START",
                "label": "Начало консультации",
                "question": None,
                "options": [],
                "source_rules": [],
                "action_details": None,
            })
        elif start_count > 1:
            logger.warning(f"Stage3: {start_count} START nodes found, keeping first only")
            seen_start = False
            for node in data["nodes"]:
                if node["type"] == "START":
                    if seen_start:
                        node["type"] = "DECISION"
                    seen_start = True

        logger.info(f"Stage 3: built {len(data['nodes'])} nodes")
        return data

    def run(self, rules: dict) -> dict:
        return self._execute_with_retry(self.build_prompt(rules))
