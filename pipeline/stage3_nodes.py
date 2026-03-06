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
ТИПЫ УЗЛОВ:
═══════════════════════════════════════════

START — точка входа, ровно один:
  • question=null, options=[], action_details=null

DECISION — вопрос с ветвлением:
  • question (текст вопроса), options (все возможные варианты)
  • action_details=null

ACTION — рекомендуемое лечение:
  • question=null, options=[]
  • action_details обязателен

WARNING — предупреждение:
  • question=null, options=[]

END — завершение ветки (может быть несколько):
  • question=null, options=[], action_details=null

═══════════════════════════════════════════
КРИТИЧЕСКИ ВАЖНОЕ ПРАВИЛО — ПАРАМЕТРЫ-ГЛОБАЛЬНЫЕ VS ЛОКАЛЬНЫЕ:
═══════════════════════════════════════════

Проанализируй все параметры ветвления из правил.
Раздели их на две группы:

ГЛОБАЛЬНЫЕ параметры — используются В НЕСКОЛЬКИХ разных ветках алгоритма
(например, возраст влияет на тактику при Pipkin II И при 31A2 И при Garden):
  → Такой параметр ДОЛЖЕН быть вынесен в начало алгоритма, ПЕРЕД разветвлением
  → Создай ОДИН DECISION-узел для него (не несколько одинаковых)
  → Структура: START → DECISION(глобальный) → DECISION(специфический) → ACTION

ЛОКАЛЬНЫЕ параметры — используются только в одной конкретной ветке:
  → DECISION-узел создаётся внутри этой ветки

ПРИМЕР ПРАВИЛЬНОЙ СТРУКТУРЫ (если возраст глобальный):
  START
    └→ DECISION "Возраст" [<60, ≥60]
         ├→ DECISION "Тип перелома" [для <60]  → ... → ACTION
         └→ DECISION "Тип перелома" [для ≥60]  → ... → ACTION

ПРИМЕР ПРАВИЛЬНОЙ СТРУКТУРЫ (если тип перелома первичен):
  START
    └→ DECISION "Тип перелома" [Pipkin, Garden, 31A]
         ├→ DECISION "Подтип Pipkin" [I, II, III, IV]
         │    ├→ Pipkin I → ACTION
         │    ├→ Pipkin II → DECISION "Возраст" → ACTION  (возраст только здесь)
         │    └→ ...
         └→ DECISION "Тип Garden" [I-II, III-IV] → ACTION

АНТИПАТТЕРН (запрещено):
  ❌ Два разных узла "Возраст пациента" в разных ветках одного алгоритма
  ❌ Один и тот же вопрос задаётся более одного раза на одном пути
  ❌ Пациент проходит через один и тот же тип вопроса дважды
  ❌ Один DECISION-узел достигается из двух веток с разными значениями одного параметра
     (например, node_003 достигается и через "Младше 60" и через "60+") — это
     создаёт неопределённость: узел не знает какая тактика нужна.
     РЕШЕНИЕ: создай два отдельных узла — node_003_young и node_003_old

═══════════════════════════════════════════
ДРУГИЕ ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
═══════════════════════════════════════════
1. START не имеет question/options
2. Каждая ветка завершается END
3. ACTION и WARNING — листовые узлы
4. Не дублируй одну процедуру в разных узлах без клинической причины

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

        # Warn about duplicate DECISION questions
        questions = {}
        for node in data["nodes"]:
            if node.get("type") == "DECISION" and node.get("question"):
                q = node["question"].strip().lower()
                if q in questions:
                    logger.warning(
                        f"Stage3: duplicate question '{node['question']}' "
                        f"in nodes {questions[q]} and {node['id']} — "
                        f"consider merging into one DECISION node"
                    )
                else:
                    questions[q] = node["id"]

        logger.info(f"Stage 3: built {len(data['nodes'])} nodes")
        return data

    def run(self, rules: dict) -> dict:
        return self._execute_with_retry(self.build_prompt(rules))
