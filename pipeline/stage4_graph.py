"""Stage 4: Generate graph JSON from algorithm structure and extracted methods."""
import json
import logging
from .base import BasePipelineStage, PipelineError

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Ты — эксперт-аналитик клинических рекомендаций. "
    "Отвечай ТОЛЬКО валидным JSON без пояснений, комментариев и markdown-разметки."
)

_PROMPT = """\
Задача: преобразовать алгоритм принятия решений в граф (nodes + edges).

## ВХОДНЫЕ ДАННЫЕ

### Алгоритм из Stage 3 (АВТОРИТЕТНЫЙ ИСТОЧНИК — строго соблюдай его структуру):
{algorithm_json}

### Методы остеосинтеза:
{osteosynthesis_json}

### Методы эндопротезирования:
{arthroplasty_json}

### Лечение по типам переломов:
{fracture_json}

### Противопоказания:
{contraindications_json}

## ПРАВИЛА УЗЛОВ
- START  — ровно один; question=null, options=[], action_details=null
- DECISION — question (текст вопроса) + options (≥ 2 варианта); action_details=null
- ACTION — question=null, options=[]; action_details ОБЯЗАТЕЛЕН
- WARNING — question=null, options=[]; ставится ДО ACTION, не после
- END    — question=null, options=[], action_details=null

## ПРАВИЛА РЁБЕР
- START → первый DECISION: label=null, condition=null
- Каждый вариант из DECISION.options → РОВНО ОДНО исходящее ребро;
  label ребра должен точно совпадать с текстом варианта
- ACTION → END: единственное исходящее ребро, label=null
- Нет дублей (одинаковые from+to), нет самопетель, нет циклов
- Все узлы достижимы из START

## СТРОГО ЗАПРЕЩЕНО

### Запрет 1 — Схлопывание нозологий
Один ACTION-узел НЕ МОЖЕТ принимать входящие рёбра из разных нозологий.
Для каждой уникальной комбинации (тип перелома × факторы пациента) — отдельный ACTION.

### Запрет 2 — Пустые ветки
Каждый вариант из DECISION.options обязан иметь исходящее ребро.
Количество options == количество исходящих рёбер. Без исключений.

### Запрет 3 — Отклонение от алгоритма Stage 3
Топология веток (порядок факторов, набор вариантов) определяется алгоритмом
из Stage 3 — не изменяй её. Не переставляй факторы, не добавляй развилки,
которых нет в алгоритме.

### Запрет 4 — Общий DECISION для разных нозологий
Нельзя использовать один узел-развилку для двух разных типов переломов.
Каждый тип перелома имеет собственные DECISION-узлы для всех последующих вопросов.

## ACTION.action_details — обязательные поля
{{
  "procedure": "точное название операции из входных данных",
  "implant": "имплантат из входных данных или null",
  "timing": "срочность из входных данных или null",
  "evidence_level": "уровень доказательности из входных данных или null",
  "contraindications": [],
  "notes": "примечания из входных данных или null"
}}

## ВЫХОДНОЙ ФОРМАТ (строго JSON, только nodes + edges)
{{
  "nodes": [
    {{
      "id": "уникальный_id",
      "type": "START | DECISION | ACTION | WARNING | END",
      "label": "краткое название",
      "question": "текст вопроса или null",
      "options": [],
      "action_details": null
    }}
  ],
  "edges": [
    {{
      "id": "edge_001",
      "from": "id узла-источника",
      "to": "id узла-назначения",
      "label": "текст варианта или null",
      "condition": {{"field": "...", "operator": "==", "value": "..."}}
    }}
  ]
}}"""


class Stage4Graph(BasePipelineStage):
    stage_name = "stage4_graph"
    MAX_OUTPUT_TOKENS = 65536  # full graph can exceed 8192 tokens

    def build_prompt(
        self,
        algorithm: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> str:
        def _j(d, key, n=2500):
            raw = json.dumps(d.get(key, []), ensure_ascii=False)[:n]
            return raw.replace("{", "{{").replace("}", "}}")

        def _safe(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        alg_json = json.dumps(algorithm.get("algorithm", {}), ensure_ascii=False)
        if len(alg_json) > 8000:
            logger.warning(
                f"Stage4: algorithm JSON is {len(alg_json)} chars — truncating to 8000. "
                f"Some branches may be lost."
            )
        return _PROMPT.format(
            algorithm_json=_safe(alg_json[:8000]),
            osteosynthesis_json=_j(osteosynthesis, "osteosynthesis_methods"),
            arthroplasty_json=_j(arthroplasty, "arthroplasty_methods"),
            fracture_json=_j(fracture_treatments, "fracture_treatments"),
            contraindications_json=_j(factors, "contraindications"),
        )

    def parse_response(self, response_text: str) -> dict:
        data = json.loads(self._clean_json(response_text))
        if "nodes" not in data or "edges" not in data:
            raise PipelineError("Stage4: missing 'nodes' or 'edges'")

        # Enforce START constraints
        starts = [n for n in data["nodes"] if n.get("type") == "START"]
        if not starts:
            data["nodes"].insert(0, {
                "id": "start_001", "type": "START", "label": "Начало консультации",
                "question": None, "options": [], "action_details": None,
            })
        for n in data["nodes"]:
            if n.get("type") == "START":
                n["question"] = None
                n["options"] = []
                n["action_details"] = None

        # Remove self-loops and duplicates
        seen: set[tuple] = set()
        clean = []
        for e in data["edges"]:
            pair = (e.get("from"), e.get("to"))
            if pair[0] == pair[1]:
                logger.warning(f"Stage4: self-loop on {pair[0]} removed")
                continue
            if pair in seen:
                logger.warning(f"Stage4: duplicate edge {pair} removed")
                continue
            seen.add(pair)
            clean.append(e)
        data["edges"] = clean

        node_type = {n["id"]: n.get("type") for n in data["nodes"]}
        n_nodes = len(data["nodes"])
        n_actions = sum(1 for n in data["nodes"] if n.get("type") == "ACTION")

        # Detect ACTION nodes shared between multiple incoming sources (nozology collapse)
        incoming: dict[str, list[str]] = {}
        for e in data["edges"]:
            dst, src = e.get("to"), e.get("from")
            if dst and src:
                incoming.setdefault(dst, []).append(src)
        for nid, srcs in incoming.items():
            if len(srcs) > 1 and node_type.get(nid) == "ACTION":
                logger.warning(
                    f"Stage4: ACTION '{nid}' reached from {len(srcs)} sources {srcs} — "
                    f"possible nozology collapse."
                )

        # Check DECISION options coverage
        out_edges: dict[str, list[dict]] = {}
        for e in data["edges"]:
            out_edges.setdefault(e.get("from"), []).append(e)

        uncovered_total = []
        for n in data["nodes"]:
            if n.get("type") != "DECISION":
                continue
            opts = n.get("options") or []
            out = out_edges.get(n["id"], [])
            edge_labels = {(e.get("label") or "").strip().lower() for e in out}
            edge_values: set[str] = set()
            for e in out:
                cond = e.get("condition") or {}
                v = cond.get("value")
                if v:
                    edge_values.add(str(v).lower())
            uncovered = [
                o for o in opts
                if o.strip().lower() not in edge_labels
                and o.strip().lower() not in edge_values
            ]
            if uncovered:
                uncovered_total.extend(uncovered)
                logger.warning(
                    f"Stage4: DECISION '{n['id']}' has uncovered options: {uncovered}"
                )

        # Check that all expected fracture types from algorithm made it into the graph.
        # This is a WARNING, not an error — a partial graph is better than None.
        # Missing types will be handled by Stage 5b which has the Stage 3 algorithm.
        expected = getattr(self, "_expected_fracture_types", [])
        if expected:
            all_labels = set()
            for e in data["edges"]:
                lbl = (e.get("label") or "").strip().lower()
                if lbl:
                    all_labels.add(lbl)
            for n in data["nodes"]:
                lbl = (n.get("label") or "").strip().lower()
                if lbl:
                    all_labels.add(lbl)

            missing_in_graph = [
                ft for ft in expected
                if not any(
                    ft.strip().lower() in lbl or lbl in ft.strip().lower()
                    for lbl in all_labels
                )
            ]
            if missing_in_graph:
                # Store on instance so main.py can log it if needed
                self.missing_fracture_types = missing_in_graph
                logger.warning(
                    f"Stage4: {len(missing_in_graph)} fracture type(s) missing from graph "
                    f"(likely hit output token limit): {missing_in_graph}. "
                    f"Stage 5b will complete the graph."
                )
            else:
                self.missing_fracture_types = []
                logger.info(
                    f"Stage4 coverage: all {len(expected)} fracture types present in graph"
                )

        logger.info(f"Stage 4: {n_nodes} nodes ({n_actions} actions), {len(data['edges'])} edges")
        return data

    def run(
        self,
        algorithm: dict,
        osteosynthesis: dict,
        arthroplasty: dict,
        fracture_treatments: dict,
        factors: dict,
    ) -> dict:
        # Store expected fracture types for coverage check in parse_response
        alg_branches = algorithm.get("algorithm", {}).get("branches", [])
        root = next((b for b in alg_branches if b.get("level") == "primary"), None)
        self._expected_fracture_types = list(root.get("options", [])) if root else []

        prompt = self.build_prompt(algorithm, osteosynthesis, arthroplasty,
                                    fracture_treatments, factors)
        return self._execute_with_retry(prompt)
