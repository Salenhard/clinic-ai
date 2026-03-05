# Clinical Graph Builder

Программа для извлечения клинических алгоритмов принятия решений из PDF-рекомендаций
с помощью каскадных запросов к Claude API.

## Запуск через Docker (рекомендуется)

### Быстрый старт

```bash
# 1. Скопируйте шаблон окружения и укажите API-ключ
cp .env.example .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 2. Положите PDF в ./data/input/
mkdir -p data/input data/output data/cache
cp /path/to/guidelines.pdf data/input/

# 3. Соберите образ
docker compose build
# или: make build

# 4. Запустите
docker compose run --rm clinical-graph-builder \
  --input  /data/input/guidelines.pdf \
  --output /data/output/graph.json \
  --metrics /data/output/metrics.json \
  --section "переломы шейки бедра" \
  --verbose

# Результаты появятся в ./data/output/
```

### Через Makefile (удобные сокращения)

```bash
make build        # собрать образ
make run          # запустить с настройками из .env
make run-verbose  # то же, но с подробным логом
make run-cached   # повторный запуск без повторного извлечения (быстро)
make shell        # bash внутри контейнера для отладки
make clean        # удалить образ и кэш
```

### Через docker run (без Compose)

```bash
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/data/input:/data/input:ro \
  -v $(pwd)/data/output:/data/output \
  -v $(pwd)/data/cache:/app/pipeline_cache \
  clinical-graph-builder:latest \
  --input  /data/input/guidelines.pdf \
  --output /data/output/graph.json \
  --metrics /data/output/metrics.json \
  --section "переломы шейки бедра" \
  --verbose
```

### Переменные окружения (.env)

| Переменная | Обязательна | Описание |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Ключ Claude API |
| `INPUT_FILE` | — | Имя PDF внутри `./data/input/` (default: `guidelines.pdf`) |
| `OUTPUT_FILE` | — | Имя выходного JSON (default: `graph.json`) |
| `METRICS_FILE` | — | Имя файла метрик (default: `metrics.json`) |
| `MODEL` | — | Claude модель (default: `claude-sonnet-4-20250514`) |
| `SECTION` | — | Раздел для фокусировки (пусто = весь документ) |
| `VERBOSE` | — | Любое значение включает `--verbose` |
| `USE_CACHE` | — | Любое значение включает `--use-cache` |

### Монтируемые тома

```
./data/input/   → /data/input   (read-only)   — входные PDF
./data/output/  → /data/output               — graph.json, metrics.json
./data/cache/   → /app/pipeline_cache        — промежуточные этапы
```

Кэш `./data/cache/` сохраняется между запусками. При `--use-cache` пропускаются уже завершённые этапы — удобно при отладке или повторной обработке того же документа.

---

## Локальная установка (без Docker)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_api_key_here"
```

### Базовый запуск
```bash
python main.py --input guidelines.pdf --output graph.json
```

### С указанием раздела и метриками
```bash
python main.py \
  --input guidelines.pdf \
  --output graph.json \
  --metrics metrics.json \
  --section "переломы шейки бедра" \
  --verbose
```

### С использованием кэша (для повторных запусков)
```bash
python main.py --input guidelines.pdf --use-cache
```

### Все параметры
```
--input, -i      Путь к входному PDF (обязательно)
--output, -o     Путь к выходному JSON-графу (default: graph.json)
--metrics, -m    Путь к файлу метрик (default: metrics.json)
--section, -s    Раздел документа для фокусировки (опционально)
--model          Claude модель (default: claude-sonnet-4-20250514)
--verbose, -v    Подробный лог
--use-cache      Использовать кэш из pipeline_cache/
```

## Архитектура каскада

```
PDF → text extraction
         │
    Stage 1: Entity Extraction      → pipeline_cache/stage1_entities.json
         │
    Stage 2: Rule Extraction        → pipeline_cache/stage2_rules.json
         │
    Stage 3: Node Construction      → pipeline_cache/stage3_nodes.json
         │
    Stage 4: Edge Construction      → pipeline_cache/stage4_edges.json
         │
    Stage 5: Assembly & Enrichment  → pipeline_cache/stage5_assembly.json
         │
    Stage 6: Clinical Verification  → pipeline_cache/stage6_verify.json
         │
    graph.json + metrics.json
```

## Выходной формат (graph.json)

```json
{
  "metadata": {
    "source_document": "guidelines.pdf",
    "created_at": "2025-01-01T00:00:00+00:00",
    "version": "1.0",
    "topic": "переломы шейки бедра"
  },
  "graph": {
    "nodes": [
      {
        "id": "node_001",
        "type": "START | DECISION | ACTION | WARNING | END",
        "label": "...",
        "question": "...",
        "options": [],
        "action_details": {
          "procedure": "...",
          "implant": "...",
          "timing": "...",
          "evidence_level": "1A",
          "contraindications": [],
          "notes": "..."
        }
      }
    ],
    "edges": [
      {
        "id": "edge_001",
        "from": "node_001",
        "to": "node_002",
        "label": "Возраст < 60 лет",
        "condition": {
          "field": "age",
          "operator": "<",
          "value": "60"
        }
      }
    ]
  }
}
```

## Структура проекта

```
clinical_graph_builder/
├── Dockerfile               # Multi-stage сборка (builder + runtime)
├── docker-compose.yml       # Compose с volume-маппингом и .env
├── .env.example             # Шаблон переменных окружения
├── .dockerignore
├── Makefile                 # Удобные сокращения (build/run/shell/clean)
├── main.py                  # CLI точка входа
├── requirements.txt
├── data/                    # (создаётся локально, не в репо)
│   ├── input/               # Входные PDF
│   ├── output/              # graph.json, metrics.json
│   └── cache/               # Кэш промежуточных этапов
├── pipeline/
│   ├── __init__.py
│   ├── base.py              # Базовый класс с retry и JSON repair
│   ├── stage1_entities.py   # Извлечение сущностей
│   ├── stage2_rules.py      # Извлечение правил
│   ├── stage3_nodes.py      # Построение узлов
│   ├── stage4_edges.py      # Построение рёбер
│   ├── stage5_assembly.py   # Финальная сборка
│   └── stage6_verify.py     # Клиническая верификация
├── metrics/
│   ├── __init__.py
│   └── graph_metrics.py     # Метрики через networkx
└── schemas/
    └── graph_schema.json    # JSON Schema для валидации
```

## Метрики

Программа автоматически рассчитывает:
- **Структурные**: число узлов, рёбер, глубина, ветвление
- **Связность**: связен ли граф, изолированные узлы, циклы
- **Полнота**: покрытие правил из текста, наличие уровней доказательности
- **LLM**: потраченные токены, завершённые этапы
- **Клинические**: оценка точности, пропущенные сценарии
