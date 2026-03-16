# Clinical Graph Builder

Инструмент для автоматического извлечения клинических графов принятия решений из PDF-документов с медицинскими рекомендациями. На входе — PDF, на выходе — валидированный JSON-граф, пригодный для интеграции в медицинские системы поддержки принятия решений.

## Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Быстрый старт](#быстрый-старт)
- [Установка без Docker](#установка-без-docker)
- [Конфигурация](#конфигурация)
- [Формат выходного графа](#формат-выходного-графа)
- [Валидатор графа](#валидатор-графа)
- [Сравнение версий pipeline](#сравнение-версий-pipeline)
- [Структура проекта](#структура-проекта)
- [Известные ограничения](#известные-ограничения)

---

## Обзор

Приложение использует каскад LLM-промптов на базе Google Gemini для поэтапного извлечения и структурирования клинических алгоритмов. Каждый этап специализируется на одном аспекте (методы, факторы, логика, граф), что позволяет обрабатывать большие документы через разбивку на чанки и переиспользовать результаты отдельных стадий через кэш.

**Пример задачи:** клинические рекомендации по переломам проксимального отдела бедренной кости (Pipkin I-IV, Garden I-IV, 31A) → граф с 13–15 клиническими маршрутами.

---

## Архитектура

### Pipeline — 5 этапов, 8 стадий

Предметно-ориентированный каскад: каждый этап Этапа 1 извлекает отдельный класс данных, Этап 3 строит логический алгоритм перед генерацией графа.

```
Stage 1a  Методы остеосинтеза        (chunk-aware)
Stage 1b  Методы эндопротезирования  (chunk-aware)
Stage 1c  Тактика по типам переломов (chunk-aware)
Stage 2   Факторы и классификации    (chunk-aware)
Stage 3   Структура алгоритма        (IF-THEN логика, уровни вложенности)
Stage 4   Генерация графа JSON
Stage 5a  Валидация полноты
Stage 5b  Исправление графа          (validate → fix, до 2 итераций)
          → Финальная структурная валидация
```

**Рекомендуемая температура:** `0.1` — оптимальный баланс между детерминизмом и полнотой (при `0.0` модель склонна оставлять незакрытые ветки и недостижимые узлы).

---

## Быстрый старт

### Docker (рекомендуется)

```bash
# 1. Скопировать конфиг
cp .env.example .env
# Вписать GEMINI_API_KEY в .env

# 2. Положить PDF
cp guidelines.pdf data/input/

# 3. Собрать и запустить
docker compose run --rm clinical-graph-builder \
  --input  /data/input/guidelines.pdf \
  --output /data/output/graph.json \
  --metrics /data/output/metrics.json \
  --max-fix-iterations 5
```

Результаты появятся в `data/output/graph.json` и `data/output/metrics.json`.

```bash
# Запуск с подробным логом
make run-verbose

# Повторный запуск без повторного обращения к API (используется кэш стадий)
make run-cached

# Консоль внутри контейнера
make shell
```

---

## Установка без Docker

**Требования:** Python 3.12+

```bash
pip install -r requirements.txt
```

### Pipeline

```bash
python main.py \
  --input  data/input/guidelines.pdf \
  --output data/output/graph.json \
  --metrics data/output/metrics_v2.json \
  --topic "clinical guidelines" \
  --section "переломы шейки бедра" \
  --model gemini-3.1-flash-lite-preview \
  --rpm 15 \
  --cache-dir data/cache \
  --use-cache \
  --verbose
```

### Все параметры CLI

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--input` | — | Путь к входному PDF (обязательный) |
| `--output` | — | Путь к выходному JSON графа (обязательный) |
| `--metrics` | `metrics.json` / `metrics_v2.json` | Путь к файлу метрик |
| `--section` | `""` | Подсказка для фокусировки на нужном разделе PDF |
| `--topic` | `"clinical guidelines"` | Метка темы в metadata графа (v2) |
| `--model` | `gemini-3.1-flash-lite-preview` | Имя модели Gemini |
| `--rpm` | `15` | Ограничение запросов в минуту |
| `--chunk-size` | `12000` | Максимум символов в одном чанке |
| `--overlap` | `400` | Перекрытие между соседними чанками (символов) |
| `--cache-dir` | `data/cache` | Директория кэша стадий |
| `--use-cache` | `False` | Загружать результаты стадий из кэша |
| `--api-key` | `$GEMINI_API_KEY` | API ключ (иначе из переменной окружения) |
| `--verbose` | `False` | Подробное логирование (уровень DEBUG) |

---

## Конфигурация

Скопируйте `.env.example` в `.env` и заполните:

```dotenv
# Обязательно
GEMINI_API_KEY=AIza...

# Модель и квота
MODEL=gemini-3.1-flash-lite   # free tier: 15 RPM
RPM=15

# Разбивка документа
CHUNK_SIZE=12000          # ~3000 токенов — безопасно для всех моделей Gemini
OVERLAP=400               # перекрытие для сохранения контекста на границах

# Файлы
INPUT_FILE=guidelines.pdf
OUTPUT_FILE=graph.json
METRICS_FILE=metrics.json

# Опционально
SECTION=                  # например: "переломы шейки бедра"
VERBOSE=                  # любое непустое значение = включить
USE_CACHE=                # любое непустое значение = включить
```

**Выбор модели:**

| Модель | RPM (free) | Качество | Рекомендация |
|---|---|---|---|
| `gemini-2.0-flash` | 15 | ★★★★ | По умолчанию |
| `gemini-2.0-flash-lite` | 30 | ★★★ | При нехватке квоты |
| `gemini-3.1-flash-lite-preview` | 30 | ★★★ | При нехватке квоты |
| `gemini-1.5-pro` | 2 | ★★★★★ | Для сложных документов |

---

## Формат выходного графа

```json
{
  "metadata": {
    "source_document": "guidelines.pdf",
    "created_at": "2026-03-07T19:00:09Z",
    "version": "1.1",
    "topic": "clinical guidelines",
    "pipeline_version": "v2"
  },
  "graph": {
    "nodes": [
      {
        "id": "start_001",
        "type": "START",
        "label": "Начало алгоритма",
        "question": null,
        "options": [],
        "action_details": null
      },
      {
        "id": "dec_fracture_type",
        "type": "DECISION",
        "label": "Тип перелома",
        "question": "Выберите тип перелома:",
        "options": ["Pipkin", "Garden I-II", "31A1.3/31A2"],
        "action_details": null
      },
      {
        "id": "act_pipkin_i",
        "type": "ACTION",
        "label": "Удаление фрагмента",
        "question": null,
        "options": [],
        "action_details": {
          "procedure": "Удаление фрагмента головки БК",
          "implant": null,
          "timing": "Срочно",
          "evidence_level": "C",
          "contraindications": ["Нестабильность"],
          "notes": "Pipkin I"
        }
      },
      {
        "id": "end_001",
        "type": "END",
        "label": "Конец",
        "question": null,
        "options": [],
        "action_details": null
      }
    ],
    "edges": [
      {
        "id": "edge_001",
        "from": "start_001",
        "to": "dec_fracture_type",
        "label": null,
        "condition": null
      },
      {
        "id": "edge_002",
        "from": "dec_fracture_type",
        "to": "act_pipkin_i",
        "label": "Pipkin I",
        "condition": {
          "field": "fracture_type",
          "operator": "==",
          "value": "Pipkin I"
        }
      }
    ]
  },
  "changelog": [
    {
      "action": "added",
      "element": "node",
      "id": "end_001",
      "reason": "Автоисправление: добавлен узел END"
    }
  ]
}
```

### Типы узлов

| Тип | `question` | `options` | `action_details` | Описание |
|---|---|---|---|---|
| `START` | `null` | `[]` | `null` | Ровно один, точка входа |
| `DECISION` | текст | ≥ 2 варианта | `null` | Развилка — вопрос врачу |
| `ACTION` | `null` | `[]` | обязателен | Конкретная операция / назначение |
| `WARNING` | `null` | `[]` | обязателен | Предупреждение о противопоказаниях; стоит **до** ACTION |
| `END` | `null` | `[]` | `null` | Конец маршрута (может быть несколько) |

---

## Валидатор графа

`pipeline/graph_validator.py` выполняет **15 детерминированных проверок** без обращения к LLM:

| # | Проверка |
|---|---|
| 1 | Ровно один START-узел |
| 2 | Хотя бы один END-узел |
| 3 | START без question/options/action_details |
| 4 | Каждый DECISION имеет question + ≥ 2 options |
| 5 | Каждый ACTION имеет action_details |
| 6 | Нет дублирующих рёбер (одинаковые from+to) |
| 7 | Нет самопетель |
| 8 | Все узлы из рёбер существуют в nodes |
| 9 | Каждый вариант DECISION.options покрыт исходящим ребром |
| 10 | Каждый ACTION/WARNING имеет ребро к END; ACTION ведёт **только** к END |
| 11 | Нет недостижимых узлов из START |
| 12 | Нет циклов (DAG-проверка обходом в глубину) |
| 13 | Число исходящих рёбер DECISION ≤ len(options) + 1 |
| 14 | Одно поле не проверяется дважды на одном пути START→END |
| 15 | Ребро не проверяет поле, уже определённое на пути к его источнику |

Запуск отдельно:

```python
from pipeline.graph_validator import validate
import json

graph = json.load(open("graph.json"))
issues = validate(graph)
for issue in issues:
    print(f"[{issue['severity'].upper()}] {issue['description']}")
    print(f"  → {issue['suggestion']}")
```

---

## Структура проекта

```
clinical_graph_builder/
│
├── main.py                      
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── Makefile
│
│
├── pipeline/                 
│   ├── base.py                  
│   ├── rate_limiter.py
│   ├── chunker.py
│   ├── graph_validator.py
│   ├── stage1a_osteosynthesis.py
│   ├── stage1b_arthroplasty.py
│   ├── stage1c_fracture_specific.py
│   ├── stage2_factors.py
│   ├── stage3_algorithm.py
│   ├── stage4_graph.py
│   ├── stage5a_validate.py
│   └── stage5b_fix.py
│
├── metrics/
│   └── graph_metrics.py
│
├── schemas/
│   └── graph_schema.json        # JSON Schema для валидации выходного графа
│
└── data/
    ├── input/                   # Входные PDF (монтируется в Docker read-only)
    ├── output/                  # Выходные JSON графы и метрики
    └── cache/                   # Кэш результатов отдельных стадий
```

---
