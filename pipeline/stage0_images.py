"""Stage 0: Extract clinical information from PDF page images using Gemini vision.

Processes pages classified as "figure", "table", or "mixed" — i.e. pages that
contain flowcharts, decision trees, classification tables, or implant diagrams
that pdfplumber cannot reliably extract as text.

Returns structured data in the same format as Stages 1a/1b/1c/2, which is
merged with text-extracted data before Stage 3.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .base import BasePipelineStage, PipelineError
from .pdf_images import PageImage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── System instruction override for vision calls ──────────────────────────────
_VISION_SYSTEM = (
    "Ты анализатор содержимого картинок."
    "Ты анализируешь страницы медицинских рекомендаций на русском языке."
    "Отвечай только валидным JSON без пояснений."
)

# ── Per-image prompt ──────────────────────────────────────────────────────────
_PROMPT_FIGURE = """\
Это страница {page_num} из клинических рекомендаций по хирургическому лечению переломов.
Тип страницы: {page_type}.
{text_hint}

Проанализируй изображение и извлеки ВСЮ клинически значимую информацию.

Верни СТРОГО JSON со следующими полями (оставь пустым то, чего нет на странице):
{{
  "page_number": {page_num},
  "page_type": "{page_type}",
  "description": "краткое описание содержимого страницы",

  "fracture_types": [
    {{
      "name": "название типа перелома",
      "classification": "система классификации (Garden/Pipkin/AO/31A и т.д.)",
      "subtypes": ["I", "II", ...],
      "notes": "особенности"
    }}
  ],

  "treatment_methods": [
    {{
      "method": "название метода",
      "implant": "имплант или null",
      "indication": "показание",
      "contraindication": "противопоказание или null",
      "timing": "срочно/планово/null",
      "evidence_level": "уровень доказательности или null"
    }}
  ],

  "decision_rules": [
    {{
      "condition": "если ...",
      "action": "то ...",
      "notes": "комментарий или null"
    }}
  ],

  "thresholds": [
    {{
      "parameter": "название параметра (возраст, TAD и т.д.)",
      "operator": "< | > | <= | >= | ==",
      "value": "числовое значение или текст",
      "implication": "клинический смысл порогового значения"
    }}
  ],

  "warnings": [
    "текст предупреждения или противопоказания"
  ],

  "flowchart_paths": [
    "текстовое описание ветки алгоритма, например: Pipkin I → удаление фрагмента"
  ]
}}"""

# ── Merge logic ───────────────────────────────────────────────────────────────

def _merge_image_results(results: list[dict]) -> dict:
    """Merge per-page vision results into unified structure."""
    merged = {
        "fracture_types": [],
        "treatment_methods": [],
        "decision_rules": [],
        "thresholds": [],
        "warnings": [],
        "flowchart_paths": [],
        "pages_analyzed": [],
    }

    seen_methods: set[str] = set()
    seen_rules: set[str] = set()
    seen_fractures: set[str] = set()

    for r in results:
        merged["pages_analyzed"].append({
            "page": r.get("page_number"),
            "type": r.get("page_type"),
            "description": r.get("description", ""),
        })

        for ft in r.get("fracture_types", []):
            key = ft.get("name", "").lower().strip()
            if key and key not in seen_fractures:
                seen_fractures.add(key)
                merged["fracture_types"].append(ft)

        for tm in r.get("treatment_methods", []):
            key = (tm.get("method", "") + tm.get("indication", "")).lower().strip()
            if key and key not in seen_methods:
                seen_methods.add(key)
                merged["treatment_methods"].append(tm)

        for dr in r.get("decision_rules", []):
            key = (dr.get("condition", "") + dr.get("action", "")).lower().strip()[:80]
            if key and key not in seen_rules:
                seen_rules.add(key)
                merged["decision_rules"].append(dr)

        merged["thresholds"].extend(r.get("thresholds", []))
        merged["warnings"].extend(r.get("warnings", []))
        merged["flowchart_paths"].extend(r.get("flowchart_paths", []))

    # Deduplicate simple lists
    merged["warnings"] = list(dict.fromkeys(merged["warnings"]))
    merged["flowchart_paths"] = list(dict.fromkeys(merged["flowchart_paths"]))

    return merged


class Stage0Images(BasePipelineStage):
    stage_name = "stage0_images"

    # Vision calls are heavier — reduce parallelism
    CHUNK_WORKERS = 1

    def build_prompt(self, page: PageImage) -> str:
        text_hint = (
            f'Текст на странице (фрагмент): "{page.text_snippet}"'
            if page.text_snippet
            else "Текст на странице отсутствует или нечитаем."
        )
        return _PROMPT_FIGURE.format(
            page_num=page.page_number,
            page_type=page.page_type,
            text_hint=text_hint,
        )

    def _call_vision(self, page: PageImage) -> dict:
        """Send one page image to Gemini vision and parse result."""
        from google.genai import types as genai_types
        import base64 as _base64
        self._limiter.acquire()

        prompt_text = self.build_prompt(page)
        image_bytes = _base64.b64decode(page.base64_png)
        contents = [
           genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=image_bytes)),
           genai_types.Part(text=prompt_text)
        ]

        config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            system_instruction=_VISION_SYSTEM,
        )

        import time
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                meta = getattr(response, "usage_metadata", None)
                if meta:
                    self.tokens_used += getattr(meta, "prompt_token_count", 0)
                    self.tokens_used += getattr(meta, "candidates_token_count", 0)

                raw = response.text
                return json.loads(self._clean_json(raw))

            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "quota" in exc_str or "resource_exhausted" in exc_str:
                    logger.warning(
                        f"Stage0 page {page.page_number}: quota — "
                        f"backing off {self.RATE_LIMIT_BACKOFF}s"
                    )
                    time.sleep(self.RATE_LIMIT_BACKOFF)
                elif attempt == self.MAX_RETRIES:
                    raise PipelineError(
                        f"Stage0 page {page.page_number} failed after "
                        f"{self.MAX_RETRIES} attempts: {exc}"
                    )
                else:
                    logger.warning(
                        f"Stage0 page {page.page_number} attempt {attempt}: {exc}"
                    )
                    time.sleep(self.RETRY_DELAY * attempt)

        raise PipelineError(f"Stage0 page {page.page_number}: all retries exhausted")

    def parse_response(self, response_text: str) -> dict:
        # Not used directly — _call_vision handles parsing
        return json.loads(self._clean_json(response_text))

    def run(self, pages: list[PageImage]) -> dict:
        """Process all visual pages and return merged structured data."""
        if not pages:
            logger.info("Stage 0: no visual pages — skipping")
            return {
                "fracture_types": [], "treatment_methods": [],
                "decision_rules": [], "thresholds": [],
                "warnings": [], "flowchart_paths": [], "pages_analyzed": [],
            }

        logger.info(f"Stage 0: processing {len(pages)} visual pages via Gemini vision")

        results = []
        for page in pages:
            logger.info(
                f"  Stage 0: page {page.page_number} ({page.page_type}, "
                f"{page.width_px}x{page.height_px}px)"
            )
            try:
                result = self._call_vision(page)
                results.append(result)
                n_paths = len(result.get("flowchart_paths", []))
                n_methods = len(result.get("treatment_methods", []))
                logger.info(
                    f"    → {n_methods} methods, {n_paths} flowchart paths"
                )
            except PipelineError as e:
                logger.error(f"  Stage 0 page {page.page_number} skipped: {e}")

        merged = _merge_image_results(results)
        logger.info(
            f"Stage 0 complete: "
            f"{len(merged['fracture_types'])} fracture types, "
            f"{len(merged['treatment_methods'])} treatment methods, "
            f"{len(merged['decision_rules'])} decision rules, "
            f"{len(merged['flowchart_paths'])} flowchart paths"
        )
        return merged
