"""PDF image extraction utilities.

Renders PDF pages as images and classifies them as:
  - "figure"  : page that is primarily a figure / flowchart / diagram
  - "table"   : page with a dominant table structure
  - "text"    : page that is predominantly text (skip for vision stage)
  - "mixed"   : page with substantial text AND visual elements

Only "figure", "table", and "mixed" pages are returned for LLM vision processing.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PageImage:
    """One rendered PDF page ready for multimodal LLM input."""
    page_number: int          # 1-based
    page_type: str            # "figure" | "table" | "mixed" | "text"
    base64_png: str           # base64-encoded PNG
    text_snippet: str = ""    # first 200 chars of text on this page (for context)
    width_px: int = 0
    height_px: int = 0


def extract_page_images(
    pdf_path: str,
    resolution: int = 150,
    max_pages: int = 60,
    skip_text_only: bool = True,
) -> list[PageImage]:
    """
    Render each PDF page to PNG and classify it.

    Parameters
    ----------
    pdf_path      : path to the PDF file
    resolution    : DPI for rendering (150 is a good balance of quality vs size)
    max_pages     : safety cap — skip pages beyond this index
    skip_text_only: if True, pages classified as "text" are excluded

    Returns
    -------
    List of PageImage objects for non-text pages (or all pages if skip_text_only=False)
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is required: pip install pdfplumber")

    results: list[PageImage] = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        logger.info(f"PDF has {total} pages, rendering up to {min(total, max_pages)}")

        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text() or ""
            images_on_page = page.images or []
            tables_on_page = page.find_tables() or []

            page_type = _classify_page(text, images_on_page, tables_on_page)

            if skip_text_only and page_type == "text":
                continue

            try:
                img = page.to_image(resolution=resolution)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                w, h = img.original.size
            except Exception as e:
                logger.warning(f"Page {i+1}: render failed — {e}")
                continue

            results.append(PageImage(
                page_number=i + 1,
                page_type=page_type,
                base64_png=b64,
                text_snippet=text[:300].strip(),
                width_px=w,
                height_px=h,
            ))
            logger.info(
                f"  Page {i+1}/{total}: type={page_type}, "
                f"size={w}x{h}, "
                f"images={len(images_on_page)}, tables={len(tables_on_page)}"
            )

    logger.info(
        f"Extracted {len(results)} visual pages from {min(total, max_pages)} processed"
    )
    return results


def _classify_page(
    text: str,
    images: list,
    tables: list,
) -> str:
    """Heuristic page classifier."""
    has_images = len(images) > 0
    has_tables = len(tables) > 0
    text_len = len(text.strip())

    # Mostly empty or very short text with images → figure
    if has_images and text_len < 300:
        return "figure"

    # Table dominant
    if has_tables and not has_images:
        return "table"

    # Table + image
    if has_tables and has_images:
        return "mixed"

    # Image + substantial text
    if has_images and text_len >= 300:
        return "mixed"

    # Pure text
    return "text"
