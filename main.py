#!/usr/bin/env python3
"""
clinical_graph_builder v2 — 5-stage cascade pipeline.

Stage flow:
  1a  Extract osteosynthesis methods   (chunk-aware)
  1b  Extract arthroplasty methods     (chunk-aware)
  1c  Extract fracture treatments      (chunk-aware)
  2   Extract decision factors         (chunk-aware)
  3   Build algorithm structure        (single LLM call)
  4   Generate graph JSON              (single LLM call)
  5a  Validate completeness + structure
  5b  Fix graph (validate→fix loop)    (up to 2 LLM calls)
  ──  Final structural validation
  ──  Assemble output JSON
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai>=1.0.0")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from pipeline import (
    PipelineError,
    Stage1aOsteosynthesis, Stage1bArthroplasty, Stage1cFractureSpecific,
    Stage2Factors, Stage3Algorithm, Stage4Graph,
    Stage5aValidate, Stage5bFix,
    validate_graph_structure, configure_limiter,
)
from pipeline.chunker import TextChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_v2")


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pdf_text(path: str, section_hint: str = "") -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    full = "\n\n".join(text_parts)
    if section_hint:
        low = full.lower()
        hint = section_hint.lower()
        idx = low.find(hint)
        if idx != -1:
            start = max(0, idx - 200)
            return full[start:start + 60000]
    return full[:80000]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, stage_name: str) -> Path:
    return Path(cache_dir) / f"{stage_name}.json"


def load_cache(cache_dir: str, stage_name: str):
    p = _cache_path(cache_dir, stage_name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def save_cache(cache_dir: str, stage_name: str, data: dict):
    p = _cache_path(cache_dir, stage_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Stage runner ───────────────────────────────────────────────────────────────

def run_stage(name: str, fn, cache_dir: str, use_cache: bool,
              failed_stages: list, stages_completed_ref: list):
    """Run a stage with cache support and error handling. Returns result or None."""
    if use_cache:
        cached = load_cache(cache_dir, name)
        if cached:
            logger.info(f"  {name}: loaded from cache")
            return cached
    try:
        result = fn()
        save_cache(cache_dir, name, result)
        stages_completed_ref.append(name)
        return result
    except PipelineError as e:
        logger.error(f"  {name} FAILED: {e}")
        failed_stages.append(name)
        return None
    except Exception as e:
        logger.error(f"  {name} unexpected error: {e}", exc_info=True)
        failed_stages.append(name)
        return None


# ── Output assembly ────────────────────────────────────────────────────────────

def assemble_output(
    graph_data: dict,
    source_path: str,
    topic: str,
    changelog: list,
) -> dict:
    """Wrap graph in the canonical output schema."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    return {
        "metadata": {
            "source_document": Path(source_path).name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "topic": topic or "clinical guidelines",
            "pipeline_version": "v2",
        },
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "changelog": changelog,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(
    output_path: str,
    metrics_path: str,
    stages_completed: list,
    failed_stages: list,
    final_graph: dict,
    validation: dict,
    structural_issues: list,
):
    n_nodes = len(final_graph.get("graph", {}).get("nodes", []))
    n_edges = len(final_graph.get("graph", {}).get("edges", []))
    n_actions = sum(
        1 for n in final_graph.get("graph", {}).get("nodes", [])
        if n.get("type") == "ACTION"
    )
    completeness = validation.get("completeness_score", "N/A") if validation else "N/A"
    s_crits = sum(1 for i in structural_issues if i["severity"] == "critical")
    s_warns = sum(1 for i in structural_issues if i["severity"] == "warning")

    if HAS_RICH:
        console = Console()
        console.print("\n[bold cyan]═══ РЕЗУЛЬТАТ (pipeline v2) ═══[/bold cyan]")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_row("Стадий выполнено", f"[green]{len(stages_completed)}/8[/green]")
        if failed_stages:
            t.add_row("Ошибки стадий", f"[red]{', '.join(failed_stages)}[/red]")
        t.add_row("Узлов в графе", str(n_nodes))
        t.add_row("Рёбер в графе", str(n_edges))
        t.add_row("ACTION-узлов (маршрутов)", str(n_actions))
        t.add_row("Полнота (LLM оценка)", str(completeness))
        t.add_row(
            "Структурная валидация",
            f"[red]{s_crits} critical[/red], [yellow]{s_warns} warning[/yellow]"
            if (s_crits or s_warns) else "[green]✅ Валиден[/green]"
        )
        console.print(t)
        if validation:
            miss = validation.get("missing_fracture_types", [])
            if miss:
                console.print(f"[yellow]Непокрытые типы переломов: {miss}[/yellow]")
            comment = validation.get("overall_comment", "")
            if comment:
                console.print(f"[dim]{comment[:200]}[/dim]")
        console.print(f"\n[green]Output:[/green]  {output_path}")
        console.print(f"[green]Metrics:[/green] {metrics_path}")
    else:
        print(f"\n{'='*55}")
        print(f"Pipeline v2 — РЕЗУЛЬТАТ")
        print(f"  Стадий: {len(stages_completed)}/8  |  Ошибок: {len(failed_stages)}")
        print(f"  Узлов: {n_nodes}  |  Рёбер: {n_edges}  |  Маршрутов: {n_actions}")
        print(f"  Полнота: {completeness}")
        print(f"  Структура: {s_crits} critical, {s_warns} warning")
        print(f"  Output:  {output_path}")
        print(f"  Metrics: {metrics_path}")
        print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clinical Graph Builder v2 — 5-stage cascade pipeline"
    )
    parser.add_argument("--input",    required=True,  help="Input PDF path")
    parser.add_argument("--output",   required=True,  help="Output JSON path")
    parser.add_argument("--metrics",  default="metrics_v2.json", help="Metrics JSON path")
    parser.add_argument("--section",  default="",     help="Section hint for PDF extraction")
    parser.add_argument("--topic",    default="clinical guidelines", help="Graph topic label")
    parser.add_argument("--model",    default="gemini-3.1-flash-lite-preview", help="Gemini model name")
    parser.add_argument("--rpm",      type=int, default=15, help="Requests per minute limit")
    parser.add_argument("--chunk-size", type=int, default=12000)
    parser.add_argument("--overlap",    type=int, default=400)
    parser.add_argument("--cache-dir",  default="data/cache_v2")
    parser.add_argument("--use-cache",  action="store_true")
    parser.add_argument("--api-key",    default=None)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # API key
    import os
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    configure_limiter(args.rpm)

    stage_kwargs = dict(model=args.model, requests_per_minute=args.rpm)
    stages = {
        "1a": Stage1aOsteosynthesis(client, **stage_kwargs),
        "1b": Stage1bArthroplasty(client, **stage_kwargs),
        "1c": Stage1cFractureSpecific(client, **stage_kwargs),
        "2":  Stage2Factors(client, **stage_kwargs),
        "3":  Stage3Algorithm(client, **stage_kwargs),
        "4":  Stage4Graph(client, **stage_kwargs),
        "5a": Stage5aValidate(client, **stage_kwargs),
        "5b": Stage5bFix(client, **stage_kwargs),
    }

    stages_completed: list = []
    failed_stages: list = []
    metrics: dict = {"started_at": datetime.now(timezone.utc).isoformat()}

    def _run(name, fn):
        return run_stage(name, fn, args.cache_dir, args.use_cache,
                         failed_stages, stages_completed)

    # ── Extract text ──────────────────────────────────────────────────────────
    logger.info("Extracting PDF text...")
    text = extract_pdf_text(args.input, args.section)
    logger.info(f"  Extracted {len(text):,} characters")

    chunker = TextChunker(max_chars=args.chunk_size, overlap_chars=args.overlap)
    chunks = chunker.split(text)
    logger.info(f"  Split into {len(chunks)} chunks")

    # ── Stage 1a ──────────────────────────────────────────────────────────────
    logger.info("Stage 1a: Extracting osteosynthesis methods...")
    osteosynthesis = _run("stage1a", lambda: stages["1a"].run(chunks)) or {"osteosynthesis_methods": []}
    metrics["stage1a_methods"] = len(osteosynthesis.get("osteosynthesis_methods", []))

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    logger.info("Stage 1b: Extracting arthroplasty methods...")
    arthroplasty = _run("stage1b", lambda: stages["1b"].run(chunks)) or {"arthroplasty_methods": []}
    metrics["stage1b_methods"] = len(arthroplasty.get("arthroplasty_methods", []))

    # ── Stage 1c ──────────────────────────────────────────────────────────────
    logger.info("Stage 1c: Extracting fracture-specific treatments...")
    fracture_treatments = _run("stage1c", lambda: stages["1c"].run(chunks)) or {"fracture_treatments": []}
    metrics["stage1c_fractures"] = len(fracture_treatments.get("fracture_treatments", []))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    logger.info("Stage 2: Extracting decision factors...")
    factors = _run("stage2", lambda: stages["2"].run(chunks)) or {
        "patient_factors": [], "fracture_classifications": [],
        "decision_criteria": [], "contraindications": [],
    }

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    logger.info("Stage 3: Building algorithm structure...")
    algorithm = _run("stage3", lambda: stages["3"].run(
        osteosynthesis, arthroplasty, fracture_treatments, factors
    )) or {"algorithm": {"branches": []}}
    metrics["stage3_branches"] = len(algorithm.get("algorithm", {}).get("branches", []))

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    logger.info("Stage 4: Generating graph JSON...")
    raw_graph = _run("stage4", lambda: stages["4"].run(
        algorithm, osteosynthesis, arthroplasty, fracture_treatments, factors
    ))
    if raw_graph is None:
        logger.error("Stage 4 failed — cannot continue")
        sys.exit(1)
    metrics["stage4_nodes"] = len(raw_graph.get("nodes", []))
    metrics["stage4_edges"] = len(raw_graph.get("edges", []))

    # ── Stage 5a ──────────────────────────────────────────────────────────────
    logger.info("Stage 5a: Validating completeness...")
    validation = _run("stage5a", lambda: stages["5a"].run(
        raw_graph, osteosynthesis, arthroplasty, fracture_treatments, text
    )) or {}
    metrics["stage5a_completeness"] = validation.get("completeness_score")
    metrics["stage5a_missing"] = validation.get("missing_fracture_types", [])

    # ── Stage 5b ──────────────────────────────────────────────────────────────
    logger.info("Stage 5b: Fixing graph...")
    fixed_data = _run("stage5b", lambda: stages["5b"].run(raw_graph, validation, text)) or {
        "nodes": raw_graph.get("nodes", []),
        "edges": raw_graph.get("edges", []),
        "changelog": [],
    }

    # ── Final structural validation ───────────────────────────────────────────
    final_graph_doc = assemble_output(fixed_data, args.input, args.topic,
                                      fixed_data.get("changelog", []))
    structural_issues = validate_graph_structure(final_graph_doc)
    metrics["final_structural_critical"] = sum(
        1 for i in structural_issues if i["severity"] == "critical"
    )
    metrics["final_structural_warnings"] = sum(
        1 for i in structural_issues if i["severity"] == "warning"
    )
    metrics["final_structural_issues"] = structural_issues
    for iss in structural_issues:
        level = logging.WARNING if iss["severity"] == "critical" else logging.INFO
        logger.log(level, f"Final validation [{iss['severity'].upper()}]: {iss['description']}")

    # ── Token accounting ──────────────────────────────────────────────────────
    total_tokens = sum(s.tokens_used for s in stages.values())
    metrics["total_tokens_used"] = total_tokens
    metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
    metrics["failed_stages"] = failed_stages
    metrics["stages_completed"] = stages_completed

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_path = args.output
    metrics_path = args.metrics
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(final_graph_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved graph → {output_path}")
    logger.info(f"Saved metrics → {metrics_path}")

    print_report(
        output_path, metrics_path,
        stages_completed, failed_stages,
        final_graph_doc, validation, structural_issues,
    )


if __name__ == "__main__":
    main()
