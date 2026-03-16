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
    Stage0Images, Stage1aOsteosynthesis,
    Stage1bArthroplasty,
    Stage1cFractureSpecific,
    Stage2Factors, Stage3Algorithm, Stage4Graph,
    Stage5aValidate, Stage5bFix,
    validate_graph_structure, configure_limiter,
)
from pipeline.chunker import TextChunker
from pipeline.pdf_images import extract_page_images
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
    parser.add_argument("--images",     action="store_true", default=False,
                        help="Enable Stage 0: analyse PDF page images via Gemini vision")
    parser.add_argument("--image-resolution", type=int, default=150,
                        help="DPI for PDF page rendering (default: 150)")
    parser.add_argument("--image-max-pages", type=int, default=60,
                        help="Maximum pages to render for vision analysis (default: 60)")
    parser.add_argument("--max-fix-iterations", type=int, default=3,
                        help="Max validate→fix loop iterations for Stage 5a/5b (default: 3)")
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
        "0":  Stage0Images(client, **stage_kwargs),
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

    # ── Stage 0: Vision analysis of PDF pages (optional) ────────────────────
    image_findings: dict = {
        "fracture_types": [], "treatment_methods": [],
        "decision_rules": [], "thresholds": [],
        "warnings": [], "flowchart_paths": [], "pages_analyzed": [],
    }
    if args.images:
        logger.info("Stage 0: Extracting page images from PDF...")
        try:
            pages = extract_page_images(
                args.input,
                resolution=args.image_resolution,
                max_pages=args.image_max_pages,
                skip_text_only=True,
            )
            metrics["stage0_pages_found"] = len(pages)
            logger.info(f"  Found {len(pages)} visual pages to analyse")

            if pages:
                image_findings = _run(
                    "stage0",
                    lambda: stages["0"].run(pages),
                ) or image_findings
                metrics["stage0_methods_found"] = len(
                    image_findings.get("treatment_methods", [])
                )
                metrics["stage0_flowchart_paths"] = len(
                    image_findings.get("flowchart_paths", [])
                )
                logger.info(
                    f"  Stage 0: {metrics['stage0_methods_found']} methods, "
                    f"{metrics['stage0_flowchart_paths']} flowchart paths from images"
                )
            else:
                logger.info("  Stage 0: no visual pages found — skipping vision")
        except Exception as e:
            logger.warning(f"Stage 0 failed (non-fatal): {e}")
    else:
        logger.info("Stage 0: image analysis disabled (use --images to enable)")
        metrics["stage0_pages_found"] = 0

    # ── Stages 1a / 1b / 1c (parallel) ──────────────────────────────────────
    logger.info("Stages 1a/1b/1c: Extracting methods in parallel...")
    import concurrent.futures

    def _run_1a(): return _run("stage1a", lambda: stages["1a"].run(chunks))
    def _run_1b(): return _run("stage1b", lambda: stages["1b"].run(chunks))
    def _run_1c(): return _run("stage1c", lambda: stages["1c"].run(chunks))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_1a = executor.submit(_run_1a)
        fut_1b = executor.submit(_run_1b)
        fut_1c = executor.submit(_run_1c)
        osteosynthesis      = fut_1a.result() or {"osteosynthesis_methods": []}
        arthroplasty        = fut_1b.result() or {"arthroplasty_methods": []}
        fracture_treatments = fut_1c.result() or {"fracture_treatments": []}

    # Merge image-extracted treatment methods into stage 1a/1b results
    if image_findings.get("treatment_methods"):
        img_methods = image_findings["treatment_methods"]
        # Partition: osteosynthesis vs arthroplasty vs fracture-specific
        for m in img_methods:
            method_name = m.get("method", "").lower()
            implant = m.get("implant", "") or ""
            if any(kw in method_name or kw in implant.lower()
                   for kw in ("эндопротез", "тэтс", "гемиэндо", "артропласт")):
                arthroplasty.setdefault("arthroplasty_methods", []).append({
                    "method": m.get("method"), "implant": m.get("implant"),
                    "indication": m.get("indication"), "timing": m.get("timing"),
                    "evidence_level": m.get("evidence_level"),
                    "source": "image",
                })
            else:
                osteosynthesis.setdefault("osteosynthesis_methods", []).append({
                    "method": m.get("method"), "implant": m.get("implant"),
                    "indication": m.get("indication"), "timing": m.get("timing"),
                    "evidence_level": m.get("evidence_level"),
                    "source": "image",
                })
        logger.info(
            f"  Merged {len(img_methods)} image-extracted methods into stages 1a/1b"
        )

    # Append image-found flowchart paths to fracture treatments as extra context
    if image_findings.get("flowchart_paths"):
        fracture_treatments.setdefault("flowchart_paths_from_images", []).extend(
            image_findings["flowchart_paths"]
        )
        logger.info(
            f"  Merged {len(image_findings['flowchart_paths'])} "
            f"flowchart paths from images into fracture_treatments"
        )


    metrics["stage1a_methods"]   = len(osteosynthesis.get("osteosynthesis_methods", []))
    metrics["stage1b_methods"]   = len(arthroplasty.get("arthroplasty_methods", []))
    metrics["stage1c_fractures"] = len(fracture_treatments.get("fracture_treatments", []))
    logger.info(
        f"  1a={metrics['stage1a_methods']} osteosynthesis, "
        f"1b={metrics['stage1b_methods']} arthroplasty, "
        f"1c={metrics['stage1c_fractures']} fracture treatments"
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    logger.info("Stage 2: Extracting decision factors...")
    factors = _run("stage2", lambda: stages["2"].run(chunks)) or {
        "patient_factors": [], "fracture_classifications": [],
        "decision_criteria": [], "contraindications": [],
    }

    if image_findings.get("thresholds") or image_findings.get("decision_rules"):
        factors.setdefault("image_thresholds", []).extend(
            image_findings.get("thresholds", [])
        )
        factors.setdefault("image_decision_rules", []).extend(
            image_findings.get("decision_rules", [])
        )


    # ── Stage 3 ───────────────────────────────────────────────────────────────
    logger.info("Stage 3: Building algorithm structure...")
    algorithm = _run("stage3", lambda: stages["3"].run(
        osteosynthesis, arthroplasty, fracture_treatments, factors,
        source_text=text,
    )) or {"algorithm": {"branches": []}}
    metrics["stage3_branches"] = len(algorithm.get("algorithm", {}).get("branches", []))

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    logger.info("Stage 4: Generating graph JSON...")
    raw_graph = _run("stage4", lambda: stages["4"].run(
        algorithm, osteosynthesis, arthroplasty, fracture_treatments, factors,
    ))
    if raw_graph is None:
        logger.error("Stage 4 failed — cannot continue")
        sys.exit(1)
    metrics["stage4_nodes"] = len(raw_graph.get("nodes", []))
    metrics["stage4_edges"] = len(raw_graph.get("edges", []))

    # ── Branch coverage check (deterministic) ────────────────────────────────
    if raw_graph and algorithm:
        branches = algorithm.get("algorithm", {}).get("branches", [])
        all_terminals = []
        for b in branches:
            all_terminals.extend(b.get("terminals", {}).values())
        graph_action_labels = {
            n["label"].lower() for n in raw_graph.get("nodes", [])
            if n.get("type") == "ACTION"
        }
        missing_terminals = [
            t for t in all_terminals
            if not any(t.lower() in lbl or lbl in t.lower() for lbl in graph_action_labels)
        ]
        if missing_terminals:
            logger.warning(
                f"Branch coverage: {len(missing_terminals)} Stage-3 terminals "
                f"not found in Stage-4 graph: {missing_terminals[:5]}"
            )
            metrics["missing_branch_terminals"] = missing_terminals
        else:
            logger.info("Branch coverage: all Stage-3 terminals present in graph ✓")
            metrics["missing_branch_terminals"] = []

    # ── Validate → Fix loop (Stage 5a ↔ Stage 5b) ───────────────────────────
    MAX_FIX_ITERATIONS = args.max_fix_iterations
    FIX_SCORE_THRESHOLD = 0.9   # stop early if completeness >= this AND 0 critical issues

    current_graph  = raw_graph          # dict with nodes + edges (no metadata wrapper)
    accumulated_changelog: list = []
    validation:     dict = {}
    metrics["fix_iterations"] = []

    logger.info(
        f"Starting validate→fix loop (max {MAX_FIX_ITERATIONS} iterations, "
        f"threshold={FIX_SCORE_THRESHOLD})"
    )

    for iteration in range(1, MAX_FIX_ITERATIONS + 1):
        iter_label = f"iter{iteration}"
        logger.info(f"── Iteration {iteration}/{MAX_FIX_ITERATIONS} ──────────────────")

        # ── 5a: Validate ──────────────────────────────────────────────────────
        logger.info(f"  [5a] Validating graph (iteration {iteration})...")
        validation = _run(
            f"stage5a_{iter_label}",
            lambda g=current_graph: stages["5a"].run(
                g, osteosynthesis, arthroplasty, fracture_treatments, text
            ),
        ) or {}

        score       = validation.get("completeness_score", 0) or 0
        crit_struct = validation.get("structural_critical", 0) or 0
        crit_llm    = sum(
            1 for i in validation.get("issues", [])
            if i.get("severity") == "critical"
        )
        n_missing   = len(validation.get("missing_fracture_types", []))

        iter_metrics = {
            "iteration":          iteration,
            "completeness_score": score,
            "structural_critical": crit_struct,
            "llm_critical_issues": crit_llm,
            "missing_fracture_types": validation.get("missing_fracture_types", []),
        }
        metrics["fix_iterations"].append(iter_metrics)

        logger.info(
            f"  [5a] score={score:.2f}  struct_crit={crit_struct}  "
            f"llm_crit={crit_llm}  missing={n_missing}"
        )

        # ── Early exit: graph is good enough ─────────────────────────────────
        if score >= FIX_SCORE_THRESHOLD and crit_struct == 0 and crit_llm == 0:
            logger.info(
                f"  ✅ Graph satisfactory after iteration {iteration} — "
                f"stopping loop"
            )
            break

        # ── Early exit: last iteration reached ───────────────────────────────
        if iteration == MAX_FIX_ITERATIONS:
            logger.warning(
                f"   Max iterations reached ({MAX_FIX_ITERATIONS}) — "
                f"using best graph so far (score={score:.2f})"
            )
            break

        # ── 5b: Fix ───────────────────────────────────────────────────────────
        logger.info(f"  [5b] Fixing graph (iteration {iteration})...")
        fixed = _run(
            f"stage5b_{iter_label}",
            lambda g=current_graph, v=validation: stages["5b"].run(
                g, v, text, algorithm=algorithm
            ),
        )

        if fixed is None:
            logger.error(f"  [5b] Fix failed at iteration {iteration} — keeping previous graph")
            break

        # Accumulate changelog from this fix iteration
        iter_changelog = fixed.get("changelog", [])
        for entry in iter_changelog:
            entry.setdefault("fix_iteration", iteration)
        accumulated_changelog.extend(iter_changelog)

        # Update current graph for next iteration
        current_graph = {
            "nodes":     fixed.get("nodes", current_graph.get("nodes", [])),
            "edges":     fixed.get("edges", current_graph.get("edges", [])),
            "changelog": accumulated_changelog,
        }

        n_nodes = len(current_graph["nodes"])
        n_edges = len(current_graph["edges"])
        n_act   = sum(1 for n in current_graph["nodes"] if n.get("type") == "ACTION")
        logger.info(
            f"  [5b] Graph updated: {n_nodes} nodes ({n_act} actions), "
            f"{n_edges} edges, {len(iter_changelog)} changelog entries"
        )

    fixed_data = current_graph
    # Preserve full accumulated changelog
    if accumulated_changelog:
        fixed_data["changelog"] = accumulated_changelog

    # Summary metrics
    metrics["stage5a_completeness"] = validation.get("completeness_score")
    metrics["stage5a_missing"]      = validation.get("missing_fracture_types", [])
    metrics["stage5b_skipped"]      = len(metrics["fix_iterations"]) == 1 and (
        metrics["fix_iterations"][0]["completeness_score"] >= FIX_SCORE_THRESHOLD
        and metrics["fix_iterations"][0]["structural_critical"] == 0
    )
    metrics["total_fix_iterations"] = len(metrics["fix_iterations"])

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