#!/usr/bin/env python3
"""
Clinical Graph Builder — Main Entry Point
==========================================
Extracts clinical decision graphs from PDF guidelines
using a cascaded LLM pipeline (Google Gemini via google-genai SDK).

Usage:
    python main.py --input guidelines.pdf --output graph.json \
        --section "переломы шейки бедра" --verbose
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types

from pipeline import (
    PipelineError,
    Stage1Entities,
    Stage2Rules,
    Stage3Nodes,
    Stage4Edges,
    Stage5Assembly,
    Stage6Verify,
    Stage7Fix,
    validate,
    configure_limiter,
)
from pipeline.chunker import TextChunker, chunk_summary
from metrics import compute_graph_metrics

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(pages)
            logging.getLogger(__name__).info(
                f"pdfplumber: {len(text)} chars from {len(pages)} pages"
            )
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"pdfplumber failed: {e}, trying pypdf")

    if HAS_PYPDF:
        reader = pypdf.PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        logging.getLogger(__name__).info(
            f"pypdf: {len(text)} chars from {len(pages)} pages"
        )
        return text

    raise RuntimeError("No PDF library available. Install: pip install pdfplumber pypdf")


def filter_section(text: str, section: Optional[str]) -> str:
    if not section:
        return text
    idx = text.lower().find(section.lower())
    if idx == -1:
        logging.getLogger(__name__).warning(
            f"Section '{section}' not found — using full text"
        )
        return text
    return text[idx: idx + 15000]


# ── Cache helpers ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("pipeline_cache")


def save_cache(stage: str, data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{stage}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_cache(stage: str) -> Optional[dict]:
    path = CACHE_DIR / f"{stage}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ── Schema validation ─────────────────────────────────────────────────────────
def validate_graph(graph: dict) -> list:
    if not HAS_JSONSCHEMA:
        return []
    schema_path = Path(__file__).parent / "schemas" / "graph_schema.json"
    if not schema_path.exists():
        return []
    schema = json.loads(schema_path.read_text())
    return [
        f"{list(e.path)}: {e.message}"
        for e in jsonschema.Draft7Validator(schema).iter_errors(graph)
    ]


# ── Console output ────────────────────────────────────────────────────────────
def _print_plain(metrics, verification, output_path, metrics_path, stages_total, failed):
    print("\n" + "=" * 55)
    print("   CLINICAL GRAPH BUILDER — REPORT")
    print("=" * 55)
    completed = metrics.get("cascade_stages_completed", 0)
    print(f"\n Stages: {completed}/{stages_total}")
    if failed:
        print(f"   Failed: {', '.join(failed)}")

    print("\nGRAPH STRUCTURE")
    print(f"  Nodes total:        {metrics['node_count']}")
    print(f"  DECISION:           {metrics['decision_nodes']}")
    print(f"  ACTION:             {metrics['action_nodes']}")
    print(f"  WARNING:            {metrics.get('warning_nodes', 0)}")
    print(f"  END:                {metrics['terminal_nodes']}")
    print(f"  Edges total:        {metrics['edge_count']}")

    print("\nCONNECTIVITY")
    conn = metrics.get("is_connected")
    if conn is None:
        print("  Connectivity check skipped (networkx unavailable)")
    else:
        print(f"  {'Connected' if conn else 'NOT connected'}")
    hc = metrics.get("has_cycles")
    if hc is not None:
        print(f"  Cycles: {'YES' if hc else 'none'}")
    print(f"  Max depth:          {metrics.get('max_depth', 'N/A')}")
    print(f"  Avg branching:      {metrics.get('avg_branching_factor', 'N/A')}")

    print("\nCOMPLETENESS")
    print(f"  Entities found:     {metrics['entities_extracted']}")
    print(f"  Rules extracted:    {metrics['rules_extracted']}")
    rt = metrics['rules_extracted']
    rm = metrics['rules_mapped_to_nodes']
    pct = f"({metrics['extraction_completeness']*100:.1f}%)" if rt else ""
    print(f"  Rules in graph:     {rm}  {pct}")
    ae, at = metrics['action_nodes_with_evidence'], metrics['action_nodes']
    epct = f"({ae/at*100:.1f}%)" if at else ""
    print(f"  Actions w/evidence: {ae}/{at}  {epct}")
    print(f"  Total tokens used:  {metrics['total_tokens_used']:,}")

    if verification:
        score = verification.get("clinical_accuracy_score", "N/A")
        issues = verification.get("issues", [])
        print(f"\nCLINICAL VALIDATION  score={score}  issues={len(issues)}")
        for iss in issues[:5]:
            sev = "CRIT" if iss.get("severity") == "critical" else "WARN"
            print(f"  [{sev}] {iss.get('description', '')}")
        for ms in verification.get("missing_scenarios", [])[:3]:
            print(f"  MISSING: {ms.get('description', '')}")

    print(f"\n Output: {output_path}")
    print(f" Metrics: {metrics_path}")
    print("=" * 55)


def _print_rich(metrics, verification, output_path, metrics_path, stages_total, failed, source):
    completed = metrics.get("cascade_stages_completed", 0)
    console.print(Panel.fit(
        f"[bold cyan]CLINICAL GRAPH BUILDER[/bold cyan]  [dim](google-genai)[/dim]\n"
        f"[dim]Source: {source}[/dim]   [dim]Stages: {completed}/{stages_total}[/dim]"
        + (f"\n[red]Failed: {', '.join(failed)}[/red]" if failed else ""),
        border_style="cyan"
    ))

    t = Table(title="Graph Structure", box=box.SIMPLE)
    t.add_column("Type", style="bold")
    t.add_column("Count", justify="right")
    for label, key in [("DECISION","decision_nodes"),("ACTION","action_nodes"),
                        ("WARNING","warning_nodes"),("END","terminal_nodes")]:
        t.add_row(label, str(metrics.get(key, 0)))
    t.add_row("[bold]TOTAL NODES[/bold]", f"[bold]{metrics['node_count']}[/bold]")
    t.add_row("[bold]TOTAL EDGES[/bold]", f"[bold]{metrics['edge_count']}[/bold]")
    console.print(t)

    conn = metrics.get("is_connected")
    if conn is not None:
        color = "green" if conn else "red"
        console.print(
            f"[{color}]{'Connected' if conn else 'NOT connected'}[/{color}]  "
            f"Cycles: {'yes' if metrics.get('has_cycles') else 'none'}  "
            f"Depth: {metrics.get('max_depth','N/A')}  "
            f"Branching: {metrics.get('avg_branching_factor','N/A')}"
        )

    t2 = Table(title="Completeness", box=box.SIMPLE)
    t2.add_column("Metric")
    t2.add_column("Value", justify="right")
    t2.add_row("Entities", str(metrics["entities_extracted"]))
    t2.add_row("Rules extracted", str(metrics["rules_extracted"]))
    rt = metrics["rules_extracted"]
    pct = f"{metrics['extraction_completeness']*100:.1f}%" if rt else "N/A"
    t2.add_row("Rules in graph", f"{metrics['rules_mapped_to_nodes']}  ({pct})")
    ae, at = metrics["action_nodes_with_evidence"], metrics["action_nodes"]
    epct = f"{ae/at*100:.1f}%" if at else "N/A"
    t2.add_row("Actions w/ evidence", f"{ae}/{at}  ({epct})")
    t2.add_row("Tokens used", f"{metrics['total_tokens_used']:,}")
    console.print(t2)

    if verification:
        score = verification.get("clinical_accuracy_score", "N/A")
        issues = verification.get("issues", [])
        console.print(f"\n[bold]Clinical Validation[/bold]  score=[cyan]{score}[/cyan]  issues={len(issues)}")
        for iss in issues[:5]:
            color = "red" if iss.get("severity") == "critical" else "yellow"
            console.print(f"  [{color}]{iss.get('description','')}[/{color}]")
        for ms in verification.get("missing_scenarios", [])[:3]:
            console.print(f"  [dim]MISSING: {ms.get('description','')}[/dim]")

    console.print(f"\n[green]Output:[/green]  {output_path}")
    console.print(f"[green]Metrics:[/green] {metrics_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    input_pdf: str,
    output_path: str,
    metrics_path: str,
    section: Optional[str],
    model: str,
    verbose: bool,
    use_cache: bool,
    requests_per_minute: int = 15,
    chunk_size: int = 25_000,
    overlap: int = 400,
) -> None:
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # ── Init google-genai Client ──────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    api_key = "AIzaSyCwCS_kDXKySsMD659KpiQqw6f6iQZnLwQ"
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    # google-genai SDK: create a Client, then use client.models.generate_content()
    client = genai.Client(api_key=api_key)

    # Configure shared rate limiter
    configure_limiter(requests_per_minute)
    logger.info(
        f"Rate limiter: {requests_per_minute} req/min "
        f"(~{60/requests_per_minute:.1f}s between requests)"
    )

    # Init stages (all share the same global limiter)
    stage_kwargs = dict(model=model, requests_per_minute=requests_per_minute)
    stages = {
        "stage1": Stage1Entities(client, **stage_kwargs),
        "stage2": Stage2Rules(client, **stage_kwargs),
        "stage3": Stage3Nodes(client, **stage_kwargs),
        "stage4": Stage4Edges(client, **stage_kwargs),
        "stage5": Stage5Assembly(client, **stage_kwargs),
        "stage6": Stage6Verify(client, **stage_kwargs),
        "stage7": Stage7Fix(client, **stage_kwargs),
    }

    failed_stages = []
    stages_completed = 0
    verification_result = {}

    # ── Extract PDF ───────────────────────────────────────────────────────────
    logger.info(f"Extracting text from: {input_pdf}")
    full_text = extract_text_from_pdf(input_pdf)
    text = filter_section(full_text, section)
    logger.info(f"Text length: {len(text)} chars")

    # ── Chunk the text ────────────────────────────────────────────────────────
    chunker = TextChunker(max_chars=chunk_size, overlap_chars=overlap)
    chunks = chunker.split(text)
    logger.info(f"Text split: {chunk_summary(chunks)}")

    # ── Stage 1: Entities ─────────────────────────────────────────────────────
    logger.info("Stage 1: Entity extraction")
    entities = load_cache("stage1_entities") if use_cache else None
    if entities:
        logger.info("  (loaded from cache)")
    else:
        try:
            entities = stages["stage1"].run(chunks, section)
            save_cache("stage1_entities", entities)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 1 failed: {e}")
            failed_stages.append("stage1")
            entities = {"entities": []}

    # ── Stage 2: Rules ────────────────────────────────────────────────────────
    logger.info("Stage 2: Rule extraction")
    rules = load_cache("stage2_rules") if use_cache else None
    if rules:
        logger.info("  (loaded from cache)")
    else:
        try:
            rules = stages["stage2"].run(chunks, entities)
            save_cache("stage2_rules", rules)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 2 failed: {e}")
            failed_stages.append("stage2")
            rules = {"rules": []}

    # ── Stage 3: Nodes ────────────────────────────────────────────────────────
    logger.info("Stage 3: Node construction")
    nodes = load_cache("stage3_nodes") if use_cache else None
    if nodes:
        logger.info("  (loaded from cache)")
    else:
        try:
            nodes = stages["stage3"].run(rules)
            save_cache("stage3_nodes", nodes)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 3 failed: {e}")
            failed_stages.append("stage3")
            nodes = {"nodes": []}

    # ── Stage 4: Edges ────────────────────────────────────────────────────────
    logger.info("Stage 4: Edge construction")
    edges = load_cache("stage4_edges") if use_cache else None
    if edges:
        logger.info("  (loaded from cache)")
    else:
        try:
            edges = stages["stage4"].run(nodes)
            save_cache("stage4_edges", edges)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 4 failed: {e}")
            failed_stages.append("stage4")
            edges = {"edges": []}

    # ── Stage 5: Assembly ─────────────────────────────────────────────────────
    logger.info("Stage 5: Final assembly & enrichment")
    enriched = load_cache("stage5_assembly") if use_cache else None
    if enriched:
        logger.info("  (loaded from cache)")
    else:
        try:
            enriched = stages["stage5"].run(nodes, edges, text)
            save_cache("stage5_assembly", enriched)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 5 failed: {e}")
            failed_stages.append("stage5")
            enriched = {"enriched_nodes": nodes.get("nodes", [])}

    final_graph = stages["stage5"].assemble_final_graph(
        enriched_nodes=enriched.get("enriched_nodes", nodes.get("nodes", [])),
        edges=edges.get("edges", []),
        source_document=Path(input_pdf).name,
        topic=section or "clinical guidelines",
    )


    # ── Schema validation ─────────────────────────────────────────────────────
    for err in validate_graph(final_graph)[:5]:
        logger.warning(f"Schema: {err}")

    # ── Stage 6: Verification ─────────────────────────────────────────────────
    logger.info("Stage 6: Clinical verification")
    try:
        verification_result = stages["stage6"].run(final_graph, text)
        save_cache("stage6_verify", verification_result)
        stages_completed += 1
    except PipelineError as e:
        logger.error(f"Stage 6 failed: {e}")
        failed_stages.append("stage6")


    # ── Stage 7: Graph repair ─────────────────────────────────────────────────
    logger.info("Stage 7: Graph repair based on verification report")
    fixed_cache = load_cache("stage7_fix") if use_cache else None
    if fixed_cache:
        final_graph = fixed_cache
        logger.info("  (loaded from cache)")
    else:
        try:
            final_graph = stages["stage7"].run(final_graph, verification_result, text)
            save_cache("stage7_fix", final_graph)
            stages_completed += 1
            changelog = final_graph.get("changelog", [])
            if changelog:
                logger.info(f"  Changelog: {len(changelog)} changes applied")
            else:
                logger.info("  No changes needed — graph returned unchanged")
        except PipelineError as e:
            logger.error(f"Stage 7 failed: {e}")
            failed_stages.append("stage7")

    # ── Metrics ───────────────────────────────────────────────────────────────
    total_tokens = sum(s.tokens_used for s in stages.values())
    rules_extracted = len(rules.get("rules", []))
    rules_mapped = len(final_graph["graph"]["nodes"])

    metrics = compute_graph_metrics(
        graph=final_graph,
        entities_count=len(entities.get("entities", [])),
        rules_count=rules_extracted,
        rules_mapped=rules_mapped,
        stages_completed=stages_completed,
        total_tokens=total_tokens,
        failed_stages=failed_stages,
    )
    if verification_result:
        metrics["clinical_accuracy_score"] = verification_result.get("clinical_accuracy_score")
        metrics["clinical_issues"] = verification_result.get("issues", [])
        metrics["missing_scenarios"] = verification_result.get("missing_scenarios", [])
    metrics["changelog"] = final_graph.get("changelog", [])
    metrics["changelog_count"] = len(final_graph.get("changelog", []))
    metrics["elapsed_seconds"] = round(time.time() - start_time, 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(final_graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(metrics_path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Final structural validation ──────────────────────────────────────────
    final_structural_issues = validate(final_graph)
    metrics["final_structural_issues"] = final_structural_issues
    metrics["final_structural_critical"] = sum(
        1 for i in final_structural_issues if i["severity"] == "critical"
    )
    metrics["final_structural_warnings"] = sum(
        1 for i in final_structural_issues if i["severity"] == "warning"
    )
    if final_structural_issues:
        for iss in final_structural_issues:
            logger.warning(f"Final validation [{iss['severity'].upper()}]: {iss['description']}")
    else:
        logger.info("Final validation: graph is structurally valid")

    # ── Report ────────────────────────────────────────────────────────────────
    source_name = Path(input_pdf).name
    if HAS_RICH:
        _print_rich(metrics, verification_result, output_path, metrics_path, 7, failed_stages, source_name)
    else:
        _print_plain(metrics, verification_result, output_path, metrics_path, 7, failed_stages)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clinical decision graphs from PDF guidelines (google-genai / Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
  python main.py --input guidelines.pdf --section "переломы шейки бедра"
  python main.py --input guidelines.pdf --model gemini-3.1-flash-lite-preview --rpm 15
  python main.py --input guidelines.pdf --use-cache --verbose
        """,
    )
    parser.add_argument("--input",   "-i", required=True, help="Input PDF file")
    parser.add_argument("--output",  "-o", default="graph.json",   help="Output graph JSON")
    parser.add_argument("--metrics", "-m", default="metrics.json", help="Output metrics JSON")
    parser.add_argument("--section", "-s", default=None, help="Focus on this document section")
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model name (default: gemini-3.1-flash-preview)",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=15,
        metavar="N",
        help="Max requests per minute (default: 15 for free tier)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25_000,
        metavar="N",
        help="Max chars per text chunk sent to LLM (default: 12000 ≈ 3000 tokens)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        metavar="N",
        help="Overlap chars between consecutive chunks (default: 400)",
    )
    parser.add_argument("--verbose",   "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--use-cache",       action="store_true", help="Load cached pipeline stages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            input_pdf=args.input,
            output_path=args.output,
            metrics_path=args.metrics,
            section=args.section,
            model=args.model,
            verbose=args.verbose,
            use_cache=args.use_cache,
            requests_per_minute=args.rpm,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
