#!/usr/bin/env python3
"""
Clinical Graph Builder — Main Entry Point
==========================================
Extracts clinical decision graphs from PDF guidelines
using a cascaded LLM pipeline (Claude API).

Usage:
    python main.py --input guidelines.pdf --output graph.json \
        --metrics metrics.json --section "переломы шейки бедра" --verbose
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import anthropic

from pipeline import (
    PipelineError,
    Stage1Entities,
    Stage2Rules,
    Stage3Nodes,
    Stage4Edges,
    Stage5Assembly,
    Stage6Verify,
)
from metrics import compute_graph_metrics

# ── Optional dependencies ────────────────────────────────────────────────────
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
    """Extract text from PDF using pdfplumber or pypdf."""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(pages)
            logging.getLogger(__name__).info(
                f"pdfplumber extracted {len(text)} chars from {len(pages)} pages"
            )
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"pdfplumber failed: {e}, trying pypdf")

    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            logging.getLogger(__name__).info(
                f"pypdf extracted {len(text)} chars from {len(pages)} pages"
            )
            return text
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}") from e

    raise RuntimeError(
        "No PDF library available. Install: pip install pdfplumber pypdf"
    )


def filter_section(text: str, section: Optional[str]) -> str:
    """Optionally narrow text to relevant section."""
    if not section:
        return text

    lower = text.lower()
    section_lower = section.lower()
    idx = lower.find(section_lower)
    if idx == -1:
        logging.getLogger(__name__).warning(
            f"Section '{section}' not found in document — using full text"
        )
        return text

    # Take from section start ~15000 chars
    return text[idx: idx + 15000]


# ── Cache helpers ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("pipeline_cache")


def save_cache(stage: str, data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{stage}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.getLogger(__name__).debug(f"Cached {stage} → {path}")


def load_cache(stage: str) -> Optional[dict]:
    path = CACHE_DIR / f"{stage}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ── Schema validation ─────────────────────────────────────────────────────────
def validate_graph(graph: dict) -> list:
    """Validate against JSON schema; return list of error strings."""
    if not HAS_JSONSCHEMA:
        return []
    schema_path = Path(__file__).parent / "schemas" / "graph_schema.json"
    if not schema_path.exists():
        return []
    schema = json.loads(schema_path.read_text())
    errors = []
    validator = jsonschema.Draft7Validator(schema)
    for err in validator.iter_errors(graph):
        errors.append(f"{list(err.path)}: {err.message}")
    return errors


# ── Console output ────────────────────────────────────────────────────────────
def _print_plain(metrics: dict, verification: dict, output_path: str, metrics_path: str,
                  stages_total: int, failed: list) -> None:
    print("\n" + "=" * 55)
    print("   CLINICAL GRAPH BUILDER — REPORT")
    print("=" * 55)

    completed = metrics.get("cascade_stages_completed", 0)
    print(f"\n🔗 Stages completed: {completed}/{stages_total}")
    if failed:
        print(f"   ⚠  Failed: {', '.join(failed)}")

    print("\nGRAPH STRUCTURE")
    print(f"  Nodes total:        {metrics['node_count']}")
    print(f"  ├─ DECISION:        {metrics['decision_nodes']}")
    print(f"  ├─ ACTION:          {metrics['action_nodes']}")
    print(f"  ├─ WARNING:         {metrics.get('warning_nodes', 0)}")
    print(f"  └─ END:             {metrics['terminal_nodes']}")
    print(f"  Edges total:        {metrics['edge_count']}")

    print("\nCONNECTIVITY")
    connected = metrics.get("is_connected")
    if connected is None:
        print("  ⚪ Connectivity check skipped (networkx unavailable)")
    elif connected:
        print("  ✅ Graph is connected")
    else:
        print("  ❌ Graph is NOT connected")

    isolated = metrics.get("isolated_nodes", [])
    print(f"  Isolated nodes:     {len(isolated)}" + (f"  {isolated}" if isolated else ""))

    has_cycles = metrics.get("has_cycles")
    if has_cycles is None:
        print("  ⚪ Cycle check skipped")
    elif has_cycles:
        print("  ❌ Cycles detected!")
    else:
        print("  ✅ No cycles detected")

    print(f"  Max depth:          {metrics.get('max_depth', 'N/A')}")
    print(f"  Avg branching:      {metrics.get('avg_branching_factor', 'N/A')}")

    print("\nCOMPLETENESS")
    print(f"  Entities found:     {metrics['entities_extracted']}")
    print(f"  Rules extracted:    {metrics['rules_extracted']}")
    rmap = metrics['rules_mapped_to_nodes']
    rtot = metrics['rules_extracted']
    pct = f"({metrics['extraction_completeness']*100:.1f}%)" if rtot else ""
    print(f"  Rules in graph:     {rmap}  {pct}")
    ae = metrics['action_nodes_with_evidence']
    at = metrics['action_nodes']
    epct = f"({ae/at*100:.1f}%)" if at else ""
    print(f"  Actions with evidence: {ae}/{at}  {epct}")
    print(f"  Total tokens used:  {metrics['total_tokens_used']:,}")

    if verification:
        score = verification.get("clinical_accuracy_score", "N/A")
        issues = verification.get("issues", [])
        print("\nCLINICAL VALIDATION")
        print(f"  Accuracy score:   {score}")
        print(f"  Issues found:     {len(issues)}")
        for iss in issues[:5]:
            sev = "⚠" if iss.get("severity") != "critical" else "❌"
            print(f"    {sev} {iss.get('description', '')}")
        missing = verification.get("missing_scenarios", [])
        if missing:
            print(f"  Missing scenarios: {len(missing)}")
            for ms in missing[:3]:
                print(f"    • {ms.get('description', '')}")

    print(f"\n💾 Output saved: {output_path}")
    print(f"📊 Metrics saved: {metrics_path}")
    print("=" * 55)


def _print_rich(metrics: dict, verification: dict, output_path: str, metrics_path: str,
                stages_total: int, failed: list, source: str) -> None:
    completed = metrics.get("cascade_stages_completed", 0)

    # Header panel
    console.print(Panel.fit(
        f"[bold cyan]CLINICAL GRAPH BUILDER[/bold cyan]\n"
        f"[dim]📄 Source: {source}[/dim]\n"
        f"[dim]🔗 Stages completed: {completed}/{stages_total}[/dim]"
        + (f"\n[red]⚠  Failed: {', '.join(failed)}[/red]" if failed else ""),
        border_style="cyan"
    ))

    # Structure table
    t = Table(title="Graph Structure", box=box.SIMPLE)
    t.add_column("Type", style="bold")
    t.add_column("Count", justify="right")
    t.add_row("DECISION", str(metrics["decision_nodes"]))
    t.add_row("ACTION", str(metrics["action_nodes"]))
    t.add_row("WARNING", str(metrics.get("warning_nodes", 0)))
    t.add_row("END", str(metrics["terminal_nodes"]))
    t.add_row("[bold]TOTAL NODES[/bold]", f"[bold]{metrics['node_count']}[/bold]")
    t.add_row("[bold]TOTAL EDGES[/bold]", f"[bold]{metrics['edge_count']}[/bold]")
    console.print(t)

    # Connectivity
    conn = metrics.get("is_connected")
    if conn is None:
        console.print("[dim]⚪ Connectivity check skipped[/dim]")
    elif conn:
        console.print("[green]✅ Graph is connected[/green]")
    else:
        console.print("[red]❌ Graph is NOT connected[/red]")

    hc = metrics.get("has_cycles")
    if hc is None:
        pass
    elif hc:
        console.print("[red]❌ Cycles detected![/red]")
    else:
        console.print("[green]✅ No cycles detected[/green]")

    console.print(
        f"  Max depth: [cyan]{metrics.get('max_depth', 'N/A')}[/cyan]  "
        f"Avg branching: [cyan]{metrics.get('avg_branching_factor', 'N/A')}[/cyan]"
    )

    # Completeness table
    t2 = Table(title="Completeness", box=box.SIMPLE)
    t2.add_column("Metric")
    t2.add_column("Value", justify="right")
    t2.add_row("Entities extracted", str(metrics["entities_extracted"]))
    t2.add_row("Rules extracted", str(metrics["rules_extracted"]))
    pct = f"{metrics['extraction_completeness']*100:.1f}%" if metrics["rules_extracted"] else "N/A"
    t2.add_row("Rules → graph", f"{metrics['rules_mapped_to_nodes']}  ({pct})")
    ae = metrics["action_nodes_with_evidence"]
    at = metrics["action_nodes"]
    epct = f"{ae/at*100:.1f}%" if at else "N/A"
    t2.add_row("Actions w/ evidence", f"{ae}/{at}  ({epct})")
    t2.add_row("Tokens used", f"{metrics['total_tokens_used']:,}")
    console.print(t2)

    # Clinical validation
    if verification:
        score = verification.get("clinical_accuracy_score", "N/A")
        issues = verification.get("issues", [])
        console.print(f"\n[bold]Clinical Validation[/bold]  Accuracy: [cyan]{score}[/cyan]")
        for iss in issues[:5]:
            color = "red" if iss.get("severity") == "critical" else "yellow"
            console.print(f"  [{color}]⚠[/{color}] {iss.get('description', '')}")
        missing = verification.get("missing_scenarios", [])
        if missing:
            console.print(f"  [dim]Missing scenarios: {len(missing)}[/dim]")
            for ms in missing[:3]:
                console.print(f"  [dim]• {ms.get('description', '')}[/dim]")

    console.print(f"\n[green]💾 Output saved:[/green] {output_path}")
    console.print(f"[green]📊 Metrics saved:[/green] {metrics_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    input_pdf: str,
    output_path: str,
    metrics_path: str,
    section: Optional[str],
    model: str,
    verbose: bool,
    use_cache: bool,
) -> None:
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Init Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)

    # Init stages
    stages = {
        "stage1": Stage1Entities(client, model),
        "stage2": Stage2Rules(client, model),
        "stage3": Stage3Nodes(client, model),
        "stage4": Stage4Edges(client, model),
        "stage5": Stage5Assembly(client, model),
        "stage6": Stage6Verify(client, model),
    }

    failed_stages = []
    stages_completed = 0
    verification_result = {}

    # ── Extract PDF text ──────────────────────────────────────────────────────
    logger.info(f"Extracting text from: {input_pdf}")
    full_text = extract_text_from_pdf(input_pdf)
    text = filter_section(full_text, section)
    logger.info(f"Text length: {len(text)} chars")

    # ── Stage 1: Entities ─────────────────────────────────────────────────────
    logger.info("── Stage 1: Entity extraction")
    cached = load_cache("stage1_entities") if use_cache else None
    if cached:
        entities = cached
        logger.info("  (loaded from cache)")
    else:
        try:
            entities = stages["stage1"].run(text, section)
            save_cache("stage1_entities", entities)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 1 failed: {e}")
            failed_stages.append("stage1")
            entities = {"entities": []}

    # ── Stage 2: Rules ────────────────────────────────────────────────────────
    logger.info("── Stage 2: Rule extraction")
    cached = load_cache("stage2_rules") if use_cache else None
    if cached:
        rules = cached
        logger.info("  (loaded from cache)")
    else:
        try:
            rules = stages["stage2"].run(text, entities)
            save_cache("stage2_rules", rules)
            stages_completed += 1
        except PipelineError as e:
            logger.error(f"Stage 2 failed: {e}")
            failed_stages.append("stage2")
            rules = {"rules": []}

    # ── Stage 3: Nodes ────────────────────────────────────────────────────────
    logger.info("── Stage 3: Node construction")
    cached = load_cache("stage3_nodes") if use_cache else None
    if cached:
        nodes = cached
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
    logger.info("── Stage 4: Edge construction")
    cached = load_cache("stage4_edges") if use_cache else None
    if cached:
        edges = cached
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
    logger.info("── Stage 5: Final assembly & enrichment")
    cached = load_cache("stage5_assembly") if use_cache else None
    if cached:
        enriched = cached
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

    # Build final graph structure
    final_graph = stages["stage5"].assemble_final_graph(
        enriched_nodes=enriched.get("enriched_nodes", nodes.get("nodes", [])),
        edges=edges.get("edges", []),
        source_document=Path(input_pdf).name,
        topic=section or "clinical guidelines",
    )

    # ── Schema validation ──────────────────────────────────────────────────────
    schema_errors = validate_graph(final_graph)
    if schema_errors:
        logger.warning(f"Schema validation errors ({len(schema_errors)}):")
        for err in schema_errors[:5]:
            logger.warning(f"  {err}")

    # ── Stage 6: Verification ─────────────────────────────────────────────────
    logger.info("── Stage 6: Clinical verification")
    try:
        verification_result = stages["stage6"].run(final_graph, text)
        save_cache("stage6_verify", verification_result)
        stages_completed += 1
    except PipelineError as e:
        logger.error(f"Stage 6 failed: {e}")
        failed_stages.append("stage6")
        verification_result = {}

    # ── Compute total tokens ───────────────────────────────────────────────────
    total_tokens = sum(s.tokens_used for s in stages.values())

    # ── Compute metrics ────────────────────────────────────────────────────────
    rules_extracted = len(rules.get("rules", []))
    rules_mapped = len(final_graph["graph"]["nodes"])  # approximation

    metrics = compute_graph_metrics(
        graph=final_graph,
        entities_count=len(entities.get("entities", [])),
        rules_count=rules_extracted,
        rules_mapped=rules_mapped,
        stages_completed=stages_completed,
        total_tokens=total_tokens,
        failed_stages=failed_stages,
    )

    # Add verification scores to metrics
    if verification_result:
        metrics["clinical_accuracy_score"] = verification_result.get("clinical_accuracy_score")
        metrics["clinical_issues"] = verification_result.get("issues", [])
        metrics["missing_scenarios"] = verification_result.get("missing_scenarios", [])

    # Add timing
    metrics["elapsed_seconds"] = round(time.time() - start_time, 1)

    # ── Save outputs ───────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(final_graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(metrics_path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved graph → {output_path}")
    logger.info(f"Saved metrics → {metrics_path}")

    # ── Print report ───────────────────────────────────────────────────────────
    source_name = Path(input_pdf).name
    if HAS_RICH:
        _print_rich(metrics, verification_result, output_path, metrics_path, 6, failed_stages, source_name)
    else:
        _print_plain(metrics, verification_result, output_path, metrics_path, 6, failed_stages)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clinical decision graphs from PDF guidelines using LLM cascade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input PDF file path")
    parser.add_argument("--output", "-o", default="graph.json", help="Output JSON file (default: graph.json)")
    parser.add_argument("--metrics", "-m", default="metrics.json", help="Metrics JSON file (default: metrics.json)")
    parser.add_argument("--section", "-s", default=None, help="Optional: focus on this section of the document")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load intermediate results from pipeline_cache/ if available"
    )
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
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
