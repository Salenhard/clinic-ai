"""Graph structural metrics using networkx."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("networkx not installed — structural metrics will be limited")


def compute_graph_metrics(
    graph: dict,
    entities_count: int = 0,
    rules_count: int = 0,
    rules_mapped: int = 0,
    stages_completed: int = 0,
    total_tokens: int = 0,
    failed_stages: list = None,
) -> dict:
    nodes = graph.get("graph", {}).get("nodes", [])
    edges = graph.get("graph", {}).get("edges", [])

    node_ids = {n["id"] for n in nodes}
    node_type_map = {n["id"]: n["type"] for n in nodes}

    # Count by type
    type_counts = {}
    for n in nodes:
        t = n.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Action nodes without evidence
    action_no_evidence = [
        n["id"] for n in nodes
        if n.get("type") == "ACTION"
        and not (n.get("action_details") or {}).get("evidence_level")
    ]

    # Decision nodes with no outgoing edges
    edge_sources = {e["from"] for e in edges}
    decision_no_edges = [
        n["id"] for n in nodes
        if n.get("type") == "DECISION" and n["id"] not in edge_sources
    ]

    metrics = {
        # Coverage
        "node_count": len(nodes),
        "edge_count": len(edges),
        "decision_nodes": type_counts.get("DECISION", 0),
        "action_nodes": type_counts.get("ACTION", 0),
        "warning_nodes": type_counts.get("WARNING", 0),
        "terminal_nodes": type_counts.get("END", 0),

        # Completeness
        "decision_nodes_with_no_options": decision_no_edges,
        "action_nodes_without_evidence": action_no_evidence,
        "action_nodes_with_evidence": type_counts.get("ACTION", 0) - len(action_no_evidence),
        "coverage_score": round(rules_mapped / rules_count, 3) if rules_count else 0.0,

        # Extraction
        "entities_extracted": entities_count,
        "rules_extracted": rules_count,
        "rules_mapped_to_nodes": rules_mapped,
        "extraction_completeness": round(rules_mapped / rules_count, 3) if rules_count else 0.0,

        # LLM pipeline
        "cascade_stages_completed": stages_completed,
        "total_tokens_used": total_tokens,
        "failed_stages": failed_stages or [],
    }

    if HAS_NETWORKX:
        metrics.update(_networkx_metrics(nodes, edges, node_type_map))
    else:
        metrics.update({
            "is_connected": None,
            "isolated_nodes": [],
            "unreachable_nodes": [],
            "has_cycles": None,
            "max_depth": None,
            "avg_branching_factor": None,
        })

    return metrics


def _networkx_metrics(nodes: list, edges: list, node_type_map: dict) -> dict:
    """Compute networkx-based structural metrics."""
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], node_type=n.get("type"))
    for e in edges:
        if e.get("from") and e.get("to"):
            G.add_edge(e["from"], e["to"])

    # Connectivity on undirected version
    undirected = G.to_undirected()
    is_connected = nx.is_connected(undirected) if undirected.number_of_nodes() > 0 else True
    isolated = [n for n in G.nodes() if G.degree(n) == 0]

    # Find START node
    start_nodes = [n for n, t in node_type_map.items() if t == "START"]
    unreachable = []
    max_depth = 0

    if start_nodes:
        start = start_nodes[0]
        reachable = nx.descendants(G, start) | {start}
        unreachable = [n for n in G.nodes() if n not in reachable]

        # BFS depth
        try:
            lengths = nx.single_source_shortest_path_length(G, start)
            max_depth = max(lengths.values()) if lengths else 0
        except Exception:
            max_depth = 0

    # Cycle detection
    try:
        has_cycles = not nx.is_directed_acyclic_graph(G)
    except Exception:
        has_cycles = False

    # Average branching factor for DECISION nodes
    decision_nodes = [n for n, t in node_type_map.items() if t == "DECISION"]
    branching_factors = [G.out_degree(n) for n in decision_nodes if G.out_degree(n) > 0]
    avg_branching = round(sum(branching_factors) / len(branching_factors), 2) if branching_factors else 0.0

    return {
        "is_connected": is_connected,
        "isolated_nodes": isolated,
        "unreachable_nodes": unreachable,
        "has_cycles": has_cycles,
        "max_depth": max_depth,
        "avg_branching_factor": avg_branching,
    }
