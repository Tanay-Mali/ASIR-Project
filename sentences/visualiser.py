"""
graph_builder/visualiser.py
=============================
Final Visualiser aligned with ASIR-RAG Graph Retriever

Includes:
✔ Subgraph safety (.copy())
✔ Query-weight visibility (actual PPR weights)
✔ Edge direction clarity
✔ Dynamic node labeling
✔ Config-driven boosted relations
✔ Frequency-aware edge thickness
✔ Cleaner debugging + demo insights
"""

from __future__ import annotations

import logging
import networkx as nx

log = logging.getLogger(__name__)

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False
    log.warning("PyVis not installed. Install with: pip install pyvis")

# ─────────────────────────────────────────────────────────────────────
# Config import (for consistency with retriever)
# ─────────────────────────────────────────────────────────────────────
try:
    from config.core_config import GRAPH_RELATION_SEMANTIC_WEIGHTS
    _BOOSTED_VERBS = set(GRAPH_RELATION_SEMANTIC_WEIGHTS.keys())
except Exception:
    _BOOSTED_VERBS = set()

# ─────────────────────────────────────────────────────────────────────
# Colour scheme
# ─────────────────────────────────────────────────────────────────────

_ENTITY_COLOURS = {
    "legislation":        "#FF9800",
    "regulatory_body":    "#9C27B0",
    "digital_right":      "#4CAF50",
    "actor":              "#2196F3",
    "principle":          "#00BCD4",
    "mechanism":          "#8BC34A",
    "concept":            "#607D8B",
    "event":              "#E91E63",
}
_DEFAULT_COLOUR = "#9E9E9E"

# ─────────────────────────────────────────────────────────────────────

def visualise_subgraph(
    graph: nx.DiGraph,
    center_node: str,
    hops: int = 2,
    output_path: str = "./graph_viz.html",
    show_weights: bool = True,
) -> None:

    if not _PYVIS_AVAILABLE:
        log.error("PyVis not installed")
        return

    if center_node not in graph:
        log.error("Node '%s' not found", center_node)
        return

    # ── Extract subgraph (FIX 1: copy for safety) ───────────────────
    nodes = {center_node}
    frontier = {center_node}

    for _ in range(hops):
        new_frontier = set()
        for n in frontier:
            new_frontier.update(graph.successors(n))
            new_frontier.update(graph.predecessors(n))
        nodes.update(new_frontier)
        frontier = new_frontier

    subgraph = graph.subgraph(nodes).copy()

    log.info(
        "Visualising %d nodes / %d edges",
        subgraph.number_of_nodes(),
        subgraph.number_of_edges()
    )

    # ── Build PyVis network ───────────────────────────────────────
    net = Network(
        height="850px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
    )

    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 2, "shadow": true},
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "dynamic"}
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 150
        }
      }
    }
    """)

    # ─────────────────────────────────────────────────────────
    # NODES
    # ─────────────────────────────────────────────────────────

    for node in subgraph.nodes():
        attrs = subgraph.nodes[node]

        label_type = attrs.get("label", "concept")
        colour = _ENTITY_COLOURS.get(label_type, _DEFAULT_COLOUR)

        chunk_count = len(attrs.get("chunk_texts", attrs.get("chunk_ids", [])))
        size = 12 + min(chunk_count * 3, 40)

        if node == center_node:
            colour = "#FFD700"
            size = 45

        # FIX 5: Better label handling
        label = node if len(node) < 30 else node[:27] + "..."

        hover = (
            f"Entity: {node}\n"
            f"Type: {label_type}\n"
            f"Context Chunks: {chunk_count}"
        )

        net.add_node(
            node,
            label=label,
            color=colour,
            size=size,
            title=hover,
        )

    # ─────────────────────────────────────────────────────────
    # EDGES
    # ─────────────────────────────────────────────────────────

    for u, v, data in subgraph.edges(data=True):

        # FIX 2: Show query_weight (actual PPR weight)
        base_weight = data.get("edge_weight", 0.1)
        query_weight = data.get("query_weight", base_weight)

        relation = data.get("relation", "related_to")
        method = data.get("method", "unknown")
        frequency = data.get("frequency", 1)

        # FIX 7: Frequency-aware width
        is_boosted = relation in _BOOSTED_VERBS
        width = (1 + query_weight * 5) + min(frequency * 0.3, 2)

        if is_boosted:
            colour = "#FF5722"
            width += 2  # extra emphasis
        else:
            colour = {
                "dependency_parse": "#4CAF50",
                "cooccurrence": "#607D8B",
                "llm": "#2196F3",
            }.get(method, "#9E9E9E")

        # FIX 2: Detailed hover with both weights
        hover = (
            f"{u} → {v}\n"
            f"Relation: {relation}\n"
            f"Base Weight: {base_weight:.4f}\n"
            f"Query Weight: {query_weight:.4f}\n"
            f"Frequency: {frequency}\n"
            f"Method: {method}"
        )

        # FIX 1: Respect show_weights toggle
        label = f"{relation} ({query_weight:.2f})" if show_weights else relation

        # FIX 4: Strong direction visibility
        net.add_edge(
            u,
            v,
            label=label[:30],
            title=hover,
            color=colour,
            width=width,
            arrows="to",
            font={"size": 10, "color": "#aaaaaa"}
        )

    net.save_graph(output_path)

    print(f"\n✅ Graph visualisation saved → {output_path}")
