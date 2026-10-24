"""
graph_builder/graph_store.py
==============================
Knowledge graph storage using NetworkX DiGraph.

NODE SCHEMA:
  Each node represents one canonical entity.
  Attributes:
    - entity_type    : str   (person, constitutional_article, legislation, ...)
    - chunk_ids      : list  (all chunk IDs where this entity was mentioned)
    - chunk_texts    : list  (context text for each chunk_id)
    - mention_count  : int   (total times seen across all documents)

EDGE SCHEMA:
  Each directed edge represents a relationship between two entities.
  Attributes:
    - relation       : str   (semantic label, e.g. "led_to", "amended")
    - confidence     : float (extraction confidence: dep=0.90, llm=0.75, cooc=0.40)
    - method         : str   ("dependency_parse" | "llm" | "cooccurrence")
    - chunk_ids      : list  (source chunks where this relation was found)
    - frequency      : int   (how many times this exact triple was seen)
    - edge_weight    : float (computed from confidence + relation type + frequency)

EDGE WEIGHT COMPUTATION:
  The edge_weight is the key addition over a standard knowledge graph.
  It is used by the Weighted PPR in the graph retriever to bias traversal
  toward semantically strong, frequently-observed relationships.

  edge_weight = (
      confidence  * CONFIDENCE_MULTIPLIER  +
      rel_strength * RELATION_MULTIPLIER   +
      freq_norm   * FREQUENCY_MULTIPLIER
  )

  Where:
    confidence   : extraction confidence (0.40 to 0.90)
    rel_strength : semantic weight of the relation type (from config)
    freq_norm    : log-normalised frequency (prevents hub domination)

  All three components sum to 1.0 (the multipliers from config sum to 1.0).
  Result is clamped to [GRAPH_EDGE_WEIGHT_MINIMUM, 1.0].
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

import config as C
from graph_builder.document_loader import TextUnit
from graph_builder.entity_extractor import ExtractedEntity
from extraction.relationship_extractor import ExtractedRelation

log = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    NetworkX-based directed knowledge graph with weighted edges.

    Instantiated by the pipeline and saved/loaded via pickle.
    Loaded by the GraphRetriever at startup.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._max_edge_frequency: int = 1   # tracked for frequency normalisation

    # ──────────────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────────────

    def add_entity(
        self,
        entity:   ExtractedEntity,
        unit:     TextUnit,
    ) -> None:
        """
        Add or update an entity node in the graph.

        If the node already exists, appends the new chunk reference
        and increments the mention count.
        """
        canonical   = entity.canonical
        entity_type = entity.entity_type
        chunk_id    = unit.chunk_id
        context     = unit.context

        if canonical not in self.graph:
            self.graph.add_node(
                canonical,
                entity_type=entity_type,
                chunk_ids=[chunk_id],
                chunk_texts=[context],
                mention_count=1,
            )
        else:
            node = self.graph.nodes[canonical]
            if chunk_id not in node["chunk_ids"]:
                node["chunk_ids"].append(chunk_id)
                node["chunk_texts"].append(context)
            node["mention_count"] = node.get("mention_count", 0) + 1

    def add_relation(
        self,
        relation: ExtractedRelation,
        chunk_id: str,
    ) -> None:
        """
        Add or update a directed edge in the graph.

        If the edge already exists:
          - Keeps the higher-confidence version's relation label
          - Merges chunk_ids
          - Increments frequency
          - Recomputes edge_weight

        Only adds the edge if both nodes exist.
        Only adds edges with confidence >= GRAPH_MIN_EDGE_CONFIDENCE.
        """
        subj = relation.subject
        obj  = relation.object_

        # Guard: both nodes must exist
        if subj not in self.graph or obj not in self.graph:
            return

        # Guard: minimum confidence
        if relation.confidence < C.GRAPH_MIN_EDGE_CONFIDENCE:
            return

        if self.graph.has_edge(subj, obj):
            edge = self.graph[subj][obj]

            # Prefer higher-confidence extraction method
            if relation.confidence > edge.get("confidence", 0.0):
                edge["relation"]   = relation.relation
                edge["confidence"] = relation.confidence
                edge["method"]     = relation.method

            # Merge chunk references
            if chunk_id not in edge.get("chunk_ids", []):
                edge.setdefault("chunk_ids", []).append(chunk_id)

            # Increment frequency
            edge["frequency"] = edge.get("frequency", 1) + 1
            self._max_edge_frequency = max(
                self._max_edge_frequency, edge["frequency"]
            )

        else:
            # New edge
            self.graph.add_edge(
                subj, obj,
                relation=relation.relation,
                confidence=relation.confidence,
                method=relation.method,
                chunk_ids=[chunk_id],
                frequency=1,
                edge_weight=0.0,    # computed after insertion
            )

        # Recompute edge weight after every add/update
        self._compute_edge_weight(subj, obj)

    def recompute_all_edge_weights(self) -> None:
        """
        Recompute edge_weight for every edge in the graph.
        Called once after all documents have been processed,
        since frequency normalisation depends on max_edge_frequency.
        """
        for subj, obj in self.graph.edges():
            self._compute_edge_weight(subj, obj)

    # ──────────────────────────────────────────────────────────────────
    # Edge weight computation — the core addition for weighted PPR
    # ──────────────────────────────────────────────────────────────────

    def _compute_edge_weight(self, subj: str, obj: str) -> None:
        """
        Compute and store the edge_weight for a given edge.

        edge_weight = (
            confidence  * CONFIDENCE_MULTIPLIER  +
            rel_strength * RELATION_MULTIPLIER   +
            freq_norm    * FREQUENCY_MULTIPLIER
        )

        Component breakdown:
        ─────────────────────────────────────────────────────────────
        1. CONFIDENCE (60% weight by default):
           Raw extraction confidence mapped to [0, 1].
             dependency_parse → 0.90
             llm              → 0.75
             cooccurrence     → 0.40

        2. RELATION SEMANTIC STRENGTH (30% weight by default):
           How semantically meaningful is this relation type?
           "amended" (1.00) >> "co_occurs_with" (0.25)
           From config.GRAPH_RELATION_SEMANTIC_WEIGHTS.

        3. FREQUENCY (10% weight by default):
           How many times was this relationship observed across documents?
           log-normalised: log(1 + frequency) / log(1 + max_frequency)
           Prevents very frequent co-occurrence edges from dominating
           over rare but strong causal edges.
        ─────────────────────────────────────────────────────────────
        """
        if not self.graph.has_edge(subj, obj):
            return

        edge = self.graph[subj][obj]

        # Component 1: confidence
        confidence = float(edge.get("confidence", 0.5))

        # Component 2: relation semantic strength
        relation     = edge.get("relation", "co_occurs_with")
        rel_strength = C.GRAPH_RELATION_SEMANTIC_WEIGHTS.get(
            relation,
            C.GRAPH_RELATION_DEFAULT_WEIGHT,
        )

        # Component 3: frequency (log-normalised)
        frequency     = int(edge.get("frequency", 1))
        max_frequency = max(self._max_edge_frequency, 1)
        freq_norm     = (
            math.log(1 + frequency) / math.log(1 + max_frequency)
        )

        # Weighted combination
        weight = (
            confidence   * C.GRAPH_EDGE_WEIGHT_CONFIDENCE_MULTIPLIER
            + rel_strength * C.GRAPH_EDGE_WEIGHT_RELATION_MULTIPLIER
            + freq_norm   * C.GRAPH_EDGE_WEIGHT_FREQUENCY_MULTIPLIER
        )

        # Clamp to valid range
        weight = max(C.GRAPH_EDGE_WEIGHT_MINIMUM, min(1.0, weight))

        edge["edge_weight"] = round(weight, 6)

    # ──────────────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────────────

    def get_node(self, canonical: str) -> Optional[dict]:
        """Return node attributes or None if node doesn't exist."""
        if canonical in self.graph:
            return dict(self.graph.nodes[canonical])
        return None

    def get_statistics(self) -> dict:
        """Return graph summary statistics."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        # Edge weight distribution
        weights = [
            d.get("edge_weight", 0)
            for _, _, d in self.graph.edges(data=True)
        ]
        avg_weight = sum(weights) / max(len(weights), 1)

        # Method distribution
        methods: dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            m = d.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1

        return {
            "nodes":                   n_nodes,
            "edges":                   n_edges,
            "avg_edge_weight":         round(avg_weight, 4),
            "max_edge_frequency":      self._max_edge_frequency,
            "weakly_connected_components":
                nx.number_weakly_connected_components(self.graph),
            "edge_methods":            methods,
        }

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save graph to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "graph":               self.graph,
                "max_edge_frequency":  self._max_edge_frequency,
            }, f)
        log.info("Graph saved to %s. Stats: %s", path, self.get_statistics())

    def load(self, path: str) -> None:
        """Load graph from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.graph              = data["graph"]
            self._max_edge_frequency = data.get("max_edge_frequency", 1)
        else:
            # Legacy format — plain graph object
            self.graph = data
        log.info("Graph loaded from %s. Stats: %s", path, self.get_statistics())

    # ──────────────────────────────────────────────────────────────────
    # Neo4j export
    # ──────────────────────────────────────────────────────────────────

    def export_to_neo4j(
        self,
        uri:      str = C.GRAPH_NEO4J_URI,
        user:     str = C.GRAPH_NEO4J_USER,
        password: str = C.GRAPH_NEO4J_PASS,
    ) -> None:
        """
        Export the full graph to Neo4j.
        Clears existing data before writing.
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Install neo4j: pip install neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info(
            "Exporting graph to Neo4j: %d nodes, %d edges...",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        with driver.session() as session:
            # Clear existing
            session.run("MATCH (n) DETACH DELETE n")

            # Create nodes in batches of 500
            nodes = list(self.graph.nodes(data=True))
            for i in range(0, len(nodes), 500):
                batch = nodes[i:i+500]
                session.run(
                    """
                    UNWIND $batch AS row
                    CREATE (n:Entity {
                        name:         row.name,
                        entity_type:  row.entity_type,
                        mentions:     row.mentions
                    })
                    """,
                    batch=[
                        {
                            "name":        node,
                            "entity_type": attrs.get("entity_type", "unknown"),
                            "mentions":    attrs.get("mention_count", 0),
                        }
                        for node, attrs in batch
                    ],
                )

            # Create edges in batches of 500
            edges = list(self.graph.edges(data=True))
            for i in range(0, len(edges), 500):
                batch = edges[i:i+500]
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (a:Entity {name: row.src})
                    MATCH (b:Entity {name: row.dst})
                    CREATE (a)-[:RELATION {
                        type:        row.relation,
                        confidence:  row.confidence,
                        edge_weight: row.edge_weight,
                        frequency:   row.frequency,
                        method:      row.method
                    }]->(b)
                    """,
                    batch=[
                        {
                            "src":         src,
                            "dst":         dst,
                            "relation":    attrs.get("relation", "related_to"),
                            "confidence":  attrs.get("confidence", 0.5),
                            "edge_weight": attrs.get("edge_weight", 0.0),
                            "frequency":   attrs.get("frequency", 1),
                            "method":      attrs.get("method", "unknown"),
                        }
                        for src, dst, attrs in batch
                    ],
                )

        driver.close()
        log.info("Neo4j export complete.")
