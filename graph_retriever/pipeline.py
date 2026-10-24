"""
graph_retriever/pipeline.py
============================
Final Graph Retrieval Pipeline for ASIR-RAG

Includes:
- Semantic pre-filtering
- Query-aware traversal (correct directional handling)
- Edge boosting (Digital Rights aware)
- Hybrid scoring (Graph + Embedding)
- Distribution normalization + sharpening
- BFS reasoning paths (relation-aware + diversity)
- Adaptive early exit
- Chunk-aware embeddings
- Legal citation propagation
"""

from __future__ import annotations

import logging
import re
import hashlib
from pathlib import Path
from typing import List, Dict

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

import config.core_config as C
import config.router_config as RC
from shared.models import RetrievedChunk, StrategyCall

log = logging.getLogger(__name__)


class GraphRetrieverPipeline:
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        self._graph: nx.DiGraph = self._load_graph()

        self._embedder = SentenceTransformer(C.EMBED_MODEL_NAME)
        self._node_embeddings = self._build_node_embeddings()

    # ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, strategy_call: StrategyCall) -> List[RetrievedChunk]:
        query_type = strategy_call.query_type or "relational"

        query_vec = self._embedder.encode(query, normalize_embeddings=True)

        # 1. Semantic filtering
        top_nodes = self._top_k_nodes(query_vec, k=200)
        
        # ── FIX 4: Pass query_vec to seed node finder ──
        seed_nodes = self._find_seed_nodes(query, query_vec)

        log.debug("Seeds: %s", seed_nodes)

        expanded_nodes = set(top_nodes) | set(seed_nodes)

        for s in seed_nodes:
            expanded_nodes.update(list(self._graph.successors(s))[:15])
            expanded_nodes.update(list(self._graph.predecessors(s))[:15])

        query_graph = self._graph.subgraph(expanded_nodes).copy()

        # 2. Factual bypass
        if query_type == "factual":
            return self._factual_retrieval(query_graph, seed_nodes)

        # 3. Correct traversal handling
        if query_type == "causal":
            graph_for_ppr = query_graph.reverse(copy=True)

        elif query_type == "exploratory":
            graph_for_ppr = query_graph.to_undirected()

        elif query_type == "comparative":
            graph_for_ppr = query_graph.to_undirected()  # ✅ IMPORTANT

        else:
            graph_for_ppr = query_graph

        # 4. Edge boosting
        self._apply_edge_boosting(graph_for_ppr, query_type)

        # 5. Run PPR
        try:
            ppr_scores = nx.pagerank(
                graph_for_ppr,
                alpha=C.GRAPH_PPR_ALPHA,
                max_iter=C.GRAPH_PPR_MAX_ITER,
                tol=C.GRAPH_PPR_TOL,
                weight="query_weight"
            )
        except nx.PowerIterationFailedConvergence:
            log.warning("PPR failed → fallback uniform")
            ppr_scores = {n: 1.0 / len(graph_for_ppr) for n in graph_for_ppr.nodes}

        # 6. Adaptive early exit
        thresholds = {
            "relational": 0.1,
            "causal": 0.1,
            "exploratory": 0.05
        }
        if max(ppr_scores.values()) < thresholds.get(query_type, 0.1):
            log.debug("Weak graph signal — continuing with reduced confidence")

        # 7. Normalize
        ppr_scores = self._normalize_scores(ppr_scores)

        # 8. Build chunks
        return self._build_chunks(ppr_scores, query_vec, strategy_call, seed_nodes, query_type)

    # ─────────────────────────────────────────────────────────────
    # EDGE BOOSTING
    # ─────────────────────────────────────────────────────────────

    def _apply_edge_boosting(self, graph, query_type):
        for u, v, data in graph.edges(data=True):
            w = data.get("edge_weight", 0.1)
            rel = data.get("relation", "")

            if query_type == "causal" and rel in {"caused", "led_to", "resulted_in", "penalized", "breached"}:
                w *= 1.3
            elif query_type == "relational" and rel in {"related_to", "amended", "applies_to", "mandates", "regulates"}:
                w *= 1.2

            data["query_weight"] = min(w, 1.0)

    # ─────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────

    def _normalize_scores(self, scores):
        vals = list(scores.values())
        min_s, max_s = min(vals), max(vals)
        epsilon = 1e-7

        # FIX: flatline fallback → let cosine dominate
        if abs(max_s - min_s) < epsilon:
            # Return uniform normalized scores, not raw PPR
            uniform = 1.0 / len(scores)
            return {k: uniform for k in scores}

        return {
            k: max(((v - min_s) / (max_s - min_s + epsilon)) ** 1.2, 0.01)
            for k, v in scores.items()
        }

    # ─────────────────────────────────────────────────────────────
    # CHUNK BUILDING
    # ─────────────────────────────────────────────────────────────

    def _build_chunks(self, scores, query_vec, strategy_call, seed_nodes, query_type):
        chunks = []
        max_per_node = C.GRAPH_MAX_CHUNKS_PER_NODE
        top_k = RC.STRATEGY_TOP_K.get("graph", 5)   # RC is router_config — correct

        for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if len(chunks) >= top_k:
                break

            emb = self._node_embeddings.get(node)
            if emb is None:
                continue

            cosine = min((self._cosine(query_vec, emb) + 1) / 2, 0.9)

            ratio = self._blend_ratio(query_type)
            final = (ratio["ppr"] * score) + (ratio["embed"] * cosine)
            final = min(final * 1.2, 1.0)
            final *= strategy_call.weight

            # Seed dominance
            if node in seed_nodes:
                final = max(final, 0.7)

            node_data = self._graph.nodes[node]
            texts = node_data.get("chunk_texts", [])
            citations = node_data.get("legal_citations", [])

            paths = self._extract_paths(node, seed_nodes)

            for text in texts[:max_per_node]:
                chunks.append(RetrievedChunk(
                    chunk_id=f"graph_{node}_{hashlib.md5(text.encode()).hexdigest()[:6]}",
                    content=text,
                    source="knowledge_graph",
                    page_number=0,
                    score=final,
                    final_score=final,
                    rank=0,
                    strategy="graph",
                    metadata={"source_node": node},
                    relation_paths=paths,
                    legal_citation=citations,
                    graph_node=node
                ))

                if len(chunks) >= top_k:
                    break

        return chunks
    
    # ─────────────────────────────────────────────────────────────
    # BFS PATHS (FIXED)
    # ─────────────────────────────────────────────────────────────

    def _extract_paths(self, target, seeds):
        if target in seeds:
            return []

        scored_paths = []

        for seed in seeds:
            try:
                path = nx.shortest_path(self._graph, source=seed, target=target)

                if len(path) > 3:
                    continue

                path_with_rel = []
                weight_sum = 0

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = self._graph.get_edge_data(u, v) or {}
                    rel = edge.get("relation", "")
                    w = edge.get("edge_weight", 0.1)

                    if not rel:
                        continue

                    path_with_rel.extend([u, rel])
                    weight_sum += w

                path_with_rel.append(path[-1])

                score = weight_sum / (len(path) ** 1.5)
                scored_paths.append((score, path_with_rel))

            except nx.NetworkXNoPath:
                continue

        scored_paths.sort(key=lambda x: x[0], reverse=True)

        # FIX: diversity enforcement
        final_paths = []
        for _, p in scored_paths:
            if not final_paths:
                final_paths.append(p)
            else:
                overlap = set(p) & set(final_paths[0])
                if len(overlap) / min(len(p), len(final_paths[0])) < 0.5:
                    final_paths.append(p)
                    break

        return final_paths

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _blend_ratio(self, qt):
        return {
            "factual": {"ppr": 0.9, "embed": 0.1},
            "relational": {"ppr": 0.8, "embed": 0.2},
            "causal": {"ppr": 0.85, "embed": 0.15},
            "exploratory": {"ppr": 0.5, "embed": 0.5},

            # ✅ ADD THIS
            "comparative": {"ppr": 0.6, "embed": 0.4},

        }.get(qt, {"ppr": 0.6, "embed": 0.4})

    # ── FIX 4: Replace brittle string match with semantic lookup ──
    def _find_seed_nodes(self, query_text: str, query_vec: list[float], k=5) -> list[str]:
        """
        Finds entry points into the knowledge graph using embedding similarity 
        rather than exact string matching. This bridges the semantic gap (e.g. 
        "companies" resolving to the "Body Corporate" node).
        """
        candidates = self._top_k_nodes(query_vec, k=k)

        # Strict cosine threshold to ensure we don't start traversals from irrelevant nodes
        return [
            n for n in candidates
            if self._cosine(query_vec, self._node_embeddings[n]) > 0.5
        ]

    def _top_k_nodes(self, qv, k=200):
        return sorted(
            self._node_embeddings,
            key=lambda n: self._cosine(qv, self._node_embeddings[n]),
            reverse=True
        )[:k]

    def _build_node_embeddings(self):
        embeddings = {}
        for node in self._graph.nodes:
            data = self._graph.nodes[node]
            text = " ".join(data.get("chunk_texts", [])[:2]) or str(node)
            embeddings[node] = self._embedder.encode(text, normalize_embeddings=True)
        return embeddings

    def _cosine(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _factual_retrieval(self, graph, seeds):
        chunks = []
        max_per_node = C.GRAPH_MAX_CHUNKS_PER_NODE
        top_k = C.STRATEGY_TOP_K["graph"]

        processed_nodes = set()

        def _add_chunks(n, base_score):
            if n in processed_nodes:
                return
            processed_nodes.add(n)

            node_data = self._graph.nodes[n]
            texts = node_data.get("chunk_texts", [])
            citations = node_data.get("legal_citations", [])

            for text in texts[:max_per_node]:
                chunks.append(RetrievedChunk(
                    chunk_id=f"graph_{n}_{hashlib.md5(text.encode()).hexdigest()[:6]}",
                    content=text,
                    source="knowledge_graph",
                    page_number=0,
                    score=base_score,
                    final_score=base_score,
                    rank=0,
                    strategy="graph",
                    metadata={"source_node": n},
                    relation_paths=[],
                    legal_citation=citations,
                    graph_node=n
                ))

        for s in seeds:
            if len(chunks) >= top_k:
                break

            _add_chunks(s, 1.0)

            for nbr in graph.successors(s):
                if len(chunks) >= top_k:
                    break
                _add_chunks(nbr, 0.2)

        return sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]

    def _load_graph(self):
        import pickle

        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {self.graph_path}. Run build_graph.py first"
            )            

        with open(self.graph_path, "rb") as f:
            data = pickle.load(f)

        return data["graph"] if isinstance(data, dict) else data
