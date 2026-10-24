"""
router/merger.py
================
Reciprocal Rank Fusion (RRF) merger for multi-strategy retrieval results.

WHAT RRF DOES:
  Takes result lists from multiple retrieval strategies and merges them
  into a single ranked list. Unlike score-based merging, RRF uses only
  rank positions — making it robust to score scale differences between
  strategies (a vector score of 0.82 and BM25 score of 14.3 cannot
  be directly compared, but rank positions can).

THE RRF FORMULA:
  For each chunk c and strategy s:
    RRF_score(c) = Σ_s [ weight(s) / (k + rank(c, s)) ]

  Where:
    k       = RRF_K (60) — constant preventing top ranks from dominating
    rank    = 1-based position of chunk in strategy's result list
    weight  = StrategyCall.weight for that strategy

  A chunk that appears in multiple strategies' results gets contributions
  from each, making cross-strategy agreement a strong positive signal.

DEDUPLICATION:
  After merging, near-duplicate chunks (Jaccard similarity > RRF_DEDUP_THRESHOLD)
  are removed. The higher-ranked chunk is kept.

EXAMPLE:
  Vector returns: [ChunkA(rank=1), ChunkB(rank=2), ChunkC(rank=3)]
  BM25 returns:   [ChunkB(rank=1), ChunkD(rank=2), ChunkA(rank=3)]

  RRF scores (k=60, weights both 0.5):
    ChunkA: 0.5/(60+1) + 0.5/(60+3) = 0.00820 + 0.00794 = 0.01614
    ChunkB: 0.5/(60+2) + 0.5/(60+1) = 0.00806 + 0.00820 = 0.01626
    ChunkC: 0.5/(60+3) = 0.00794
    ChunkD: 0.5/(60+2) = 0.00806

  Final ranking: B, A, D, C  (ChunkB wins — appeared highly in both)
"""

from __future__ import annotations

import logging
import time
import copy
from collections import defaultdict

import config.router_config as C
from shared.models import MergedResults, RetrievedChunk, RouterDecision

log = logging.getLogger(__name__)


class Merger:
    """
    Merges multi-strategy retrieval results using Reciprocal Rank Fusion.

    Instantiate once in the Router and reuse.
    """

    def merge(
        self,
        strategy_results: dict[str, list[RetrievedChunk]],
        router_decision:  RouterDecision,
    ) -> MergedResults:
        """
        Merge results from multiple strategies into a single ranked list.

        Parameters
        ----------
        strategy_results : dict[str, list[RetrievedChunk]]
            Keys are strategy names, values are result lists from each retriever.
            e.g. {"vector": [...], "bm25": [...]}

        router_decision : RouterDecision
            Used to get strategy weights and query_id for logging.

        Returns
        -------
        MergedResults
            Single ranked list trimmed to ROUTER_FINAL_TOP_K chunks.
        """
        t_start = time.perf_counter()

        # ── Filter to non-empty results ───────────────────────────────
        non_empty = {k: v for k, v in strategy_results.items() if v}
        strategies_used = list(non_empty.keys())

        if not non_empty:
            log.warning(
                "query_id=%s: all strategies returned empty results.",
                router_decision.query_id
            )
            return self._empty_result(router_decision, t_start)

        # ── Single strategy — no merging needed ───────────────────────
        if len(non_empty) == 1:
            strategy = strategies_used[0]
            chunks   = non_empty[strategy][:C.ROUTER_FINAL_TOP_K]
            # Re-assign ranks sequentially
            for i, chunk in enumerate(chunks):
                chunk.rank = i + 1

            t_end = time.perf_counter()
            return MergedResults(
                query_id=router_decision.query_id,
                chunks=chunks,
                strategies_used=strategies_used,
                total_before_merge=len(non_empty[strategy]),
                total_after_merge=len(chunks),
                merge_strategy="none",
                retrieval_time_ms=round((t_end - t_start) * 1000, 2),
                fallback_used=False,
                fallback_reason="",
            )

        # ── Build strategy weight map ─────────────────────────────────
        # Map from strategy name → weight from the RouterDecision
        strategy_weights = {
            sc.strategy: sc.weight
            for sc in router_decision.primary_strategies
        }

        # ── Run RRF ───────────────────────────────────────────────────
        total_before = sum(len(v) for v in non_empty.values())
        rrf_scores, chunk_registry = self._compute_rrf_scores(
            non_empty, strategy_weights
        )

        # Sort chunks by RRF score descending
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        # ── Deduplication ─────────────────────────────────────────────
        deduplicated = self._deduplicate(ranked_ids, chunk_registry)

        # ── Trim to final top-k ───────────────────────────────────────
        final_chunks = []
        for i, chunk_id in enumerate(deduplicated[:C.ROUTER_FINAL_TOP_K]):
            original_chunk = chunk_registry[chunk_id]
            chunk = copy.copy(original_chunk)

            chunk.rank  = i + 1
            chunk.score = round(rrf_scores[chunk_id], 6)

            final_chunks.append(chunk)
        t_end = time.perf_counter()

        log.debug(
            "query_id=%s RRF merge: %d → %d chunks (dedup removed %d) in %.1fms",
            router_decision.query_id,
            total_before,
            len(final_chunks),
            total_before - len(deduplicated),
            (t_end - t_start) * 1000,
        )

        return MergedResults(
            query_id=router_decision.query_id,
            chunks=final_chunks,
            strategies_used=strategies_used,
            total_before_merge=total_before,
            total_after_merge=len(final_chunks),
            merge_strategy=router_decision.merge_strategy,
            retrieval_time_ms=round((t_end - t_start) * 1000, 2),
            fallback_used=False,
            fallback_reason="",
        )

    # ──────────────────────────────────────────────────────────────────
    # Core RRF computation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rrf_scores(
        strategy_results: dict[str, list[RetrievedChunk]],
        strategy_weights: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, RetrievedChunk]]:
        """
        Compute RRF score for every unique chunk across all strategies.

        Returns
        -------
        rrf_scores     : dict[chunk_id, float]
        chunk_registry : dict[chunk_id, RetrievedChunk]
            Single canonical object for each chunk.
        """
        rrf_scores: dict[str, float]        = defaultdict(float)
        chunk_registry: dict[str, RetrievedChunk] = {}

        for strategy, chunks in strategy_results.items():
            weight = strategy_weights.get(strategy, 1.0)

            for rank_0based, chunk in enumerate(chunks):
                rank_1based = rank_0based + 1
                chunk_id    = chunk.chunk_id

                # RRF contribution: weight / (k + rank)
                rrf_scores[chunk_id] += weight / (C.RRF_K + rank_1based)

                # Register chunk (first occurrence wins for deduplication identity)
                if chunk_id not in chunk_registry:
                    chunk_registry[chunk_id] = chunk

        return dict(rrf_scores), chunk_registry

    @staticmethod
    def _deduplicate(
        ranked_ids: list[str],
        chunk_registry: dict[str, RetrievedChunk],
    ) -> list[str]:

        kept = []
        kept_texts = []

        for chunk_id in ranked_ids:
            chunk = chunk_registry[chunk_id]

            # ✅ SAFE TEXT EXTRACTION
            if isinstance(chunk.content, str):
                text = chunk.content.lower()
            elif isinstance(chunk.content, dict):
                text = chunk.content.get("text", "").lower()
            else:
                text = str(chunk.content).lower()

            if not text.strip():
                continue

            is_duplicate = False
            for kept_text in kept_texts:
                if _jaccard_similarity(text, kept_text) > C.RRF_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(chunk_id)
                kept_texts.append(text)

        return kept

    @staticmethod
    def _empty_result(router_decision: RouterDecision, t_start: float) -> MergedResults:
        t_end = time.perf_counter()
        return MergedResults(
            query_id=router_decision.query_id,
            chunks=[],
            strategies_used=[],
            total_before_merge=0,
            total_after_merge=0,
            merge_strategy="none",
            retrieval_time_ms=round((t_end - t_start) * 1000, 2),
            fallback_used=False,
            fallback_reason="all strategies returned empty results",
        )


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute character-level Jaccard similarity between two strings.

    Uses sets of 3-character shingles for robustness to minor wording differences.
    Fast enough for small texts without any ML models.

    Returns float in [0.0, 1.0]. 1.0 = identical, 0.0 = no overlap.
    """
    def shingles(text: str, n: int = 5) -> set[str]:
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    set_a = shingles(text_a)
    set_b = shingles(text_b)

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0
