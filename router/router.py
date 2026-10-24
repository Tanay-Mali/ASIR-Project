"""
router/router.py
====================================================
Adaptive confidence-band router for Digital Rights & Laws.
REFACTORED: Global timeouts removed for full retrieval completion.
"""

from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Optional

import config.router_config as C
from router.confidence_bands import evaluate_band
from router.fallback import FallbackHandler, is_sufficient
from router.merger import Merger
from router.strategy_selector import StrategySelector
from router.weight_store import WeightStore
from reranker.reranker import Reranker, RerankResult
from shared.models import (
    ClassifierOutput,
    MergedResults,
    RetrievedChunk,
    RouterDecision,
    StrategyCall,
)

log = logging.getLogger(__name__)

RetrievalFn = Callable[[StrategyCall], list[RetrievedChunk]]

class Router:
    def __init__(
        self,
        retrieval_registry: dict[str, RetrievalFn],
        weight_store:       Optional[WeightStore] = None,
        reranker:           Optional[Reranker]    = None,
    ) -> None:
        self._registry  = retrieval_registry
        self._weights   = weight_store or WeightStore()
        self._selector  = StrategySelector(self._weights)
        self._merger    = Merger()
        self._fallback  = FallbackHandler(retrieval_registry)
        self._reranker  = reranker   
        self._query_count = 0

        log.info("Router initialised in Unlimited Execution mode (No Timeouts).")

    # ──────────────────────────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────────────────────────

    def route(self, classifier_output: ClassifierOutput) -> tuple[RouterDecision, MergedResults]:

        t_start = time.perf_counter()
        self._query_count += 1

    # feedback loop disabled
    # self._maybe_recalculate_weights()

        if classifier_output.needs_decomposition and classifier_output.sub_queries:
            return self._route_decomposed(classifier_output, t_start)

        return self._route_single(classifier_output, t_start)

    # ──────────────────────────────────────────────────────────────────
    # Internal Logic
    # ──────────────────────────────────────────────────────────────────

    def _route_single(
        self,
        classifier_output: ClassifierOutput,
        t_start: float,
        query_override: Optional[str] = None,
    ) -> tuple[RouterDecision, MergedResults]:
        
        band_result = evaluate_band(classifier_output)
        primary, fallback, merge_strategy = self._selector.select(
            classifier_output, band_result, query_override=query_override
        )

        decision = RouterDecision(
            query_id=classifier_output.query_id,
            session_id=classifier_output.session_id,
            raw_text=query_override or classifier_output.raw_text,
            query_type=classifier_output.query_type,
            confidence=classifier_output.confidence,
            routing_mode=band_result.routing_mode,
            confidence_band=band_result.band,
            primary_strategies=primary,
            fallback_strategies=fallback,
            merge_strategy=merge_strategy,
            weights_used=self._weights.get_weights_for_type(classifier_output.query_type),
            is_decomposed=False,
            routing_time_ms=0.0, 
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if band_result.routing_mode == "direct_llm":
            return decision, self._empty_merged_results(classifier_output.query_id)

        # Execute retrieval (TIMEOUT REMOVED)
        if not primary:
            log.warning("No primary strategies selected. Defaulting to fallback handler.")
            strategy_results, fallback_used, fallback_reason = \
                self._fallback.execute_with_fallback({}, decision)
        else:
            strategy_results = self._execute_retrieval(primary)

            if not is_sufficient(strategy_results):
                strategy_results, fallback_used, fallback_reason = \
                    self._fallback.execute_with_fallback(strategy_results, decision)
            else:
                fallback_used, fallback_reason = False, ""

        merged = self._merger.merge(strategy_results, decision)
        merged.fallback_used = fallback_used
        merged.fallback_reason = fallback_reason

        if merged.chunks:
            merged = self._apply_reranking(
                query_text=query_override or classifier_output.raw_text,
                merged=merged,
                top_k=C.ROUTER_FINAL_TOP_K
            )

        decision.routing_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return decision, merged

    # ──────────────────────────────────────────────────────────────────
    # Execution Logic (No Timeouts)
    # ──────────────────────────────────────────────────────────────────

    def _execute_retrieval(self, strategy_calls: list[StrategyCall]) -> dict[str, list[RetrievedChunk]]:
        """Executes retrievers in parallel and waits for all to finish."""
        results: dict[str, list[RetrievedChunk]] = {}
        
        with ThreadPoolExecutor(max_workers=C.ROUTER_MAX_PARALLEL_STRATEGIES) as executor:
            future_to_strategy = {
                executor.submit(self._call_strategy, sc): sc.strategy
                for sc in strategy_calls
            }
            
            # [TIMEOUT REMOVED] Iterates through futures as they complete, indefinitely
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    results[strategy] = future.result()
                except Exception as exc:
                    log.error("Strategy '%s' raised an unhandled exception: %s", strategy, exc)
                    results[strategy] = []

        return results

    def _call_strategy(self, strategy_call: StrategyCall) -> list[RetrievedChunk]:
        """Wrapper for a single strategy call with latency logging."""
        strategy = strategy_call.strategy
        if strategy not in self._registry:
            log.error("Strategy '%s' not in registry.", strategy)
            return []

        t0 = time.perf_counter()
        try:
            chunks = self._registry[strategy](strategy_call) or []
            latency = (time.perf_counter() - t0) * 1000
            log.info("Retriever %s: %d chunks | latency=%.2fms", strategy, len(chunks), latency)
            print(f"[{strategy_call.strategy.upper()}] fetched {len(chunks)} chunks.")
            for c in chunks[:2]:
                print(f"   -> {c.score:.4f} | {c.content[:100]}")
            return chunks
        except Exception as exc:
            log.error("Execution error in %s: %s", strategy, exc)
            return []

    def _route_decomposed(
        self,
        classifier_output: ClassifierOutput,
        t_start: float,
    ) -> tuple[RouterDecision, MergedResults]:
        """Route each sub-query independently in parallel without timeouts."""
        sub_queries         = classifier_output.sub_queries
        sub_classifications = classifier_output.sub_query_classifications
        sub_results: list[tuple] = [None] * len(sub_queries)

        with ThreadPoolExecutor(max_workers=C.ROUTER_MAX_PARALLEL_SUBQUERIES) as executor:
            futures = {
                executor.submit(
                    self._route_single,
                    sq_classification,
                    time.perf_counter(),
                    sq_text,
                ): i
                for i, (sq_text, sq_classification) in enumerate(
                    zip(sub_queries, sub_classifications)
                )
            }

            # [TIMEOUT REMOVED] Waits for all sub-queries to resolve
            for future in as_completed(futures):
                i = futures[future]
                try:
                    sub_results[i] = future.result()
                except Exception as exc:
                    log.error("Sub-query %d failed: %s", i, exc)
                    sub_results[i] = (None, self._empty_merged_results(classifier_output.query_id))

        # ── Aggregate sub-query results into a single MergedResults ───
        all_chunks: list[RetrievedChunk] = []
        all_strategies: set[str] = set()
        total_before = 0
        any_fallback_used = False
        fallback_reasons: list[str] = []

        for idx, result_pair in enumerate(sub_results):
            if result_pair is None:
                log.warning("Sub-query %d returned None; skipping.", idx)
                continue

            sub_decision, sub_merged = result_pair
            if sub_merged is None:
                continue

            all_chunks.extend(sub_merged.chunks)
            all_strategies.update(sub_merged.strategies_used)
            total_before += sub_merged.total_before_merge

            if sub_merged.fallback_used:
                any_fallback_used = True
                if sub_merged.fallback_reason:
                    fallback_reasons.append(
                        f"sq{idx}: {sub_merged.fallback_reason}"
                    )

        # De-duplicate chunks by chunk_id, keeping the highest-scored copy
        seen: dict[str, RetrievedChunk] = {}
        for chunk in all_chunks:
            cid = chunk.chunk_id
            if cid not in seen or chunk.score > seen[cid].score:
                seen[cid] = chunk
        unique_chunks = sorted(seen.values(), key=lambda c: c.score, reverse=True)

        # Re-assign ranks
        for i, chunk in enumerate(unique_chunks):
            chunk.rank = i + 1

        merged = MergedResults(
            query_id=classifier_output.query_id,
            chunks=unique_chunks,
            strategies_used=sorted(all_strategies),
            total_before_merge=total_before,
            total_after_merge=len(unique_chunks),
            merge_strategy="decomposed_rrf",
            retrieval_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
            fallback_used=any_fallback_used,
            fallback_reason="; ".join(fallback_reasons) if fallback_reasons else "",
        )

        # Re-rank the combined pool through the cross-encoder
        if merged.chunks:
            merged = self._apply_reranking(
                query_text=classifier_output.raw_text,
                merged=merged,
                top_k=C.ROUTER_FINAL_TOP_K,
            )

        # Build the parent RouterDecision for the decomposed query
        first_valid_decision = next(
            (d for d, _ in sub_results if d is not None),
            None,
        )

        decision = RouterDecision(
            query_id=classifier_output.query_id,
            session_id=classifier_output.session_id,
            raw_text=classifier_output.raw_text,
            query_type=classifier_output.query_type,
            confidence=classifier_output.confidence,
            routing_mode=first_valid_decision.routing_mode if first_valid_decision else "decomposed",
            confidence_band=first_valid_decision.confidence_band if first_valid_decision else "unknown",
            merge_strategy="decomposed_rrf",
            primary_strategies=(
                first_valid_decision.primary_strategies if first_valid_decision else []
            ),
            fallback_strategies=(
                first_valid_decision.fallback_strategies if first_valid_decision else []
            ),
            weights_used=self._weights.get_weights_for_type(classifier_output.query_type),
            is_decomposed=True,
            routing_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return decision, merged

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _empty_merged_results(self, query_id: str) -> MergedResults:
        """Return an empty MergedResults placeholder."""
        return MergedResults(
            query_id=query_id,
            chunks=[],
            strategies_used=[],
            total_before_merge=0,
            total_after_merge=0,
            merge_strategy="none",
            retrieval_time_ms=0.0,
            fallback_used=False,
            fallback_reason="",
        )

    def _apply_reranking(self, query_text, merged, top_k, *args, **kwargs):
        """
        Passes the merged chunks to the BAAI Cross-Encoder ML model 
        for intelligent re-scoring and sorting.
        """
        # 1. Initialize the ML Reranker (only happens once)
        if not hasattr(self, '_reranker_engine'):
            self._reranker_engine = Reranker()

        # 2. Guard against empty chunks
        if not hasattr(merged, 'chunks') or not merged.chunks:
            return merged

        # 3. Ask the ML model to read the chunks and re-grade them!
        rerank_result = self._reranker_engine.rerank(
            query_text=query_text,
            chunks=merged.chunks,
            top_k=top_k
        )
        
        # 4. Overwrite the old chunks with the smart, ML-sorted chunks
        merged.chunks = rerank_result.chunks
        return merged
