"""
router/fallback.py
==================
Fallback chain handler for the Router.

WHAT THIS DOES:
  When a primary retrieval strategy returns fewer than STRATEGY_MIN_RESULTS
  chunks (or times out), the fallback chain activates.

  The fallback chain works through a pre-ordered list of alternative
  strategies until one succeeds or all are exhausted.

  If all strategies fail, the system uses FALLBACK_FINAL ("direct_llm"),
  meaning the LLM answers from its parametric memory with a warning
  to the user that no documents were retrieved.

WHY A FALLBACK CHAIN MATTERS FOR INDUSTRIAL QUALITY:
  A RAG system that silently returns hallucinated answers when retrieval
  fails is dangerous in real applications. The fallback chain ensures:
    1. The system always attempts to retrieve something.
    2. The LLM generator always knows whether context was retrieved.
    3. Every failure is logged for debugging.

FALLBACK ORDER (from router_config.FALLBACK_STRATEGY_ORDER):
  ["vector", "bm25", "hierarchical", "graph"]

  Vector is first because it is the most reliable and broadly applicable
  fallback — it works for any query type and any document structure.
"""

from __future__ import annotations

import logging
from typing import Callable

import config.router_config as C
from shared.models import MergedResults, RetrievedChunk, RouterDecision, StrategyCall

log = logging.getLogger(__name__)

# Type alias for a retrieval function that the fallback handler calls.
# Each retrieval module provides a function matching this signature.
RetrievalFn = Callable[[StrategyCall], list[RetrievedChunk]]


class FallbackHandler:
    """
    Manages the fallback chain when primary retrieval is insufficient.

    This class is injected with the actual retrieval functions at
    application startup. It calls them in fallback order until
    sufficient results are obtained.

    Example
    -------
    >>> handler = FallbackHandler(retrieval_registry={
    ...     "vector": vector_retriever.retrieve,
    ...     "bm25":   bm25_retriever.retrieve,
    ... })
    >>> results, used_fallback, reason = handler.execute_with_fallback(
    ...     primary_results={"graph": []},
    ...     router_decision=decision,
    ... )
    """

    def __init__(
        self,
        retrieval_registry: dict[str, RetrievalFn],
    ) -> None:
        """
        Parameters
        ----------
        retrieval_registry : dict[str, callable]
            Maps strategy name → retrieval function.
            e.g. {"vector": vector_module.retrieve, "bm25": bm25_module.retrieve}
        """
        self._registry = retrieval_registry

    def execute_with_fallback(
        self,
        primary_results:  dict[str, list[RetrievedChunk]],
        router_decision:  RouterDecision,
    ) -> tuple[dict[str, list[RetrievedChunk]], bool, str]:
        """
        Check if primary results are sufficient; if not, run fallback chain.

        Parameters
        ----------
        primary_results : dict[str, list[RetrievedChunk]]
            Results already obtained from primary strategies.
        router_decision : RouterDecision
            Full routing decision (contains fallback_strategies).

        Returns
        -------
        (final_results, fallback_was_used, fallback_reason)
            final_results      : augmented results dict
            fallback_was_used  : True if any fallback was triggered
            fallback_reason    : human-readable reason if fallback ran
        """
        # ── Check if primary results are sufficient ────────────────────
        total_primary = sum(len(v) for v in primary_results.values())

        if total_primary >= C.STRATEGY_MIN_RESULTS:
            # Primary succeeded — no fallback needed
            return primary_results, False, ""

        # ── Primary failed — log and activate fallback chain ──────────
        failed_strategies = [
            sc.strategy for sc in router_decision.primary_strategies
            if not primary_results.get(sc.strategy)
        ]

        log.warning(
            "query_id=%s: primary strategies %s returned insufficient results "
            "(total=%d, needed=%d). Activating fallback chain.",
            router_decision.query_id,
            failed_strategies,
            total_primary,
            C.STRATEGY_MIN_RESULTS,
        )

        # ── Try each fallback strategy in order ───────────────────────
        augmented_results = dict(primary_results)
        fallback_reason   = f"Primary failed: {failed_strategies}."

        for fallback_call in router_decision.fallback_strategies:
            strategy = fallback_call.strategy

            if strategy not in self._registry:
                log.debug(
                    "Fallback strategy '%s' not registered. Skipping.", strategy
                )
                continue

            try:
                log.info(
                    "query_id=%s: trying fallback strategy '%s'.",
                    router_decision.query_id, strategy
                )
                fallback_chunks = self._registry[strategy](fallback_call)

                if fallback_chunks:
                    augmented_results[strategy] = fallback_chunks
                    fallback_reason += f" Fallback '{strategy}' succeeded with {len(fallback_chunks)} results."
                    log.info(
                        "query_id=%s: fallback '%s' returned %d chunks.",
                        router_decision.query_id, strategy, len(fallback_chunks)
                    )
                    return augmented_results, True, fallback_reason

                log.debug(
                    "query_id=%s: fallback strategy '%s' also returned empty.",
                    router_decision.query_id, strategy
                )

            except Exception as exc:
                log.error(
                    "query_id=%s: fallback strategy '%s' raised exception: %s",
                    router_decision.query_id, strategy, exc
                )
                continue

        # ── All fallbacks exhausted ────────────────────────────────────
        final_reason = fallback_reason + " All fallbacks exhausted — using direct LLM."
        log.error(
            "query_id=%s: ALL retrieval strategies failed. Routing to direct LLM.",
            router_decision.query_id
        )

        # Mark the router decision as direct_llm
        router_decision.routing_mode = "direct_llm"

        return augmented_results, True, final_reason


def is_sufficient(results: dict[str, list[RetrievedChunk]]) -> bool:
    """
    Return True if the results contain at least STRATEGY_MIN_RESULTS chunks.
    Utility function used by the Router to gate the fallback decision.
    """
    total = sum(len(v) for v in results.values())
    return total >= C.STRATEGY_MIN_RESULTS
