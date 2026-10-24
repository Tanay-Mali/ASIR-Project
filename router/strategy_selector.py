"""
router/strategy_selector.py
============================
Translates a BandResult + dynamic weights into a concrete list of StrategyCall objects.

THIS IS WHERE THE TWO LAYERS COMBINE:
  BandResult  → tells us how many strategies to use (1, 2, or all)
  WeightStore → tells us which strategies are preferred for this query type

The strategy selector multiplies classifier scores by dynamic weights
to produce adjusted priority scores, then builds StrategyCall objects
for the top-N strategies according to the band.

ADJUSTED PRIORITY SCORE:
  adjusted_score(strategy) =
    classifier_score(query_type) * dynamic_weight(query_type, strategy)

  classifier_score: from ClassifierOutput.all_scores
  dynamic_weight:   from WeightStore.get_weight(query_type, strategy)

  Example:
    classifier says: {factual: 0.72, relational: 0.18}
    dynamic weights: {vector: 0.82, graph: 0.55, bm25: 0.71, hierarchical: 0.48}

    For HIGH band (query_type = "factual"):
      primary strategy of type "factual" = vector (default)
      adjusted = 0.72 * 0.82 = 0.590  ← vector
                 0.72 * 0.71 = 0.511  ← bm25
                 0.72 * 0.55 = 0.396  ← graph
                 0.72 * 0.48 = 0.346  ← hierarchical

      → Single routing: vector strategy is chosen

    For MEDIUM band (top-2 needed):
      → Hybrid: vector + bm25 (top-2 by adjusted score)
"""

from __future__ import annotations

import logging
from typing import Optional

import config.router_config as C
from router.confidence_bands import BandResult
from router.weight_store import WeightStore
from shared.models import ClassifierOutput, StrategyCall
from router.query_expander import expand_query

log = logging.getLogger(__name__)

# All strategies the system can route to
_ALL_STRATEGIES = ["vector", "bm25", "hierarchical", "graph"]

# Default strategy ordering per query type — used as tiebreaker when
# adjusted scores are equal (rare but must be handled).
_DEFAULT_STRATEGY_PRIORITY = {
    "factual":     ["vector",       "bm25",   "hierarchical", "graph"],
    "relational":  ["graph",        "vector", "bm25",         "hierarchical"],
    "comparative": ["vector",       "bm25",   "graph",        "hierarchical"],
    "exploratory": ["hierarchical", "vector", "bm25",         "graph"],
}


class StrategySelector:
    """
    Converts BandResult + WeightStore into a list of StrategyCall objects.

    Instantiate once in the Router and reuse.
    """

    def __init__(self, weight_store: WeightStore) -> None:
        self._weights = weight_store

    def select(
        self,
        classifier_output: ClassifierOutput,
        band_result:       BandResult,
        query_override:    Optional[str] = None,
    ) -> tuple[list[StrategyCall], list[StrategyCall], str]:
        """
        Build primary and fallback StrategyCall lists.

        Parameters
        ----------
        classifier_output : ClassifierOutput
        band_result       : BandResult from confidence_bands.evaluate_band()
        query_override    : str, optional
            If provided, use this text as the query for all StrategyCall objects.
            Used for sub-queries in decomposed routing.

        Returns
        -------
        (primary_strategies, fallback_strategies, merge_strategy)
            primary_strategies  : list[StrategyCall] — strategies to run first
            fallback_strategies : list[StrategyCall] — if primary fails
            merge_strategy      : "none" | "rrf" | "weighted_rrf"
        """
        query_text = query_override or classifier_output.raw_text

        # ── No-retrieval path ─────────────────────────────────────────
        if band_result.band == "no_retrieval":
            return [], [], "none"

        # ── Compute Adjusted Scores ───────────────────────────────────
        # (Replaces the TEMP FIX that set everything to 1.0)
        adjusted_scores = self._compute_adjusted_scores(
            classifier_output, band_result, query_text
        )

        log.debug(
            "query_id=%s base scores (weighted + token adjusted): %s",
            classifier_output.query_id, adjusted_scores
        )

        # ── Build primary strategies based on band ─────────────────────
        if band_result.routing_mode == "single":
            primary = self._build_single(
                adjusted_scores, band_result, query_text
            )
            merge = "none"

        elif band_result.routing_mode == "hybrid":
            primary = self._build_hybrid(
                adjusted_scores, band_result, query_text
            )
            merge = "weighted_rrf"

        else:  # ensemble
            primary = self._build_ensemble(
                adjusted_scores, band_result, query_text
            )
            merge = "weighted_rrf"

        # ── Build fallback strategies ─────────────────────────────────
        fallback = self._build_fallback(primary, query_text)

        return primary, fallback, merge

    # ──────────────────────────────────────────────────────────────────
    # Adjusted score computation
    # ──────────────────────────────────────────────────────────────────

    def _compute_adjusted_scores(
        self,
        classifier_output: ClassifierOutput,
        band_result:       BandResult,
        query_text:        str,
    ) -> dict[str, float]:
        """
        For each strategy, compute:
            adjusted_score = classifier_type_score * dynamic_weight
        Then apply token-length heuristics to boost BM25 or Vector.
        """
        query_type = classifier_output.query_type
        all_scores = classifier_output.all_scores or {}

        # Get dynamic weights for this query type
        dynamic_weights = self._weights.get_weights_for_type(query_type)

        strategy_type_affinity = {
            "vector":       "factual",
            "bm25":         "factual",
            "graph":        "relational",
            "hierarchical": "exploratory",
        }

        adjusted: dict[str, float] = {}

        for strategy in _ALL_STRATEGIES:
            preferred_type = strategy_type_affinity[strategy]
            type_score     = all_scores.get(preferred_type, 0.0)
            dynamic_weight = dynamic_weights.get(strategy, C.ROUTER_WEIGHT_INITIAL)
            adjusted[strategy] = round(type_score * dynamic_weight, 6)

        # Ensure the primary type's default strategies are never zero-scored.
        primary_strategies = _DEFAULT_STRATEGY_PRIORITY.get(query_type, _ALL_STRATEGIES)
        for strategy in primary_strategies[:2]:
            if adjusted[strategy] == 0.0:
                adjusted[strategy] = dynamic_weights.get(strategy, C.ROUTER_WEIGHT_INITIAL)

        # ── Token-Based Boosting Logic ────────────────────────────────
        query_token_count = len(query_text.split())
        
        if query_token_count < 4:
            # Boost BM25 for short exact-term queries (e.g. "Section 72A", "data fiduciary")
            adjusted["bm25"] = round(adjusted.get("bm25", 0.0) * 1.5, 6)
            log.debug("Boosted BM25 score (Query length < 4 tokens)")
            
        elif query_token_count > 8:
            # Boost Vector for longer, conceptual queries
            adjusted["vector"] = round(adjusted.get("vector", 0.0) * 1.3, 6)
            log.debug("Boosted Vector score (Query length > 8 tokens)")

        return adjusted

    # ──────────────────────────────────────────────────────────────────
    # Strategy list builders
    # ──────────────────────────────────────────────────────────────────

    def _build_single(
        self,
        adjusted_scores: dict[str, float],
        band_result:     BandResult,
        query_text:      str,
    ) -> list[StrategyCall]:
        """
        HIGH band: one strategy only.
        Select the strategy with the highest adjusted score.
        """
        best_strategy = self._rank_strategies(
            adjusted_scores,
            preferred_order=_DEFAULT_STRATEGY_PRIORITY.get(band_result.top_type, [])
        )[0]

        return [self._make_strategy_call(best_strategy, weight=1.0, query_text=query_text)]

    def _build_hybrid(
        self,
        adjusted_scores: dict[str, float],
        band_result:     BandResult,
        query_text:      str,
    ) -> list[StrategyCall]:
        
        ranked = self._rank_strategies(
            adjusted_scores,
            preferred_order=_DEFAULT_STRATEGY_PRIORITY.get(band_result.top_type, [])
        )
        top2 = ranked[:2]

        # Fix: Default missing scores to 1.0 so they get equal weight, not zero weight!
        s0 = adjusted_scores.get(top2[0], 1.0)
        s1 = adjusted_scores.get(top2[1], 1.0) if len(top2) > 1 else 0.0
        total = s0 + s1 if (s0 + s1) > 0 else 1.0

        calls = []
        for i, strategy in enumerate(top2):
            # Fix: Default this to 1.0 as well
            score = adjusted_scores.get(strategy, 1.0)
            calls.append(
                self._make_strategy_call(
                    strategy,
                    weight=round(score / total, 4),
                    query_text=query_text,
                )
            )
        return calls
    
    def _build_ensemble(
        self,
        adjusted_scores: dict[str, float],
        band_result:     BandResult,
        query_text:      str,
    ) -> list[StrategyCall]:
        """
        LOW band: all strategies with non-zero adjusted scores.
        Weights are normalised to sum to 1.0.
        """
        ranked = self._rank_strategies(
            adjusted_scores,
            preferred_order=_DEFAULT_STRATEGY_PRIORITY.get(band_result.top_type, [])
        )

        # Include strategies with adjusted score > 0.0
        active = [s for s in ranked if adjusted_scores[s] > 0.0]

        # Always include at least 2 strategies even if scores are zero
        if len(active) < 2:
            active = ranked[:2]

        total = sum(adjusted_scores[s] for s in active)
        if total == 0:
            # All equal — distribute evenly
            total = len(active)
            weights = {s: 1.0 / total for s in active}
        else:
            weights = {s: adjusted_scores[s] / total for s in active}

        return [
            self._make_strategy_call(s, weight=round(weights[s], 4), query_text=query_text)
            for s in active
        ]

    def _build_fallback(
        self,
        primary_strategies: list[StrategyCall],
        query_text: str,
    ) -> list[StrategyCall]:
        """
        Build a fallback chain from strategies NOT in the primary list.
        Ordered by FALLBACK_STRATEGY_ORDER from config.router_config.
        """
        primary_names = {sc.strategy for sc in primary_strategies}
        fallbacks     = []

        for strategy in C.FALLBACK_STRATEGY_ORDER:
            if strategy not in primary_names:
                fallbacks.append(
                    self._make_strategy_call(
                        strategy, weight=1.0, query_text=query_text
                    )
                )

        return fallbacks

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rank_strategies(
        adjusted_scores:  dict[str, float],
        preferred_order:  list[str],
    ) -> list[str]:
        """
        Sort strategies by adjusted score descending.
        Tiebreaker: preferred_order from the default strategy priority.
        """
        def sort_key(strategy: str) -> tuple:
            score = adjusted_scores.get(strategy, 0.0)
            tie_rank = (
                preferred_order.index(strategy)
                if strategy in preferred_order
                else len(preferred_order)
            )
            return (-score, tie_rank)

        return sorted(_ALL_STRATEGIES, key=sort_key)

    @staticmethod
    def _make_strategy_call(
        strategy: str,
        weight: float,
        query_text: str,
    ) -> StrategyCall:
        
        # ── STATIC SYNONYM EXPANSION (FIX 3) ──
        # Only expand the query for keyword-based strategies. 
        # Vector/Graph receive the unpolluted original text.
        final_query_text = query_text
        if strategy == "bm25":
            final_query_text = expand_query(query_text)
            if final_query_text != query_text:
                log.debug("Expanded BM25 Query: '%s' -> '%s'", query_text, final_query_text)

        return StrategyCall(
            strategy=strategy,
            top_k=C.STRATEGY_TOP_K.get(strategy, 5),
            weight=weight,
            query_text=final_query_text, # Passes the expanded string to BM25
        )
