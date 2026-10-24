"""
router/confidence_bands.py
===========================
Evaluates the classifier output and assigns a confidence band.

The confidence band determines the routing mode:
  HIGH band   → single best strategy
  MEDIUM band → hybrid of top-2 strategies
  LOW band    → ensemble of all relevant strategies
  NO_RETRIEVAL→ bypass all retrieval

WHY BANDS MATTER:
  A classifier confidence of 0.72 means the model is reasonably sure
  but not certain. Forcing a single strategy for this query ignores
  the 28% probability that a different strategy is actually better.
  Routing to a hybrid in this case is more robust.

  The gap check is equally important. A query scored:
    factual: 0.86, relational: 0.84
  looks high-confidence on the top score alone (0.86 > threshold)
  but is genuinely ambiguous — the gap is only 0.02.
  The gap threshold prevents misrouting on thin margins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config.router_config as C
from shared.models import ClassifierOutput

log = logging.getLogger(__name__)


@dataclass
class BandResult:
    """
    The result of confidence band evaluation.
    Consumed by the strategy selector to determine routing mode.
    """
    band:              str    # "high" | "medium" | "low" | "no_retrieval"
    routing_mode:      str    # "single" | "hybrid" | "ensemble" | "direct_llm"
    top_type:          str    # highest-scoring query type
    top_score:         float
    second_type:       str    # second-highest query type
    second_score:      float
    gap:               float  # top_score - second_score
    top_n_types:       list[tuple[str, float]]  # top types with scores, ordered


def evaluate_band(classifier_output: ClassifierOutput) -> BandResult:
    """
    Evaluate the classifier output and return the confidence band.

    Parameters
    ----------
    classifier_output : ClassifierOutput

    Returns
    -------
    BandResult with band, routing_mode, and ranked type scores.
    """

    # ── No-retrieval bypass ───────────────────────────────────────────
    if classifier_output.query_type == "no_retrieval":
        return BandResult(
            band="no_retrieval",
            routing_mode="direct_llm",
            top_type="no_retrieval",
            top_score=classifier_output.confidence,
            second_type="",
            second_score=0.0,
            gap=1.0,
            top_n_types=[("no_retrieval", classifier_output.confidence)],
        )

    # ── Sort all_scores descending ────────────────────────────────────
    scores = classifier_output.all_scores or {}

    # Filter out no_retrieval from ranking — it has its own path above
    scores = {k: v for k, v in scores.items() if k != "no_retrieval"}

    if not scores:
        # Fallback: use the classifier's top-level type at face value
        scores = {classifier_output.query_type: classifier_output.confidence}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_type,    top_score    = sorted_scores[0]
    second_type, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0.0)
    gap = top_score - second_score

    # ── HIGH band ─────────────────────────────────────────────────────
    # Conditions: top score above threshold AND gap above threshold.
    # Both must hold. A high score with a tiny gap is still ambiguous.
    if (top_score >= C.ROUTER_HIGH_CONFIDENCE_THRESHOLD and
            gap >= C.ROUTER_HIGH_GAP_THRESHOLD):
        log.debug(
            "query_id=%s → HIGH band (top=%.2f, gap=%.2f) → single routing",
            classifier_output.query_id, top_score, gap
        )
        return BandResult(
            band="high",
            routing_mode="single",
            top_type=top_type,
            top_score=top_score,
            second_type=second_type,
            second_score=second_score,
            gap=gap,
            top_n_types=sorted_scores[:2],
        )

    # ── MEDIUM band ───────────────────────────────────────────────────
    # Conditions: top score reasonably high but not fully confident,
    # OR gap is small (two types are close in score).
    # Route to hybrid of top-2 strategies.
    if top_score >= C.ROUTER_MEDIUM_CONFIDENCE_THRESHOLD:
        log.debug(
            "query_id=%s → MEDIUM band (top=%.2f, gap=%.2f) → hybrid routing",
            classifier_output.query_id, top_score, gap
        )
        return BandResult(
            band="medium",
            routing_mode="hybrid",
            top_type=top_type,
            top_score=top_score,
            second_type=second_type,
            second_score=second_score,
            gap=gap,
            top_n_types=sorted_scores[:2],
        )

    # ── LOW band ──────────────────────────────────────────────────────
    # Classifier is genuinely uncertain. Route to all relevant strategies.
    # The merger will weight each strategy's contribution by score.
    log.debug(
        "query_id=%s → LOW band (top=%.2f, gap=%.2f) → ensemble routing",
        classifier_output.query_id, top_score, gap
    )
    return BandResult(
        band="low",
        routing_mode="ensemble",
        top_type=top_type,
        top_score=top_score,
        second_type=second_type,
        second_score=second_score,
        gap=gap,
        top_n_types=sorted_scores,   # all types for ensemble
    )
