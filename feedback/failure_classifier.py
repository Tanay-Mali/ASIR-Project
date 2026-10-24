import logging
from typing import Iterable

from sklearn.metrics import precision_recall_curve

log = logging.getLogger(__name__)


def classify_failure(
    rating: str,
    failure_reason: str,
    llm_not_found: bool,
    grounding_score: float,
    citation_valid: bool,
    uncited_answer: bool,
    qa_similarity: float,
    max_chunk_sim: float,
    rapid_rephrase: bool,
    dwell_time: float,
    fallback_used: bool,
    thresh_high: float = 0.72,
    thresh_low: float = 0.52,
) -> tuple[str, float]:
    """
    Returns (failure_class, confidence).

    Failure classes:
    confirmed_success, system_error_retrieval, system_error_generation,
    system_error_hallucination, system_error_knowledge_gap, user_error, ambiguous.
    """
    try:
        if rating == "helpful":
            return "confirmed_success", 0.90

        # PATH C0: explicit wrong citation report
        if failure_reason == "wrong_citation":
            return "system_error_hallucination", 0.90

        if llm_not_found:
            if failure_reason == "not_in_doc":
                return "system_error_knowledge_gap", 0.95
            return "system_error_retrieval", 0.92

        # Legal-domain grounding threshold: 0.60
        if grounding_score < 0.60:
            return "system_error_hallucination", 0.85

        if uncited_answer:
            if grounding_score > 0.60:
                return "system_error_generation", 0.72
            return "system_error_hallucination", 0.78

        if not citation_valid:
            return "system_error_hallucination", 0.80

        if max_chunk_sim < thresh_low:
            return "system_error_retrieval", 0.78

        if fallback_used and rating == "not_helpful":
            return "system_error_retrieval", 0.75

        if rating == "not_helpful":
            if failure_reason == "not_in_doc":
                return "system_error_knowledge_gap", 0.95

            if failure_reason == "incomplete":
                if max_chunk_sim > thresh_high:
                    return "system_error_generation", 0.78
                return "system_error_retrieval", 0.72

            if failure_reason == "irrelevant":
                # Dual gate (never QA-similarity alone)
                if qa_similarity > thresh_high and grounding_score > 0.65 and citation_valid:
                    return "user_error", 0.78
                if qa_similarity > thresh_high and grounding_score < 0.60:
                    return "system_error_hallucination", 0.82
                if qa_similarity < thresh_low and max_chunk_sim < thresh_low:
                    return "system_error_retrieval", 0.85
                return "ambiguous", 0.40

        if rapid_rephrase:
            return "system_error_retrieval", 0.65

        if dwell_time > 20.0 and not rapid_rephrase:
            return "confirmed_success", 0.60

        return "ambiguous", 0.30
    except Exception as exc:
        log.exception("classify_failure failed: %s", exc)
        return "ambiguous", 0.30


def calibrate_thresholds(labeled_examples: Iterable[dict]) -> tuple[float, float]:
    """
    Finds similarity thresholds that separate confirmed_success from system_error_retrieval.
    Run after first 30 labels, then monthly.
    """
    examples = list(labeled_examples or [])
    if len(examples) < 30:
        return 0.72, 0.52

    sims = [float(e.get("max_chunk_sim", 0.0)) for e in examples]
    outcomes = [1 if e.get("failure_class") == "confirmed_success" else 0 for e in examples]
    precision, _recall, thresholds = precision_recall_curve(outcomes, sims)

    thresh_high = 0.72
    for i, p in enumerate(precision):
        if p >= 0.80 and i < len(thresholds):
            thresh_high = float(thresholds[i])
            break

    thresh_low = thresh_high - 0.20
    return round(thresh_high, 3), round(thresh_low, 3)
