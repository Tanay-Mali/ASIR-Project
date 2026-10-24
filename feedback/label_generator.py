import logging
import re

from classifier.preprocessor import _extract_structural_features, _NLP

log = logging.getLogger(__name__)
# Changed from 0.65 — Layer 1 fallback produces lower confidence scores
# Raise back to 0.65 once LLM classifier is stable
MIN_CONFIDENCE = 0.55


def generate_training_label(
    failure_class: str,
    confidence: float,
    query_text: str,
    query_type: str,
    signals: dict,
) -> dict:
    """
    Returns label payload for model training.
    """
    try:
        if confidence < MIN_CONFIDENCE:
            logging.info(
                f"[label_generator] Discarded: confidence {confidence:.3f} "
                f"below {MIN_CONFIDENCE} | class={failure_class}"
            )
            return {"label": None}

        label_confidence = float(confidence)
        if failure_class in ("confirmed_success", "user_error"):
            true_type = query_type
        elif failure_class in ("system_error_retrieval", "system_error_generation"):
            true_type, uncertain = infer_correct_query_type(query_text, query_type)
            if uncertain:
                label_confidence *= 0.7
        else:
            logging.info(
                f"[label_generator] Discarded: class={failure_class} not eligible"
            )
            return {"label": None}

        # Process the text through spaCy first
        doc = _NLP(query_text)
        
        return {
            "label": true_type,
            "true_query_type": true_type,
            "label_confidence": round(label_confidence, 4),
            "failure_class": failure_class,
            "query_text": query_text,
            "feature_vector": _extract_structural_features(doc, query_text),
        }
    except Exception as exc:
        log.exception("generate_training_label failed: %s", exc)
        return {"label": None}


def infer_correct_query_type(query_text: str, predicted_type: str) -> tuple[str, bool]:
    """
    Returns (query_type, uncertain_guess).
    """
    text = (query_text or "").lower()

    if re.search(r"\b(section|clause|sub-section|rule)\s+\d+[a-z]?(?:\.\d+)*\b", text):
        return "factual", False

    if any(w in text for w in ["relationship", "between", "connect", "difference"]):
        return "relational", False

    if any(w in text for w in ["meaning", "definition", "define", "what is"]):
        return "factual", False

    if any(w in text for w in ["explain", "overview", "summarize", "describe", "elaborate"]):
        return "exploratory", False

    if any(w in text for w in ["compare", "versus", "vs", "stricter", "better"]):
        return "comparative", False

    flip = {
        "factual": "exploratory",
        "relational": "factual",
        "exploratory": "relational",
        "comparative": "relational",
    }
    return flip.get(predicted_type, "factual"), True
