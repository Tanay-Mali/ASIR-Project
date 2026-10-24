def run_semantic_audit(
    stored_signals: dict,
) -> dict:
    """
    Returns pre-computed similarity values from stored signals.
    No re-encoding needed — values computed at query time by auto_signals().
    """
    return {
        "qa_similarity": stored_signals.get("qa_similarity", 0.0),
        "max_chunk_sim": stored_signals.get("max_chunk_sim", 0.0),
        "mean_chunk_sim": stored_signals.get("mean_chunk_sim", 0.0),
    }
