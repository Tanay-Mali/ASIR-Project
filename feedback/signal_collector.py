import logging
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QuerySignals:
    query_id: str
    session_id: str
    query_text: str
    timestamp: str
    query_type: str
    classifier_used: str
    classifier_conf: float
    routing_mode: str
    strategies_used: list
    fallback_triggered: bool
    top_chunk_scores: list
    reranker_top: float
    reranker_gap: float
    chunks_filtered: int
    chunk_ranks: dict
    llm_not_found: bool
    citation_valid: bool
    uncited_answer: bool
    hallucinated_refs: list
    grounding_score: float
    answer_text: str
    qa_similarity: float = 0.0
    max_chunk_sim: float = 0.0
    mean_chunk_sim: float = 0.0
    max_query_chunk_sim: float = 0.0
    dwell_time_seconds: float = 0.0
    rapid_rephrase: bool = False
    followed_by_rephrase: bool = False
    sources_clicked: list = field(default_factory=list)


def compute_grounding_score(
    query_text: str,
    answer_text: str,
    chunks: list,
    embedder,
    query_type: str = "factual",
) -> dict:
    """
    Three-component grounding score.
    Component 1: answer vs chunks  — did LLM use retrieved chunks?
    Component 2: query vs answer   — did answer address the query?
    Component 3: query vs chunks   — were the right chunks retrieved?
    Weights vary by query type.
    """
    query_vec = embedder.encode(query_text or "", normalize_embeddings=True)
    answer_vec = embedder.encode(answer_text or "", normalize_embeddings=True)

    chunk_texts = []
    for c in chunks or []:
        text = getattr(c, "content", None) if not isinstance(c, dict) else c.get("text", "")
        if text is None:
            continue
        chunk_texts.append(str(text))

    chunk_vecs = [embedder.encode(text, normalize_embeddings=True) for text in chunk_texts]

    chunk_answer_sims = [float(np.dot(answer_vec, cv)) for cv in chunk_vecs]
    max_chunk_sim = max(chunk_answer_sims) if chunk_answer_sims else 0.0
    mean_chunk_sim = sum(chunk_answer_sims) / len(chunk_answer_sims) if chunk_answer_sims else 0.0

    qa_similarity = float(np.dot(query_vec, answer_vec))

    chunk_query_sims = [float(np.dot(query_vec, cv)) for cv in chunk_vecs]
    max_query_chunk_sim = max(chunk_query_sims) if chunk_query_sims else 0.0

    if query_type == "factual":
        composite = (
            0.50 * max_chunk_sim +
            0.35 * qa_similarity +
            0.15 * max_query_chunk_sim
        )
    elif query_type == "exploratory":
        composite = (
            0.35 * mean_chunk_sim +
            0.40 * qa_similarity +
            0.25 * max_query_chunk_sim
        )
    elif query_type == "relational":
        composite = (
            0.40 * max_chunk_sim +
            0.35 * qa_similarity +
            0.25 * max_query_chunk_sim
        )
    else:  # comparative
        composite = (
            0.40 * mean_chunk_sim +
            0.35 * qa_similarity +
            0.25 * max_query_chunk_sim
        )

    return {
        "grounding_score": round(composite, 4),
        "max_chunk_sim": round(max_chunk_sim, 4),
        "mean_chunk_sim": round(mean_chunk_sim, 4),
        "qa_similarity": round(qa_similarity, 4),
        "max_query_chunk_sim": round(max_query_chunk_sim, 4),
    }


def auto_signals(
    query_text: str,
    answer_text: str,
    chunks: list,
    embedder,
    query_type: str = "factual",
) -> dict:
    """
    Fires automatically after every generation — no user click needed.
    Reuses the sentence-transformer embedder already loaded in graph_retriever/pipeline.py.
    """
    try:
        llm_not_found = "could not find enough information" in (answer_text or "").lower()
        grounding_result = compute_grounding_score(
            query_text, answer_text, chunks, embedder, query_type
        )
        citation_result = verify_legal_citations(answer_text or "", chunks or [])

        return {
            "llm_not_found": llm_not_found,
            "grounding_score": grounding_result["grounding_score"],
            "max_chunk_sim": grounding_result["max_chunk_sim"],
            "mean_chunk_sim": grounding_result["mean_chunk_sim"],
            "qa_similarity": grounding_result["qa_similarity"],
            "max_query_chunk_sim": grounding_result["max_query_chunk_sim"],
            "citation_valid": citation_result["citation_valid"],
            "uncited_answer": citation_result["uncited_answer"],
            "hallucinated_refs": citation_result["hallucinated_refs"],
        }
    except Exception as exc:
        log.exception("auto_signals failed: %s", exc)
        return {
            "llm_not_found": False,
            "grounding_score": 0.0,
            "max_chunk_sim": 0.0,
            "mean_chunk_sim": 0.0,
            "qa_similarity": 0.0,
            "max_query_chunk_sim": 0.0,
            "citation_valid": True,
            "uncited_answer": True,
            "hallucinated_refs": [],
        }


def verify_legal_citations(answer: str, chunks: list) -> dict:
    """
    Legal-domain citation check.
    Looks for Section/Clause/Article references in the answer
    and verifies they exist in the retrieved chunks.
    """
    legal_ref_pattern = re.compile(
        r"((?:Section|Clause|Article|Sub-section|Schedule|Rule)\s+\d+[a-zA-Z]?(?:\.\d+)*)",
        re.IGNORECASE,
    )

    answer_refs = set(r.lower().strip() for r in legal_ref_pattern.findall(answer or ""))

    if not answer_refs:
        return {
            "citation_valid": True,
            "uncited_answer": True,
            "hallucinated_refs": [],
            "answer_refs": [],
        }

    chunk_refs = set()
    for chunk in chunks:
        section_ref = getattr(chunk, "section_ref", None) if not isinstance(chunk, dict) else chunk.get("section_ref")
        if section_ref:
            chunk_refs.add(section_ref.lower().strip())

        content = getattr(chunk, "content", None) if not isinstance(chunk, dict) else chunk.get("text", "")
        for ref in legal_ref_pattern.findall(content or ""):
            chunk_refs.add(ref.lower().strip())

    hallucinated = answer_refs - chunk_refs
    return {
        "citation_valid": len(hallucinated) == 0,
        "uncited_answer": False,
        "hallucinated_refs": list(hallucinated),
        "answer_refs": list(answer_refs),
    }
