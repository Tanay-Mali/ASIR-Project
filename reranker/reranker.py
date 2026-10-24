"""
reranker/reranker.py
=====================
Cross-encoder re-ranker for ASIR-RAG.

POSITION IN PIPELINE:
  RRF Merger output (15-20 chunks)
    → Re-ranker (scores all chunks against the query)
      → Top-K chunks passed to LLM generator

WHY CROSS-ENCODER:
  The RRF merger already handles scale differences between retrievers
  (vector cosine similarity vs BM25 scores vs PPR scores) by using
  rank positions. But RRF cannot evaluate actual semantic relevance —
  it only knows which retriever preferred which chunk.

  A cross-encoder reads the query and each chunk TOGETHER as one input,
  allowing the model to capture:
    - Exact semantic match between query intent and chunk content
    - Subtle paraphrase relationships that bi-encoders miss
    - Cross-retriever comparison (graph chunk vs vector chunk vs BM25 chunk)

  The output is a single relevance score per (query, chunk) pair,
  directly comparable across all chunks regardless of which retriever
  produced them.

MODEL CHOICE — BAAI/bge-reranker-base:
  - Runs fully on CPU — no GPU required
  - ~280MB download — manageable for local development
  - Latency: 50–200ms for 15 chunks on MacBook Air M-series
  - Outperforms cross-encoder/ms-marco-MiniLM-L-6-v2 on multilingual
    and domain-specific text (important for UPSC which mixes English
    with Indian political terminology)
  - Produces logit scores (can be negative) — higher = more relevant

FALLBACK:
  If the re-ranker model fails to load or throws during scoring,
  the input chunks are returned in their original RRF order.
  This ensures the pipeline never breaks due to re-ranker failure.

INTEGRATION:
  The re-ranker is called from the Router after RRF merging:
    merged   = merger.merge(strategy_results, router_decision)
    reranked = reranker.rerank(query_text, merged.chunks)
    # pass reranked to LLM generator
"""

from __future__ import annotations

import logging
import time
import math
from dataclasses import dataclass
from typing import Optional

import config.reranker_config as C
from shared.models import RetrievedChunk

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RerankResult:
    """
    Output of the re-ranker.
    Contains the re-ordered chunks and metadata about the re-ranking.
    """
    chunks:            list[RetrievedChunk]   # re-ordered, trimmed to top_k
    query_text:        str
    input_count:       int                    # chunks received
    output_count:      int                    # chunks returned (≤ top_k)
    rerank_time_ms:    float
    model_used:        str
    fallback_used:     bool                   # True if model failed → original order kept
    fallback_reason:   str


# ─────────────────────────────────────────────────────────────────────
# Re-ranker class
# ─────────────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder re-ranker using BAAI/bge-reranker-base.

    Instantiate once at application startup. The model is loaded
    lazily on first use and then reused for every query.

    Example
    -------
    >>> reranker = Reranker()
    >>> result = reranker.rerank(
    ...     query_text="How did the Rowlatt Act lead to the Non-Cooperation Movement?",
    ...     chunks=merged_chunks,    # list[RetrievedChunk] from RRF merger
    ...     top_k=5,
    ... )
    >>> for chunk in result.chunks:
    ...     print(chunk.rank, chunk.metadata["rerank_score"], chunk.content[:60])
    """

    def __init__(
        self,
        model_name: str = None,
    ) -> None:
        self._model_name   = model_name or getattr(
            C, "RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"
        )
        self._model        = None    # lazy loaded on first rerank() call
        self._tokenizer    = None
        self._load_failed  = False   # if True, always use fallback

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query_text: str,
        chunks:     list[RetrievedChunk],
        top_k:      Optional[int] = None,
    ) -> RerankResult:
        """
        Re-rank a list of retrieved chunks against a query.

        Parameters
        ----------
        query_text : The original user query string.
        chunks     : Merged chunks from the RRF merger.
                     Typically 10–20 chunks from multiple retrievers.
        top_k      : How many chunks to return after re-ranking.
                     Defaults to config.RERANKER_TOP_K (5).

        Returns
        -------
        RerankResult containing re-ordered chunks trimmed to top_k.
        """
        t_start = time.perf_counter()
        k       = top_k or getattr(C, "RERANKER_TOP_K", 5)

        # Guard: nothing to rerank
        if not chunks:
            return RerankResult(
                chunks=[], query_text=query_text,
                input_count=0, output_count=0,
                rerank_time_ms=0.0, model_used=self._model_name,
                fallback_used=False, fallback_reason="",
            )

        # Guard: fewer chunks than top_k — no reranking needed
        if len(chunks) <= k and len(chunks) <= 3:
            log.debug(
                "Only %d chunks — skipping re-rank, returning as-is.", len(chunks)
            )
            return RerankResult(
                chunks=chunks[:k], query_text=query_text,
                input_count=len(chunks), output_count=len(chunks),
                rerank_time_ms=0.0, model_used="passthrough",
                fallback_used=False, fallback_reason="too_few_chunks",
            )

        # Load model if not already loaded
        if self._model is None and not self._load_failed:
            self._load_model()

        # Use fallback if model loading failed
        if self._load_failed:
            trimmed = chunks[:k]
            _attach_fallback_metadata(trimmed)
            t_end = time.perf_counter()
            return RerankResult(
                chunks=trimmed, query_text=query_text,
                input_count=len(chunks), output_count=len(trimmed),
                rerank_time_ms=round((t_end - t_start) * 1000, 2),
                model_used="fallback",
                fallback_used=True,
                fallback_reason="model_load_failed",
            )

        # Score all chunks
        try:
            scores = self._score_chunks(query_text, chunks)
        except Exception as exc:
            log.error("Re-ranker scoring failed: %s. Using original order.", exc)
            trimmed = chunks[:k]
            _attach_fallback_metadata(trimmed)
            t_end = time.perf_counter()
            return RerankResult(
                chunks=trimmed, query_text=query_text,
                input_count=len(chunks), output_count=len(trimmed),
                rerank_time_ms=round((t_end - t_start) * 1000, 2),
                model_used="fallback",
                fallback_used=True,
                fallback_reason=f"scoring_error:{exc}",
            )

        # Attach scores and re-rank
        min_score = getattr(C, "RERANKER_MIN_SCORE", -10.0)
        reranked  = self._apply_scores(chunks, scores, min_score)

        # Trim to top_k
        final = reranked[:k]

        t_end = time.perf_counter()

        log.info(
            "Re-ranked %d → %d chunks in %.0fms | "
            "top score=%.3f | bottom score=%.3f",
            len(chunks), len(final),
            (t_end - t_start) * 1000,
            final[0].metadata.get("rerank_score", 0) if final else 0,
            final[-1].metadata.get("rerank_score", 0) if final else 0,
        )

        return RerankResult(
            chunks=final, query_text=query_text,
            input_count=len(chunks), output_count=len(final),
            rerank_time_ms=round((t_end - t_start) * 1000, 2),
            model_used=self._model_name,
            fallback_used=False, fallback_reason="",
        )

    # ──────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the cross-encoder model and tokenizer.
        Uses the HuggingFace transformers library directly
        (not sentence-transformers CrossEncoder) for more control.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            log.info("Loading re-ranker model: %s", self._model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model     = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            self._model.eval()   # inference mode — disables dropout etc.

            # Use MPS (Apple Silicon) if available, else CPU
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
                log.info("Re-ranker using MPS (Apple Silicon GPU)")
            else:
                self._device = torch.device("cpu")
                log.info("Re-ranker using CPU")

            self._model = self._model.to(self._device)
            log.info("Re-ranker model loaded successfully.")

        except ImportError:
            log.error(
                "transformers or torch not installed. "
                "Run: pip install transformers torch"
            )
            self._load_failed = True

        except Exception as exc:
            log.error("Failed to load re-ranker model '%s': %s", self._model_name, exc)
            self._load_failed = True

    # ──────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────

    def _score_chunks(
        self,
        query_text: str,
        chunks:     list[RetrievedChunk],
    ) -> list[float]:
        """
        Score all chunks against the query using the cross-encoder.

        The cross-encoder processes (query, chunk_text) pairs and
        returns a logit score for each. Higher score = more relevant.

        Processes chunks in batches to avoid OOM on long chunk lists.

        Returns
        -------
        list[float] — one score per chunk, same order as input.
        """
        import torch

        batch_size = getattr(C, "RERANKER_BATCH_SIZE", 16)
        all_scores: list[float] = []

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]

            # Build (query, passage) pairs for the cross-encoder
            # The cross-encoder expects query and passage concatenated
            # with a separator token — the tokenizer handles this.
            pairs = [
                [query_text, chunk.content]
                for chunk in batch_chunks
            ]

            # Tokenize
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,       # BAAI/bge-reranker-base max is 512
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Score — no gradient computation needed for inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits  = outputs.logits

            # BAAI/bge-reranker-base outputs a single logit per pair
            # Shape: (batch_size, 1) or (batch_size,)
            if logits.dim() == 2:
                batch_scores = logits.squeeze(-1).tolist()
            else:
                batch_scores = logits.tolist()

            # Ensure it's a list of floats even for batch_size=1
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            all_scores.extend(batch_scores)

        return all_scores

    # ──────────────────────────────────────────────────────────────────
    # Score application
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_scores(
        chunks: list[RetrievedChunk],
        scores: list[float],
        min_score: float,
    ) -> list[RetrievedChunk]:
        """
        Attach normalized rerank scores and sort chunks.

        Converts raw logits → [0, 1] using sigmoid so scores are comparable.
        """

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        # ── Attach normalized scores ─────────────────────────────
        for chunk, score in zip(chunks, scores):
            raw = float(score)
            normalized = round(sigmoid(raw), 6)

            chunk.metadata["rerank_score"]   = normalized
            chunk.metadata["rerank_logit"]   = round(raw, 6)   # keep raw for debug
            chunk.metadata["original_rank"]  = chunk.rank
            chunk.metadata["original_score"] = chunk.score

            chunk.score = normalized   # ✅ unified scoring scale

        # ── Sort by normalized score ─────────────────────────────
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.metadata["rerank_score"],
            reverse=True
        )

        # ✅ FIXED: Removed the aggressive [:2] truncation!
        # Just return the sorted chunks. The parent rerank() method 
        # will safely trim the list down to ROUTER_FINAL_TOP_K (15 chunks).

        # ── Update ranks ─────────────────────────────────────────
        for new_rank, chunk in enumerate(sorted_chunks, start=1):
            chunk.metadata["rerank_rank"] = new_rank
            chunk.rank = new_rank

        return sorted_chunks

# ─────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────

def _attach_fallback_metadata(chunks: list[RetrievedChunk]) -> None:
    """
    When the re-ranker falls back to original order, still attach
    metadata fields so downstream code has consistent chunk structure.
    """
    for chunk in chunks:
        chunk.metadata.setdefault("rerank_score",   None)
        chunk.metadata.setdefault("rerank_rank",    chunk.rank)
        chunk.metadata.setdefault("original_rank",  chunk.rank)
        chunk.metadata.setdefault("original_score", chunk.score)
