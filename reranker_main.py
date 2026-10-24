"""
reranker_main.py
=================
End-to-end demo of the ASIR-RAG re-ranker.

Demonstrates:
  1. Re-ranker standalone (scores mock chunks)
  2. Re-ranker integrated with the Router
  3. Before/after comparison showing rank changes

RUN:
  python reranker_main.py

PREREQUISITES:
  pip install transformers torch
  (sentence-transformers is NOT needed — we use transformers directly)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from reranker.reranker import Reranker
from shared.models import RetrievedChunk


# ─────────────────────────────────────────────────────────────────────
# Mock chunk builder
# ─────────────────────────────────────────────────────────────────────

def make_chunk(
    content:  str,
    strategy: str,
    rank:     int,
    score:    float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{strategy}_{uuid.uuid4().hex[:8]}",
        content=content,
        source=f"{strategy}_doc.pdf",
        page_number=1,
        score=score,
        rank=rank,
        strategy=strategy,
        metadata={"strategy": strategy},
    )


# ─────────────────────────────────────────────────────────────────────
# Demo 1: Standalone re-ranker
# ─────────────────────────────────────────────────────────────────────

def demo_standalone() -> None:
    print("\n" + "═" * 70)
    print("  DEMO 1 — Standalone Re-ranker")
    print("  Query: How did the Rowlatt Act lead to the Non-Cooperation Movement?")
    print("═" * 70)

    query = "How did the Rowlatt Act lead to the Non-Cooperation Movement?"

    # Simulate chunks from multiple retrievers — deliberately mixed quality
    mock_chunks = [
        # Highly relevant — from graph retriever
        make_chunk(
            "The Rowlatt Act of 1919 gave the British government sweeping powers "
            "to arrest Indians without trial. Gandhi viewed this as a fundamental "
            "violation of civil liberties and called for a nationwide hartal. "
            "This protest eventually evolved into the Non-Cooperation Movement of 1920.",
            strategy="graph", rank=1, score=0.000012,
        ),
        # Moderately relevant — from vector retriever
        make_chunk(
            "The Non-Cooperation Movement was launched by Gandhi in 1920. "
            "It called for Indians to boycott British goods, courts, and institutions. "
            "The movement marked a turning point in India's independence struggle.",
            strategy="vector", rank=1, score=0.82,
        ),
        # Low relevance — from BM25 (keyword match but wrong context)
        make_chunk(
            "The Rowlatt Act was also called the Black Act by Indian nationalists. "
            "It was passed by the Imperial Legislative Council in March 1919. "
            "The Act extended emergency measures from the First World War.",
            strategy="bm25", rank=1, score=14.3,
        ),
        # Irrelevant — keyword match on 'movement' only
        make_chunk(
            "The Swadeshi Movement preceded the Non-Cooperation Movement by two decades. "
            "It arose in response to the partition of Bengal in 1905. "
            "Tilak and Gokhale were prominent leaders of this earlier movement.",
            strategy="vector", rank=2, score=0.71,
        ),
        # Moderately relevant — provides context
        make_chunk(
            "Gandhi's response to the Rowlatt Act was to call for satyagraha — "
            "non-violent resistance. The Jallianwala Bagh massacre of April 1919 "
            "further radicalised public opinion and strengthened support for non-cooperation.",
            strategy="hierarchical", rank=1, score=0.68,
        ),
    ]

    print("\n  BEFORE RE-RANKING (RRF order from merger):")
    print(f"  {'Rank':<6} {'Strategy':<14} {'Score':<12} Content preview")
    print(f"  {'─'*6} {'─'*14} {'─'*12} {'─'*40}")
    for chunk in mock_chunks:
        print(
            f"  [{chunk.rank}]   {chunk.strategy:<14} {chunk.score:<12.6f} "
            f"{chunk.content[:55]}..."
        )

    # Re-rank
    reranker = Reranker()
    result   = reranker.rerank(query, mock_chunks, top_k=4)

    print(f"\n  AFTER RE-RANKING (cross-encoder order):")
    print(f"  Re-rank time: {result.rerank_time_ms:.0f}ms | Model: {result.model_used}")
    print(f"  {'New Rank':<10} {'Old Rank':<10} {'Rerank Score':<14} {'Strategy':<14} Content preview")
    print(f"  {'─'*10} {'─'*10} {'─'*14} {'─'*14} {'─'*40}")
    for chunk in result.chunks:
        print(
            f"  [{chunk.metadata['rerank_rank']}]       "
            f"[{chunk.metadata['original_rank']}]       "
            f"{chunk.metadata['rerank_score']:<14.4f} "
            f"{chunk.metadata.get('strategy', '?'):<14} "
            f"{chunk.content[:55]}..."
        )

    print(
        f"\n  ✅ Result: {result.input_count} chunks → {result.output_count} returned | "
        f"Fallback: {result.fallback_used}"
    )


# ─────────────────────────────────────────────────────────────────────
# Demo 2: Multi-retriever re-ranking
# ─────────────────────────────────────────────────────────────────────

def demo_multi_retriever() -> None:
    print("\n" + "═" * 70)
    print("  DEMO 2 — Multi-retriever Re-ranking")
    print("  Shows cross-retriever comparison (graph vs vector vs BM25)")
    print("═" * 70)

    query = "What is the relationship between Fundamental Rights and Directive Principles?"

    # Mix of chunks from different retrievers with different score scales
    chunks = [
        make_chunk(
            "The Fundamental Rights and Directive Principles of State Policy "
            "represent the twin pillars of the Indian Constitution. "
            "While Fundamental Rights are justiciable, Directive Principles guide state policy.",
            strategy="vector", rank=1, score=0.91,
        ),
        make_chunk(
            "Article 37 of the Constitution states that the Directive Principles "
            "shall not be enforceable by any court, but the principles therein laid down "
            "are nevertheless fundamental in the governance of the country.",
            strategy="graph", rank=1, score=0.0000089,
        ),
        make_chunk(
            "In the Kesavananda Bharati case, the Supreme Court held that both "
            "Fundamental Rights and Directive Principles must be read harmoniously. "
            "Neither can be used to destroy the other.",
            strategy="graph", rank=2, score=0.0000071,
        ),
        make_chunk(
            "Fundamental Rights are enforceable in courts under Article 32. "
            "Citizens can approach the Supreme Court directly for their enforcement. "
            "The right to constitutional remedies is itself a Fundamental Right.",
            strategy="bm25", rank=1, score=18.7,
        ),
        make_chunk(
            "The Directive Principles were inspired by the Irish Constitution. "
            "They include provisions for equal pay for equal work, free legal aid, "
            "and protection of the environment.",
            strategy="hierarchical", rank=1, score=0.74,
        ),
        make_chunk(
            "Part III of the Indian Constitution deals with Fundamental Rights "
            "covering Articles 12 to 35. Part IV deals with Directive Principles "
            "covering Articles 36 to 51.",
            strategy="vector", rank=2, score=0.79,
        ),
    ]

    reranker = Reranker()
    result   = reranker.rerank(query, chunks, top_k=4)

    print(f"\n  Query: {query}")
    print(f"\n  Input: {result.input_count} chunks from 4 retrievers")
    print(f"  Output: {result.output_count} chunks after re-ranking")
    print(f"  Time: {result.rerank_time_ms:.0f}ms\n")

    print(f"  {'#':<4} {'Rerank Score':<14} {'Strategy':<14} {'Orig Rank':<10} Preview")
    print(f"  {'─'*4} {'─'*14} {'─'*14} {'─'*10} {'─'*50}")

    for chunk in result.chunks:
        strat = chunk.metadata.get("strategy", "?")
        print(
            f"  {chunk.rank:<4} "
            f"{chunk.metadata['rerank_score']:<14.4f} "
            f"{strat:<14} "
            f"[{chunk.metadata['original_rank']}]       "
            f"{chunk.content[:55]}..."
        )


# ─────────────────────────────────────────────────────────────────────
# Demo 3: Router with re-ranker integrated
# ─────────────────────────────────────────────────────────────────────

def demo_router_integration() -> None:
    print("\n" + "═" * 70)
    print("  DEMO 3 — Router with Re-ranker (stub retrievers)")
    print("═" * 70)

    from router.router import Router
    from shared.models import ClassifierOutput

    # Stub retrievers
    def stub_vector(sc):
        return [
            make_chunk(
                f"[VECTOR] Semantic context for: {sc.query_text[:40]}",
                "vector", i + 1, 0.9 - i * 0.08
            )
            for i in range(sc.top_k)
        ]

    def stub_graph(sc):
        return [
            make_chunk(
                f"[GRAPH] Relational context for: {sc.query_text[:40]}",
                "graph", i + 1, 0.001 / (i + 1)
            )
            for i in range(sc.top_k)
        ]

    def stub_bm25(sc):
        return [
            make_chunk(
                f"[BM25] Keyword match for: {sc.query_text[:40]}",
                "bm25", i + 1, 12.5 - i * 1.5
            )
            for i in range(sc.top_k)
        ]

    def stub_hierarchical(sc):
        return [
            make_chunk(
                f"[HIER] Hierarchical context for: {sc.query_text[:40]}",
                "hierarchical", i + 1, 0.7 - i * 0.1
            )
            for i in range(sc.top_k)
        ]

    reranker = Reranker()

    router = Router(
        retrieval_registry={
            "vector":       stub_vector,
            "bm25":         stub_bm25,
            "hierarchical": stub_hierarchical,
            "graph":        stub_graph,
        },
        reranker=reranker,
    )

    # Simulate a ClassifierOutput for a relational query
    now = datetime.now(timezone.utc).isoformat()
    co  = ClassifierOutput(
        query_id=str(uuid.uuid4()),
        session_id="demo",
        raw_text="How does colonialism influence India's economy?",
        cleaned_text="how does colonialism influence india's economy?",
        timestamp=now,
        query_type="relational",
        confidence=0.74,
        classified_by="demo",
        all_scores={
            "factual": 0.10, "relational": 0.74,
            "comparative": 0.10, "exploratory": 0.06,
        },
        llm_reasoning="Demo",
        phase="phase1_llm",
    )

    print(f"\n  Query: {co.raw_text}")
    print(f"  Type: {co.query_type} | Confidence: {co.confidence}")

    decision, merged = router.route(co)

    print(f"\n  Band: {decision.confidence_band} | Mode: {decision.routing_mode}")
    print(f"  Strategies: {[sc.strategy for sc in decision.primary_strategies]}")
    print(f"  Chunks after re-ranking: {len(merged.chunks)}")
    print()

    for chunk in merged.chunks:
        rerank_score = chunk.metadata.get("rerank_score", "N/A")
        orig_rank    = chunk.metadata.get("original_rank", "?")
        print(
            f"  [{chunk.rank}] {chunk.strategy:<14} "
            f"rerank_score={rerank_score:<10} "
            f"(was rank {orig_rank}) | "
            f"{chunk.content[:50]}..."
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ASIR-RAG Re-ranker Demo")
    print("=" * 70)

    demo_standalone()
    demo_multi_retriever()
    demo_router_integration()
