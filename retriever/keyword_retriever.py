"""
keyword_retriever.py
────────────────────
BM25 based keyword retriever.

How it differs from vector_retriever.py:
    vector_retriever  → finds chunks by SEMANTIC similarity
                        works well for conceptual questions
                        example: "How does the tree organize documents?"

    keyword_retriever → finds chunks by EXACT WORD MATCH
                        works well for specific term queries
                        example: "BM25 algorithm" or "contrastive learning"

When to use keyword retrieval:
    - Query contains specific technical terms, model names, acronyms
    - Query contains exact phrases you expect to appear in the document
    - Vector retriever returns low scores or irrelevant chunks

How BM25 scoring works:
    BM25 scores each chunk based on:
        1. Term frequency  — how many times query words appear in the chunk
        2. Inverse document frequency — rare words score higher than common words
        3. Document length normalization — longer chunks are not unfairly favored
    Higher score = better keyword match.
    Unlike vector scores (0 to 1), BM25 scores have no fixed range.
    A score of 0 means no query words found in that chunk.
"""

import os
import sys

# Make sure root folder is in path when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.keyword_indexer import load_keyword_index, tokenize
from rag_config import RAGConfig

from shared.models import RetrievedChunk


# ---------------------------
# Config
# ---------------------------
# KEYWORD_DIR = "./database/keyword"


# ---------------------------
# Core retrieval function
# ---------------------------

def keyword_retrieve(
    query: str,
    bm25,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Search chunks using BM25 keyword matching.

    Args:
        query     : user's natural language question
        bm25      : loaded BM25Okapi index
        chunks    : loaded chunk list (same order as BM25 was built on)
        top_k     : how many chunks to return
        min_score : minimum BM25 score to include a chunk
                    default 0.0 means return top_k regardless of score
                    set higher (e.g. 1.0) to filter weak matches

    Returns:
        List of result dicts each containing:
            text         : chunk text
            score        : BM25 score (higher = better keyword match)
            source       : source filename
            chunk_index  : position of chunk in document
            total_chunks : total chunks from that document
            node_id      : ChromaDB ID for traceability
    """

    # Guard — empty query
    if not query or not query.strip():
        print("  ⚠️  Empty query received. Skipping.")
        return []

    # Step 1 — tokenize query same way chunks were tokenized
    # Critical — must use same tokenizer as keyword_indexer.py
    # If chunks were tokenized with lowercase + punctuation removal,
    # query must be tokenized the same way or scores will be wrong
    query_tokens = tokenize(query)

    if not query_tokens:
        print("  ⚠️  Query produced no tokens after cleaning. Skipping.")
        return []

    # Step 2 — get BM25 scores for all chunks
    scores = bm25.get_scores(query_tokens)

    # Step 3 — pair each chunk with its score
    scored_chunks = [
        (score, chunk)
        for score, chunk in zip(scores, chunks)
    ]

    # Step 4 — sort by score descending (highest first)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Step 5 — take top_k and apply min_score filter
    results = []

    for score, chunk in scored_chunks:

        if score < min_score:
            break

        results.append(
            RetrievedChunk(
                chunk_id=f"bm25_{chunk.get('node_id', hash(chunk['text']))}",
                content=chunk["text"],
                source=chunk.get("source", "keyword_db"),
                page_number=0,

                score=float(score),
                final_score=float(score),
                rank=0,

                strategy="bm25",
                metadata={
                    "chunk_index": chunk.get("chunk_index", -1),
                    "total_chunks": chunk.get("total_chunks", -1),
                },

                relation_paths=[],
                legal_citation=[],
                graph_node=None
            )
        )

        if len(results) >= top_k:
            break

    return results


# ---------------------------
# Display results
# ---------------------------

def display_results(results: list[dict], query: str) -> None:
    """Print retrieved chunks in readable format for testing."""

    print(f"\n🔑 Query    : {query}")
    print(f"📚 Retrieved: {len(results)} chunk(s)\n")

    if not results:
        print("  ℹ️  No chunks matched the query keywords.")
        return

    for i, chunk in enumerate(results):
        print(f"--- Chunk {i + 1} ---")
        print(f"Source     : {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']} / {chunk['total_chunks']}")
        print(f"BM25 Score : {chunk['score']}")
        print(f"Retriever  : {chunk['retriever']}")
        clean_preview = chunk['text'][:300].replace("\n", " ").strip()
        print(f"Text       : {clean_preview}")
        print()


# ---------------------------
# Test directly
# ---------------------------

if __name__ == "__main__":

    cfg = RAGConfig()

    # Step 1 — load keyword index from disk
    print("📂 Loading keyword index...")
    bm25, chunks = load_keyword_index(cfg.keyword_dir)

    # Step 2 — test queries
    # These are keyword-style queries — specific terms expected in document
    test_queries = [
        "BM25 contrastive learning retrieval",
        "ReTreever tree depth binary split function",
        "FAISS index nearest neighbor search",
    ]

    for query in test_queries:
        results = keyword_retrieve(
            query=query,
            bm25=bm25,
            chunks=chunks,
            top_k=5,
            min_score=0.0,
        )
        display_results(results, query)
        print("=" * 60)
