"""
hierarchical_retriever.py
─────────────────────────
Retrieves chunks using the multi-level hierarchy built by hierarchical_indexer.py

How it works:
    Uses LlamaIndex index.as_retriever() with metadata level filters.
    This guarantees the SAME embedding pipeline as ingestion —
    no vector space mismatch, no distance scaling issues.

    Level 0 → raw chunks (precise, specific)
    Level 1+ → cluster/domain summaries (broad context)

    Both levels searched separately, merged by score, deduplicated.
    Scores are already 0-1 from LlamaIndex — no conversion needed.

Storage it reads from:
    ChromaDB rag_collection — same collection as vector retriever
    Metadata field "level" differentiates chunk types
"""

import logging
import os

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_config import RAGConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# LOAD INDEX
# ══════════════════════════════════════════════════════

def load_hierarchical_index(cfg: RAGConfig = None) -> VectorStoreIndex:
    """
    Connect to ChromaDB and load VectorStoreIndex.

    Uses the SAME embed_model as hierarchical_indexer.py so vectors
    are in the same space during both storage and retrieval.

    Returns:
        VectorStoreIndex connected to the same ChromaDB collection
        used during ingestion.

    Raises:
        FileNotFoundError: if database folder does not exist.
        ValueError: if collection not found or is empty.
    """

    if cfg is None:
        cfg = RAGConfig()

    if not os.path.exists(cfg.db_path):
        raise FileNotFoundError(
            f"Database not found at '{cfg.db_path}'. "
            f"Run ingest.py first."
        )

    # Same model name as used in hierarchical_indexer.py
    embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model_name)

    chroma_client = chromadb.PersistentClient(path=cfg.db_path)

    try:
        collection = chroma_client.get_collection(
            name=cfg.collection_name
        )
    except Exception:
        raise ValueError(
            f"Collection '{cfg.collection_name}' not found at '{cfg.db_path}'. "
            f"Run ingest.py first."
        )

    if collection.count() == 0:
        raise ValueError(
            f"Collection '{cfg.collection_name}' is empty. "
            f"Run ingest.py first."
        )

    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context = storage_context,
        embed_model     = embed_model,
    )

    total = collection.count()
    logger.info(f"Hierarchical index loaded — {total} total nodes in collection.")
    print(f"  ✅ Hierarchical index loaded — {total} total nodes.")

    return index


# ══════════════════════════════════════════════════════
# CORE RETRIEVAL
# ══════════════════════════════════════════════════════

def hierarchical_retrieve(
    query: str,
    index: VectorStoreIndex,
    cfg: RAGConfig = None,
) -> list[dict]:
    """
    Retrieve chunks and summaries using level metadata filters.

    Two separate searches:
        1. Summary nodes (level >= 1) — broad context
        2. Chunk nodes  (level == 0) — precise details

    Results merged and sorted by score descending.
    Scores come directly from LlamaIndex (0 to 1 range).
    No conversion or rescaling needed.

    Args:
        query : user question
        index : VectorStoreIndex from load_hierarchical_index()
        cfg   : RAGConfig

    Returns:
        List of dicts ordered by score descending:
        {
            text         : str   — node text content
            score        : float — similarity score 0 to 1 (higher = better)
            source       : str   — source filename
            level        : int   — 0=chunk, 1=cluster, 2=super, 3+=domain
            chunk_index  : int   — position in document (-1 for summaries)
            total_chunks : int   — total chunks in document (-1 for summaries)
            retriever    : str   — always "hierarchical"
        }
    """

    if cfg is None:
        cfg = RAGConfig()

    if not query or not query.strip():
        print("  ⚠️  Empty query. Skipping.")
        return []

    summary_count = min(cfg.raptor_summary_count, cfg.similarity_top_k)
    chunk_count   = max(1, cfg.similarity_top_k - summary_count)

    results = []

    # ── Search summary nodes (level >= 1) ─────────────────────────────
    if summary_count > 0:
        try:
            summary_filter = MetadataFilters(filters=[
                MetadataFilter(key="level", value=1, operator=">=")
            ])
            summary_retriever = index.as_retriever(
                similarity_top_k = summary_count,
                filters          = summary_filter,
            )
            summary_nodes = summary_retriever.retrieve(query)

            for node in summary_nodes:
                score = float(node.score) if node.score is not None else 0.0
                results.append({
                    "text":         node.node.text,
                    "score":        round(score, 4),
                    "source":       node.node.metadata.get("source", "unknown"),
                    "level":        node.node.metadata.get("level", 1),
                    "chunk_index":  -1,
                    "total_chunks": -1,
                    "retriever":    "hierarchical",
                })

            print(f"  🔵 Summary search: {len(summary_nodes)} result(s).")

        except Exception as e:
            logger.warning(f"Summary search failed: {e}")
            print(f"  ⚠️  Summary search failed: {e}")

    # ── Search chunk nodes (level == 0) ───────────────────────────────
    if chunk_count > 0:
        try:
            chunk_filter = MetadataFilters(filters=[
                MetadataFilter(key="level", value=0)
            ])
            chunk_retriever = index.as_retriever(
                similarity_top_k = chunk_count,
                filters          = chunk_filter,
            )
            chunk_nodes = chunk_retriever.retrieve(query)

            for node in chunk_nodes:
                score = float(node.score) if node.score is not None else 0.0
                results.append({
                    "text":         node.node.text,
                    "score":        round(score, 4),
                    "source":       node.node.metadata.get("source", "unknown"),
                    "level":        node.node.metadata.get("level", 0),
                    "chunk_index":  node.node.metadata.get("chunk_index", -1),
                    "total_chunks": node.node.metadata.get("total_chunks", -1),
                    "retriever":    "hierarchical",
                })

            print(f"  🟢 Chunk search: {len(chunk_nodes)} result(s).")

        except Exception as e:
            logger.warning(f"Chunk search failed: {e}")
            print(f"  ⚠️  Chunk search failed: {e}")

    # ── Sort by score descending (best first) ─────────────────────────
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Deduplicate by text fingerprint ───────────────────────────────
    seen   = set()
    unique = []
    for r in results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    actual_summaries = sum(1 for r in unique if r["level"] >= 1)
    actual_chunks    = sum(1 for r in unique if r["level"] == 0)

    if unique:
        logger.info(
            f"Query: '{query}' — "
            f"{actual_summaries} summary, {actual_chunks} chunk. "
            f"Top score: {unique[0]['score']}, "
            f"Bottom score: {unique[-1]['score']}"
        )
        print(f"  ✅ {len(unique)} result(s) — "
              f"{actual_summaries} summary, {actual_chunks} chunk. "
              f"Score range: {unique[-1]['score']} – {unique[0]['score']}")
    else:
        logger.warning(f"Query: '{query}' — no results.")
        print(f"  ⚠️  No results returned.")

    return unique


# ══════════════════════════════════════════════════════
# TEST DIRECTLY
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "When were the Digital Personal Data Protection (DPDP) Rules, 2025 notified, and how many inputs were received during the consultation process to shape them?"

    cfg   = RAGConfig()
    index = load_hierarchical_index(cfg)

    print(f"\nQuery: {query}\n")
    print("-" * 60)

    results = hierarchical_retrieve(query, index, cfg)

    for i, r in enumerate(results, 1):
        label   = "summary" if r["level"] >= 1 else "chunk"
        preview = r["text"][:200].replace("\n", " ").strip()
        print(f"\nResult {i} — Level {r['level']} ({label})")
        print(f"Source : {r['source']}")
        print(f"Score  : {r['score']}")
        print(f"Text   : {preview}...")
        print("-" * 60)
