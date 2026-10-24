"""
keyword_indexer.py
──────────────────
Builds and saves a BM25 keyword index from chunks already stored in ChromaDB.

Why BM25:
    BM25 (Best Match 25) is the industry standard keyword search algorithm.
    It is what powers Elasticsearch and Solr under the hood.
    Unlike vector search which finds semantically similar chunks,
    BM25 finds chunks that contain the exact words from the query.
    Both approaches complement each other which is why your RAG has both.

How it works:
    1. Reads all chunks already stored in ChromaDB (no duplicate storage)
    2. Tokenizes each chunk into individual words
    3. Builds a BM25 index over all chunks
    4. Saves the index to database/keyword/ as a .pkl file

Storage:
    database/
    ├── chromadb/       ← vector embeddings (chunking.py)
    └── keyword/        ← BM25 index (this file)
"""

import os
import re
import pickle
import chromadb

from rank_bm25 import BM25Okapi
from ingestion.chunking import RAGConfig


# ---------------------------
# Tokenizer
# ---------------------------

def tokenize(text: str) -> list[str]:
    """
    Convert a chunk of text into a list of clean lowercase words.

    What this does step by step:
        1. Lowercase everything so 'RAG' and 'rag' are treated the same
        2. Remove punctuation and special characters
        3. Split into individual words
        4. Remove very short words under 2 characters (noise like 'a', 'i')

    Example:
        Input : "ReTreever organizes documents into a tree."
        Output: ["retreeever", "organizes", "documents", "into", "tree"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


# ---------------------------
# Read chunks from ChromaDB
# ---------------------------

def fetch_chunks_from_chromadb(cfg: RAGConfig) -> list[dict]:
    """
    Reads all chunks already stored in ChromaDB.
    Returns them as a list of dicts with text and metadata.

    Why we read from ChromaDB instead of re-chunking:
        chunking.py already split and stored everything cleanly.
        Reading from ChromaDB ensures keyword index uses
        exactly the same chunks as vector retriever —
        no inconsistency, no duplicate storage.

    Returns:
        List of dicts, each with:
            text         : chunk text content
            source       : source filename
            chunk_index  : position of chunk in document
            total_chunks : total chunks from that document
            node_id      : ChromaDB ID for this chunk
            act          : Parent act metadata
            chapter      : Parent chapter metadata
            section      : Parent section metadata
    """
    chroma_client = chromadb.PersistentClient(path=cfg.db_path)
    collection = chroma_client.get_or_create_collection(name=cfg.collection_name)

    total = collection.count()
    if total == 0:
        raise RuntimeError(
            "ChromaDB is empty. Run ingest.py (vector step) before building keyword index."
        )

    # Fetch everything — text and metadata
    data = collection.get(include=["documents", "metadatas"])

    chunks = []
    for node_id, text, meta in zip(
        data["ids"],
        data["documents"],
        data["metadatas"]
    ):
        chunks.append({
            "node_id":     node_id,
            "text":        text,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "total_chunks": meta.get("total_chunks", -1),
            # NEW: Extract the legal structure metadata that chunking.py created
            "act":         meta.get("act", ""),
            "chapter":     meta.get("chapter", ""),
            "section":     meta.get("section", ""),
        })

    print(f"  📦 Fetched {len(chunks)} chunk(s) from ChromaDB.")
    return chunks


# ---------------------------
# Build BM25 index
# ---------------------------

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk dicts.

    BM25Okapi is the standard variant of BM25.
    It takes a list of tokenized documents and builds
    the internal frequency tables needed for scoring.

    Args:
        chunks: list of chunk dicts from fetch_chunks_from_chromadb()

    Returns:
        BM25Okapi index object ready for searching
    """
    tokenized_chunks = []
    
    for chunk in chunks:
        # NEW: Inject legal metadata into the text before tokenizing
        meta_prefix_parts = []
        
        if chunk.get("act"):
            meta_prefix_parts.append(f"Act: {chunk['act']}")
        if chunk.get("chapter"):
            meta_prefix_parts.append(f"Chapter {chunk['chapter']}")
        if chunk.get("section"):
            meta_prefix_parts.append(f"Section {chunk['section']}")
            
        meta_prefix = " | ".join(meta_prefix_parts)
        
        if meta_prefix:
            # Inject tags at the top for BM25 to tokenize
            indexed_text = f"[{meta_prefix}]\n{chunk['text']}"
        else:
            indexed_text = chunk["text"]
            
        tokenized_chunks.append(tokenize(indexed_text))

    # Guard — BM25 needs at least one non-empty tokenized document
    if not any(tokenized_chunks):
        raise ValueError("All chunks produced empty token lists. Check your text quality.")

    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


# ---------------------------
# Save index to disk
# ---------------------------

def save_keyword_index(
    bm25: BM25Okapi,
    chunks: list[dict],
    keyword_dir: str,
) -> None:
    """
    Save the BM25 index and chunk metadata to database/keyword/.

    We save two things:
        bm25_index.pkl   — the BM25 index object (for scoring)
        chunks.pkl       — the chunk list (for retrieving text after scoring)

    Why we need both:
        BM25 only stores token frequencies for scoring.
        It does not store the original text.
        So we save chunks separately to retrieve text after scoring.

    Args:
        bm25        : built BM25Okapi index
        chunks      : list of chunk dicts (same order as BM25 was built on)
        keyword_dir : folder path to save into
    """
    os.makedirs(keyword_dir, exist_ok=True)

    bm25_path   = os.path.join(keyword_dir, "bm25_index.pkl")
    chunks_path = os.path.join(keyword_dir, "chunks.pkl")

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  ✅ BM25 index saved   → {bm25_path}")
    print(f"  ✅ Chunks saved       → {chunks_path}")


# ---------------------------
# Load index from disk
# ---------------------------

def load_keyword_index(keyword_dir: str) -> tuple[BM25Okapi, list[dict]]:
    """
    Load a previously saved BM25 index and chunk list from disk.

    This is what keyword_retriever.py calls at startup —
    same pattern as vector_retriever.py calling load_index().

    Args:
        keyword_dir: folder path where index was saved

    Returns:
        bm25   : BM25Okapi index ready for searching
        chunks : list of chunk dicts in same order as BM25
    """
    bm25_path   = os.path.join(keyword_dir, "bm25_index.pkl")
    chunks_path = os.path.join(keyword_dir, "chunks.pkl")

    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"BM25 index not found at '{bm25_path}'.\n"
            f"Run ingest.py first to build the keyword index."
        )

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_path}'.\n"
            f"Run ingest.py first to build the keyword index."
        )

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"  ✅ Keyword index loaded — {len(chunks)} chunk(s) available.")
    return bm25, chunks


# ---------------------------
# Main build function
# called from ingest.py
# ---------------------------

def build_keyword_index(cfg: RAGConfig, keyword_dir: str) -> None:
    """
    Full pipeline — fetch chunks from ChromaDB, build BM25, save to disk.

    This is the only function ingest.py needs to call.

    Args:
        cfg         : RAGConfig (for ChromaDB connection details)
        keyword_dir : where to save the index e.g. "./database/keyword"

    Flow:
        ChromaDB → fetch chunks → tokenize → build BM25 → save to disk
    """

    print("\n🔑 Building keyword index...")

    # Step 1 — check if index already exists (deduplication equivalent)
    bm25_path = os.path.join(keyword_dir, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        print(f"  ℹ️  Keyword index already exists at '{keyword_dir}'.")
        print(f"  ℹ️  Delete '{keyword_dir}' folder to rebuild from scratch.")
        return

    # Step 2 — fetch chunks from ChromaDB
    chunks = fetch_chunks_from_chromadb(cfg)

    # Step 3 — build BM25 index
    print(f"  ⚙️  Building BM25 index over {len(chunks)} chunk(s)...")
    bm25 = build_bm25_index(chunks)

    # Step 4 — save to disk
    save_keyword_index(bm25, chunks, keyword_dir)

    print(f"  ✅ Keyword index built successfully — {len(chunks)} chunk(s) indexed.")
