"""
ingest.py
─────────
Run this once to load all documents into all indexes.

Order of operations:
    1. Load all PDFs from data_file/
    2. Clean all documents
    3. Vector ingestion — one doc at a time into ChromaDB
    4. Keyword ingestion — reads from ChromaDB, builds BM25
    5. Hierarchical ingestion — ALL docs together, dynamic levels

Usage:
    python ingest.py

Make sure Ollama is running with Mistral pulled before running this.
"""

import os
from file_loader.file_loader import load_documents_from_folder
from file_loader.text_cleaner import clean_text
from ingestion.chunking import (
    get_chroma_store,
    get_embed_model,
    build_index,
    inspect_database,
)
from ingestion.keyword_indexer import build_keyword_index
from ingestion.hierarchical_indexer import build_hierarchical_index
from rag_config import RAGConfig

def main():

    cfg = RAGConfig()

    # ── Step 1: Load all PDFs from data_file/ ────────────────────────
    print("📂 Loading all documents from folder...")
    print(f"   Folder: {cfg.data_dir}")

    raw_docs = load_documents_from_folder(cfg.data_dir)

    # Filter PDF only — skip anything else accidentally in the folder
    raw_docs = [
        doc for doc in raw_docs
        if doc["file_name"].lower().endswith(".pdf")
    ]

    if not raw_docs:
        print(f"❌ No PDF documents found in '{cfg.data_dir}'.")
        print(f"   Place your PDF files there and try again.")
        return

    print(f"✅ Found {len(raw_docs)} PDF document(s):")
    for doc in raw_docs:
        pages = doc["metadata"].get("total_pages", "N/A")
        print(f"   • {doc['file_name']} ({pages} pages)")

    # ── Step 2: Clean all documents ───────────────────────────────────
    print("\n🧹 Cleaning all documents...")

    cleaned_docs = {}   # {source_name: cleaned_text}

    for doc in raw_docs:
        cleaned = clean_text(doc["content"])

        if not cleaned.strip():
            print(f"  ⚠️  '{doc['file_name']}': Empty after cleaning. Skipping.")
            continue

        cleaned_docs[doc["file_name"]] = cleaned
        print(f"  ✅ '{doc['file_name']}': {len(cleaned):,} characters.")

    if not cleaned_docs:
        print("❌ No valid documents after cleaning. Aborting.")
        return

    print(f"\n✅ {len(cleaned_docs)} document(s) ready for ingestion.")

    # ── Step 3: Setup shared resources ───────────────────────────────
    print("\n⚙️  Setting up ChromaDB and embedding model...")
    vector_store, collection = get_chroma_store(cfg)
    embed_model              = get_embed_model(cfg)
    print("✅ Setup complete.")

    # ── Step 4: Vector ingestion — one document at a time ─────────────
    print("\n" + "═" * 55)
    print("  VECTOR INGESTION")
    print("═" * 55)

    for source_name, cleaned_text in cleaned_docs.items():
        print(f"\n  🔄 Processing: {source_name}")
        build_index(
            text         = cleaned_text,
            source_name  = source_name,
            cfg          = cfg,
            embed_model  = embed_model,
            vector_store = vector_store,
            collection   = collection,
        )

    print("\n✅ Vector ingestion complete.")

    # ── Step 5: Keyword ingestion ─────────────────────────────────────
    # Must run AFTER vector ingestion
    # Reads chunks directly from ChromaDB — no duplicate storage
    print("\n" + "═" * 55)
    print("  KEYWORD INGESTION")
    print("═" * 55)

    build_keyword_index(cfg, cfg.keyword_dir)
    print("\n✅ Keyword ingestion complete.")

    # ── Step 6: Hierarchical ingestion — ALL docs together ────────────
    # Pass ALL cleaned documents at once
    # Clustering happens across all documents together
    # Dynamic levels decided based on total chunk count
    print("\n" + "═" * 55)
    print("  HIERARCHICAL INGESTION")
    print("═" * 55)
    print("  ℹ️  All documents are processed together for cross-document clustering.")
    print("  ℹ️  Mistral summarization will take several minutes.")
    print("  ℹ️  Make sure Ollama is running: ollama run mistral")

    build_hierarchical_index(
        documents   = cleaned_docs,
        cfg         = cfg,
        embed_model = embed_model,
    )

    print("\n✅ Hierarchical ingestion complete.")

    # ── Step 7: Final verification ────────────────────────────────────
    print("\n" + "═" * 55)
    print("  DATABASE INSPECTION")
    print("═" * 55)

    inspect_database(cfg)

    # Print hierarchical build stats if available
    marker_path = os.path.join(cfg.hierarchical_dir, "build_complete.txt")
    if os.path.exists(marker_path):
        print("\n📊 Hierarchical index stats:")
        with open(marker_path, "r") as f:
            print(f.read())

    print("🎉 All ingestion complete. Run main.py to start asking questions.")


if __name__ == "__main__":
    main()
