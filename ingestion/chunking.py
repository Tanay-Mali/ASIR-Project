import re
import hashlib
import chromadb

from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from collections import Counter
from rag_config import RAGConfig


# ---------------------------
# Step 1: ChromaDB setup
# ---------------------------
def get_chroma_store(cfg: RAGConfig):
    chroma_client = chromadb.PersistentClient(path=cfg.db_path)
    collection = chroma_client.get_or_create_collection(
        name=cfg.collection_name
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store, collection


# ---------------------------
# Step 2: Embedding model
# ---------------------------
def get_embed_model(cfg: RAGConfig):
    return HuggingFaceEmbedding(model_name=cfg.embed_model_name)


# ---------------------------
# NEW HELPER: Extract Legal Metadata
# ---------------------------
def extract_legal_metadata(text_block: str, current_act: str) -> dict:
    """Extracts Chapter and Section identifiers from the top of a text block."""
    metadata = {"act": current_act, "chapter": "", "section": ""}
    
    # Look for "Chapter X" in the first 200 chars
    chapter_match = re.search(r'(?i)(?:Chapter\s+([A-Z0-9IVX]+))', text_block[:200])
    if chapter_match:
        metadata["chapter"] = chapter_match.group(1).upper()
        
    # Look for "Section 72A" OR "72A." at the start of lines
    section_match = re.search(r'(?i)(?:Section\s+(\d+[A-Z]?)|(\d+[A-Z]?)\.\s+)', text_block[:200])
    if section_match:
        metadata["section"] = section_match.group(1) or section_match.group(2)
        
    return metadata


# ---------------------------
# Step 3: Chunking (UPDATED for Two-Pass Structure Aware Splitting)
# ---------------------------
def create_nodes(text: str, source_name: str, cfg: RAGConfig) -> list[TextNode]:

    if not text or not text.strip():
        print(f"  ⚠️  '{source_name}': Empty text. Skipping.")
        return []

    # Count approximate tokens (words / 0.75 is a rough token estimate)
    approx_tokens = len(text.split()) / 0.75

    # If document is small, reduce chunk size proportionally
    if approx_tokens < cfg.chunk_size:
        actual_chunk_size = max(int(approx_tokens / 3), cfg.min_chunk_length)
        actual_overlap = max(int(actual_chunk_size * cfg.overlap_ratio), 10)
    else:
        actual_chunk_size = cfg.chunk_size
        actual_overlap = cfg.chunk_overlap

    # Fallback token splitter for long sections
    splitter = SentenceSplitter(
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_overlap,
    )

    print(f"  ℹ️  Chunk size used: {actual_chunk_size} | Overlap: {actual_overlap}")

    # PASS 1: Split at structural boundaries
    split_pattern = r'(?=\n(?:CHAPTER\s+[IVXLCDM]+|[A-Z]{2}[A-Z\s]+\n|\d{1,3}[A-Z]?\.\s+[A-Z][a-z]))'
    structural_blocks = re.split(split_pattern, text)

    nodes = []
    chunk_counter = 0
    current_chapter = ""
    current_section = ""

    for block in structural_blocks:
        block = block.strip()
        if not block:
            continue
            
        # Update running state based on the block's header
        meta = extract_legal_metadata(block, source_name)
        if meta["chapter"]: current_chapter = meta["chapter"]
        if meta["section"]: current_section = meta["section"]
        
        # PASS 2: Token splitting for the section block
        sub_texts = splitter.split_text(block)
        sub_texts = [c for c in sub_texts if len(c.strip()) >= cfg.min_chunk_length]

        for sub_text in sub_texts:
            chunk_hash = hashlib.md5(sub_text.encode()).hexdigest()
            node = TextNode(
                id_=f"{source_name}__chunk_{chunk_counter}__{chunk_hash[:8]}",
                text=sub_text,
                metadata={
                    "source": source_name,
                    "chunk_index": chunk_counter,
                    "chunk_hash": chunk_hash,
                    "act": source_name,
                    "chapter": current_chapter,  
                    "section": current_section   
                },
            )
            nodes.append(node)
            chunk_counter += 1

    if not nodes:
        print(f"  ⚠️  '{source_name}': No valid chunks after filtering.")
        return []

    # Backfill the total_chunks count now that we know exactly how many were produced
    total_chunks = len(nodes)
    for node in nodes:
        node.metadata["total_chunks"] = total_chunks

    return nodes


# ---------------------------
# Step 4: Deduplication
# ---------------------------
def filter_existing_nodes(nodes: list[TextNode], collection) -> list[TextNode]:
    if collection.count() == 0:
        return nodes

    try:
        result = collection.get(ids=[n.id_ for n in nodes])
        existing_ids = set(result["ids"])
    except Exception:
        existing_ids = set()
    new_nodes = [n for n in nodes if n.id_ not in existing_ids]

    skipped = len(nodes) - len(new_nodes)
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} duplicate chunk(s) already in DB.")

    return new_nodes


# ---------------------------
# Step 5: Build index
# ---------------------------
def build_index(
    text: str,
    source_name: str,
    cfg: RAGConfig,
    embed_model,
    vector_store,
    collection,
) -> VectorStoreIndex:

    nodes = create_nodes(text, source_name, cfg)

    if not nodes:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )

    print(f"  📄 '{source_name}': {len(nodes)} chunks created.")

    nodes = filter_existing_nodes(nodes, collection)
    print(f"  ✅ {len(nodes)} new chunk(s) to insert.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not nodes:
        print("  ℹ️  Nothing new to insert. Loading existing index.")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


# ---------------------------
# Step 6: Inspect database
# ---------------------------
def inspect_database(cfg: RAGConfig) -> None:
    _, collection = get_chroma_store(cfg)
    total = collection.count()
    print(f"\n📦 ChromaDB '{cfg.collection_name}': {total} chunk(s) stored.")

    if total == 0:
        print("  (empty)")
        return

    all_data = collection.get(include=["documents", "metadatas"])

    # Per source count
    source_counts = Counter(
        m.get("source", "unknown") for m in all_data["metadatas"]
    )
    print("\n📊 Chunks per source:")
    for src, count in sorted(source_counts.items()):
        print(f"   {src:<40} {count:>4} chunk(s)")

    # Sample chunks
    print("\n📋 Sample chunks (up to 3):")
    for i, (doc, meta) in enumerate(
        zip(all_data["documents"][:3], all_data["metadatas"][:3])
    ):
        print(f"\n  [{i+1}] {meta.get('source')} "
              f"(chunk {meta.get('chunk_index')}/{meta.get('total_chunks')})")
        print(f"       {doc[:160].strip()}...")
