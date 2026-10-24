"""
hierarchical_indexer.py
───────────────────────
Builds a DYNAMIC multi-level hierarchy across ALL documents together.

Number of levels is decided automatically based on data size:
    Less than 200 chunks  → 2 levels (0 + 1)
    200 to 1000 chunks    → 3 levels (0 + 1 + 2)
    More than 1000 chunks → 4 levels (0 + 1 + 2 + 3)

Stopping condition:
    If a level produces 3 or fewer nodes, stop building higher levels
    and create the domain summary from the current top level.

Why dynamic:
    Fixed 4 levels on 1-2 documents wastes Mistral calls summarizing
    only 3 things. Dynamic levels build only what the data justifies.

Levels:
    Level 0 → Raw chunks (256 tokens, leaf nodes)
    Level 1 → Cluster summaries
    Level 2 → Super-cluster summaries (only if enough data)
    Level 3 → Domain summary (root, always the final level)

Storage:
    ChromaDB rag_collection:
        All nodes stored with "level" field in metadata
        Level 0 nodes  → level: 0
        Level 1 nodes  → level: 1
        Level 2 nodes  → level: 2 (if created)
        Domain node    → level: top_level + 1

    database/hierarchyDB/
        hierarchy_map.pkl  → child_id to parent_id for all levels
        build_complete.txt → marker file with build stats
"""

import os
import pickle
import hashlib
import numpy as np
import chromadb
import ollama

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize  # Added to prevent BGE-Large math crashes
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_config import RAGConfig


# ══════════════════════════════════════════════════════
# CONSTANTS — dynamic level thresholds
# ══════════════════════════════════════════════════════

MIN_NODES_TO_BUILD_NEXT_LEVEL = 3   # stop if current level has 3 or fewer nodes
MAX_LEVELS = 4                      # never go beyond level 3 + domain


# ══════════════════════════════════════════════════════
# LEVEL 0 — Create child chunks from all documents
# ══════════════════════════════════════════════════════

def create_all_child_chunks(
    documents: dict[str, str],
    cfg: RAGConfig,
) -> list[TextNode]:
    """
    Create Level 0 child chunks from ALL documents pooled together.

    Every chunk from every document goes into one flat list.
    Source name stored in metadata so origin is always traceable.
    """
    all_nodes = []

    for source_name, text in documents.items():
        if not text or not text.strip():
            print(f"  ⚠️  '{source_name}': Empty text. Skipping.")
            continue

        splitter = SentenceSplitter(
            chunk_size    = cfg.child_chunk_size,
            chunk_overlap = cfg.child_chunk_overlap,
        )

        chunks = splitter.split_text(text)
        chunks = [c for c in chunks if len(c.strip()) >= cfg.min_chunk_length]

        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            node_id    = f"{source_name}__child_{i}__{chunk_hash[:8]}"

            node = TextNode(
                id_  = node_id,
                text = chunk,
                metadata = {
                    "source":       source_name,
                    "level":        0,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "chunk_hash":   chunk_hash,
                    "parent_id":    "",
                },
            )
            all_nodes.append(node)

        print(f"  📄 '{source_name}': {len(chunks)} chunk(s) created.")

    print(f"\n  ✅ Total Level 0 chunks across all documents: {len(all_nodes)}")
    return all_nodes


# ══════════════════════════════════════════════════════
# EMBEDDING — Get vectors for any list of nodes
# ══════════════════════════════════════════════════════

def embed_nodes(
    nodes: list[TextNode],
    embed_model,
    level: int,
) -> np.ndarray:
    """Embed a list of TextNodes and return numpy array of vectors."""
    print(f"  🔢 Embedding {len(nodes)} Level {level} node(s)...")

    vectors = []
    for node in nodes:
        vector = embed_model.get_text_embedding(node.text)
        vectors.append(vector)

    return np.array(vectors)


# ══════════════════════════════════════════════════════
# CLUSTERING — Group nodes by semantic similarity
# ══════════════════════════════════════════════════════

def cluster_nodes(
    nodes: list[TextNode],
    vectors: np.ndarray,
    cfg: RAGConfig,
    level: int,
) -> dict[int, list[TextNode]]:
    """Group nodes into clusters using K-Means on embedding vectors."""
    num_nodes    = len(nodes)
    num_clusters = max(cfg.min_clusters, num_nodes // cfg.chunks_per_cluster)
    num_clusters = min(num_clusters, num_nodes)  # cannot exceed node count

    print(f"  🔵 Clustering {num_nodes} Level {level} node(s) "
          f"into {num_clusters} cluster(s)...")

    # FIX: Normalize vectors to prevent divide by zero/NaN errors with BGE-Large
    normalized_vectors = normalize(vectors)

    kmeans = KMeans(
        n_clusters   = num_clusters,
        random_state = 42,
        n_init       = 10,
    )
    kmeans.fit(normalized_vectors)

    cluster_groups: dict[int, list[TextNode]] = {}
    for node, label in zip(nodes, kmeans.labels_):
        label = int(label)
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(node)

    print(f"  ✅ Level {level} clustering done — {num_clusters} cluster(s).")
    return cluster_groups


# ══════════════════════════════════════════════════════
# SUMMARIZATION — Mistral generates summary per cluster
# ══════════════════════════════════════════════════════

def summarize_cluster(
    cluster_nodes_list: list[TextNode],
    cluster_index: int,
    level: int,
) -> str:
    """Send all node texts in one cluster to Mistral via Ollama."""
    combined_text = "\n\n".join([
        f"Text {i+1}:\n{node.text.strip()}"
        for i, node in enumerate(cluster_nodes_list)
    ])

    prompt = f"""You are a document summarizer. Below are {len(cluster_nodes_list)} related text segments from a document collection.
Write a concise summary (3-5 sentences) capturing the main topic and key information shared across these segments.
Rules:
- Do NOT add information not present in the texts
- Do NOT use bullet points
- Write in plain paragraph form
- Be specific about the topic

TEXT SEGMENTS:
{combined_text}

SUMMARY:"""

    try:
        response = ollama.chat(
            model    = "mistral", # FIX: standard name to prevent 404 errors
            messages = [{"role": "user", "content": prompt}],
            options  = {"num_ctx": 16384}, # FIX: Increase context window to handle large clusters
        )
        summary = response["message"]["content"].strip()
        print(f"    ✅ Level {level} cluster {cluster_index} summarized "
              f"({len(cluster_nodes_list)} nodes → {len(summary)} chars).")
        return summary

    except Exception as e:
        print(f"    ⚠️  Mistral failed for Level {level} "
              f"cluster {cluster_index}: {e}")
        print(f"    ℹ️  Using first node text as fallback.")
        return cluster_nodes_list[0].text


# ══════════════════════════════════════════════════════
# BUILD ONE LEVEL — Cluster + Summarize → new nodes
# ══════════════════════════════════════════════════════

def build_one_level(
    child_nodes: list[TextNode],
    vectors: np.ndarray,
    source_prefix: str,
    current_level: int,
    parent_level: int,
    cfg: RAGConfig,
    hierarchy_map: dict,
    embed_model,
) -> tuple[list[TextNode], np.ndarray]:
    """Build one level of the hierarchy above the given child nodes."""
    print(f"\n  {'─'*50}")
    print(f"  Building Level {parent_level} from "
          f"{len(child_nodes)} Level {current_level} node(s)...")

    cluster_groups = cluster_nodes(child_nodes, vectors, cfg, current_level)
    num_clusters   = len(cluster_groups)

    print(f"\n  🤖 Summarizing {num_clusters} cluster(s) with Mistral...")

    parent_nodes = []

    for cluster_index, members in cluster_groups.items():
        summary = summarize_cluster(members, cluster_index, current_level)

        summary_hash = hashlib.md5(summary.encode()).hexdigest()
        parent_id    = (f"{source_prefix}__L{parent_level}_"
                        f"cluster{cluster_index}__{summary_hash[:8]}")

        # FIX: Collect real sources safely into a list to prevent metadata string corruption
        all_child_sources = []
        for n in members:
            # Look for contributing docs first (if parent is already a summary), fallback to source (if parent is raw text)
            src_data = n.metadata.get("contributing_docs") or n.metadata.get("source", "unknown")
            if isinstance(src_data, list):
                all_child_sources.extend(src_data)
            else:
                all_child_sources.extend([s.strip() for s in src_data.split(",")])
        
        unique_sources = list(set(all_child_sources))

        parent_node = TextNode(
            id_  = parent_id,
            text = summary,
            metadata = {
                "source":            f"Summary_Level_{parent_level}", # Clean, descriptive string
                "level":             parent_level,
                "contributing_docs": ", ".join(unique_sources),       # FIX: Convert list to string for ChromaDB
                "cluster_index":     cluster_index,
                "num_clusters":      num_clusters,
                "child_count":       len(members),
                "parent_id":         "", 
            },
        )
        parent_nodes.append(parent_node)

        for child in members:
            child.metadata["parent_id"] = parent_id
            hierarchy_map[child.id_]    = parent_id

    parent_vectors = embed_nodes(parent_nodes, embed_model, parent_level)

    return parent_nodes, parent_vectors


# ══════════════════════════════════════════════════════
# DOMAIN SUMMARY — Root node for entire collection
# ══════════════════════════════════════════════════════

def build_domain_summary(
    top_level_nodes: list[TextNode],
    top_level: int,
    source_prefix: str,
    hierarchy_map: dict,
) -> TextNode:
    """Create the root node — one summary for the entire collection."""
    domain_level = top_level + 1

    print(f"\n  🌐 Building domain summary at Level {domain_level}...")
    print(f"     Summarizing {len(top_level_nodes)} top-level node(s)...")

    combined = "\n\n".join([
        f"Topic {i+1}:\n{node.text.strip()}"
        for i, node in enumerate(top_level_nodes)
    ])

    prompt = f"""You are a document summarizer. Below are high-level topic summaries from a document collection.
Write a concise overview (4-6 sentences) describing the entire collection — what topics it covers, what domain it belongs to, and what a reader would learn.
Rules:
- Do NOT add information not present in the texts
- Do NOT use bullet points
- Write in plain paragraph form

TOPIC SUMMARIES:
{combined}

COLLECTION OVERVIEW:"""

    try:
        response = ollama.chat(
            model    = "mistral", # FIX: standard name
            messages = [{"role": "user", "content": prompt}],
            options  = {"num_ctx": 16384}, # FIX: Increase context window to handle large domain inputs
        )
        summary = response["message"]["content"].strip()
        print(f"  ✅ Domain summary created at Level {domain_level} "
              f"({len(summary)} chars).")

    except Exception as e:
        print(f"  ⚠️  Mistral failed for domain summary: {e}")
        summary = top_level_nodes[0].text

    domain_hash = hashlib.md5(summary.encode()).hexdigest()
    domain_id   = f"{source_prefix}__L{domain_level}_domain__{domain_hash[:8]}"

    # FIX: Collect domain sources safely
    all_child_sources = []
    for n in top_level_nodes:
        src_data = n.metadata.get("contributing_docs") or n.metadata.get("source", "unknown")
        if isinstance(src_data, list):
            all_child_sources.extend(src_data)
        else:
            all_child_sources.extend([s.strip() for s in src_data.split(",")])
            
    unique_sources = list(set(all_child_sources))

    domain_node = TextNode(
        id_  = domain_id,
        text = summary,
        metadata = {
            "source":            "Domain_Summary",                  # Clean root string
            "level":             domain_level,
            "contributing_docs": ", ".join(unique_sources),         # FIX: Convert list to string for ChromaDB
            "child_count":       len(top_level_nodes),
            "parent_id":         "",
        },
    )

    for node in top_level_nodes:
        node.metadata["parent_id"] = domain_id
        hierarchy_map[node.id_]    = domain_id

    return domain_node


# ══════════════════════════════════════════════════════
# DEDUPLICATION
# ══════════════════════════════════════════════════════

def filter_existing_nodes(
    nodes: list[TextNode],
    collection,
) -> list[TextNode]:
    """Remove nodes already stored in ChromaDB."""
    if collection.count() == 0:
        return nodes

    existing_ids = set(collection.get()["ids"])
    new_nodes    = [n for n in nodes if n.id_ not in existing_ids]

    skipped = len(nodes) - len(new_nodes)
    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} duplicate node(s).")

    return new_nodes


# ══════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════

def save_hierarchy_map(hierarchy_map: dict, hierarchical_dir: str) -> None:
    """Save complete child → parent mapping to disk."""
    os.makedirs(hierarchical_dir, exist_ok=True)
    map_path = os.path.join(hierarchical_dir, "hierarchy_map.pkl")

    with open(map_path, "wb") as f:
        pickle.dump(hierarchy_map, f)

    print(f"  ✅ Hierarchy map saved → {map_path}")
    print(f"     {len(hierarchy_map)} node relationships stored.")


def load_hierarchy_map(hierarchical_dir: str) -> dict:
    """Load hierarchy map from disk. Called by retriever at startup."""
    map_path = os.path.join(hierarchical_dir, "hierarchy_map.pkl")

    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Hierarchy map not found at '{map_path}'.\n"
            f"Run ingest.py first to build the hierarchical index."
        )

    with open(map_path, "rb") as f:
        hierarchy_map = pickle.load(f)

    print(f"  ✅ Hierarchy map loaded — "
          f"{len(hierarchy_map)} node relationships.")
    return hierarchy_map


def save_build_complete(hierarchical_dir: str, stats: dict) -> None:
    """Save marker file with build stats."""
    os.makedirs(hierarchical_dir, exist_ok=True)
    marker_path = os.path.join(hierarchical_dir, "build_complete.txt")

    with open(marker_path, "w") as f:
        f.write("Hierarchical index build complete.\n\n")
        f.write("Stats:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")

    print(f"  ✅ Build marker saved → {marker_path}")


# ══════════════════════════════════════════════════════
# DYNAMIC LEVEL DECISION
# ══════════════════════════════════════════════════════

def decide_num_levels(total_chunks: int) -> int:
    """Decide how many intermediate levels to build based on chunk count."""
    if total_chunks < 200:
        levels = 1
        reason = f"{total_chunks} chunks < 200 threshold"
    elif total_chunks < 1000:
        levels = 2
        reason = f"200 ≤ {total_chunks} chunks < 1000 threshold"
    else:
        levels = 3
        reason = f"{total_chunks} chunks ≥ 1000 threshold"

    print(f"  📊 Dynamic level decision: {levels} intermediate level(s)")
    print(f"     Reason: {reason}")
    print(f"     Total levels including domain: {levels + 1}")

    return levels


# ══════════════════════════════════════════════════════
# MAIN BUILD FUNCTION — called by ingest.py
# ══════════════════════════════════════════════════════

def build_hierarchical_index(
    documents: dict[str, str],
    cfg: RAGConfig,
    embed_model,
) -> None:
    """Full pipeline — builds DYNAMIC multi-level hierarchy."""
    print("\n🏗️  Building dynamic multi-level hierarchical index...")
    print(f"   Documents: {len(documents)}")

    marker_path = os.path.join(cfg.hierarchical_dir, "build_complete.txt")
    if os.path.exists(marker_path):
        print(f"  ℹ️  Hierarchical index already exists.")
        print(f"  ℹ️  Delete '{cfg.hierarchical_dir}' folder to rebuild.")
        return

    chroma_client   = chromadb.PersistentClient(path=cfg.db_path)
    collection      = chroma_client.get_or_create_collection(
        name = cfg.collection_name
    )
    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    source_prefix = "collection"
    hierarchy_map: dict[str, str] = {}

    # ── Level 0 ───────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  LEVEL 0 — Creating child chunks")
    print(f"{'═'*55}")

    level0_nodes = create_all_child_chunks(documents, cfg)

    if not level0_nodes:
        print("  ❌ No child chunks created. Aborting.")
        return

    num_intermediate_levels = decide_num_levels(len(level0_nodes))

    current_vectors = embed_nodes(level0_nodes, embed_model, level=0)
    current_nodes   = level0_nodes
    current_level   = 0
    all_built_levels: list[list[TextNode]] = [level0_nodes]

    # ── Dynamic intermediate levels ───────────────────────────────────
    for level_num in range(1, num_intermediate_levels + 1):

        print(f"\n{'═'*55}")
        print(f"  LEVEL {level_num} — Clustering and summarizing")
        print(f"{'═'*55}")

        if len(current_nodes) <= MIN_NODES_TO_BUILD_NEXT_LEVEL:
            print(f"  ℹ️  Only {len(current_nodes)} node(s) at Level "
                  f"{current_level}.")
            print(f"  ℹ️  Too few to cluster further. Stopping here.")
            break

        parent_nodes, parent_vectors = build_one_level(
            child_nodes   = current_nodes,
            vectors       = current_vectors,
            source_prefix = source_prefix,
            current_level = current_level,
            parent_level  = level_num,
            cfg           = cfg,
            hierarchy_map = hierarchy_map,
            embed_model   = embed_model,
        )

        all_built_levels.append(parent_nodes)
        current_nodes   = parent_nodes
        current_vectors = parent_vectors
        current_level   = level_num

    # ── Domain summary — always built on top ─────────────────────────
    print(f"\n{'═'*55}")
    print(f"  DOMAIN SUMMARY — Root node")
    print(f"{'═'*55}")

    domain_node = build_domain_summary(
        top_level_nodes = current_nodes,
        top_level       = current_level,
        source_prefix   = source_prefix,
        hierarchy_map   = hierarchy_map,
    )

    # ── Store everything in ChromaDB ──────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  STORING all nodes in ChromaDB")
    print(f"{'═'*55}")

    all_nodes = []
    for level_nodes in all_built_levels:
        all_nodes.extend(level_nodes)
    all_nodes.append(domain_node)

    print(f"  📦 Total nodes before dedup: {len(all_nodes)}")
    for i, level_nodes in enumerate(all_built_levels):
        print(f"     Level {i}: {len(level_nodes)} node(s)")
    print(f"     Domain : 1 node")

    all_nodes = filter_existing_nodes(all_nodes, collection)
    print(f"  📦 Nodes after dedup: {len(all_nodes)}")

    if all_nodes:
        print(f"  💾 Embedding and storing in ChromaDB...")
        VectorStoreIndex(
            all_nodes,
            storage_context = storage_context,
            embed_model     = embed_model,
            show_progress   = True,
        )
        print(f"  ✅ All nodes stored in ChromaDB.")

    # ── Save hierarchy map ────────────────────────────────────────────
    save_hierarchy_map(hierarchy_map, cfg.hierarchical_dir)

    # ── Save build marker with stats ──────────────────────────────────
    stats = {
        "total_documents":         len(documents),
        "total_levels":            len(all_built_levels) + 1,
        "hierarchy_relationships": len(hierarchy_map),
    }
    for i, level_nodes in enumerate(all_built_levels):
        stats[f"level_{i}_nodes"] = len(level_nodes)
    stats["domain_node"] = 1

    save_build_complete(cfg.hierarchical_dir, stats)

    print(f"\n{'═'*55}")
    print(f"  ✅ Hierarchical index built successfully.")
    print(f"     Documents   : {len(documents)}")
    print(f"     Total levels: {len(all_built_levels) + 1}")
    for i, level_nodes in enumerate(all_built_levels):
        print(f"     Level {i}     : {len(level_nodes)} node(s)")
    print(f"     Domain      : 1 node")
    print(f"     Map entries : {len(hierarchy_map)}")
    print(f"{'═'*55}")
