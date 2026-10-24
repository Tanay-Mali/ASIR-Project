from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Paths
    db_path:          str = "./database/ChromaDB"
    keyword_dir:      str = "./database/keyword"
    hierarchical_dir: str = "./database/hierarchyDB"
    graph_dir:        str = "./database/graph"
    data_dir:         str = "./data_file"

    # ChromaDB
    collection_name: str = "rag_collection"

    # Embedding
    embed_model_name: str = "BAAI/bge-large-en-v1.5"

    # Vector chunking
    chunk_size:       int   = 256
    chunk_overlap:    int   = 40
    min_chunk_length: int   = 30
    overlap_ratio:    float = 0.15

    # Hierarchical chunking
    child_chunk_size:    int = 256
    child_chunk_overlap: int = 40

    # Clustering
    min_clusters:       int = 3
    chunks_per_cluster: int = 10

    # Retrieval
    similarity_top_k: int = 15

    # Hierarchical retrieval
    max_distance_threshold: float = 1.0
    # max_distance_threshold: float = 0.75
    raptor_summary_count:   int   = 3
    raptor_summary_boost:   float = 0.85
