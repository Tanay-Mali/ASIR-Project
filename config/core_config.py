"""
config/core_config.py
=====================
Shared foundation for ASIR-RAG.
Contains paths, shared model names, intent types, and global configs.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_NAME = "ASIR-RAG"
VERSION      = "1.1.0"
RANDOM_SEED  = 42
TEST_MODE    = True  # Useful for debugging and disabling constraints

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_DIR = os.path.join(BASE_DIR, "database")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

FEEDBACK_DB_PATH    = "./database/feedback.db"
VECTOR_DB_PATH      = "./database/chroma_db"
XGBOOST_MODEL_PATH  = "./models/xgb_classifier.pkl"
LABEL_ENCODER_PATH  = "./models/xgb_label_encoder.pkl"
FEATURE_SCALER_PATH = "./models/xgb_scaler.pkl"
GRAPH_PICKLE_PATH   = "./database/graph/graph.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# INTENT & STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
VALID_QUERY_TYPES = [
    "factual",
    "relational",
    "causal",
    "comparative",
    "exploratory",
    "no_retrieval",
]

# Centralized strategy mapping (USED BY BOTH CLASSIFIER + ROUTER)
STRATEGY_MAP = {
    "factual":      "vector",
    "relational":   "graph",
    "causal":       "graph",
    "comparative":  "hybrid",       # vector + bm25
    "exploratory":  "hierarchical",
    "no_retrieval": "none",
}

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_FULL_DIM = 384
EMBEDDING_PCA_DIM  = 32
USE_EMBEDDING_NORMALIZATION = True

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH GLOBAL CONFIG (IMPORTANT)
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_PPR_ALPHA = 0.85
GRAPH_PPR_MAX_ITER = 200
GRAPH_PPR_TOL = 1e-6

GRAPH_MAX_SEED_NODES = 5
GRAPH_MAX_CHUNKS_PER_NODE = 2

GRAPH_RELATION_DEFAULT_WEIGHT = 0.4
GRAPH_PPR_MIN_NODE_SCORE = 1e-7
