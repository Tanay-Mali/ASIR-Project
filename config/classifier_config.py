"""
config/classifier_config.py
Complete configuration for ASIR-RAG classifier
"""
import os
# ─────────────────────────────────────────────────────────────
# LLM CONFIG
# ─────────────────────────────────────────────────────────────

LLM_CONFIG = {
    "model": "llama3",
    "host": "http://localhost:11434",
    "temperature": 0.1,
    "max_tokens": 300,
    "format": "json",
}

# Flattened compatibility layer
LLM_MODEL       = LLM_CONFIG["model"]
LLM_FORMAT      = LLM_CONFIG["format"]
LLM_TEMPERATURE = LLM_CONFIG["temperature"]
LLM_MAX_TOKENS  = LLM_CONFIG["max_tokens"]
LLM_HOST        = LLM_CONFIG["host"]

# Timeout (REQUIRED by llm_classifier)
LLM_TIMEOUT_SEC = 30


# ─────────────────────────────────────────────────────────────
# SIGNAL WEIGHTS
# ─────────────────────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "comparative_marker": 1.4,
    "relational_marker":  1.3,
    "verb_type":          1.1,
    "question_word":      1.0,
    "entity_type":        0.9,
    "query_length":       0.5,
}


# ─────────────────────────────────────────────────────────────
# QUERY TYPES
# ─────────────────────────────────────────────────────────────

VALID_QUERY_TYPES = [
    "factual",
    "relational",
    "causal",
    "comparative",
    "exploratory",
    "no_retrieval",
]


# ─────────────────────────────────────────────────────────────
# DOMAIN MARKERS
# ─────────────────────────────────────────────────────────────

RELATIONAL_MARKERS = {
    "relationship between": 0.92,
    "impact of": 0.85,
    "how does": 0.70,
    "mandated by": 0.90,
    "enforced by": 0.90,
    "applies to": 0.80
}

CAUSAL_MARKERS = {
    "leads to": 0.90,
    "result in": 0.85,
    "consequence of": 0.88,
    "penalties for": 0.92,
    "liability for": 0.90
}

COMPARATIVE_MARKERS = {
    "difference between": 0.95,
    "compare": 0.90,
    "vs": 0.85,
    "better than": 0.88,
    "worse than": 0.88
}


# ─────────────────────────────────────────────────────────────
# STRUCTURAL SIGNALS
# ─────────────────────────────────────────────────────────────

QUESTION_WORD_SIGNALS = {
    "what": {"factual": 0.9},
    "why": {"causal": 0.9},
    "how": {"relational": 0.8},
    "when": {"factual": 0.7},
    "where": {"factual": 0.7},
    "who": {"factual": 0.8},
    "which": {"comparative": 0.7},
}


VERB_TYPE_SIGNALS = {
    "define": {"factual": 0.9},
    "explain": {"exploratory": 0.8},
    "compare": {"comparative": 0.9},
    "affect": {"causal": 0.8},
    "impact": {"causal": 0.8},
}


ENTITY_TYPE_SIGNALS_SINGLE = {
    "LAW": {"factual": 0.7},
    "ORG": {"relational": 0.6},
}

ENTITY_TYPE_SIGNALS_MULTIPLE = {
    "LAW": {"comparative": 0.8},
    "ORG": {"relational": 0.7},
}


# ─────────────────────────────────────────────────────────────
# LENGTH SIGNALS
# ─────────────────────────────────────────────────────────────

LENGTH_KEYWORD_MAX = 3
LENGTH_SHORT_MAX = 8
LENGTH_STANDARD_MAX = 20


# ─────────────────────────────────────────────────────────────
# LAYER 1 THRESHOLDS
# ─────────────────────────────────────────────────────────────

LAYER1_CONFIDENCE_THRESHOLD = 0.6
LAYER1_GAP_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────
# DECOMPOSITION
# ─────────────────────────────────────────────────────────────

DECOMPOSITION_MIN_SENTENCES = 2
DECOMPOSITION_MIN_TOKENS = 20

DECOMPOSITION_CONJUNCTIONS = ["and", "or", "also", "along with"]


# ─────────────────────────────────────────────────────────────
# NO RETRIEVAL
# ─────────────────────────────────────────────────────────────

NO_RETRIEVAL_SYSTEM_PHRASES = [
    "your name",
    "who are you",
    "what can you do",
]


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEEDBACK_DB_PATH = os.path.join(DATABASE_DIR, "feedback.db")
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "xgb_label_encoder.pkl")
FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, "xgb_scaler.pkl")

TRANSITION_MIN_TOTAL_SAMPLES = 50
TRANSITION_MIN_PER_CLASS = 8
TRANSITION_MAX_CLASS_DOMINANCE = 0.60

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_PCA_DIM = 32

XGB_TRAIN_TEST_SPLIT = 0.20
XGB_RANDOM_STATE = 42
XGB_N_ESTIMATORS = 250
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE_BYTREE = 0.9
XGB_EVAL_METRIC = "mlogloss"

QUESTION_WORD_ENCODING = {
    "what": 1,
    "why": 2,
    "how": 3,
    "when": 4,
    "where": 5,
    "who": 6,
    "which": 7,
    "other": 0,
    None: 0,
}

VERB_ENCODING = {
    "define": 1,
    "explain": 2,
    "compare": 3,
    "affect": 4,
    "impact": 5,
    "other": 0,
    None: 0,
}

STRATEGY_MAP = {
    "factual": "vector",
    "relational": "graph",
    "causal": "graph",
    "comparative": "hybrid",
    "exploratory": "hierarchical",
    "no_retrieval": "none",
}


