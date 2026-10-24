"""
config/router_config.py
=======================
Configurations for the Adaptive Router and RRF Merger.
"""

import os
import config.core_config as C

ROUTER_WEIGHTS_DB = os.path.join(C.DATABASE_DIR, "router_weights.db")

# Get the absolute path to the root of your project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define the feedback database path (Required for WeightStore initialization)
FEEDBACK_DB_PATH = os.path.join(BASE_DIR, "database", "feedback.db")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BANDS
# ─────────────────────────────────────────────────────────────────────────────
ROUTER_HIGH_CONFIDENCE_THRESHOLD = 0.85
ROUTER_HIGH_GAP_THRESHOLD        = 0.35

ROUTER_MEDIUM_CONFIDENCE_THRESHOLD = 0.55
ROUTER_MEDIUM_GAP_THRESHOLD        = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
ROUTER_WEIGHT_INITIAL         = 1
ROUTER_WEIGHT_ALPHA           = 0.10
ROUTER_WEIGHT_RECALC_INTERVAL = 50
ROUTER_WEIGHT_MIN             = 0.2
ROUTER_WEIGHT_MAX             = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_TOP_K = {
    "vector":       15,
    "bm25":         15,
    "hybrid":       15,   # FIXED
    "hierarchical": 5,
    "graph":        10,
}

ROUTER_FINAL_TOP_K = 15

# NOTE: Timeout intentionally unchanged (as requested)
STRATEGY_TOTAL_TIMEOUT_MS = 60000

# ─────────────────────────────────────────────────────────────────────────────
# RRF MERGING
# ─────────────────────────────────────────────────────────────────────────────
RRF_K = 60
RRF_DEDUP_THRESHOLD = 0.98

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_STRATEGY_ORDER = ["vector", "bm25", "hierarchical", "graph"]

# ─────────────────────────────────────────────────────────────
# ROUTER WEIGHT LEARNING CONFIG
# ─────────────────────────────────────────────────────────────

# Number of queries before adaptive weights kick in
ROUTER_WEIGHT_WARMUP_SAMPLES = 3

# Learning rate for feedback updates
ROUTER_WEIGHT_LEARNING_RATE = 0.1

# Minimum confidence to trust feedback
ROUTER_FEEDBACK_CONFIDENCE_THRESHOLD = 0.6

# Default weights (fallback)
ROUTER_DEFAULT_WEIGHTS = {
    "vector": 1.0,
    "bm25": 0.8,
    "hierarchical": 0.9,
    "graph": 0.85
}

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# Maximum number of databases to search simultaneously
ROUTER_MAX_PARALLEL_STRATEGIES = 4
# Minimum number of retrieved chunks required to consider a search successful 
# (If it finds fewer than this, it triggers the fallback strategies)
STRATEGY_MIN_RESULTS = 1
ROUTER_MAX_PARALLEL_SUBQUERIES = 4
