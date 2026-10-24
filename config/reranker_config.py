"""
config/reranker_config.py
=========================
Settings for Cross-Encoder semantic re-ranking.
"""

import config.core_config as C

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Must match router final output
RERANKER_TOP_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
RERANKER_BATCH_SIZE = 16

# Permissive floor for logits
RERANKER_MIN_SCORE = 0.30
