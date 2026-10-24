"""
router/weight_store.py
======================
Persistent dynamic weight store for the Router.

WHAT IS STORED:
  For every (query_type, strategy) pair, this store tracks:
    - current_weight : float — the live routing weight (0.10 to 0.95)
    - sample_count   : int   — how many feedback signals have updated this pair
    - last_updated   : str   — ISO timestamp of the most recent update
    - cumulative_rating : float — sum of all normalised ratings (for recalc)

HOW WEIGHTS ARE UPDATED:
  Exponential Moving Average:
    new_weight = alpha * feedback_signal + (1 - alpha) * current_weight

  Where:
    alpha          = ROUTER_WEIGHT_ALPHA (0.10) — learning rate
    feedback_signal = user_rating / 5.0          — normalised to 0.0–1.0

  The weight is clamped to [ROUTER_WEIGHT_MIN, ROUTER_WEIGHT_MAX] after update.

HOW WEIGHTS INFLUENCE ROUTING:
  The strategy_selector reads weights from this store and multiplies them
  with classifier scores to produce adjusted priority scores for each strategy.
  Higher weight = higher priority = preferred in routing decision.

WARMUP:
  Until ROUTER_WEIGHT_WARMUP_SAMPLES feedback signals have arrived for a
  (query_type, strategy) pair, the weight is treated as ROUTER_WEIGHT_INITIAL
  (neutral 0.50) regardless of what the database says.
  This prevents early over-fitting on sparse data.

PERIODIC RECALCULATION:
  Every ROUTER_WEIGHT_RECALC_INTERVAL queries, the router triggers a full
  recalculation of all weights from the raw feedback history.
  This corrects any drift that accumulated through incremental updates.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import config.router_config as C

log = logging.getLogger(__name__)

# All (query_type, strategy) pairs tracked.
# Only valid combinations — graph is not a fallback for factual, for example.
_TRACKED_PAIRS: list[tuple[str, str]] = [
    ("factual",     "vector"),
    ("factual",     "bm25"),
    ("factual",     "hierarchical"),
    ("factual",     "graph"),
    ("relational",  "vector"),
    ("relational",  "bm25"),
    ("relational",  "graph"),
    ("relational",  "hierarchical"),
    ("comparative", "vector"),
    ("comparative", "bm25"),
    ("comparative", "hierarchical"),
    ("comparative", "graph"),
    ("exploratory", "vector"),
    ("exploratory", "bm25"),
    ("exploratory", "hierarchical"),
    ("exploratory", "graph"),
]

def get_current_weight(conn, query_type: str, strategy: str) -> float:
    """
    Always reads the actual current weight from the database.
    Never use a hardcoded default — always call this before computing updates.
    """
    conn.execute("""
        INSERT OR IGNORE INTO routing_weights
        (query_type, strategy, current_weight, sample_count, cumulative_rating, last_updated)
        VALUES (?, ?, 1.0, 0, 0.0, ?)
    """, (query_type, strategy, datetime.utcnow().isoformat()))
    # No commit here — caller is responsible for committing the full transaction.

    row = conn.execute("""
        SELECT current_weight FROM routing_weights
        WHERE query_type = ? AND strategy = ?
    """, (query_type, strategy)).fetchone()

    return row[0] if row else 1.0




class WeightStore:
    """
    Manages dynamic routing weights with SQLite persistence.

    Instantiate once in the Router and reuse for the application lifetime.
    The store initialises all weights to ROUTER_WEIGHT_INITIAL on first run,
    then updates them as user feedback arrives.

    Example
    -------
    >>> store = WeightStore()
    >>> w = store.get_weight("factual", "vector")
    >>> 0.10 <= w <= 0.95  # always within valid range
    True
    >>> store.update_weight("factual", "vector", user_rating=4)
    """

    def __init__(self, db_path: str = C.ROUTER_WEIGHTS_DB) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialise_db()
        log.info("WeightStore ready. DB: %s", db_path)

    # ──────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────

    def get_weight(self, query_type: str, strategy: str) -> float:
        """
        Return the current routing weight for a (query_type, strategy) pair.

        If the pair has fewer than WARMUP_SAMPLES feedback signals,
        returns ROUTER_WEIGHT_INITIAL regardless of stored value.
        This prevents premature weight biasing from sparse early data.

        Parameters
        ----------
        query_type : str   e.g. "factual"
        strategy   : str   e.g. "vector"

        Returns
        -------
        float in [ROUTER_WEIGHT_MIN, ROUTER_WEIGHT_MAX]
        """
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT current_weight, sample_count
                FROM routing_weights
                WHERE query_type = ? AND strategy = ?
            """, (query_type, strategy)).fetchone()
        finally:
            conn.close()

        if row is None:
            return C.ROUTER_WEIGHT_INITIAL

        current_weight, sample_count = row

        # Warmup: treat as neutral until enough samples exist
        if sample_count < C.ROUTER_WEIGHT_WARMUP_SAMPLES:
            return C.ROUTER_WEIGHT_INITIAL

        return float(current_weight)

    def get_all_weights(self) -> dict[tuple[str, str], float]:
        """
        Return all weights as a flat dict keyed by (query_type, strategy).
        Warmup logic is applied per-pair.
        Used by the Router for logging and auditability.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT query_type, strategy, current_weight, sample_count "
                "FROM routing_weights"
            ).fetchall()
        finally:
            conn.close()

        result = {}
        for query_type, strategy, weight, count in rows:
            if count < C.ROUTER_WEIGHT_WARMUP_SAMPLES:
                result[(query_type, strategy)] = C.ROUTER_WEIGHT_INITIAL
            else:
                result[(query_type, strategy)] = float(weight)

        return result

    def get_weights_for_type(self, query_type: str) -> dict[str, float]:
        """
        Return all strategy weights for one query type.
        Used by the strategy selector for a specific routing decision.

        Returns
        -------
        dict mapping strategy name → weight float
        e.g. {"vector": 0.73, "bm25": 0.61, "graph": 0.42, "hierarchical": 0.55}
        """
        all_strategies = list(set(s for _, s in _TRACKED_PAIRS))
        weights = {}
        for strategy in all_strategies:
            weights[strategy] = self.get_weight(query_type, strategy)
        return weights

    # ──────────────────────────────────────────────────────────────────
    # Write — incremental update
    # ──────────────────────────────────────────────────────────────────

    def update_weight(
        self,
        query_type:  str,
        strategy:    str,
        user_rating: int,
    ) -> float:
        """
        Update the weight for a (query_type, strategy) pair using EMA.

        new_weight = alpha * (rating/5) + (1 - alpha) * current_weight

        Parameters
        ----------
        query_type  : str  e.g. "factual"
        strategy    : str  e.g. "vector"
        user_rating : int  1 (very bad) to 5 (excellent)

        Returns
        -------
        float — the new weight after update
        """
        feedback_signal = user_rating / 5.0   # normalise to [0.0, 1.0]
        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT current_weight, sample_count, cumulative_rating
                FROM routing_weights
                WHERE query_type = ? AND strategy = ?
            """, (query_type, strategy)).fetchone()

            if row is None:
                # First feedback for this pair — insert with initial weight
                new_weight   = C.ROUTER_WEIGHT_INITIAL
                sample_count = 0
                cumulative   = 0.0
            else:
                new_weight, sample_count, cumulative = row

            # EMA update
            alpha      = C.ROUTER_WEIGHT_ALPHA
            new_weight = alpha * feedback_signal + (1.0 - alpha) * float(new_weight)

            # Clamp to valid range
            new_weight = max(C.ROUTER_WEIGHT_MIN,
                             min(C.ROUTER_WEIGHT_MAX, new_weight))

            new_count      = sample_count + 1
            new_cumulative = float(cumulative) + feedback_signal

            conn.execute("""
                INSERT INTO routing_weights
                    (query_type, strategy, current_weight,
                     sample_count, cumulative_rating, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_type, strategy) DO UPDATE SET
                    current_weight  = excluded.current_weight,
                    sample_count    = excluded.sample_count,
                    cumulative_rating = excluded.cumulative_rating,
                    last_updated    = excluded.last_updated
            """, (
                query_type, strategy,
                round(new_weight, 6),
                new_count, round(new_cumulative, 6), now,
            ))
            conn.commit()

            log.debug(
                "Weight updated (%s, %s): %.4f → %.4f (n=%d, signal=%.2f)",
                query_type, strategy,
                float(row[0]) if row else C.ROUTER_WEIGHT_INITIAL,
                new_weight, new_count, feedback_signal,
            )

        finally:
            conn.close()

        return new_weight

    def update_from_feedback(
        self,
        query_type:       str,
        strategies_used:  list[str],
        user_rating:      int,
    ) -> dict[str, float]:
        """
        Update weights for all strategies used in a single interaction.

        All strategies that contributed to the response receive the same
        feedback signal. This is the method called by the feedback processor.

        Parameters
        ----------
        query_type      : str
        strategies_used : list[str]  — all strategies that returned results
        user_rating     : int        — 1 to 5

        Returns
        -------
        dict mapping strategy → new_weight for all updated pairs
        """
        updated = {}
        for strategy in strategies_used:
            new_w = self.update_weight(query_type, strategy, user_rating)
            updated[strategy] = new_w

        return updated

    # ──────────────────────────────────────────────────────────────────
    # Periodic full recalculation from feedback history
    # ──────────────────────────────────────────────────────────────────

    def recalculate_from_history(
        self,
        feedback_db_path: str = C.FEEDBACK_DB_PATH,
    ) -> None:
        """
        Recompute all weights from the complete feedback history.

        This corrects any drift from incremental EMA updates.
        Called every ROUTER_WEIGHT_RECALC_INTERVAL queries.

        For each (query_type, strategy) pair, computes:
            weight = average(user_rating / 5.0) across all interactions
                     where that strategy was used for that query_type.
        The result is then clamped to [WEIGHT_MIN, WEIGHT_MAX].
        """
        log.info("Recalculating all router weights from feedback history...")

        try:
            fconn = sqlite3.connect(feedback_db_path)
            rows  = fconn.execute("""
                SELECT query_type, retrieval_strategy_used,
                       AVG(user_rating / 5.0) as avg_signal,
                       COUNT(*) as n
                FROM classification_log
                WHERE user_rating IS NOT NULL
                  AND retrieval_strategy_used != ''
                GROUP BY query_type, retrieval_strategy_used
            """).fetchall()
            fconn.close()
        except Exception as exc:
            log.warning("Recalculation failed — could not read feedback DB: %s", exc)
            return

        now  = datetime.now(timezone.utc).isoformat()
        conn = self._connect()

        try:
            for query_type, strategy, avg_signal, n in rows:
                if n < C.ROUTER_WEIGHT_WARMUP_SAMPLES:
                    continue   # not enough data — leave at initial

                new_weight = max(
                    C.ROUTER_WEIGHT_MIN,
                    min(C.ROUTER_WEIGHT_MAX, float(avg_signal))
                )

                conn.execute("""
                    INSERT INTO routing_weights
                        (query_type, strategy, current_weight,
                         sample_count, cumulative_rating, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_type, strategy) DO UPDATE SET
                        current_weight    = excluded.current_weight,
                        sample_count      = excluded.sample_count,
                        cumulative_rating = excluded.cumulative_rating,
                        last_updated      = excluded.last_updated
                """, (
                    query_type, strategy,
                    round(new_weight, 6), n,
                    round(float(avg_signal) * n, 6), now,
                ))

            conn.commit()
            log.info("Weight recalculation complete. Updated %d pairs.", len(rows))

        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Inspection
    # ──────────────────────────────────────────────────────────────────

    def get_weight_report(self) -> dict:
        """
        Return a full weight report for the analytics dashboard.
        Shows current weights, sample counts, and warmup status.
        """
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT query_type, strategy, current_weight,
                       sample_count, last_updated
                FROM routing_weights
                ORDER BY query_type, strategy
            """).fetchall()
        finally:
            conn.close()

        report = {}
        for qt, st, w, n, updated in rows:
            if qt not in report:
                report[qt] = {}
            report[qt][st] = {
                "weight":       round(float(w), 4),
                "sample_count": n,
                "warmed_up":    n >= C.ROUTER_WEIGHT_WARMUP_SAMPLES,
                "last_updated": updated,
            }
        return report

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _initialise_db(self) -> None:
        """Create routing_weights table and seed all tracked pairs."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_weights (
                    query_type        TEXT NOT NULL,
                    strategy          TEXT NOT NULL,
                    current_weight    REAL NOT NULL DEFAULT 0.50,
                    sample_count      INTEGER NOT NULL DEFAULT 0,
                    cumulative_rating REAL NOT NULL DEFAULT 0.0,
                    last_updated      TEXT,
                    PRIMARY KEY (query_type, strategy)
                )
            """)

            # Seed all tracked pairs with initial neutral weight.
            # INSERT OR IGNORE means existing rows are untouched on restart.
            for qt, st in _TRACKED_PAIRS:
                conn.execute("""
                    INSERT OR IGNORE INTO routing_weights
                        (query_type, strategy, current_weight, sample_count,
                         cumulative_rating)
                    VALUES (?, ?, ?, 0, 0.0)
                """, (qt, st, C.ROUTER_WEIGHT_INITIAL))

            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
@dataclass
class WeightUpdateConfig:
    eta: float = 0.05
    w_max: float = 2.0
    w_min: float = 0.2
    decay_rate: float = 0.97
    w_initial: float = 1.0


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(C.ROUTER_WEIGHTS_DB, check_same_thread=False)


def reset_weights_to_prior():
    """
    Resets all strategy weights to 1.0.
    CALL ONCE from app.py startup to fix 0.50 initialization bug.
    Remove the call after first run.
    """
    query_types = ["factual", "relational", "comparative", "exploratory"]
    strategies = ["vector", "bm25", "graph", "hierarchical"]
    with get_connection() as conn:
        for qt in query_types:
            for strategy in strategies:
                conn.execute(
                    """
                    UPDATE routing_weights
                    SET current_weight = 1.0
                    WHERE query_type = ? AND strategy = ?
                    """,
                    (qt, strategy),
                )
        conn.commit()
    print("[weight_store] All weights reset to 1.0 — remove this call from app.py")


def apply_credit_attribution(
    failure_class,
    confidence,
    strategies_used,
    chunk_ranks,
    weight_store,
    query_type,
    config,
):
    """
    Rank-weighted credit attribution.
    Supports both WeightStore objects and plain dict stores.
    """
    try:
        effective_eta = float(getattr(config, "eta", 0.05)) * float(confidence or 0.0)
        w_max = float(getattr(config, "w_max", 2.0))
        w_min = float(getattr(config, "w_min", 0.2))
        all_strategies = ["vector", "bm25", "graph", "hierarchical"]
        conn = weight_store._connect() if hasattr(weight_store, "_connect") else None

        def _get_weight(s: str) -> float:
            if conn is not None:
                return float(get_current_weight(conn, query_type, s))
            return float(weight_store.get(query_type, {}).get(s, 1.0))

        def _set_weight(s: str, new_w: float) -> None:
            bounded = max(w_min, min(w_max, float(new_w)))
            if conn is not None:
                old_weight = get_current_weight(conn, query_type, s)
                timestamp = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO routing_weights
                        (query_type, strategy, current_weight, sample_count, cumulative_rating, last_updated)
                    VALUES (?, ?, ?, 0, 0.0, ?)
                    """,
                    (query_type, s, round(bounded, 6), timestamp),
                )
                conn.execute(
                    """
                    UPDATE routing_weights
                    SET current_weight = ?,
                        sample_count = sample_count + 1,
                        last_updated = ?
                    WHERE query_type = ? AND strategy = ?
                    """,
                    (round(bounded, 6), timestamp, query_type, s),
                )

                # Log weight change to weight_history table (additive, fail-safe)
                try:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS weight_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query_type   TEXT,
                            strategy     TEXT,
                            old_weight   REAL,
                            new_weight   REAL,
                            failure_class TEXT,
                            timestamp    TEXT
                        )
                    """)
                    conn.execute("""
                        INSERT INTO weight_history
                        (query_type, strategy, old_weight, new_weight, failure_class, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (query_type, s, round(old_weight, 6), round(bounded, 6), failure_class,
                          timestamp))
                except sqlite3.Error:
                    pass  # Weight history write failure doesn't break attribution
            else:
                weight_store.setdefault(query_type, {})
                weight_store[query_type][s] = bounded

        try:
            if failure_class == "confirmed_success":
                for strategy in strategies_used or []:
                    rank = int((chunk_ranks or {}).get(strategy, 5))
                    delta = effective_eta * (1.0 / (1 + rank))
                    _set_weight(strategy, _get_weight(strategy) + delta)

            elif failure_class == "system_error_retrieval":
                for strategy in strategies_used or []:
                    rank = int((chunk_ranks or {}).get(strategy, 5))
                    delta = effective_eta * (1.0 / (1 + rank)) * 0.5
                    _set_weight(strategy, _get_weight(strategy) - delta)

                unused = [s for s in all_strategies if s not in (strategies_used or [])]
                for strategy in unused:
                    _set_weight(strategy, _get_weight(strategy) + (effective_eta * 0.15))

            elif failure_class == "system_error_generation":
                return weight_store

            elif failure_class == "system_error_hallucination":
                for strategy in strategies_used or []:
                    _set_weight(strategy, _get_weight(strategy) - (effective_eta * 0.2))

            elif failure_class in ("system_error_knowledge_gap", "ambiguous"):
                return weight_store

            elif failure_class == "user_error":
                for strategy in strategies_used or []:
                    _set_weight(strategy, _get_weight(strategy) + (effective_eta * 0.2))

            if conn is not None:
                conn.commit()
            return weight_store
        finally:
            if conn is not None:
                conn.close()
    except Exception as exc:
        log.exception("apply_credit_attribution failed: %s", exc)
        return weight_store


def apply_weekly_decay(weight_store, config):
    """
    Pull all weights toward prior of 1.0:
      w = w * 0.97 + 1.0 * 0.03
    """
    try:
        decay_rate = float(getattr(config, "decay_rate", 0.97))
        prior = float(getattr(config, "w_initial", 1.0))
        query_types = ["factual", "relational", "comparative", "exploratory"]
        strategies = ["vector", "bm25", "graph", "hierarchical"]
        conn = weight_store._connect() if hasattr(weight_store, "_connect") else get_connection()
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            for qt in query_types:
                for st in strategies:
                    w = float(get_current_weight(conn, qt, st))
                    new_w = (w * decay_rate) + (prior * (1.0 - decay_rate))
                    conn.execute(
                        """
                        UPDATE routing_weights
                        SET current_weight = ?,
                            last_updated = ?
                        WHERE query_type = ? AND strategy = ?
                        """,
                        (round(new_w, 6), timestamp, qt, st),
                    )
            conn.commit()
        finally:
            conn.close()
        return weight_store
    except Exception as exc:
        log.exception("apply_weekly_decay failed: %s", exc)
        return weight_store
