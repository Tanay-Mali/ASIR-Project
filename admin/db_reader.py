import sqlite3
from pathlib import Path

FEEDBACK_DB = Path("database/feedback.db")
WEIGHTS_DB  = Path("database/router_weights.db")


def get_connection(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_feedback_summary() -> dict:
    with get_connection(FEEDBACK_DB) as conn:
        rows = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN rating = 'helpful'     THEN 1 ELSE 0 END) as helpful,
                SUM(CASE WHEN rating = 'not_helpful' THEN 1 ELSE 0 END) as not_helpful
            FROM explicit_feedback
        """).fetchone()

        failure_breakdown = conn.execute("""
            SELECT failure_class, COUNT(*) as count
            FROM classification_log
            WHERE failure_class IS NOT NULL
            GROUP BY failure_class
            ORDER BY count DESC
        """).fetchall()

        return {
            "total":             rows["total"]       or 0,
            "helpful":           rows["helpful"]     or 0,
            "not_helpful":       rows["not_helpful"] or 0,
            "failure_breakdown": [
                {"class": r["failure_class"], "count": r["count"]}
                for r in failure_breakdown
            ]
        }


def get_recent_feedback(limit: int = 20) -> list:
    with get_connection(FEEDBACK_DB) as conn:
        rows = conn.execute("""
            SELECT
                ef.query_id,
                ef.rating,
                ef.failure_reason,
                ef.timestamp,
                cl.raw_text        AS query_text,
                cl.query_type,
                cl.failure_class,
                cl.label_confidence,
                cl.grounding_score,
                cl.llm_not_found,
                cl.citation_valid,
                cl.uncited_answer,
                cl.hallucinated_refs,
                cl.retrieval_strategy_used AS strategies_used,
                cl.classified_by           AS classifier_used
            FROM explicit_feedback ef
            LEFT JOIN classification_log cl ON ef.query_id = cl.query_id
            ORDER BY ef.timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_grounding_distribution() -> list:
    with get_connection(FEEDBACK_DB) as conn:
        rows = conn.execute("""
            SELECT
                ROUND(grounding_score, 1) as bucket,
                COUNT(*) as count
            FROM classification_log
            WHERE grounding_score IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()
        return [dict(r) for r in rows]


def get_knowledge_gaps(limit: int = 10) -> list:
    try:
        with get_connection(FEEDBACK_DB) as conn:
            rows = conn.execute("""
                SELECT query_id, query_type, query_text, timestamp
                FROM knowledge_gaps
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_label_stats() -> dict:
    with get_connection(FEEDBACK_DB) as conn:
        total = conn.execute("""
            SELECT COUNT(*) as c
            FROM classification_log
            WHERE true_query_type IS NOT NULL
        """).fetchone()["c"]

        per_class = conn.execute("""
            SELECT true_query_type, COUNT(*) as count
            FROM classification_log
            WHERE true_query_type IS NOT NULL
            GROUP BY true_query_type
        """).fetchall()

        return {
            "total_labels":      total,
            "ready_for_xgboost": total >= 50,
            "per_class": [
                {"type": r["true_query_type"], "count": r["count"]}
                for r in per_class
            ]
        }


def get_current_weights() -> list:
    try:
        with get_connection(WEIGHTS_DB) as conn:
            rows = conn.execute("""
                SELECT query_type, strategy, current_weight AS weight
                FROM routing_weights
                ORDER BY query_type, strategy
            """).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_weight_history(limit: int = 50) -> list:
    try:
        with get_connection(WEIGHTS_DB) as conn:
            rows = conn.execute("""
                SELECT query_type, strategy, old_weight, new_weight,
                       failure_class, timestamp
                FROM weight_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
