"""
feedback/logger.py  (FIXED VERSION)
=====================================
Writes every classification decision and its feature vector to SQLite.

FIX IN THIS VERSION:
─────────────────────────────────────────────────────────────────────
BUG:  "49 values for 50 columns" error in save_feature_vector()

CAUSE:
  placeholders = ", ".join("?" * (17 + C.EMBEDDING_PCA_DIM))
                                   ^^
  This calculated 17 + 32 = 49 placeholders.
  But the INSERT column list has 18 fixed fields + 32 embedding dims
  = 50 total values.

  The 18 fixed fields are:
    query_id, token_count, sentence_count,
    is_question, is_multi_part,
    has_comparative, comparative_strength,
    has_relational, relational_strength,
    entity_count, has_multiple_entities,
    dominant_entity_type,
    question_word_code, verb_type_code,
    layer1_factual, layer1_relational,
    layer1_comparative, layer1_exploratory
    = 18 fields (the original code counted 17, missing one)

FIX:
  Changed 17 → 18:
  placeholders = ", ".join("?" * (18 + C.EMBEDDING_PCA_DIM))
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

import config.classifier_config as C
from shared.models import ClassifierOutput, FeatureVector

log = logging.getLogger(__name__)


class FeedbackLogger:
    """
    Handles all database writes for the feedback system.
    Instantiate once in QueryClassifier and reuse.
    """

    def __init__(self, db_path: str = C.FEEDBACK_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialise_tables()
        log.info("FeedbackLogger ready. DB: %s", db_path)

    # ──────────────────────────────────────────────────────────────────
    # Table initialisation
    # ──────────────────────────────────────────────────────────────────

    def _initialise_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(self._get_schema_sql())
            self._migrate_schema(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _get_schema_sql() -> str:
        embedding_cols = "\n".join(
            f"    embedding_dim_{i}     REAL DEFAULT 0.0,"
            for i in range(C.EMBEDDING_PCA_DIM)
        )

        return f"""
        CREATE TABLE IF NOT EXISTS classification_log (
            log_id                   TEXT PRIMARY KEY,
            query_id                 TEXT NOT NULL,
            session_id               TEXT,
            raw_text                 TEXT NOT NULL,
            cleaned_text             TEXT,
            query_type               TEXT NOT NULL,
            confidence               REAL,
            classified_by            TEXT,
            all_scores               TEXT,
            llm_reasoning            TEXT,
            llm_raw_response         TEXT,
            xgboost_confidence       REAL,
            preprocessing_ms         REAL,
            classification_ms        REAL,
            total_ms                 REAL,
            needs_decomposition      INTEGER DEFAULT 0,
            sub_queries              TEXT,
            retrieval_strategy_used  TEXT DEFAULT '',
            timestamp                TEXT,
            phase                    TEXT,

            user_rating              INTEGER,
            answer_was_correct       INTEGER,
            feedback_timestamp       TEXT,

            llm_not_found            INTEGER,
            grounding_score          REAL,
            citation_valid           INTEGER,
            uncited_answer           INTEGER,
            hallucinated_refs        TEXT,
            qa_similarity            REAL,
            max_chunk_sim            REAL,
            mean_chunk_sim           REAL,
            max_query_chunk_sim      REAL,
            failure_class            TEXT,
            label_confidence         REAL,
            true_query_type          TEXT,
            chunks_json              TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_classification_log_query_id
            ON classification_log (query_id);
        CREATE INDEX IF NOT EXISTS idx_classification_log_query_type
            ON classification_log (query_type);
        CREATE INDEX IF NOT EXISTS idx_classification_log_timestamp
            ON classification_log (timestamp);

        CREATE TABLE IF NOT EXISTS feature_vectors (
            query_id                 TEXT PRIMARY KEY,
            token_count              INTEGER,
            sentence_count           INTEGER,
            is_question              INTEGER,
            is_multi_part            INTEGER,
            has_comparative          INTEGER,
            comparative_strength     REAL,
            has_relational           INTEGER,
            relational_strength      REAL,
            entity_count             INTEGER,
            has_multiple_entities    INTEGER,
            dominant_entity_type     INTEGER,
            question_word_code       INTEGER,
            verb_type_code           INTEGER,
            layer1_factual           REAL,
            layer1_relational        REAL,
            layer1_comparative       REAL,
            layer1_exploratory       REAL,
            {embedding_cols}
            true_label               TEXT
        );

        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            query_id                 TEXT,
            query_type               TEXT,
            query_text               TEXT,
            timestamp                TEXT
        );

        CREATE TABLE IF NOT EXISTS user_error_queries (
            id                       TEXT PRIMARY KEY,
            query_text               TEXT NOT NULL,
            query_type               TEXT,
            timestamp                TEXT
        );

        CREATE TABLE IF NOT EXISTS explicit_feedback (
            id                       TEXT PRIMARY KEY,
            query_id                 TEXT NOT NULL,
            rating                   TEXT,
            failure_reason           TEXT,
            timestamp                TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id                       TEXT PRIMARY KEY,
            metric_name              TEXT NOT NULL,
            metric_value             REAL NOT NULL,
            timestamp                TEXT
        );
        """

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """
        Additive schema migration for existing databases with legacy schemas.
        """
        desired = {
            "llm_not_found": "INTEGER",
            "grounding_score": "REAL",
            "citation_valid": "INTEGER",
            "uncited_answer": "INTEGER",
            "hallucinated_refs": "TEXT",
            "qa_similarity": "REAL",
            "max_chunk_sim": "REAL",
            "mean_chunk_sim": "REAL",
            "max_query_chunk_sim": "REAL",
            "failure_class": "TEXT",
            "label_confidence": "REAL",
            "true_query_type": "TEXT",
            "chunks_json": "TEXT",
        }
        for col, typ in desired.items():
            self._ensure_column(conn, "classification_log", col, typ)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                query_id TEXT,
                query_type TEXT,
                query_text TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_error_queries (
                id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_type TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS explicit_feedback (
                id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                rating TEXT,
                failure_reason TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TEXT
            )
            """
        )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ──────────────────────────────────────────────────────────────────
    # Write: log a classification result
    # ──────────────────────────────────────────────────────────────────

    def log_classification(self, result: ClassifierOutput) -> str:
        log_id   = str(uuid.uuid4())
        strategy = C.STRATEGY_MAP.get(result.query_type, "unknown")

        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO classification_log (
                    log_id, query_id, session_id, raw_text, cleaned_text,
                    query_type, confidence, classified_by, all_scores,
                    llm_reasoning, llm_raw_response, xgboost_confidence,
                    preprocessing_ms, classification_ms, total_ms,
                    needs_decomposition, sub_queries,
                    retrieval_strategy_used, timestamp, phase
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                log_id,
                result.query_id,
                result.session_id,
                result.raw_text,
                result.cleaned_text,
                result.query_type,
                result.confidence,
                result.classified_by,
                json.dumps(result.all_scores),
                result.llm_reasoning or "",
                result.llm_raw_response or "",
                result.xgboost_confidence,
                result.preprocessing_time_ms,
                result.classification_time_ms,
                result.total_time_ms,
                int(result.needs_decomposition),
                json.dumps(result.sub_queries),
                strategy,
                result.timestamp,
                result.phase,
            ))
            conn.commit()
            log.debug("Logged classification for query_id=%s", result.query_id)
        except sqlite3.IntegrityError:
            log.warning(
                "Duplicate query_id=%s in classification_log. Skipping.",
                result.query_id
            )
        finally:
            conn.close()

        return log_id

    # ──────────────────────────────────────────────────────────────────
    # Write: save feature vector  ← BUG FIX HERE
    # ──────────────────────────────────────────────────────────────────

    def save_feature_vector(self, fv: FeatureVector) -> None:
        """
        Write a FeatureVector to the feature_vectors table.

        FIX: Changed placeholder count from (17 + PCA_DIM) to (18 + PCA_DIM).
        The fixed fields in the INSERT are:
          query_id, token_count, sentence_count,          ← 3
          is_question, is_multi_part,                     ← 2
          has_comparative, comparative_strength,           ← 2
          has_relational, relational_strength,             ← 2
          entity_count, has_multiple_entities,             ← 2
          dominant_entity_type,                            ← 1
          question_word_code, verb_type_code,              ← 2
          layer1_factual, layer1_relational,               ← 2
          layer1_comparative, layer1_exploratory           ← 2
                                                    TOTAL = 18 fixed fields
        + EMBEDDING_PCA_DIM (32) embedding dims
                                                    TOTAL = 50 placeholders
        """
        embedding_col_names = ", ".join(
            f"embedding_dim_{i}" for i in range(C.EMBEDDING_PCA_DIM)
        )

        # ✅ FIXED: was (17 + ...) which gave 49 — now correctly (18 + ...)
        placeholders = ", ".join(["?"] * (18 + C.EMBEDDING_PCA_DIM))

        # Pad/trim embedding to exactly PCA_DIM values
        embedding = (fv.embedding_dims or [])[:C.EMBEDDING_PCA_DIM]
        embedding += [0.0] * max(0, C.EMBEDDING_PCA_DIM - len(embedding))

        values = (
            fv.query_id,               # 1
            fv.token_count,            # 2
            fv.sentence_count,         # 3
            fv.is_question,            # 4
            fv.is_multi_part,          # 5
            fv.has_comparative,        # 6
            fv.comparative_strength,   # 7
            fv.has_relational,         # 8
            fv.relational_strength,    # 9
            fv.entity_count,           # 10
            fv.has_multiple_entities,  # 11
            fv.dominant_entity_type,   # 12
            fv.question_word_code,     # 13
            fv.verb_type_code,         # 14
            fv.layer1_factual,         # 15
            fv.layer1_relational,      # 16
            fv.layer1_comparative,     # 17
            fv.layer1_exploratory,     # 18  ← was being counted as 17
            *embedding,                # 19..50 (32 dims)
        )

        conn = self._connect()
        try:
            conn.execute(f"""
                INSERT OR IGNORE INTO feature_vectors (
                    query_id, token_count, sentence_count,
                    is_question, is_multi_part,
                    has_comparative, comparative_strength,
                    has_relational, relational_strength,
                    entity_count, has_multiple_entities,
                    dominant_entity_type,
                    question_word_code, verb_type_code,
                    layer1_factual, layer1_relational,
                    layer1_comparative, layer1_exploratory,
                    {embedding_col_names}
                ) VALUES ({placeholders})
            """, values)
            conn.commit()
        except Exception as exc:
            log.error(
                "Failed to save feature vector for query_id=%s: %s",
                fv.query_id, exc
            )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Write: record user feedback
    # ──────────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        query_id:           str,
        user_rating:        int,
        answer_was_correct: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            conn.execute("""
                UPDATE classification_log
                SET user_rating        = ?,
                    answer_was_correct = ?,
                    feedback_timestamp = ?
                WHERE query_id = ?
            """, (user_rating, int(answer_was_correct), now, query_id))

            if answer_was_correct:
                row = conn.execute("""
                    SELECT query_type FROM classification_log
                    WHERE query_id = ?
                """, (query_id,)).fetchone()

                if row:
                    conn.execute("""
                        UPDATE feature_vectors
                        SET true_label = ?
                        WHERE query_id = ?
                    """, (row[0], query_id))

            conn.commit()
            log.info(
                "Feedback recorded for query_id=%s | rating=%d | correct=%s",
                query_id, user_rating, answer_was_correct
            )
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────────────
    # Read: statistics for monitoring
    # ──────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM classification_log"
            ).fetchone()[0]

            labelled = conn.execute(
                "SELECT COUNT(*) FROM classification_log WHERE user_rating IS NOT NULL"
            ).fetchone()[0]

            per_type = conn.execute("""
                SELECT query_type, COUNT(*) as n
                FROM classification_log
                WHERE user_rating IS NOT NULL
                GROUP BY query_type
            """).fetchall()

            avg_rating = conn.execute(
                "SELECT AVG(user_rating) FROM classification_log "
                "WHERE user_rating IS NOT NULL"
            ).fetchone()[0]

            by_classifier = conn.execute("""
                SELECT classified_by, COUNT(*) as n
                FROM classification_log
                GROUP BY classified_by
            """).fetchall()

        finally:
            conn.close()

        return {
            "total_queries":    total,
            "labelled_queries": labelled,
            "per_type_counts":  {row[0]: row[1] for row in per_type},
            "average_rating":   round(avg_rating or 0.0, 2),
            "by_classifier":    {row[0]: row[1] for row in by_classifier},
        }

    def log_query_signals(self, signals) -> None:
        """
        Writes QuerySignals payload into classification_log.
        """
        try:
            payload = signals.__dict__ if hasattr(signals, "__dict__") else dict(signals)
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT log_id FROM classification_log WHERE query_id = ? ORDER BY ROWID DESC LIMIT 1",
                    (payload.get("query_id"),),
                ).fetchone()

                if existing:
                    conn.execute(
                        """
                        UPDATE classification_log
                        SET llm_not_found = ?,
                            grounding_score = ?,
                            citation_valid = ?,
                            uncited_answer = ?,
                            hallucinated_refs = ?,
                            qa_similarity = ?,
                            max_chunk_sim = ?,
                            mean_chunk_sim = ?,
                            max_query_chunk_sim = ?,
                            query_type = ?,
                            confidence = ?,
                            classified_by = ?,
                            retrieval_strategy_used = ?,
                            raw_text = ?,
                            session_id = ?,
                            timestamp = ?,
                            phase = ?,
                            llm_raw_response = ?,
                            chunks_json = ?
                        WHERE log_id = ?
                        """,
                        (
                            int(bool(payload.get("llm_not_found", False))),
                            float(payload.get("grounding_score", 0.0)),
                            int(bool(payload.get("citation_valid", True))),
                            int(bool(payload.get("uncited_answer", True))),
                            json.dumps(payload.get("hallucinated_refs", [])),
                            float(payload.get("qa_similarity", 0.0)),
                            float(payload.get("max_chunk_sim", 0.0)),
                            float(payload.get("mean_chunk_sim", 0.0)),
                            float(payload.get("max_query_chunk_sim", 0.0)),
                            payload.get("query_type", "factual"),
                            float(payload.get("classifier_conf", 0.0)),
                            payload.get("classifier_used", ""),
                            json.dumps(payload.get("strategies_used", [])),
                            payload.get("query_text", ""),
                            payload.get("session_id", "web_user"),
                            payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            payload.get("routing_mode", ""),
                            payload.get("answer_text", ""),
                            json.dumps(payload.get("chunks_json", [])),
                            existing[0],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO classification_log (
                            log_id, query_id, session_id, raw_text, cleaned_text, query_type,
                            confidence, classified_by, all_scores, llm_reasoning, llm_raw_response,
                            xgboost_confidence, preprocessing_ms, classification_ms, total_ms,
                            needs_decomposition, sub_queries, retrieval_strategy_used, timestamp, phase,
                            llm_not_found, grounding_score, citation_valid, uncited_answer, hallucinated_refs,
                            qa_similarity, max_chunk_sim, mean_chunk_sim, max_query_chunk_sim,
                            chunks_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            payload.get("query_id"),
                            payload.get("session_id", "web_user"),
                            payload.get("query_text", ""),
                            payload.get("query_text", ""),
                            payload.get("query_type", "factual"),
                            float(payload.get("classifier_conf", 0.0)),
                            payload.get("classifier_used", ""),
                            "{}",
                            "",
                            payload.get("answer_text", ""),
                            None,
                            0.0,
                            0.0,
                            0.0,
                            int(bool(payload.get("followed_by_rephrase", False))),
                            json.dumps([]),
                            json.dumps(payload.get("strategies_used", [])),
                            payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            payload.get("routing_mode", ""),
                            int(bool(payload.get("llm_not_found", False))),
                            float(payload.get("grounding_score", 0.0)),
                            int(bool(payload.get("citation_valid", True))),
                            int(bool(payload.get("uncited_answer", True))),
                            json.dumps(payload.get("hallucinated_refs", [])),
                            float(payload.get("qa_similarity", 0.0)),
                            float(payload.get("max_chunk_sim", 0.0)),
                            float(payload.get("mean_chunk_sim", 0.0)),
                            float(payload.get("max_query_chunk_sim", 0.0)),
                            json.dumps(payload.get("chunks_json", [])),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("log_query_signals failed: %s", exc)

    def log_explicit_feedback(self, query_id, rating, failure_reason, timestamp) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO explicit_feedback (id, query_id, rating, failure_reason, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), query_id, rating, failure_reason, timestamp),
                )
                conn.execute(
                    """
                    UPDATE classification_log
                    SET user_rating = ?, feedback_timestamp = ?
                    WHERE query_id = ?
                    """,
                    (_rating_to_int(rating), timestamp, query_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("log_explicit_feedback failed: %s", exc)

    def log_knowledge_gap(self, query_type, query_text, query_id) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO knowledge_gaps (query_id, query_type, query_text, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query_id, query_type, query_text, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("log_knowledge_gap failed: %s", exc)

    def log_user_error_query(self, query_text, query_type) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO user_error_queries (id, query_text, query_type, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), query_text, query_type, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("log_user_error_query failed: %s", exc)

    def load_query_signals(self, query_id: str):
        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT query_id, session_id, raw_text, query_type, confidence, classified_by,
                           phase, retrieval_strategy_used, llm_not_found, grounding_score,
                           citation_valid, uncited_answer, hallucinated_refs, llm_raw_response,
                           timestamp, qa_similarity, max_chunk_sim, mean_chunk_sim, max_query_chunk_sim, chunks_json
                    FROM classification_log
                    WHERE query_id = ?
                    ORDER BY ROWID DESC LIMIT 1
                    """,
                    (query_id,),
                ).fetchone()
            finally:
                conn.close()
            if not row:
                return None
            strategies = []
            try:
                strategies = json.loads(row["retrieval_strategy_used"] or "[]")
                if isinstance(strategies, str):
                    strategies = [strategies]
            except Exception:
                strategies = []
            return SimpleNamespace(
                query_id=row["query_id"],
                session_id=row["session_id"] or "web_user",
                query_text=row["raw_text"] or "",
                answer_text=row["llm_raw_response"] or "",
                query_type=row["query_type"] or "factual",
                classifier_conf=float(row["confidence"] or 0.0),
                classifier_used=row["classified_by"] or "",
                routing_mode=row["phase"] or "",
                strategies_used=strategies,
                fallback_triggered=False,
                llm_not_found=bool(row["llm_not_found"] or 0),
                grounding_score=float(row["grounding_score"] or 0.0),
                citation_valid=bool(row["citation_valid"] if row["citation_valid"] is not None else 1),
                uncited_answer=bool(row["uncited_answer"] or 0),
                hallucinated_refs=json.loads(row["hallucinated_refs"] or "[]"),
                rapid_rephrase=False,
                dwell_time_seconds=0.0,
                chunk_ranks={},
                qa_similarity=float(row["qa_similarity"] or 0.0),
                max_chunk_sim=float(row["max_chunk_sim"] or 0.0),
                mean_chunk_sim=float(row["mean_chunk_sim"] or 0.0),
                max_query_chunk_sim=float(row["max_query_chunk_sim"] or 0.0),
                chunks_json=json.loads(row["chunks_json"] or "[]"),
            )
        except Exception as exc:
            log.exception("load_query_signals failed: %s", exc)
            return None

    def load_chunks_for_query(self, query_id: str) -> list:
        """
        Returns chunk objects with `.content` from stored chunks_json.
        """
        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT chunks_json
                    FROM classification_log
                    WHERE query_id = ?
                    ORDER BY ROWID DESC LIMIT 1
                    """,
                    (query_id,),
                ).fetchone()
            finally:
                conn.close()

            if not row or not row["chunks_json"]:
                return []

            chunks = json.loads(row["chunks_json"])
            out = []
            for c in chunks:
                out.append(
                    SimpleNamespace(
                        content=c.get("content", ""),
                        source=c.get("source", "unknown"),
                        score=float(c.get("score", 0.0)),
                        strategy=c.get("strategy", ""),
                    )
                )
            return out
        except Exception as exc:
            log.exception("load_chunks_for_query failed: %s", exc)
            return []

    def save_training_label(self, label_payload: dict) -> None:
        """
        Persists generated label and confidence into existing tables.
        """
        try:
            query_text = label_payload.get("query_text", "")
            true_label = label_payload.get("label") or label_payload.get("true_query_type")
            if not true_label:
                return

            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT query_id FROM classification_log WHERE raw_text = ? ORDER BY ROWID DESC LIMIT 1",
                    (query_text,),
                ).fetchone()
                if not row:
                    return
                query_id = row[0]
                conn.execute(
                    """
                    UPDATE classification_log
                    SET true_query_type = ?, label_confidence = ?, failure_class = ?
                    WHERE query_id = ?
                    """,
                    (
                        true_label,
                        float(label_payload.get("label_confidence", 0.0)),
                        label_payload.get("failure_class"),
                        query_id,
                    ),
                )
                conn.execute(
                    "UPDATE feature_vectors SET true_label = ? WHERE query_id = ?",
                    (true_label, query_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("save_training_label failed: %s", exc)

    def count_training_labels(self) -> int:
        try:
            conn = self._connect()
            try:
                return conn.execute(
                    "SELECT COUNT(*) FROM classification_log WHERE true_query_type IS NOT NULL"
                ).fetchone()[0]
            finally:
                conn.close()
        except Exception:
            return 0

    def save_model_f1(self, f1_score: float) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO model_metrics (id, metric_name, metric_value, timestamp) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), "xgboost_f1", float(f1_score), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("save_model_f1 failed: %s", exc)

    def load_model_f1(self) -> float:
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT metric_value FROM model_metrics WHERE metric_name = 'xgboost_f1' ORDER BY ROWID DESC LIMIT 1"
                ).fetchone()
                return float(row[0]) if row else 0.0
            finally:
                conn.close()
        except Exception:
            return 0.0

    def update_feedback_diagnostics(
        self,
        query_id: str,
        failure_class: str,
        label_confidence: float,
        qa_similarity: float,
        max_chunk_sim: float,
        mean_chunk_sim: float,
        true_query_type: str | None = None,
    ) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE classification_log
                    SET failure_class = ?,
                        label_confidence = ?,
                        qa_similarity = ?,
                        max_chunk_sim = ?,
                        mean_chunk_sim = ?,
                        true_query_type = COALESCE(?, true_query_type)
                    WHERE query_id = ?
                    """,
                    (
                        failure_class,
                        float(label_confidence),
                        float(qa_similarity),
                        float(max_chunk_sim),
                        float(mean_chunk_sim),
                        true_query_type,
                        query_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.exception("update_feedback_diagnostics failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)


def _rating_to_int(rating) -> int:
    if isinstance(rating, (int, float)):
        return int(rating)
    mapping = {"helpful": 5, "partially": 3, "not_helpful": 1}
    return mapping.get(str(rating), 0)


_DEFAULT_LOGGER: FeedbackLogger | None = None


def _get_default_logger() -> FeedbackLogger:
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = FeedbackLogger()
    return _DEFAULT_LOGGER


def log_query_signals(signals) -> None:
    _get_default_logger().log_query_signals(signals)


def log_explicit_feedback(query_id, rating, failure_reason, timestamp) -> None:
    _get_default_logger().log_explicit_feedback(query_id, rating, failure_reason, timestamp)


def log_knowledge_gap(query_type, query_text, query_id) -> None:
    _get_default_logger().log_knowledge_gap(query_type, query_text, query_id)


def log_user_error_query(query_text, query_type) -> None:
    _get_default_logger().log_user_error_query(query_text, query_type)


def load_query_signals(query_id):
    return _get_default_logger().load_query_signals(query_id)


def load_chunks_for_query(query_id):
    return _get_default_logger().load_chunks_for_query(query_id)


def save_training_label(label_payload: dict) -> None:
    _get_default_logger().save_training_label(label_payload)


def count_training_labels() -> int:
    return _get_default_logger().count_training_labels()


def load_model_f1() -> float:
    return _get_default_logger().load_model_f1()


def save_model_f1(f1_score: float) -> None:
    _get_default_logger().save_model_f1(f1_score)


def update_feedback_diagnostics(
    query_id: str,
    failure_class: str,
    label_confidence: float,
    qa_similarity: float,
    max_chunk_sim: float,
    mean_chunk_sim: float,
    true_query_type: str | None = None,
) -> None:
    _get_default_logger().update_feedback_diagnostics(
        query_id=query_id,
        failure_class=failure_class,
        label_confidence=label_confidence,
        qa_similarity=qa_similarity,
        max_chunk_sim=max_chunk_sim,
        mean_chunk_sim=mean_chunk_sim,
        true_query_type=true_query_type,
    )
