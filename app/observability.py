"""
Bonus Option A - Observability.

Logs every query (question, retrieved chunk ids/scores, answer, latency)
to a local SQLite database so query patterns and latency can be reviewed
without an external logging stack.
"""
import sqlite3
import json
import time
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "query_logs.db")


def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            question TEXT,
            answer TEXT,
            sources TEXT,
            retrieved_chunks TEXT,
            latency_ms REAL
        )
        """
    )
    conn.commit()
    conn.close()


@contextmanager
def _connect():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def log_query(question: str, answer: str, sources: list, retrieved_chunks: list, latency_ms: float):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO query_logs (timestamp, question, answer, sources, retrieved_chunks, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                question,
                answer,
                json.dumps(sources),
                json.dumps(retrieved_chunks),
                latency_ms,
            ),
        )
        conn.commit()


def fetch_recent_logs(limit: int = 50):
    with _connect() as conn:
        cur = conn.execute(
            "SELECT timestamp, question, answer, sources, retrieved_chunks, latency_ms "
            "FROM query_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "timestamp": r[0],
            "question": r[1],
            "answer": r[2],
            "sources": json.loads(r[3]),
            "retrieved_chunks": json.loads(r[4]),
            "latency_ms": r[5],
        }
        for r in rows
    ]
