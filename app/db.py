"""SQLite helpers for logging queries and their telemetry (cost, latency).

Kept intentionally small — SQLite only, per MVP scope (no Postgres, no
decision-history features beyond simple logging).
"""

import sqlite3

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    question TEXT NOT NULL,
    decision TEXT,
    confidence REAL,
    latency_ms REAL,
    cost_usd REAL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
