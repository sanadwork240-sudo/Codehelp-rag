
"""SQLite helpers: chat logs, tickets, feedback, admin stats."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row

    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                question TEXT,
                standalone_question TEXT,
                answer TEXT,
                top_score REAL,
                answered INTEGER,
                sources TEXT,
                feedback INTEGER
            );

            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                question TEXT,
                status TEXT DEFAULT 'open',
                log_id INTEGER,
                category TEXT DEFAULT 'Other',
                priority TEXT DEFAULT 'Medium',
                summary TEXT DEFAULT '',
                department TEXT DEFAULT 'General Administration'
            );
            """
        )

        # Safely add new columns to existing databases.
        existing = {
            row[1]
            for row in c.execute("PRAGMA table_info(tickets)").fetchall()
        }

        migrations = {
            "category": "TEXT DEFAULT 'Other'",
            "priority": "TEXT DEFAULT 'Medium'",
            "summary": "TEXT DEFAULT ''",
            "department": "TEXT DEFAULT 'General Administration'",
        }

        for column, definition in migrations.items():
            if column not in existing:
                c.execute(
                    f"ALTER TABLE tickets ADD COLUMN {column} {definition}"
                )


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_chat(question, standalone, answer, top_score, answered, sources):
    with conn() as c:
        cur = c.execute(
            """
            INSERT INTO chat_logs (
                ts, question, standalone_question, answer,
                top_score, answered, sources
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                question,
                standalone,
                answer,
                top_score,
                int(answered),
                sources,
            ),
        )

        return cur.lastrowid


def create_ticket(
    question,
    log_id,
    category="Other",
    priority="Medium",
    summary="",
    department="General Administration",
):
    with conn() as c:
        cur = c.execute(
            """
            INSERT INTO tickets (
                ts, question, log_id, category,
                priority, summary, department
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                question,
                log_id,
                category,
                priority,
                summary,
                department,
            ),
        )

        return cur.lastrowid


def save_feedback(log_id, value):
    with conn() as c:
        c.execute(
            "UPDATE chat_logs SET feedback=? WHERE id=?",
            (value, log_id),
        )


def resolve_ticket(ticket_id):
    with conn() as c:
        c.execute(
            "UPDATE tickets SET status='resolved' WHERE id=?",
            (ticket_id,),
        )


def get_tickets(status=None):
    with conn() as c:
        if status:
            rows = c.execute(
                """
                SELECT * FROM tickets
                WHERE status=?
                ORDER BY id DESC
                """,
                (status,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM tickets ORDER BY id DESC"
            ).fetchall()

    return [dict(row) for row in rows]


def top_questions(limit=10):
    with conn() as c:
        rows = c.execute(
            """
            SELECT
                standalone_question AS question,
                COUNT(*) AS times
            FROM chat_logs
            GROUP BY LOWER(standalone_question)
            ORDER BY times DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def disliked_answers(limit=20):
    with conn() as c:
        rows = c.execute(
            """
            SELECT ts, question, answer
            FROM chat_logs
            WHERE feedback=-1
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def stats():
    with conn() as c:
        def count(sql):
            return c.execute(sql).fetchone()[0]

        total = count("SELECT COUNT(*) FROM chat_logs")
        answered = count(
            "SELECT COUNT(*) FROM chat_logs WHERE answered=1"
        )
        up = count(
            "SELECT COUNT(*) FROM chat_logs WHERE feedback=1"
        )
        down = count(
            "SELECT COUNT(*) FROM chat_logs WHERE feedback=-1"
        )
        open_tickets = count(
            "SELECT COUNT(*) FROM tickets WHERE status='open'"
        )

    rated = up + down

    return {
        "total": total,
        "answered": answered,
        "answer_rate": (
            answered / total * 100 if total else 0
        ),
        "open_tickets": open_tickets,
        "satisfaction": (
            up / rated * 100 if rated else None
        ),
        "rated": rated,
    }
