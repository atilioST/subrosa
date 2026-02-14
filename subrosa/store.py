"""Unified async SQLite store — events, memory, tasks, embeddings."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import aiosqlite
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DB_PATH = Path("~/.subrosa/subrosa.db").expanduser()

# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    importance TEXT NOT NULL DEFAULT 'normal',
    langfuse_trace_id TEXT,
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_chat_id INTEGER NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_message_at TEXT,
    agent_session_id TEXT,
    topic TEXT
);

-- Diagnostics
CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    component TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT,
    level TEXT NOT NULL DEFAULT 'info'
);
CREATE INDEX IF NOT EXISTS idx_diag_ts ON diagnostics(timestamp);

-- Memory: topics + facts + tags + attributes
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    topic_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_topics_type ON topics(topic_type);
CREATE INDEX IF NOT EXISTS idx_topics_active ON topics(active);

CREATE TABLE IF NOT EXISTS topic_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_topic_facts_topic ON topic_facts(topic_id);

CREATE TABLE IF NOT EXISTS topic_tags (
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (topic_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_topic_tags_tag ON topic_tags(tag);

CREATE TABLE IF NOT EXISTS topic_attributes (
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (topic_id, key)
);

CREATE TABLE IF NOT EXISTS topic_embeddings (
    topic_id INTEGER PRIMARY KEY REFERENCES topics(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Tasks
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    task_type TEXT NOT NULL CHECK(task_type IN ('discrete', 'recurring')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'snoozed')),
    priority TEXT CHECK(priority IN ('low', 'medium', 'high', 'urgent')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    due_date TEXT,
    completed_at TEXT,
    snoozed_until TEXT,
    recurrence_pattern TEXT,
    last_surfaced_at TEXT,
    next_surface_at TEXT,
    context TEXT,
    tags TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date) WHERE due_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_surface ON tasks(next_surface_at) WHERE next_surface_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    changed_fields TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_history_task ON task_history(task_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(
    name, facts, content='', contentless_delete=1, tokenize='porter'
);
"""


# ── Embedding helpers ───────────────────────────────────────────────────────

class EmbeddingService:
    """Singleton wrapper for sentence-transformers model."""

    _instance: EmbeddingService | None = None
    _model: SentenceTransformer | None = None
    _model_name: str = "all-MiniLM-L6-v2"

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device="cpu")
            logger.info("Embedding model loaded")
        return self._model

    def embed(self, text: str) -> bytes:
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return pickle.dumps(embedding)

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        return pickle.loads(blob)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Store ───────────────────────────────────────────────────────────────────

class Store:
    """Unified async SQLite store for all Subrosa state."""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or _DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._embeddings = EmbeddingService()

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._migrate_fts()
        await self._db.executescript(_FTS_SCHEMA)
        await self._db.commit()
        logger.info("Store initialized: %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _migrate_fts(self) -> None:
        cursor = await self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='topics_fts'"
        )
        row = await cursor.fetchone()
        if row and "contentless_delete" not in row[0]:
            logger.info("Migrating topics_fts to contentless_delete=1")
            await self._db.execute("DROP TABLE topics_fts")
            await self._db.executescript(_FTS_SCHEMA)
            topics_cursor = await self._db.execute("SELECT id FROM topics WHERE active = 1")
            topics = await topics_cursor.fetchall()
            for t in topics:
                await self._update_topic_fts(t["id"])
            await self._db.commit()

    # ── Events ──────────────────────────────────────────────────────────────

    async def log_event(
        self,
        source: str,
        event_type: str,
        summary: str = "",
        content: str = "",
        importance: str = "normal",
        langfuse_trace_id: str = "",
        session_id: str = "",
    ) -> int:
        cursor = await self._db.execute(
            """INSERT INTO events (source, event_type, summary, content, importance,
               langfuse_trace_id, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source, event_type, summary, content, importance, langfuse_trace_id, session_id),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_recent_events(
        self, hours: int = 24, source: str | None = None, limit: int = 20
    ) -> list[dict]:
        query = "SELECT * FROM events WHERE timestamp > datetime('now', ?)"
        params: list[Any] = [f"-{hours} hours"]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Conversations ───────────────────────────────────────────────────────

    async def get_or_create_conversation(self, chat_id: int) -> dict:
        cursor = await self._db.execute(
            "SELECT * FROM conversations WHERE telegram_chat_id = ? ORDER BY started_at DESC LIMIT 1",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)

        cursor = await self._db.execute(
            "INSERT INTO conversations (telegram_chat_id) VALUES (?)", (chat_id,)
        )
        await self._db.commit()
        return {
            "id": cursor.lastrowid,
            "telegram_chat_id": chat_id,
            "started_at": datetime.now(UTC).isoformat(),
            "last_message_at": None,
            "agent_session_id": "",
            "topic": "",
        }

    async def clear_all_sessions(self) -> int:
        """Clear all agent session IDs. Called on startup since old CLI subprocesses are dead."""
        cursor = await self._db.execute(
            "UPDATE conversations SET agent_session_id = '' WHERE agent_session_id != ''"
        )
        await self._db.commit()
        return cursor.rowcount

    async def update_conversation(
        self, conversation_id: int, agent_session_id: str | None = None, topic: str | None = None
    ) -> None:
        updates = ["last_message_at = datetime('now')"]
        params: list[Any] = []
        if agent_session_id is not None:
            updates.append("agent_session_id = ?")
            params.append(agent_session_id)
        if topic is not None:
            updates.append("topic = ?")
            params.append(topic)
        params.append(conversation_id)
        await self._db.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", params
        )
        await self._db.commit()

    # ── Diagnostics ─────────────────────────────────────────────────────────

    async def log_diagnostic(
        self, component: str, event: str, detail: str | None = None, level: str = "info"
    ) -> None:
        if not self._db:
            return
        try:
            await self._db.execute(
                "INSERT INTO diagnostics (component, event, detail, level) VALUES (?, ?, ?, ?)",
                (component, event, detail, level),
            )
            await self._db.commit()
        except Exception:
            logger.debug("Failed to log diagnostic", exc_info=True)

    async def get_recent_diagnostics(
        self, hours: int = 24, level: str | None = None, limit: int = 20
    ) -> list[dict]:
        if not self._db:
            return []
        query = "SELECT * FROM diagnostics WHERE timestamp > datetime('now', ?)"
        params: list[Any] = [f"-{hours} hours"]
        if level:
            query += " AND level = ?"
            params.append(level)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Memory ──────────────────────────────────────────────────────────────

    async def _load_facts(self, topic_id: int) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT fact, source, confidence, created_at FROM topic_facts WHERE topic_id = ? ORDER BY created_at DESC",
            (topic_id,),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def _load_tags(self, topic_id: int) -> list[str]:
        cursor = await self._db.execute(
            "SELECT tag FROM topic_tags WHERE topic_id = ?", (topic_id,)
        )
        return [r["tag"] for r in await cursor.fetchall()]

    async def _load_attributes(self, topic_id: int) -> dict[str, str]:
        cursor = await self._db.execute(
            "SELECT key, value FROM topic_attributes WHERE topic_id = ?", (topic_id,)
        )
        return {r["key"]: r["value"] for r in await cursor.fetchall()}

    async def _enrich_topic(self, row: aiosqlite.Row) -> dict:
        """Load full topic data as a dict."""
        topic_id = row["id"]
        facts = await self._load_facts(topic_id)
        tags = await self._load_tags(topic_id)
        attributes = await self._load_attributes(topic_id)
        fact_lines = [f"- {f['fact']}" for f in facts]
        return {
            "id": row["id"],
            "subject": row["name"],
            "topic_type": row["topic_type"],
            "content": "\n".join(fact_lines) if fact_lines else "(no facts)",
            "facts": facts,
            "tags": tags,
            "attributes": attributes,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "active": bool(row["active"]),
        }

    async def _update_topic_embedding(self, topic_id: int) -> None:
        cursor = await self._db.execute("SELECT name FROM topics WHERE id = ?", (topic_id,))
        row = await cursor.fetchone()
        if not row:
            return
        facts = await self._load_facts(topic_id)
        fact_texts = [f["fact"] for f in facts]
        text = f"{row['name']}: " + " | ".join(fact_texts[:10])
        blob = self._embeddings.embed(text)
        await self._db.execute(
            "INSERT OR REPLACE INTO topic_embeddings (topic_id, embedding, updated_at) VALUES (?, ?, datetime('now'))",
            (topic_id, blob),
        )

    async def _update_topic_fts(self, topic_id: int) -> None:
        cursor = await self._db.execute("SELECT name FROM topics WHERE id = ?", (topic_id,))
        row = await cursor.fetchone()
        if not row:
            return
        facts = await self._load_facts(topic_id)
        facts_text = " ".join(f["fact"] for f in facts)
        await self._db.execute("DELETE FROM topics_fts WHERE rowid = ?", (topic_id,))
        await self._db.execute(
            "INSERT INTO topics_fts (rowid, name, facts) VALUES (?, ?, ?)",
            (topic_id, row["name"], facts_text),
        )

    async def save_memory(
        self,
        subject: str,
        content: str,
        topic_type: str = "fact",
        source: str = "user",
        confidence: float = 1.0,
        tags: list[str] | None = None,
        attributes: dict[str, str] | None = None,
    ) -> dict:
        """Save a memory (create/update topic + add fact)."""
        cursor = await self._db.execute(
            "SELECT id FROM topics WHERE LOWER(name) = LOWER(?) AND active = 1",
            (subject,),
        )
        row = await cursor.fetchone()

        if row:
            topic_id = row["id"]
            await self._db.execute(
                "INSERT INTO topic_facts (topic_id, fact, source, confidence) VALUES (?, ?, ?, ?)",
                (topic_id, content, source, confidence),
            )
            await self._db.execute(
                "UPDATE topics SET updated_at = datetime('now') WHERE id = ?", (topic_id,)
            )
        else:
            cursor = await self._db.execute(
                "INSERT INTO topics (name, topic_type) VALUES (?, ?)", (subject, topic_type)
            )
            topic_id = cursor.lastrowid
            await self._db.execute(
                "INSERT INTO topic_facts (topic_id, fact, source, confidence) VALUES (?, ?, ?, ?)",
                (topic_id, content, source, confidence),
            )
            for tag in (tags or []):
                await self._db.execute(
                    "INSERT OR IGNORE INTO topic_tags (topic_id, tag) VALUES (?, ?)",
                    (topic_id, tag.lower()),
                )
            for k, v in (attributes or {}).items():
                await self._db.execute(
                    "INSERT OR REPLACE INTO topic_attributes (topic_id, key, value) VALUES (?, ?, ?)",
                    (topic_id, k, v),
                )

        await self._update_topic_embedding(topic_id)
        await self._update_topic_fts(topic_id)
        await self._db.commit()

        cursor = await self._db.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
        topic_row = await cursor.fetchone()
        result = await self._enrich_topic(topic_row)
        logger.info("Saved memory for topic '%s' (id=%d)", subject, topic_id)
        return result

    async def get_memory(self, memory_id: int) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM topics WHERE id = ?", (memory_id,))
        row = await cursor.fetchone()
        return await self._enrich_topic(row) if row else None

    async def delete_memory(self, memory_id: int) -> None:
        await self._db.execute(
            "UPDATE topics SET active = 0, updated_at = datetime('now') WHERE id = ?",
            (memory_id,),
        )
        await self._db.commit()

    async def list_memories(
        self, topic_type: str | None = None, active_only: bool = True, limit: int = 50
    ) -> list[dict]:
        query = "SELECT * FROM topics"
        conditions: list[str] = []
        params: list[Any] = []
        if active_only:
            conditions.append("active = 1")
        if topic_type:
            conditions.append("topic_type = ?")
            params.append(topic_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [await self._enrich_topic(r) for r in rows]

    async def search_fts(self, query_text: str, limit: int = 20) -> list[tuple[dict, float]]:
        cursor = await self._db.execute(
            """SELECT t.*, rank FROM topics_fts fts
               JOIN topics t ON t.id = fts.rowid
               WHERE topics_fts MATCH ? AND t.active = 1
               ORDER BY rank LIMIT ?""",
            (query_text, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            m = await self._enrich_topic(row)
            results.append((m, row["rank"]))
        return results

    async def semantic_search(
        self, query_text: str, limit: int = 20, threshold: float = 0.3
    ) -> list[tuple[dict, float]]:
        query_blob = self._embeddings.embed(query_text)
        query_vec = EmbeddingService.deserialize(query_blob)

        cursor = await self._db.execute(
            """SELECT t.*, e.embedding FROM topics t
               JOIN topic_embeddings e ON e.topic_id = t.id
               WHERE t.active = 1"""
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            topic_vec = EmbeddingService.deserialize(row["embedding"])
            sim = EmbeddingService.cosine_similarity(query_vec, topic_vec)
            if sim >= threshold:
                m = await self._enrich_topic(row)
                results.append((m, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_memories_by_topics(self, topics: list[str], limit: int = 20) -> list[dict]:
        if not topics:
            return []
        placeholders = ", ".join("?" for _ in topics)
        cursor = await self._db.execute(
            f"""SELECT DISTINCT t.* FROM topics t
                JOIN topic_tags tt ON tt.topic_id = t.id
                WHERE tt.tag IN ({placeholders}) AND t.active = 1
                ORDER BY t.updated_at DESC LIMIT ?""",
            [t.lower() for t in topics] + [limit],
        )
        rows = await cursor.fetchall()
        return [await self._enrich_topic(r) for r in rows]

    async def get_memories_by_subject(self, subject: str, limit: int = 10) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM topics WHERE LOWER(name) = LOWER(?) AND active = 1 ORDER BY updated_at DESC LIMIT ?",
            (subject, limit),
        )
        rows = await cursor.fetchall()
        return [await self._enrich_topic(r) for r in rows]

    async def get_all_topics(self) -> list[str]:
        cursor = await self._db.execute(
            """SELECT DISTINCT tt.tag FROM topic_tags tt
               JOIN topics t ON t.id = tt.topic_id
               WHERE t.active = 1 ORDER BY tt.tag"""
        )
        return [r["tag"] for r in await cursor.fetchall()]

    async def memory_count(self, active_only: bool = True) -> int:
        query = "SELECT COUNT(*) as cnt FROM topics"
        if active_only:
            query += " WHERE active = 1"
        cursor = await self._db.execute(query)
        row = await cursor.fetchone()
        return row["cnt"]

    # ── Tasks ───────────────────────────────────────────────────────────────

    def _calculate_next_surface(self, pattern: str, from_date: str) -> str:
        base = datetime.fromisoformat(from_date)
        deltas = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "biweekly": timedelta(weeks=2),
            "monthly": timedelta(days=30),
            "quarterly": timedelta(days=90),
        }
        delta = deltas.get(pattern)
        if not delta:
            raise ValueError(f"Unknown recurrence pattern: {pattern}")
        return (base + delta).isoformat()

    async def create_task(
        self,
        title: str,
        task_type: str = "discrete",
        description: str | None = None,
        priority: str | None = None,
        due_date: str | None = None,
        recurrence_pattern: str | None = None,
        context: dict | None = None,
        tags: str | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        next_surface = None
        if task_type == "recurring" and recurrence_pattern:
            next_surface = self._calculate_next_surface(recurrence_pattern, now)

        cursor = await self._db.execute(
            """INSERT INTO tasks (title, description, task_type, status, priority,
               created_at, due_date, recurrence_pattern, next_surface_at, context, tags, updated_at)
               VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, description, task_type, priority, now, due_date,
             recurrence_pattern, next_surface,
             json.dumps(context) if context else None, tags, now),
        )
        task_id = cursor.lastrowid
        await self._db.execute(
            "INSERT INTO task_history (task_id, action) VALUES (?, 'created')", (task_id,)
        )
        await self._db.commit()
        return task_id

    async def get_task(self, task_id: int) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("context"):
            d["context"] = json.loads(d["context"])
        return d

    async def complete_task(self, task_id: int) -> bool:
        task = await self.get_task(task_id)
        if not task:
            return False
        now = datetime.now(UTC).isoformat()

        if task["task_type"] == "recurring" and task.get("recurrence_pattern"):
            next_surface = self._calculate_next_surface(task["recurrence_pattern"], now)
            await self._db.execute(
                "UPDATE tasks SET status = 'pending', last_surfaced_at = ?, next_surface_at = ?, updated_at = ? WHERE id = ?",
                (now, next_surface, now, task_id),
            )
        else:
            await self._db.execute(
                "UPDATE tasks SET status = 'completed', completed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, task_id),
            )

        await self._db.execute(
            "INSERT INTO task_history (task_id, action) VALUES (?, 'completed')", (task_id,)
        )
        await self._db.commit()
        return True

    async def get_active_tasks(self) -> list[dict]:
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            """SELECT * FROM tasks
               WHERE status != 'completed'
               AND (status != 'snoozed' OR snoozed_until <= ?)
               ORDER BY
                   CASE priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3 WHEN 'low' THEN 4 ELSE 5 END,
                   due_date ASC, next_surface_at ASC""",
            (now,),
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("context"):
                d["context"] = json.loads(d["context"])
            results.append(d)
        return results

    async def get_tasks_due_soon(self, days: int = 7) -> list[dict]:
        threshold = (datetime.now(UTC) + timedelta(days=days)).isoformat()
        cursor = await self._db.execute(
            """SELECT * FROM tasks WHERE task_type = 'discrete'
               AND status NOT IN ('completed', 'snoozed')
               AND due_date IS NOT NULL AND due_date <= ?
               ORDER BY due_date ASC""",
            (threshold,),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_tasks_to_surface(self) -> list[dict]:
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            """SELECT * FROM tasks WHERE task_type = 'recurring'
               AND status NOT IN ('completed', 'snoozed')
               AND next_surface_at <= ?
               ORDER BY priority DESC, next_surface_at ASC""",
            (now,),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def delete_task(self, task_id: int) -> bool:
        cursor = await self._db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await self._db.commit()
        return cursor.rowcount > 0
