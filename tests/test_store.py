"""Tests for the unified store."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from subrosa1.store import Store


@pytest.fixture
def store():
    """Create a temporary store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = Store(db_path=db_path)
        asyncio.get_event_loop().run_until_complete(s.initialize())
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestEvents:
    def test_log_and_retrieve(self, store):
        event_id = run(store.log_event("test", "msg", summary="hello"))
        assert event_id > 0

        events = run(store.get_recent_events(hours=1))
        assert len(events) >= 1
        assert events[0]["summary"] == "hello"

    def test_filter_by_source(self, store):
        run(store.log_event("slack", "message", summary="s1"))
        run(store.log_event("jira", "update", summary="j1"))
        slack_events = run(store.get_recent_events(hours=1, source="slack"))
        assert all(e["source"] == "slack" for e in slack_events)


class TestConversations:
    def test_create_and_retrieve(self, store):
        conv = run(store.get_or_create_conversation(12345))
        assert conv["telegram_chat_id"] == 12345
        assert conv["id"] > 0

        conv2 = run(store.get_or_create_conversation(12345))
        assert conv2["id"] == conv["id"]

    def test_update(self, store):
        conv = run(store.get_or_create_conversation(99999))
        run(store.update_conversation(conv["id"], agent_session_id="sess-1"))


class TestDiagnostics:
    def test_log_and_retrieve(self, store):
        run(store.log_diagnostic("poller", "started"))
        diags = run(store.get_recent_diagnostics(hours=1))
        assert len(diags) >= 1
        assert diags[0]["component"] == "poller"


class TestMemory:
    def test_save_and_get(self, store):
        mem = run(store.save_memory(
            subject="Brock",
            content="PM for Compass",
            topic_type="person",
            tags=["compass"],
        ))
        assert mem["subject"] == "Brock"
        assert mem["topic_type"] == "person"
        assert "compass" in mem["tags"]

        retrieved = run(store.get_memory(mem["id"]))
        assert retrieved is not None
        assert retrieved["subject"] == "Brock"

    def test_list_memories(self, store):
        run(store.save_memory(subject="Alice", content="Engineer", topic_type="person"))
        run(store.save_memory(subject="ProjectX", content="New feature", topic_type="project"))

        all_mems = run(store.list_memories())
        assert len(all_mems) == 2

        people = run(store.list_memories(topic_type="person"))
        assert len(people) == 1

    def test_delete_memory(self, store):
        mem = run(store.save_memory(subject="ToDelete", content="temp", topic_type="fact"))
        run(store.delete_memory(mem["id"]))
        active = run(store.list_memories())
        assert all(m["subject"] != "ToDelete" for m in active)

    def test_search_fts(self, store):
        run(store.save_memory(subject="Sprint Review", content="Weekly sprint review meeting", topic_type="fact"))
        results = run(store.search_fts("sprint"))
        assert len(results) >= 1
        assert results[0][0]["subject"] == "Sprint Review"

    def test_get_by_topics(self, store):
        run(store.save_memory(subject="Compass API", content="REST API", topic_type="project", tags=["compass"]))
        results = run(store.get_memories_by_topics(["compass"]))
        assert len(results) >= 1

    def test_memory_count(self, store):
        run(store.save_memory(subject="Counted", content="test", topic_type="fact"))
        count = run(store.memory_count())
        assert count >= 1


class TestTasks:
    def test_create_and_get(self, store):
        task_id = run(store.create_task(
            title="Fix bug",
            task_type="discrete",
            priority="high",
            due_date="2026-03-01",
        ))
        assert task_id > 0

        task = run(store.get_task(task_id))
        assert task["title"] == "Fix bug"
        assert task["priority"] == "high"

    def test_complete_discrete(self, store):
        task_id = run(store.create_task(title="Done task", task_type="discrete"))
        result = run(store.complete_task(task_id))
        assert result is True

        task = run(store.get_task(task_id))
        assert task["status"] == "completed"

    def test_complete_recurring(self, store):
        task_id = run(store.create_task(
            title="Check costs",
            task_type="recurring",
            recurrence_pattern="weekly",
        ))
        run(store.complete_task(task_id))
        task = run(store.get_task(task_id))
        assert task["status"] == "pending"  # Recurring resets to pending
        assert task["next_surface_at"] is not None

    def test_get_active_tasks(self, store):
        run(store.create_task(title="Active", task_type="discrete"))
        tasks = run(store.get_active_tasks())
        assert len(tasks) >= 1

    def test_delete_task(self, store):
        task_id = run(store.create_task(title="Delete me", task_type="discrete"))
        assert run(store.delete_task(task_id)) is True
        assert run(store.get_task(task_id)) is None
