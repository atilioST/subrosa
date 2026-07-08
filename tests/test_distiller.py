"""Tests for the distillation pipeline's scheduling mechanics (no Haiku calls)."""

import asyncio
import pickle

import numpy as np
import pytest
import pytest_asyncio

from subrosa.distiller import Distiller, _call_haiku
from subrosa.store import Store


@pytest_asyncio.fixture
async def store():
    s = Store(db_path=":memory:")
    await s.initialize()
    # Avoid loading the sentence-transformers model in tests
    s._embeddings.embed = lambda text: pickle.dumps(np.zeros(3, dtype=np.float32))
    yield s
    await s.close()


@pytest.fixture
def distiller(store):
    d = Distiller(store=store, model="haiku")
    d.calls = []

    async def fake_distill(events):
        d.calls.append([e["id"] for e in events])
        return len(events)  # pretend one item per event

    d.distill = fake_distill
    return d


async def test_run_advances_cursor_and_is_idempotent(store, distiller):
    for i in range(3):
        await store.log_event(source="test", event_type="message", summary=f"e{i}")

    written = await distiller.run()
    assert written == 3
    assert len(distiller.calls) == 1

    cursor = await store.get_meta("last_distilled_event_id")
    assert cursor == "3"
    assert await store.get_meta("last_distilled_at") is not None

    # Second run: nothing new, no distill call
    written = await distiller.run()
    assert written == 0
    assert len(distiller.calls) == 1

    # New event → only that one is processed
    await store.log_event(source="test", event_type="message", summary="e3")
    written = await distiller.run()
    assert written == 1
    assert distiller.calls[-1] == [4]


async def test_run_paginates_past_fetch_limit(store, distiller):
    for i in range(7):
        await store.log_event(source="test", event_type="message", summary=f"e{i}")

    orig = store.get_events_after_id

    async def small_batches(after_id=None, limit=500):
        return await orig(after_id, limit=3)

    store.get_events_after_id = small_batches

    written = await distiller.run()
    assert written == 7
    assert [len(c) for c in distiller.calls] == [3, 3, 1]
    assert await store.get_meta("last_distilled_event_id") == "7"


async def test_first_run_limits_to_24h_window(store, distiller):
    # Backdate one event beyond the window, add one fresh event
    await store._db.execute(
        "INSERT INTO events (timestamp, source, event_type, summary) "
        "VALUES (datetime('now', '-3 days'), 'test', 'message', 'old')"
    )
    await store._db.commit()
    await store.log_event(source="test", event_type="message", summary="fresh")

    written = await distiller.run()
    assert written == 1  # only the fresh event; backfill is an explicit path


async def test_run_logs_diagnostic(store, distiller):
    await store.log_event(source="test", event_type="message", summary="e")
    await distiller.run()
    diags = await store.get_recent_diagnostics(hours=1)
    assert any(d["component"] == "distiller" for d in diags)


async def test_call_haiku_times_out(monkeypatch):
    import subrosa.distiller as d

    class FakeProc:
        returncode = 0

        def __init__(self):
            self.killed = False

        async def communicate(self):
            if self.killed:
                return b"", b""
            await asyncio.sleep(10)

        def kill(self):
            self.killed = True

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(d, "_HAIKU_TIMEOUT", 0.05)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await _call_haiku("prompt", "haiku")
    assert result == []
