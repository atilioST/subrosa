"""Tests for memory extraction."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from subrosa.memory import (
    detect_explicit_memory_request,
    create_explicit_memory,
    _detect_type,
    _extract_subject,
    _extract_topics,
)
from subrosa.store import Store


def test_detect_explicit_remember():
    assert detect_explicit_memory_request("remember that Brock is PM for Compass") == "Brock is PM for Compass"
    assert detect_explicit_memory_request("note that sprint ends Friday") == "sprint ends Friday"
    assert detect_explicit_memory_request("keep in mind that Walter is on PTO") == "Walter is on PTO"
    assert detect_explicit_memory_request("fyi: new hire starts Monday") == "new hire starts Monday"
    assert detect_explicit_memory_request("what's the status?") is None


def test_detect_type():
    assert _detect_type("Brock is the PM for Compass") == "person"
    assert _detect_type("Scout has a new API") == "project"
    assert _detect_type("I prefer TDD") == "preference"
    assert _detect_type("The meeting is at 3pm") == "fact"


def test_extract_subject():
    assert _extract_subject("Brock is PM for Compass", "person") == "Brock"
    assert _extract_subject("Scout has new features", "project") == "Scout"


def test_extract_topics():
    topics = _extract_topics("Brock works on Compass", ["scout", "compass"])
    assert "compass" in topics
    assert "brock" in topics


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = Store(db_path=db_path)
        asyncio.get_event_loop().run_until_complete(s.initialize())
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_create_explicit_memory(store):
    mem = run(create_explicit_memory(store, "Brock is the PM for Compass", known_topics=["compass"]))
    assert mem["subject"] == "Brock"
    assert mem["topic_type"] == "person"
    assert "compass" in mem["tags"]
