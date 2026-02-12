"""Tests for context building."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from subrosa1.context import (
    build_system_prompt,
    build_monitoring_prompt,
    format_memories_for_prompt,
    _extract_topics,
    _extract_subjects,
    _score_topic_match,
    _score_subject_match,
)
from subrosa1.store import Store


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_build_system_prompt():
    prompt = build_system_prompt()
    assert "Subrosa" in prompt
    assert "chief-of-staff" in prompt


def test_monitoring_prompt():
    prompt = build_monitoring_prompt(
        slack_channels=["scout-dev"],
        jira_projects=["SCOUT"],
        github_repos=[],
    )
    assert "#scout-dev" in prompt
    assert "SCOUT" in prompt
    assert "NO_INSIGHTS" in prompt


def test_extract_topics():
    topics = _extract_topics("what's happening with compass?", ["scout", "compass", "missions"])
    assert "compass" in topics


def test_extract_subjects():
    subjects = _extract_subjects("what is brock working on?")
    assert "brock" in subjects


def test_score_topic_match():
    mem = {"tags": ["compass", "scout"]}
    assert _score_topic_match(mem, ["compass"]) > 0
    assert _score_topic_match(mem, []) == 0.0


def test_score_subject_match():
    mem = {"subject": "Brock"}
    assert _score_subject_match(mem, ["brock"]) == 0.5
    assert _score_subject_match(mem, ["alice"]) == 0.0


def test_format_memories_empty():
    assert format_memories_for_prompt([]) == ""


def test_format_memories():
    scored = [
        {"memory": {"subject": "Alice", "content": "- Engineer", "topic_type": "person", "attributes": {"role": "SWE"}}, "score": 0.8, "reason": "topic"},
        {"memory": {"subject": "Scout", "content": "- Main project", "topic_type": "project", "attributes": {}}, "score": 0.6, "reason": "topic"},
        {"memory": {"subject": "TDD", "content": "- Prefer TDD", "topic_type": "preference", "attributes": {}}, "score": 0.4, "reason": "fts"},
    ]
    result = format_memories_for_prompt(scored)
    assert "Relevant Knowledge" in result
    assert "People" in result
    assert "Projects" in result
    assert "Context" in result


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = Store(db_path=db_path)
        asyncio.get_event_loop().run_until_complete(s.initialize())
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


def test_build_user_prompt(store):
    from subrosa1.context import build_user_prompt
    # Save some memories first
    run(store.save_memory(subject="Compass", content="Navigation feature", topic_type="project", tags=["compass"]))

    prompt = run(build_user_prompt("what's happening with compass?", store, known_topics=["compass"]))
    assert "compass" in prompt.lower() or "what's happening" in prompt
