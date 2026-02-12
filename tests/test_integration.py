"""Integration tests — verify wiring without connecting to Telegram."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subrosa1.agent import Agent
from subrosa1.config import Config
from subrosa1.health import Health
from subrosa1.memory import MemoryExtractor
from subrosa1.scheduler import Scheduler
from subrosa1.store import Store
from subrosa1.telegram import TelegramBot


@pytest.fixture
def config():
    return Config(
        bot_token="test:token",
        chat_id=12345,
        model="sonnet",
        max_turns=5,
        agent_timeout=60,
        morning_briefing="07:30",
        noon_briefing="12:00",
        evening_digest="18:00",
        monitoring_interval_minutes=30,
        work_hours_start="07:00",
        work_hours_end="19:00",
        timezone="America/Chicago",
    )


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "test.db")
    asyncio.get_event_loop().run_until_complete(s.initialize())
    yield s
    asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture
def health():
    return Health()


@pytest.fixture
def agent(config):
    return Agent(model=config.model, max_turns=config.max_turns)


def test_all_modules_import():
    """Every module in subrosa1 should import cleanly."""
    import subrosa1.app
    import subrosa1.agent
    import subrosa1.config
    import subrosa1.context
    import subrosa1.health
    import subrosa1.media
    import subrosa1.memory
    import subrosa1.prompt
    import subrosa1.scheduler
    import subrosa1.store
    import subrosa1.telegram


def test_wiring(config, store, health, agent):
    """Verify all components wire together without errors."""
    memory_extractor = MemoryExtractor(
        store=store,
        extraction_model=config.extraction_model,
        known_topics=config.known_topics,
    )

    bot = TelegramBot(
        config=config,
        agent=agent,
        store=store,
        health=health,
        memory_extractor=memory_extractor,
    )

    scheduler = Scheduler(
        config=config,
        agent=agent,
        bot=bot,
        store=store,
        health=health,
    )

    # Verify all references are correct
    assert bot._agent is agent
    assert bot._store is store
    assert bot._health is health
    assert bot._memory_extractor is memory_extractor
    assert scheduler._agent is agent
    assert scheduler._bot is bot


def test_bot_build_app(config, store, health, agent):
    """Bot can build a PTB Application."""
    bot = TelegramBot(config=config, agent=agent, store=store, health=health)
    app = bot.build_app()
    assert app is not None
    assert bot.app is app


def test_health_tracks_state(health):
    """Health correctly tracks component state."""
    assert health.is_healthy()
    health.record_poll()
    health.record_agent()
    health.record_response()
    status = health.status_dict()
    assert status["healthy"] is True
    assert status["last_response_ago"] is not None
    assert status["last_agent_ago"] is not None
    assert status["last_poll_ago"] is not None


@pytest.mark.asyncio
async def test_store_full_lifecycle(tmp_path):
    """Store init → write → read → close lifecycle."""
    store = Store(db_path=tmp_path / "lifecycle.db")
    await store.initialize()

    # Log event
    await store.log_event(source="test", event_type="integration", summary="test event")
    events = await store.get_recent_events(limit=1)
    assert len(events) == 1
    assert events[0]["source"] == "test"

    # Save memory
    mem = await store.save_memory(
        topic_type="person", subject="Test User",
        content="integration test memory", tags=["test"],
    )
    assert mem["subject"] == "Test User"
    retrieved = await store.get_memory(mem["id"])
    assert retrieved["subject"] == "Test User"

    # Create task
    task_id = await store.create_task(
        title="Test task", task_type="discrete", priority="medium",
    )
    task = await store.get_task(task_id)
    assert task["title"] == "Test task"

    # Log diagnostic
    await store.log_diagnostic("test", "integration test")
    diags = await store.get_recent_diagnostics(hours=1)
    assert len(diags) >= 1

    await store.close()


@pytest.mark.asyncio
async def test_memory_explicit_flow(tmp_path):
    """Explicit memory request → store → retrieve."""
    from subrosa1.memory import detect_explicit_memory_request, create_explicit_memory

    store = Store(db_path=tmp_path / "mem.db")
    await store.initialize()

    # Detect
    content = detect_explicit_memory_request("remember that Brock is the PM for Scout")
    assert content is not None
    assert "Brock" in content

    # Create — "Brock" is in _KNOWN_PEOPLE so it gets detected as person
    mem = await create_explicit_memory(store, content, known_topics=["scout"])
    assert mem["topic_type"] == "person"
    assert "brock" in mem["subject"].lower()

    # Retrieve via context
    from subrosa1.context import retrieve_relevant_memories
    results = await retrieve_relevant_memories(store, "Who is the PM for Scout?", known_topics=["scout"])
    assert len(results) >= 1

    await store.close()
