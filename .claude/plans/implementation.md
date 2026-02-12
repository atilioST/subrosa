# Subrosa1 Implementation Plan

## Overview
Ground-up rewrite of Subrosa with clean architecture. 11 modules, ~1860 lines.
Same features, radically simpler structure. One path to Claude, timeouts everywhere.

---

## Step 1: Project Scaffolding
- Create `pyproject.toml` with hatchling build system
- Create `subrosa1/__init__.py`
- Create `CLAUDE.md` for the new repo
- Install in editable mode
- **Test**: `python -c "import subrosa1"` succeeds

## Step 2: config.py — Configuration
- Single flat Config dataclass loaded from `~/.subrosa/config.toml`
- `${VAR}` env substitution (port `_resolve_deep` from subrosa)
- Drop `session.*` config (no persistent sessions)
- Add `agent_timeout`, `briefing_timeout`, `monitoring_timeout`
- **Test**: Unit test loading config from a test TOML file

## Step 3: store.py — Unified Async Database
- Merge EventStore + MemoryStore + TaskManager into one async SQLite file
- Tables: events, conversations, diagnostics, topics, topic_facts, topic_tags, topic_attributes, topic_embeddings, topics_fts, tasks, task_history
- All operations via aiosqlite
- Return dicts (no model dataclasses)
- **Test**: Unit test CRUD operations on all tables

## Step 4: health.py — Health + Observability
- Track `last_response_at`, `last_agent_at`, `last_poll_at`
- `is_healthy()` checks all three
- Langfuse init/shutdown/trace helpers
- **Test**: Unit test health state transitions

## Step 5: agent.py — Single Path to Claude
- One `invoke()` method using `query()` from claude-agent-sdk
- `resume=session_id` for conversation continuity
- Returns `AgentResponse(text, session_id, usage, is_error)`
- Never raises — catches all exceptions
- **Test**: Unit test with mocked SDK

## Step 6: prompt.py — System Prompt
- Port `SYSTEM_PROMPT` constant from subrosa
- No dependencies
- **Test**: Assert prompt is non-empty string, contains key sections

## Step 7: context.py — Prompt Building + Memory Retrieval
- Build system prompt (base + briefing doc + memories + events + tasks)
- Port multi-signal memory retrieval (semantic + FTS5 + topic + subject + recency)
- Port memory formatting with token budget
- Port embeddings (sentence-transformers all-MiniLM-L6-v2)
- Handle media context formatting
- **Test**: Unit test prompt assembly with mock store

## Step 8: memory.py — Memory Extraction
- Explicit detection (regex for "remember that" patterns)
- Implicit extraction via Haiku call
- Background task tracking with done-callbacks
- **Test**: Unit test regex detection, mock Haiku call

## Step 9: media.py — Multimedia Context
- Download photos, documents, videos, audio, voice from Telegram
- Format media metadata for prompt injection
- Direct port from subrosa/telegram/media.py
- **Test**: Unit test format functions

## Step 10: telegram.py — Bot + Handlers + WorkingIndicator
- Single file: bot setup, all command handlers, message handler, WorkingIndicator
- `asyncio.wait_for()` on every agent call
- Bounded queue (max 5)
- WorkingIndicator with exponential backoff heartbeat
- Commands: /start, /status, /briefing, /stop, /silence, /remember, /memories, /forget, /restart, /schedule
- Media handler for photos/documents/videos/audio/voice
- Markdown→HTML conversion and message chunking (port from send.py)
- **Test**: Unit test WorkingIndicator lifecycle, command parsing

## Step 11: scheduler.py — Scheduled Jobs
- APScheduler with AsyncIOScheduler
- Cron triggers for morning briefing, noon briefing, evening digest, monitoring
- Jobs call agent.invoke() directly with their own timeouts
- Work hours enforcement
- **Test**: Unit test job creation and timeout config

## Step 12: app.py — Entry Point + Wiring
- Load .env → config.toml → logging → Langfuse → Store → Agent → Telegram → Scheduler
- Start polling → send "online" → await stop_event
- Shutdown: stop scheduler → stop polling → send "offline" → close DB → flush Langfuse
- Signal handling (SIGINT, SIGTERM)
- **Test**: Smoke test — import and verify wiring logic

## Step 13: Integration Test
- Stop subrosa service
- Start subrosa1
- Send Telegram message → verify response
- Test timeout handling
- Test /status, /briefing commands
- Verify memory retrieval works
- Test media handling

## Step 14: Systemd Service
- Create `systemd/subrosa1.service`
- Install and enable
- Verify auto-restart on failure
