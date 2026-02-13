# CLAUDE.md

## What This Is

Subrosa is a personal AI chief-of-staff agent for Atilio.
It monitors Slack, Jira, and GitHub via MCP servers, generates daily briefings, and communicates through Telegram.

## Architecture

11 modules, ~1860 lines total. Flat structure, no god objects.

- `app.py` — Entry point, lifecycle, wiring
- `config.py` — Single Config dataclass from config.toml
- `telegram.py` — Bot, handlers, send, WorkingIndicator
- `agent.py` — Claude SDK wrapper, single invoke()
- `store.py` — Unified async SQLite (all tables)
- `context.py` — Prompt building, memory retrieval, embeddings
- `memory.py` — Memory extraction (explicit + implicit)
- `media.py` — Telegram media download + prompt formatting
- `scheduler.py` — APScheduler briefing/monitoring jobs
- `health.py` — Health tracking + Langfuse tracing
- `prompt.py` — System prompt constant

## Running

```bash
python -m subrosa       # or: subrosa
subrosa-cli             # CLI REPL mode
```

The systemd service runs: `python -m subrosa` as user `ati` with env from `~/.subrosa/.env`.

## Configuration

All config lives in `~/.subrosa/`:
- `.env` — secrets
- `config.toml` — app config
- `mcp.json` — MCP server definitions
- `briefing.md` — user context document
- `subrosa.db` — SQLite database (auto-created)

## Key Design Principles

1. One path to Claude — no session vs one-shot split
2. Timeouts everywhere — every external await has a deadline
3. User always gets a response — even on timeout/error
4. Flat structure — max 2 layers between handler and Claude SDK
5. One database — all state in one async SQLite file
6. Bounded everything — queue max 5, background tasks tracked

## Dependencies

- `claude-agent-sdk>=0.1.33` (NOT claude-code-sdk)
- `python-telegram-bot>=21.0`
- `langfuse>=3.0.0`
- `aiosqlite>=0.20.0`
- `apscheduler>=3.10,<4.0`
- `sentence-transformers>=2.2.0`
