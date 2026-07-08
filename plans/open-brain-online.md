# Open Brain Online — Spec

**Status:** Design
**Date:** 2026-07-07
**Predecessor:** `plans/open-brain.md` (Phases 1–4 built; this spec gets the system actually *live*)

---

## Problem

The Open Brain infrastructure exists end-to-end — schema, distiller, MCP server,
LAN auth, client docs — but it has **never produced or served a single knowledge
item** in production:

| Component | State (verified 2026-07-07) |
|---|---|
| `knowledge` table | **0 rows** |
| `last_distilled_at` watermark | never set |
| `events` table | 882 rows waiting, growing daily |
| Distiller trigger | only fires after monitoring polls — monitoring is disabled (`monitoring_interval_minutes = -1`) |
| Brain MCP server | running on `0.0.0.0:7771`, bearer auth on, reachable |
| Brain MCP clients | configured on this host (`~/.claude.json`); remote doc exists (`config/brain_mcp_remote.md`) |
| Server errors | ASGI `AssertionError` in the journal on every client disconnect (hourly) |
| Stale sessions | after a Subrosa restart, connected Claude Code sessions get `-32602` on every tool call until the client reconnects |

Root cause of the empty brain: **distillation was wired as a side effect of a
feature that got turned off.** Everything downstream is starved.

---

## Goals

1. Knowledge flows continuously: events → distiller → `knowledge` table, on its own schedule.
2. The brain MCP is dependable from any machine on the LAN: no error spam, survives Subrosa restarts, observable freshness.
3. Backfill the 882 historical events so the brain starts useful, not empty.

Non-goals: changing the ontology, the retrieval algorithm, or Telegram behavior.

---

## Phase A — Distillation on its own schedule

The distiller becomes a first-class scheduled job, decoupled from monitoring.

1. **Scheduler job** in `scheduler.py`: run `distiller.schedule()` hourly during
   work hours (reuse the person-digest cron window), plus once at startup if
   `last_distilled_at` is older than 24h. Config: `[brain] distill_interval_minutes = 60`
   (0 disables).
2. **Also fire after each person-digest job** — that's when fresh events land.
   (One line at the end of `_person_digest_job`, mirroring what `_monitoring_job` does.)
3. **Subprocess timeout.** `_call_haiku` in `distiller.py` has no timeout — a hung
   CLI blocks distillation forever. Wrap `proc.communicate()` in
   `asyncio.wait_for(..., 120)` and kill the process on expiry.
4. **Watermark correctness.** `run()` only advances `last_distilled_at` when events
   were found; with the watermark unset, `get_events_since(None)` looks back just
   24h. Backfill (Phase B) must set the watermark explicitly when done.
5. **Distillation diagnostics.** Log a `diagnostics` row per run (events in, items
   written) so `/status` can report it.

**Acceptance:** `last_distilled_at` advances every hour during work hours;
`knowledge` count grows on days with activity; a hung CLI cannot stall the pipeline.

## Phase B — Backfill + input stream

1. **Backfill script** (`scripts/backfill_distill.py` or `python -m subrosa.distiller --backfill`):
   process all historical events oldest-first in the existing 15-event batches,
   then set `last_distilled_at`. Decision: distill everything (882 events ≈ 59
   Haiku calls, trivial cost) rather than windowing — old decisions/risks are
   exactly what the brain is for. Expired/transient items are handled by
   `expires_at` and confidence filtering at query time.
2. **Richer inputs.** Today only two event sources exist: interactive Telegram
   exchanges and digest summaries truncated to 500 chars. Improvements, in order:
   - Raise the digest `log_event` content cap from 500 → 4000 chars (the digest
     text *is* the day's distilled Slack activity; truncating it starves the brain).
   - Re-enable the monitoring job in **collector mode**: a new config flag
     `monitoring_collect_only = true` runs the poll and logs events but never
     messages Telegram. This restores the Slack/Jira firehose without the noise
     that led to disabling it.

**Acceptance:** brain answers the `vp_queries.jsonl` eval questions from backfilled
data; `eval_distillation.py ≥ 0.75`, `eval_retrieval.py recall@3 ≥ 0.70` (gates
carried over from the original spec).

## Phase C — Server reliability

1. **Fix the ASGI error.** `BearerTokenMiddleware` extends Starlette's
   `BaseHTTPMiddleware`, which is known-broken with SSE/streaming responses
   (the hourly `AssertionError: Unexpected message: http.response.start`).
   Replace with a pure ASGI middleware (~15 lines, no behavior change):

   ```python
   class BearerAuthASGI:
       def __init__(self, app, token): self.app, self.token = app, token
       async def __call__(self, scope, receive, send):
           if scope["type"] == "http" and self.token:
               headers = dict(scope["headers"])
               auth = headers.get(b"authorization", b"").decode()
               if auth != f"Bearer {self.token}":
                   await send({"type": "http.response.start", "status": 401, "headers": []})
                   await send({"type": "http.response.body", "body": b"Unauthorized"})
                   return
           await self.app(scope, receive, send)
   ```

2. **Migrate SSE → streamable HTTP transport** (`mcp.streamable_http_app()`,
   endpoint `/mcp`). SSE is deprecated in the MCP spec; streamable HTTP handles
   reconnects and restarts far better — directly addresses the stale-session
   `-32602` failures observed after Subrosa restarts. Keep `/sse` mounted during
   a transition window; update `config/brain_mcp_remote.md` and local
   `~/.claude.json` entries (`"type": "http", "url": "http://<host>:7771/mcp"`).
3. **Unauthenticated `/healthz`** returning `{knowledge_count, last_distilled_at,
   uptime}` — lets any machine (and monitoring) check brain freshness with curl.
4. **`/status` in Telegram** gains: knowledge count, last distillation time/result.

**Acceptance:** 24h with zero ASGI exceptions in the journal; a Subrosa restart
does not break an open Claude Code session (or the session recovers on next call);
`curl /healthz` works from a remote machine.

## Phase D — LAN exposure hardening (mostly done, needs verification)

1. Verify the nftables rule for `192.168.0.0/24 → tcp/7771` exists and persists
   in `/etc/nftables.conf` (doc written, rule presence unverified — needs root).
2. Give SubrosaBox a stable name in client configs (static IP already:
   `192.168.0.154`; optionally mDNS `subrosabox.local`).
3. Token rotation note in the doc: token lives in `~/.subrosa/.env`
   (`SUBROSA_BRAIN_TOKEN`); rotating it requires updating every client config.

## Phase E (optional, later) — Wider MCP surface

- Expose fact-memory (`topics`/`topic_facts`) as `search_memories` / `remember`
  tools on the same server — store methods already exist; ~40 lines.
- An `ask_subrosa` tool that invokes the full agent remotely is feasible (the
  `Agent` class is transport-agnostic) but is a different product decision:
  cost, concurrency with Telegram traffic, and auth scope all need thought.
  Not part of getting the brain online.

---

## Sequencing & effort

| Phase | Effort | Depends on |
|---|---|---|
| A — scheduled distillation | ~1h | — |
| B — backfill + inputs | ~1–2h | A (watermark) |
| C — server reliability | ~2h | — (parallel with A/B) |
| D — LAN verification | ~15min + root access | C |
| E — wider surface | later | A–C |

Suggested order: **A → backfill (B1) → C → B2 → D**, with the eval gates run
after B1 and again after B2.
