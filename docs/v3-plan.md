# Subrosa v3: Experiential Learning & Long-Running Tasks

## Context

Subrosa v2 works reliably for bounded Q&A interactions (max 10 turns, 5min timeout). But she can't:
- Work on complex tasks that need 50+ tool calls
- Learn from past successes/failures and reuse that knowledge
- Orchestrate parallel work (e.g., check Slack + Jira + GitHub simultaneously)
- Adaptively decide how much autonomy to take based on task complexity

This plan adds **experiential memory** (learn from doing), **extended execution** (long-running tasks with adaptive checkpoints), and **subagent orchestration** (parallel work).

---

## 1. Extended Execution (Long-Running Tasks)

### What changes

- **`max_turns`**: Increase default from 10 → 50 in config. The SDK supports 250+.
- **`agent_timeout`**: Increase from 300s → 900s (15min) for interactive, keep briefings at 300s.
- **Task-mode detection**: When Subrosa recognizes a complex task (multiple steps, research required), she switches to "task mode" with higher turn limits and periodic progress updates.

### Adaptive checkpointing

Subrosa decides autonomy level per-task:

```
Simple query ("what's the sprint status?")
  → One-shot, no checkpoint, respond immediately

Medium task ("summarize last week's Slack activity")
  → Run autonomously, send result when done

Complex task ("investigate why deployments failed this week, check logs, PRs, and Slack")
  → Checkpoint after each phase, send progress update
  → User can steer ("skip the logs, focus on PRs") or let her continue
  → Final summary when complete
```

### Implementation

**`agent.py`** — Add a `task_mode` parameter:
```python
async def invoke(self, prompt, system_prompt, *,
                 task_mode: bool = False,       # higher max_turns
                 resume_session: str | None = None,
                 on_progress: Callable | None = None,  # checkpoint callback
                 ...) -> AgentResponse:
```

When `task_mode=True`:
- `max_turns` = `config.task_max_turns` (default 50)
- `agent_timeout` = `config.task_timeout` (default 900s)

**`telegram.py`** — Detect complex tasks in `_do_process()`:
- If the agent uses >5 tools in a single invocation, suggest task mode for next similar request
- If the user says "investigate", "deep dive", "analyze thoroughly" → auto task mode
- Progress updates via WorkingIndicator edits (already built)

**`config.toml`** — New settings:
```toml
[agent]
max_turns = 20           # default interactive
task_max_turns = 50      # task mode
task_timeout = 900       # 15 min
```

### Files modified
- `subrosa/agent.py` — task_mode parameter, configurable max_turns
- `subrosa/telegram.py` — task detection heuristic, progress callback wiring
- `subrosa/config.py` — new config fields

---

## 2. Experiential Learning

This is the core new capability. Three alternative approaches, pick one:

### Approach A: Procedural Memory (CLAUDE.md-style) — Recommended

After completing a task, Subrosa runs a **reflection step** that extracts a reusable procedure and saves it as a markdown file. On future similar tasks, matching procedures are injected into the prompt.

**How it works:**

```
1. User asks Subrosa to do something complex
2. Subrosa completes the task (using tools, multiple turns)
3. Post-task reflection (async, background):
   - Haiku analyzes the full conversation (user request + agent actions + outcome)
   - Extracts: what was the task? what steps worked? what failed? what's the procedure?
   - Writes a procedure file:
     ~/.subrosa/procedures/investigate-deployment-failures.md
4. Next time a similar task comes in:
   - Embed the user's request
   - Semantic search against procedure embeddings
   - Inject top 1-2 matching procedures into system prompt
5. Subrosa follows the procedure (or adapts it)
6. After execution, update the procedure if the approach changed
```

**Procedure file format:**
```markdown
---
title: Investigate Deployment Failures
created: 2026-02-12
updated: 2026-02-12
trigger_examples:
  - "why did deploys fail"
  - "investigate deployment issues"
  - "what's wrong with CI/CD"
success_count: 3
last_used: 2026-02-12
---

## Steps
1. Check #scout-releases Slack channel for recent deployment messages
2. Search Jira SCOUT project for blocked/failed deployment tickets
3. Cross-reference with GitHub PR merge times
4. Look for patterns (same service? same time? same author?)

## What works
- Always check Slack first — deployment bot posts there immediately
- Filter Jira by `status changed to "Failed"` in last 7 days

## What doesn't work
- Don't search all of GitHub — too noisy. Start with the specific service mentioned.

## Output format
- Lead with "X deployments failed in the last Y days"
- Group by root cause
- End with recommended actions
```

**Why this approach:**
- Human-readable and editable (you can tweak procedures manually)
- Transparent — you can see exactly what she learned
- Lightweight — just markdown files + embeddings in existing DB
- Builds on existing infrastructure (sentence-transformers, semantic search)
- Similar to how Claude Code's CLAUDE.md works, proven pattern

**Storage:** Procedures stored as files in `~/.subrosa/procedures/`. Embeddings of title + trigger_examples stored in a new `procedure_embeddings` table in the existing SQLite DB.

### Approach B: Episodic Memory (DB-backed)

Instead of generating procedures, store raw execution traces.

```
DB Table: episodes
  - id, task_description, task_embedding (BLOB)
  - actions_taken (JSON array of tool calls)
  - outcome (success/partial/failure)
  - reflection (what worked, what didn't)
  - duration_ms, num_turns, cost_usd
  - created_at
```

On new tasks: embed → find similar episodes → inject as "here's how you handled something similar before" in the prompt.

**Pros:** Fully automatic, no file management, richer data.
**Cons:** Less transparent, harder to edit, can accumulate noise. Agent sees raw traces, not distilled procedures — uses more context window.

### Approach C: Dynamic Skills

After a successful complex task, generate a full skill definition (config.json + skill.md) automatically. Skills are stored in the DB and can be triggered by name or pattern match.

**Pros:** Most reusable — skills can be scheduled, triggered by command, composed.
**Cons:** Highest implementation complexity. Generating reliable skill definitions automatically is hard — bad skills could cause loops or errors.

### Recommendation: Approach A (Procedural Memory)

Start with A. It's the simplest to implement, most transparent, and easiest to debug. If procedures prove too rigid, add episodic traces (B) as supplementary data. Skills (C) can be generated from well-tested procedures later.

### Implementation (Approach A)

**New file: `subrosa/procedures.py`** (~200 lines)
```python
class ProcedureStore:
    """Manages procedure files and their embeddings."""

    async def reflect_on_task(self, user_request: str, conversation: list[dict],
                               outcome: str, agent: Agent) -> dict | None:
        """Post-task reflection. Generates or updates a procedure."""

    async def find_relevant(self, query: str, top_k: int = 2) -> list[dict]:
        """Semantic search for matching procedures."""

    async def load_procedure(self, path: Path) -> dict:
        """Parse a procedure markdown file."""

    async def save_procedure(self, procedure: dict) -> Path:
        """Write procedure to file, update embedding."""
```

**Reflection prompt (sent to Haiku/Sonnet after task completion):**
```
Analyze this completed task and extract a reusable procedure.

User request: {request}
Tools used: {tools_summary}
Outcome: {outcome_summary}
Duration: {duration}

Extract:
1. A short title for this procedure
2. 3-5 example trigger phrases
3. Step-by-step procedure (what to do)
4. What worked well
5. What didn't work or was unnecessary
6. Recommended output format

Return as structured markdown.
```

**Integration into existing flow:**

- `context.py` — `build_user_prompt()` adds a "Relevant Procedures" section (like "Relevant Knowledge" for memories)
- `telegram.py` — `_do_process()` triggers reflection after successful tasks with >3 tool calls
- `store.py` — new table `procedure_embeddings` for semantic search index

### Files modified/created
- `subrosa/procedures.py` — **new**, ProcedureStore class
- `subrosa/context.py` — inject matching procedures into prompt
- `subrosa/telegram.py` — trigger reflection after complex tasks
- `subrosa/store.py` — procedure_embeddings table

---

## 3. Subagent Orchestration

### What it enables

Subrosa can spawn parallel subagents for independent subtasks:
- Check Slack, Jira, and GitHub **simultaneously** instead of sequentially
- Research multiple topics in parallel during briefings
- Run a "deep dive" on one topic while continuing to answer quick questions

### Architecture

The Claude Agent SDK supports subagents natively. We don't need a framework (CrewAI, LangGraph) — the SDK's `query()` can be called concurrently.

**Pattern: Fan-out / Fan-in**
```python
async def invoke_parallel(self, subtasks: list[SubTask], system_prompt: str) -> list[AgentResponse]:
    """Run multiple agent invocations concurrently."""
    coros = [
        asyncio.wait_for(
            self._do_invoke(task.prompt, system_prompt, ...),
            timeout=task.timeout,
        )
        for task in subtasks
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [r if not isinstance(r, Exception) else AgentResponse(is_error=True, ...)
            for r in results]
```

**When to fan out:**
- Briefings: check Slack + Jira + GitHub in parallel (3 subagents)
- Monitoring: poll all channels simultaneously
- "Investigate X" tasks: research from multiple sources concurrently
- Subrosa decides based on task structure (system prompt instructs her when parallel makes sense)

### Cost consideration

Each subagent is a separate `query()` call — separate context window, separate billing. For briefings that currently take 1 long call with sequential tool use, parallelizing into 3 shorter calls may actually be cheaper (less context accumulation) and faster.

### Implementation

**`agent.py`** — Add `invoke_parallel()`:
```python
@dataclass
class SubTask:
    name: str
    prompt: str
    timeout: int = 180
    max_turns: int = 10

async def invoke_parallel(self, subtasks: list[SubTask], system_prompt: str,
                          trace_name: str = "") -> list[AgentResponse]:
```

**`context.py`** — Add `build_parallel_briefing_prompts()` that returns a list of SubTasks instead of one big prompt.

**`scheduler.py`** — Use `invoke_parallel()` for briefings and monitoring.

### Files modified
- `subrosa/agent.py` — SubTask dataclass, invoke_parallel()
- `subrosa/context.py` — parallel prompt builders
- `subrosa/scheduler.py` — use parallel invocation for briefings/monitoring

---

## 4. Similarity Matching

Already mostly built. The existing memory system uses sentence-transformers (`all-MiniLM-L6-v2`) with cosine similarity for semantic search. Procedures (Section 2) will use the same infrastructure.

### Enhancements needed

1. **Procedure matching** — embed procedure titles + trigger_examples, search on incoming user messages
2. **Confidence threshold** — only inject procedures above 0.5 similarity (higher bar than memories at 0.3)
3. **Usage tracking** — increment `success_count` and update `last_used` when a procedure is used, so frequently-used procedures rank higher
4. **Decay** — procedures not used in 90 days get flagged for review (not auto-deleted)

### No new dependencies

Uses existing `sentence-transformers`, existing embedding infrastructure in `store.py`. Just a new table and a few queries.

---

## Implementation Order

| Phase | What | Files | Effort |
|-------|------|-------|--------|
| 1 | Extended execution (max_turns, task_mode, timeouts) | agent.py, config.py, telegram.py | Small |
| 2 | Procedural memory (reflection + storage + retrieval) | procedures.py (new), context.py, telegram.py, store.py | Medium |
| 3 | Procedure injection into prompts | context.py, prompt.py | Small |
| 4 | Subagent orchestration (invoke_parallel) | agent.py, context.py, scheduler.py | Medium |
| 5 | Adaptive autonomy (checkpoint vs autonomous decision) | telegram.py, agent.py | Small |

Phases 1-3 are the highest value. Phase 4 (subagents) is an optimization. Phase 5 (adaptive autonomy) builds naturally on phases 1-3 as Subrosa accumulates procedures about what tasks need checkpoints.

---

## Verification

1. Send a complex task → Subrosa uses task_mode (higher turns), completes it
2. After completion → procedure file appears in `~/.subrosa/procedures/`
3. Send a similar task later → "Relevant Procedures" section appears in logs, Subrosa follows the procedure
4. `/briefing` → runs in parallel (Slack + Jira subagents), faster than before
5. Simple question → no checkpoint, immediate response
6. Complex investigation → periodic progress updates, user can steer
7. `ls ~/.subrosa/procedures/` → human-readable markdown files you can edit
