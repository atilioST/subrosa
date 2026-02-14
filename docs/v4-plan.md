# Subrosa v4: Extended Execution & Subagent Orchestration

## Context

v3 adds procedural memory (learn from doing). v4 builds on that with **extended execution** (long-running tasks with adaptive checkpoints), **subagent orchestration** (parallel work), and **adaptive autonomy** (deciding how much independence to take per-task).

---

## 1. Extended Execution (Long-Running Tasks)

### What changes

- **`max_turns`**: Increase default from 10 to 50 in config. The SDK supports 250+.
- **`agent_timeout`**: Increase from 300s to 900s (15min) for interactive, keep briefings at 300s.
- **Task-mode detection**: When Subrosa recognizes a complex task (multiple steps, research required), she switches to "task mode" with higher turn limits and periodic progress updates.

### Adaptive checkpointing

Subrosa decides autonomy level per-task:

```
Simple query ("what's the sprint status?")
  -> One-shot, no checkpoint, respond immediately

Medium task ("summarize last week's Slack activity")
  -> Run autonomously, send result when done

Complex task ("investigate why deployments failed this week, check logs, PRs, and Slack")
  -> Checkpoint after each phase, send progress update
  -> User can steer ("skip the logs, focus on PRs") or let her continue
  -> Final summary when complete
```

### Implementation

**`agent.py`** -- Add a `task_mode` parameter:
```python
async def invoke(self, prompt, system_prompt, *,
                 task_mode: bool = False,
                 resume_session: str | None = None,
                 on_progress: Callable | None = None,
                 ...) -> AgentResponse:
```

When `task_mode=True`:
- `max_turns` = `config.task_max_turns` (default 50)
- `agent_timeout` = `config.task_timeout` (default 900s)

**`telegram.py`** -- Detect complex tasks in `_do_process()`:
- If the agent uses >5 tools in a single invocation, suggest task mode for next similar request
- If the user says "investigate", "deep dive", "analyze thoroughly" -> auto task mode
- Progress updates via WorkingIndicator edits (already built)

**`config.toml`** -- New settings:
```toml
[agent]
max_turns = 20
task_max_turns = 50
task_timeout = 900
```

### Files modified
- `subrosa/agent.py` -- task_mode parameter, configurable max_turns
- `subrosa/telegram.py` -- task detection heuristic, progress callback wiring
- `subrosa/config.py` -- new config fields

---

## 2. Subagent Orchestration

### What it enables

Subrosa can spawn parallel subagents for independent subtasks:
- Check Slack, Jira, and GitHub **simultaneously** instead of sequentially
- Research multiple topics in parallel during briefings
- Run a "deep dive" on one topic while continuing to answer quick questions

### Architecture

The Claude Agent SDK supports subagents natively. We don't need a framework -- the SDK's `query()` can be called concurrently.

**Pattern: Fan-out / Fan-in**
```python
async def invoke_parallel(self, subtasks: list[SubTask], system_prompt: str) -> list[AgentResponse]:
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
- Subrosa decides based on task structure

### Cost consideration

Each subagent is a separate `query()` call. For briefings that currently take 1 long call with sequential tool use, parallelizing into 3 shorter calls may actually be cheaper (less context accumulation) and faster.

### Implementation

**`agent.py`** -- Add `invoke_parallel()`:
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

**`context.py`** -- Add `build_parallel_briefing_prompts()` that returns a list of SubTasks.

**`scheduler.py`** -- Use `invoke_parallel()` for briefings and monitoring.

### Files modified
- `subrosa/agent.py` -- SubTask dataclass, invoke_parallel()
- `subrosa/context.py` -- parallel prompt builders
- `subrosa/scheduler.py` -- use parallel invocation

---

## 3. Adaptive Autonomy

Builds naturally on extended execution and procedural memory:

- Procedures with high `success_count` -> run autonomously
- New/unfamiliar tasks -> checkpoint more frequently
- User preferences tracked as memories (e.g., "always ask before posting to Slack")

This phase is mostly system prompt refinement + using procedure metadata to calibrate autonomy. Low implementation effort once phases 1-2 are built.

---

## Implementation Order

| Phase | What | Files | Effort |
|-------|------|-------|--------|
| 1 | Extended execution (max_turns, task_mode, timeouts) | agent.py, config.py, telegram.py | Small |
| 2 | Subagent orchestration (invoke_parallel) | agent.py, context.py, scheduler.py | Medium |
| 3 | Adaptive autonomy (checkpoint vs autonomous) | telegram.py, agent.py, prompt.py | Small |

---

## Verification

1. Send a complex task -> Subrosa uses task_mode (higher turns), completes it
2. `/briefing` -> runs in parallel (Slack + Jira subagents), faster than before
3. Simple question -> no checkpoint, immediate response
4. Complex investigation -> periodic progress updates, user can steer
