# Subrosa v3: Procedural Memory

## Context

Subrosa v2 has a working fact-based memory system (people, projects, preferences), but can't learn *how* to do things. When she successfully completes a complex multi-tool task (e.g., "investigate deployment failures"), that knowledge is lost. Next time, she starts from scratch.

v3 adds **procedural memory**: after complex tasks, Subrosa reflects on what worked and saves a reusable procedure as a human-readable markdown file. On similar future tasks, matching procedures are injected into the prompt so she can follow proven approaches.

---

## Procedural Memory (Approach A: CLAUDE.md-style)

After completing a task, Subrosa runs a **reflection step** that extracts a reusable procedure and saves it as a markdown file. On future similar tasks, matching procedures are injected into the prompt.

### How it works

```
1. User asks Subrosa to do something complex
2. Subrosa completes the task (using tools, multiple turns)
3. Post-task reflection (async, background):
   - Haiku analyzes: user request + tools used + agent narration + outcome
   - Extracts: what was the task? what steps worked? what failed?
   - Writes a procedure file to ~/.subrosa/procedures/
4. Next time a similar task comes in:
   - Embed the user's request
   - Semantic search against procedure embeddings
   - Inject top 1-2 matching procedures into user prompt
5. Subrosa follows the procedure (or adapts it)
6. After execution, update the procedure if the approach changed
```

### Procedure file format

Stored in `~/.subrosa/procedures/` as YAML-frontmatter markdown:

```markdown
---
title: Investigate Deployment Failures
created: 2026-02-13T10:00:00
updated: 2026-02-13T10:00:00
trigger_examples:
  - "why did deploys fail"
  - "investigate deployment issues"
  - "what's wrong with CI/CD"
success_count: 3
last_used: 2026-02-13T10:00:00
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

### Implementation

**New module: `subrosa/procedures.py`** (~200 lines)

```python
class ProcedureManager:
    def __init__(self, store, reflection_model, min_tool_calls,
                 similarity_threshold, update_threshold, max_per_prompt)

    def should_reflect(tools_used, is_error) -> bool
    def schedule_reflection(user_request, tools_used, narration, final_response)
    async def _reflect(...)
    async def find_relevant(query, top_k, threshold) -> list[dict]
    def _load_procedure(path) -> dict | None
    async def _save_procedure(data) -> Path
    async def _update_procedure(path, data) -> Path
```

**Config** (`config.toml`):
```toml
[procedures]
enabled = true
reflection_model = "haiku"
min_tool_calls = 3
similarity_threshold = 0.5
update_threshold = 0.7
max_per_prompt = 2
```

**Storage**: Procedure files on disk, embeddings of title + trigger_examples in `procedure_embeddings` table (existing SQLite DB).

### Integration

- `agent.py` — Capture intermediate TextBlocks as `narration` in `AgentResponse`
- `context.py` — `build_user_prompt()` injects "Relevant Procedures" section
- `prompt.py` — System prompt tells Subrosa to follow procedures when present
- `telegram.py` — Trigger reflection after tasks with 3+ tool calls
- `cli.py` — Same reflection trigger
- `store.py` — `procedure_embeddings` table for semantic search

---

## Similarity Matching

Uses existing sentence-transformers (`all-MiniLM-L6-v2`) and cosine similarity infrastructure.

### Thresholds

- **Retrieval**: 0.5 similarity (higher bar than memories at 0.3)
- **Update**: 0.7 similarity — if an existing procedure is very similar, update it instead of creating a new one
- **Max per prompt**: 2 procedures injected

### Usage tracking

- Increment `success_count` and update `last_used` when a procedure is followed
- Procedures not used in 90 days could be flagged for review (future enhancement)

---

## Verification

1. **Start Subrosa** — `python -m subrosa` boots without errors, `procedure_embeddings` table created
2. **Send a complex task** — uses 3+ tools
3. **Check logs** — "Created procedure: {filename}" after response
4. **Check filesystem** — `ls ~/.subrosa/procedures/` shows `.md` file with valid frontmatter
5. **Send similar task** — logs show procedure injection, response follows learned approach
6. **CLI mode** — `subrosa-cli`, repeat to verify CLI wiring
7. **Simple query** (< 3 tools) — does NOT trigger reflection
8. **Edit procedure manually** — Subrosa picks up edited version on next similar query
