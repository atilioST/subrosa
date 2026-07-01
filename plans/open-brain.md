# Open Brain — Spec

**Status:** Design
**Author:** Atilio / Claude
**Date:** 2026-04-01

---

## Vision

Transform Subrosa from a personal AI chief-of-staff that monitors and briefs into a **living knowledge base** — a continuously updated, semantically searchable memory of everything happening across the organization. Any AI assistant (Claude Code, Codex, or future tools) running on any machine on the local network can query it or write to it via MCP.

The system answers questions like:
- "What's the most impactful PR that went in this week?"
- "Are any engineers showing signs of burnout?"
- "What decisions have we made about Scout quality this quarter?"
- "What's currently blocking the Compass release?"

It answers from memory first — fast, free, local — then calls MCP tools (Slack, Jira, GitHub) if it needs more detail.

---

## Design Principles

1. **Memory-first, MCP-fallback.** Answer from stored knowledge. Only call live data sources when memory is thin, stale, or the query demands it. Over time the system answers more from memory and makes fewer MCP calls.
2. **Locally hosted, no cloud dependency.** Embeddings run on-device via `sentence-transformers`. Storage is SQLite. No Supabase, no OpenRouter, no per-query API cost.
3. **Extensible ontology.** Domains and primitives are database rows, not hardcoded logic. Add a new domain with an INSERT. The distillation prompt rebuilds dynamically.
4. **Single process, shared store.** The brain server is a module inside Subrosa, not a separate service. It shares the same SQLite connection and embedding model singleton.
5. **Structured knowledge, not raw summaries.** Every piece of ingested information is typed, domain-tagged, and entity-linked. This is the difference between a searchable knowledge graph and a pile of text.

---

## Architecture

```
Remote machine                          Subrosa host
┌──────────────────────┐               ┌──────────────────────────────────────┐
│  Claude Code         │               │  app.py (existing)                   │
│  Codex               │──── HTTP ────▶│  brain_server.py (new)               │
│  Future tools        │  SSE / MCP    │       │                              │
└──────────────────────┘               │       ▼                              │
                                       │  knowledge store (extended SQLite)   │
                                       │  ┌─────────────────────────────┐     │
                                       │  │ knowledge (new)             │     │
                                       │  │ knowledge_embeddings (new)  │     │
                                       │  │ ontology_domains (new)      │     │
                                       │  │ ontology_primitives (new)   │     │
                                       │  │ topics (existing)           │     │
                                       │  │ topic_facts (existing)      │     │
                                       │  │ topic_embeddings (existing) │     │
                                       │  │ events (existing)           │     │
                                       │  └─────────────────────────────┘     │
                                       │       ▲                              │
                                       │  distillation pipeline (new)         │
                                       │  runs after each monitoring poll     │
                                       └──────────────┬───────────────────────┘
                                                      │
                                       ┌──────────────┼───────────────┐
                                       │              │               │
                                    Slack MCP      Jira MCP      GitHub MCP
                                    (existing)    (existing)    (existing)
```

---

## Ontology

The ontology is the schema for what gets extracted and stored. It has three axes.

### Domains (the 11 VP responsibility areas)

Stored in `ontology_domains`. Seeded at init, extensible at runtime.

| Domain | Covers |
|--------|--------|
| Org Leadership | Structure, headcount, hiring, onboarding, manager coaching, promotions, compensation, retention, succession |
| Talent & Performance | Performance standards, coaching, feedback, PIPs, exits, reviews, calibration, recognition, growth |
| Delivery & Execution | Roadmap planning, cross-team prioritization, dependencies, sprint/release health, escalations, capacity |
| Product Outcome Ownership | Scout answer quality, engineering accountability for user trust, production behavior, quality bar |
| Technical & Quality Leadership | Architecture oversight, tech debt, engineering standards, development process, release discipline, high-risk changes |
| Operational Excellence | DORA metrics, incident process, post-mortems, bug triage, engineering cadences, tooling |
| Cross-Functional Leadership | Coordination with Product, Design, Support, Services, IT, Security; scope and tradeoff negotiation |
| Executive & Ad Hoc Support | Unplanned leadership requests, audits, compliance, board/exec communication, ambiguous asks |
| Business & Customer Stewardship | Executive escalations, customer-impact risk, field pain, AWS/infra cost, vendor management, budget |
| Reporting & Communication | Upward reporting, status/risk/decision communication, engineering updates to stakeholders |
| Strategic | Capacity allocation, long-range org planning, strategic signals from delivery friction, where to invest/defer/say no |

### Memory Primitives (the 8 knowledge types)

Stored in `ontology_primitives`. Each has a `retrieval_behavior` that guides how the brain server surfaces it.

| Primitive | Description | Retrieval behavior |
|-----------|-------------|-------------------|
| `reference` | Stable facts about entities | Return when queried about that entity |
| `state` | Current condition of something | Return when queried about current status; refresh when updated |
| `event` | Something that happened at a point in time | Return for temporal queries; rank by recency |
| `decision` | A committed choice | Surface when making related decisions |
| `commitment` | Something promised, with implied accountability | Surface when querying delivery status or risk |
| `risk` | A known threat to an outcome | Surface proactively alongside related queries |
| `task` | Actionable follow-up | Surface in briefings and relevant domain queries |
| `principle` | An enduring rule or standard | Inject into context for related decisions |

### Entity Types

Used to classify what a knowledge item is *about*. Not a separate table — a field on each knowledge item.

`person` | `team` | `product` | `system` | `initiative` | `process` | `vendor`

---

## Data Schema

### `ontology_domains`

```sql
CREATE TABLE ontology_domains (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### `ontology_primitives`

```sql
CREATE TABLE ontology_primitives (
    name                 TEXT PRIMARY KEY,
    description          TEXT NOT NULL,
    retrieval_behavior   TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### `knowledge`

The central table. One row per extracted knowledge item.

```sql
CREATE TABLE knowledge (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    domain        TEXT NOT NULL REFERENCES ontology_domains(name),
    primitive     TEXT NOT NULL REFERENCES ontology_primitives(name),
    entity_type   TEXT NOT NULL,   -- person | team | product | system | initiative | process | vendor
    entity_name   TEXT NOT NULL,
    summary       TEXT NOT NULL,   -- the distilled fact, 1-3 sentences
    source        TEXT NOT NULL,   -- slack | jira | github | telegram | manual
    source_ref    TEXT,            -- channel ID, ticket key, PR number, thread ts, etc.
    confidence    REAL NOT NULL DEFAULT 0.8,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at    TEXT,            -- null = permanent; set for time-bounded states
    active        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_knowledge_domain    ON knowledge(domain);
CREATE INDEX idx_knowledge_primitive ON knowledge(primitive);
CREATE INDEX idx_knowledge_entity    ON knowledge(entity_type, entity_name);
CREATE INDEX idx_knowledge_source    ON knowledge(source, created_at);
CREATE INDEX idx_knowledge_active    ON knowledge(active, updated_at);
```

### `knowledge_embeddings`

```sql
CREATE TABLE knowledge_embeddings (
    knowledge_id INTEGER PRIMARY KEY REFERENCES knowledge(id) ON DELETE CASCADE,
    embedding    BLOB NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
```

The embedding is generated over: `{domain} {primitive} {entity_type} {entity_name} {summary}`

This means semantic search naturally respects domain and entity context without requiring explicit filters.

---

## Distillation Pipeline

### When it runs

After every monitoring poll in `scheduler.py`. Also available as an explicit trigger.

### What it processes

New `events` rows since the last distillation run. Tracked via a `last_distilled_at` timestamp in a `_meta` key-value table.

### The Haiku prompt

Built dynamically from the ontology tables at runtime. Pseudostructure:

```
You are extracting structured knowledge from engineering activity events.

Current domains:
{SELECT name, description FROM ontology_domains}

Memory primitive types:
{SELECT name, description FROM ontology_primitives}

Entity types: person, team, product, system, initiative, process, vendor

For each event below, extract 0-3 knowledge items. Only extract items that are
genuinely informative — skip noise, routine status updates with no signal, and
things already obviously known.

For each item output:
  domain:      (one of the domain names above)
  primitive:   (one of the primitive names above)
  entity_type: (one of the entity types above)
  entity_name: (the specific person, team, product, etc. this is about)
  summary:     (1-3 sentences, precise, factual, no hedging)
  confidence:  (0.0–1.0, how certain you are this is meaningful)
  expires_at:  (ISO date if this fact is time-bounded, e.g. sprint states; null otherwise)

Events:
{batch of event summaries with source, channel, timestamp}
```

### Deduplication

Before writing, check if a knowledge item with the same `(domain, primitive, entity_type, entity_name)` and similar summary already exists (embedding cosine similarity > 0.85). If so, update `updated_at` and refresh the summary rather than creating a duplicate.

### Cost profile

Haiku at ~$0.25/MTok input. A batch of 20 events at ~500 tokens each = ~10K tokens = ~$0.0025 per poll cycle. Negligible.

---

## Brain Server

A new module `subrosa/brain_server.py` exposing a local MCP server over HTTP SSE. Starts as an asyncio task alongside the Telegram bot. Shares the `Store` instance — no second SQLite connection, no second embedding model load.

### Config additions

```toml
[brain]
enabled = true
port    = 7771
token   = "${SUBROSA_BRAIN_TOKEN}"
```

### Auth

Every request checks `Authorization: Bearer <token>`. Returns 401 otherwise. Sufficient for LAN-only exposure.

### MCP Tools

#### `search_knowledge(query, domain?, primitive?, entity_name?, days?, limit?)`

The primary query tool. Semantic search over `knowledge_embeddings`, optionally filtered by domain, primitive, entity, or time window. Returns ranked knowledge items with metadata.

Use this for: "most impactful PR this week", "what decisions have we made about Scout quality", "what is currently at risk in Delivery".

#### `query_entity(entity_name, entity_type?, domain?, days?)`

All knowledge about a specific person, team, product, or system. Combines semantic + exact entity_name match.

Use this for: "what do we know about Himanshu", "what's the current state of the Compass initiative".

#### `get_domain_brief(domain, days?)`

A structured summary across all primitive types for a given domain over a time window. Returns grouped by primitive: decisions made, risks identified, events, current states.

Use this for: "give me a Talent & Performance brief for the last two weeks".

#### `capture_thought(text, domain?, entity_name?, entity_type?, primitive?)`

Write a knowledge item manually — from a Claude Code session, a voice note processed by Telegram, or any other source. If domain/primitive/entity are omitted, Haiku classifies them.

Use this for: capturing a decision made in a meeting, recording a risk identified during a review, noting a commitment made to a customer.

#### `add_domain(name, description)`

Extend the ontology at runtime. The next distillation cycle includes the new domain.

#### `add_primitive(name, description, retrieval_behavior)`

Extend the primitive types at runtime.

---

## Query Patterns

### Memory-first, MCP-fallback

The brain server itself does not call MCP tools. The AI assistant orchestrates:

```
1. Call search_knowledge() or query_entity() on brain server
2. Evaluate whether the result is sufficient
3. If not: call the relevant MCP tool (Jira search, GitHub PR list, Slack thread)
4. Optionally: call capture_thought() to write the newly retrieved detail back to memory
```

This loop means the knowledge base self-reinforces — each MCP call that produces new insight feeds back into memory, so the same query gets cheaper over time.

### Example: "Most impactful PR this week"

```
search_knowledge(
  query="high impact merged pull request",
  source="github",
  primitive="event",
  days=7,
  limit=5
)
→ returns top PRs by embedding similarity + importance
→ if detail needed: call GitHub MCP for full PR description/diff
```

### Example: "Signs of burnout"

```
search_knowledge(
  query="engineer overloaded stressed working late capacity",
  domain="Talent & Performance",
  primitive="risk",
  entity_type="person",
  days=14
)
→ returns risk items about people from Slack/standup signals
→ cross-reference with search_knowledge(primitive="state", entity_type="person") for workload states
```

### Example: "What's blocking the Compass release"

```
query_entity("Compass", entity_type="initiative", domain="Delivery & Execution")
→ returns: commitments, risks, blockers, current state, recent events
→ if needed: call Jira MCP for open tickets in the Compass project
```

---

## Firewall / Network

The brain server binds to `0.0.0.0:7771`. On the Subrosa host, restrict to LAN:

```bash
sudo ufw allow from 192.168.x.0/24 to any port 7771
```

No changes to the systemd unit.

### Claude Code config (remote machine)

`~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "subrosa": {
      "type": "sse",
      "url": "http://192.168.x.HOST:7771/sse",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

### Codex config (remote machine)

`~/.codex/config.json` (or equivalent MCP config location):

```json
{
  "mcpServers": {
    "subrosa": {
      "type": "sse",
      "url": "http://192.168.x.HOST:7771/sse",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

---

## Eval System

Two harnesses gate phase transitions. Neither requires a running server — they work against fixtures and a seeded in-memory store.

### Fixture files (`tests/fixtures/`)

| File | Contents |
|------|----------|
| `distillation_events.jsonl` | 25 real events from Slack/Jira (supply chain attack, Scout 0.4.0 pre-release, mobile patch 100.18.1, customer escalations, Bedrock timeout, etc.) |
| `distillation_expected.jsonl` | Hand-labeled expected extractions for all 25 events with `must_contain` term lists |
| `vp_queries.jsonl` | 20 VP questions grounded in real activity with expected domains, primitives, and surface terms |

### Distillation eval (`tests/eval_distillation.py`)

Runs the Haiku distillation prompt against fixture events. Scores output using LLM-as-judge (Haiku): "did the distillation capture this expected knowledge item?" Measures recall and precision across domain/primitive/entity classification.

```bash
python -m tests.eval_distillation --verbose    # all events
python -m tests.eval_distillation --id evt_001 # single event
```

**Gate: overall score ≥ 0.75 before moving to Phase 3.**

### Retrieval eval (`tests/eval_retrieval.py`)

Seeds a test in-memory store with the fixture corpus, runs each VP query through `search_knowledge_semantic()`, measures recall@k. Fully deterministic — no LLM judge needed.

```bash
python -m tests.eval_retrieval --verbose  # all queries
python -m tests.eval_retrieval --k 5      # recall@5
python -m tests.eval_retrieval --id q_001 # single query
```

**Gate: recall@3 ≥ 0.70 before moving to Phase 4.**

---

## Build Phases

### Phase 1 — Schema, seed data & eval fixtures ✓

- Add `ontology_domains`, `ontology_primitives`, `knowledge`, `knowledge_embeddings`, `_meta` tables to `store.py`
- Seed with 11 domains and 8 primitives on first `initialize()`
- Add store methods: `insert_knowledge()`, `search_knowledge_semantic()`, `query_entity()`, `get_domain_brief()`, `upsert_ontology_domain()`, `upsert_ontology_primitive()`, `get_meta()`, `set_meta()`
- Deduplication on insert: exact (domain, primitive, entity_type, entity_name) match updates rather than duplicates
- Add `mcp>=1.9` dependency to `pyproject.toml`
- Add `[brain]` config section to `Config` and `config.toml`
- Eval fixtures: `distillation_events.jsonl`, `distillation_expected.jsonl`, `vp_queries.jsonl`
- Eval harnesses: `eval_distillation.py`, `eval_retrieval.py`

**Verify:** Run `eval_retrieval.py` against hand-built corpus from expected extractions. Schema works if retrieval harness seeds and queries without error.

No behavior change to running Subrosa. Foundation only.

### Phase 2 — Distillation pipeline + distillation eval

- Add `distill_events()` to store: builds dynamic Haiku prompt from ontology tables, extracts structured knowledge items from new events, writes to `knowledge` with embeddings
- Wire as post-poll step in `scheduler.py`

**Iterate:** Run `eval_distillation.py` after each prompt change. Tune until score ≥ 0.75.
**Then:** Run `eval_retrieval.py` against distillation-generated corpus. Verify recall@3 ≥ 0.70.

### Phase 3 — Brain server (localhost only)

- Add `brain_server.py` using `FastMCP` with HTTP SSE transport, bound to `127.0.0.1`
- Implement 6 MCP tools: `search_knowledge`, `query_entity`, `get_domain_brief`, `capture_thought`, `add_domain`, `add_primitive`
- Wire into `app.py` as an asyncio task
- Configure Claude Code on the Subrosa host machine

**Verify:** Ask Claude Code the VP questions from `vp_queries.jsonl`. Confirm tools are called and answers are grounded.

### Phase 4 — LAN exposure

- Change bind from `127.0.0.1` to `0.0.0.0`
- Add bearer token auth; configure firewall
- Configure Claude Code and Codex on remote machine

### Phase 5 — Subrosa self-integration

- Update `context.py` to pull from `knowledge` table in addition to `topics`
- `capture_thought` called at end of significant Telegram sessions

---

## Applicability to Other Systems

This pattern — structured ontology + distillation pipeline + MCP brain server — is not specific to Subrosa or VP work. The same architecture applies to any domain where:

- Events stream in from multiple sources
- You want natural language queries over accumulated knowledge
- The knowledge has meaningful structure beyond raw text
- Multiple AI tools need to share the same context

To apply to Scout or another system:
1. Define the domain-specific ontology (replace the 11 VP domains with the relevant taxonomy)
2. Define the entity types relevant to that domain
3. Keep the 8 memory primitives — they are domain-agnostic
4. Replace the Slack/Jira/GitHub event sources with whatever the target system ingests
5. The brain server, distillation prompt pattern, and schema are reusable as-is

The memory primitive layer (reference, state, event, decision, commitment, risk, task, principle) is the most transferable piece — it reflects how knowledge actually works, independent of domain.
