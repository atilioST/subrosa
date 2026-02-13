"""Context assembly — prompt building, memory retrieval, formatting."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .prompt import SYSTEM_PROMPT
from .store import Store

logger = logging.getLogger(__name__)

# Known entities for subject matching
_KNOWN_PEOPLE = {"brock", "himanshu", "jared", "bailee", "walter", "atilio"}
_KNOWN_PROJECTS = {"scout", "compass", "missions", "sdlc", "subrosa"}


# ── Briefing document ──────────────────────────────────────────────────────

def load_briefing(path: str | None = None) -> str:
    """Load the briefing document. Returns empty string if not found."""
    briefing_path = Path(path or "~/.subrosa/briefing.md").expanduser()
    if not briefing_path.exists():
        return ""
    content = briefing_path.read_text()
    logger.info("Loaded briefing (%d chars)", len(content))
    return content


# ── System prompt ───────────────────────────────────────────────────────────

def build_system_prompt(briefing_path: str | None = None) -> str:
    """Base persona + briefing document."""
    parts = [SYSTEM_PROMPT]
    briefing = load_briefing(briefing_path)
    if briefing:
        parts.append("\n\n## Briefing Document\n")
        parts.append(briefing)
    return "\n".join(parts)


# ── Memory retrieval ────────────────────────────────────────────────────────

def _extract_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _extract_topics(query: str, known_topics: list[str]) -> list[str]:
    tokens = _extract_tokens(query)
    q = query.lower()
    return [t.lower() for t in known_topics if t.lower() in q or t.lower() in tokens]


def _extract_subjects(query: str) -> list[str]:
    tokens = _extract_tokens(query)
    q = query.lower()
    return [n for n in _KNOWN_PEOPLE | _KNOWN_PROJECTS if n in q or n in tokens]


def _score_topic_match(mem: dict, query_topics: list[str]) -> float:
    if not query_topics or not mem.get("tags"):
        return 0.0
    tags = {t.lower() for t in mem["tags"]}
    return min(sum(1 for t in query_topics if t in tags) * 0.4, 1.0)


def _score_subject_match(mem: dict, query_subjects: list[str]) -> float:
    if not query_subjects:
        return 0.0
    subj = mem.get("subject", "").lower()
    for s in query_subjects:
        if s == subj or s in subj:
            return 0.5
    return 0.0


def _score_recency(mem: dict) -> float:
    updated = mem.get("updated_at")
    if not updated:
        return 0.0
    dt = datetime.fromisoformat(updated)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if datetime.now(UTC) - dt < timedelta(days=7):
        return 0.1
    return 0.0


def _normalize_bm25(rank: float, min_rank: float, max_rank: float) -> float:
    if min_rank == max_rank:
        return 0.15
    return ((rank - max_rank) / (min_rank - max_rank)) * 0.3


async def retrieve_relevant_memories(
    store: Store,
    query: str,
    known_topics: list[str] | None = None,
    max_memories: int = 10,
    max_tokens_budget: int = 2000,
) -> list[dict]:
    """Multi-signal memory retrieval. Returns list of {memory, score, reason}."""
    all_topics = known_topics or await store.get_all_topics()
    query_topics = _extract_topics(query, all_topics)
    query_subjects = _extract_subjects(query)

    candidates: dict[int, dict] = {}

    # Semantic search
    semantic_scores: dict[int, float] = {}
    try:
        for m, sim in await store.semantic_search(query, limit=30, threshold=0.3):
            candidates[m["id"]] = m
            semantic_scores[m["id"]] = sim * 0.5
    except Exception:
        logger.debug("Semantic search failed", exc_info=True)

    # Topic match
    if query_topics:
        for m in await store.get_memories_by_topics(query_topics, limit=30):
            candidates[m["id"]] = m

    # Subject match
    for subj in query_subjects:
        for m in await store.get_memories_by_subject(subj, limit=10):
            candidates[m["id"]] = m

    # FTS5
    fts_scores: dict[int, float] = {}
    try:
        fts_query = " OR ".join(re.findall(r"\w+", query))
        if fts_query:
            for m, rank in await store.search_fts(fts_query, limit=20):
                candidates[m["id"]] = m
                fts_scores[m["id"]] = rank
    except Exception:
        logger.debug("FTS search failed", exc_info=True)

    if not candidates:
        return []

    # Normalize FTS
    fts_normalized: dict[int, float] = {}
    if fts_scores:
        min_r, max_r = min(fts_scores.values()), max(fts_scores.values())
        fts_normalized = {mid: _normalize_bm25(r, min_r, max_r) for mid, r in fts_scores.items()}

    # Score all
    scored = []
    for mid, mem in candidates.items():
        sem = semantic_scores.get(mid, 0.0)
        top = _score_topic_match(mem, query_topics)
        sub = _score_subject_match(mem, query_subjects)
        fts = fts_normalized.get(mid, 0.0)
        rec = _score_recency(mem)
        total = sem + top + sub + fts + rec
        if total <= 0:
            continue

        reasons = []
        if sem > 0:
            reasons.append(f"semantic:{sem:.2f}")
        if top > 0:
            reasons.append(f"topic:{top:.1f}")
        if sub > 0:
            reasons.append(f"subject:{sub:.1f}")
        if fts > 0:
            reasons.append(f"fts:{fts:.2f}")
        if rec > 0:
            reasons.append("recent")

        scored.append({"memory": mem, "score": total, "reason": ", ".join(reasons)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    scored = scored[:max_memories]

    # Trim to token budget
    total_chars = 0
    budget_chars = max_tokens_budget * 4
    trimmed = []
    for item in scored:
        mem = item["memory"]
        chars = len(mem.get("subject", "")) + len(mem.get("content", "")) + 50
        if total_chars + chars > budget_chars:
            break
        total_chars += chars
        trimmed.append(item)

    if trimmed:
        logger.info(
            "Memory: %d injected (top: %s, score=%.2f)",
            len(trimmed), trimmed[0]["memory"]["subject"], trimmed[0]["score"],
        )

    return trimmed


# ── Memory formatting ───────────────────────────────────────────────────────

def _fmt_person(mem: dict) -> str:
    attrs = mem.get("attributes", {})
    parts = [f"**{mem['subject']}**"]
    details = []
    if "role" in attrs:
        details.append(attrs["role"])
    if "team" in attrs:
        details.append(f"{attrs['team']} team")
    if details:
        parts[0] += f" ({', '.join(details)})"
    parts.append(f": {mem['content']}")
    if "slack_channel" in attrs:
        parts.append(f" [#{attrs['slack_channel']}]")
    return "- " + "".join(parts)


def _fmt_project(mem: dict) -> str:
    attrs = mem.get("attributes", {})
    parts = [f"**{mem['subject']}**"]
    if mem.get("content"):
        parts.append(f": {mem['content']}")
    extras = []
    if "slack_channel" in attrs:
        extras.append(f"Slack: #{attrs['slack_channel']}")
    if "jira_project" in attrs:
        extras.append(f"Jira: {attrs['jira_project']}")
    if extras:
        parts.append(f" [{', '.join(extras)}]")
    return "- " + "".join(parts)


def _fmt_generic(mem: dict) -> str:
    if mem.get("subject") and mem["subject"].lower() != mem.get("content", "")[:len(mem["subject"])].lower():
        return f"- **{mem['subject']}**: {mem['content']}"
    return f"- {mem.get('content', '')}"


def format_memories_for_prompt(scored_memories: list[dict]) -> str:
    """Format scored memories as markdown section."""
    if not scored_memories:
        return ""

    people, projects, context = [], [], []
    for item in scored_memories:
        mem = item["memory"]
        mt = mem.get("topic_type", "fact")
        if mt == "person":
            people.append(_fmt_person(mem))
        elif mt == "project":
            projects.append(_fmt_project(mem))
        else:
            context.append(_fmt_generic(mem))

    sections = ["## Relevant Knowledge\n"]
    if people:
        sections.append("### People")
        sections.extend(people)
        sections.append("")
    if projects:
        sections.append("### Projects")
        sections.extend(projects)
        sections.append("")
    if context:
        sections.append("### Context")
        sections.extend(context)
        sections.append("")

    return "\n".join(sections)


# ── Task context for queries ────────────────────────────────────────────────

async def _get_task_context(query: str, store: Store) -> str | None:
    """Get task context if query is task-related."""
    q = query.lower()
    keywords = [
        "task", "todo", "deadline", "due", "overdue", "remind",
        "what do i need", "what should i", "what's on deck",
        "priority", "urgent", "recurring",
    ]
    if not any(k in q for k in keywords):
        return None

    tasks = await store.get_active_tasks()
    if not tasks:
        return "No active tasks."

    lines = ["## Active Tasks\n"]
    for t in tasks[:10]:
        priority = {"urgent": "!!", "high": "!", "medium": "", "low": ""}.get(t.get("priority", ""), "")
        prefix = f"[{priority}] " if priority else ""
        due = f" (due {t['due_date'][:10]})" if t.get("due_date") else ""
        lines.append(f"- {prefix}#{t['id']}: {t['title']}{due}")

    return "\n".join(lines)


# ── User prompt builder ─────────────────────────────────────────────────────

async def build_user_prompt(
    user_message: str,
    store: Store,
    known_topics: list[str] | None = None,
    max_memories: int = 10,
    max_memory_tokens: int = 2000,
    media_files: list[dict] | None = None,
) -> str:
    """Assemble user prompt with memory context, task context, and media."""
    parts = []

    # Memory
    try:
        scored = await retrieve_relevant_memories(
            store, user_message, known_topics, max_memories, max_memory_tokens,
        )
        section = format_memories_for_prompt(scored)
        if section:
            parts.append(section)
    except Exception:
        logger.warning("Memory retrieval failed", exc_info=True)

    # Tasks
    try:
        task_ctx = await _get_task_context(user_message, store)
        if task_ctx:
            parts.append(task_ctx)
    except Exception:
        logger.warning("Task context failed", exc_info=True)

    # Media
    if media_files:
        from .media import format_media_for_prompt
        media_section = format_media_for_prompt(media_files)
        if media_section:
            parts.append(media_section)

    parts.append(user_message)
    return "\n".join(parts)


# ── Briefing/monitoring prompts ─────────────────────────────────────────────

async def build_briefing_prompt(kind: str = "morning", store: Store | None = None) -> str:
    """Build prompt for scheduled briefings."""
    parts = []

    # Task section for morning briefing
    if kind == "morning" and store:
        try:
            tasks_due = await store.get_tasks_due_soon(days=7)
            recurring = await store.get_tasks_to_surface()
            if tasks_due or recurring:
                lines = []
                if tasks_due:
                    lines.append("## Tasks Due Soon")
                    for t in tasks_due:
                        lines.append(f"- #{t['id']}: {t['title']} (due {t.get('due_date', 'N/A')[:10]})")
                if recurring:
                    lines.append("## Recurring Items")
                    for t in recurring:
                        lines.append(f"- #{t['id']}: {t['title']}")
                parts.append("\n".join(lines))
                parts.append("\n---\n")
        except Exception:
            logger.warning("Task summary failed", exc_info=True)

    if kind == "morning":
        parts.append(
            "Generate a morning briefing for today. Include:\n"
            "1. **Sprint Status**: Current sprint health, days remaining, any at-risk items\n"
            "2. **Key Activity**: Important Slack discussions, decisions, or escalations from overnight\n"
            "3. **PR Status**: PRs awaiting review, recently merged significant changes\n"
            "4. **Today's Focus**: Top 3 things to pay attention to today\n"
            "5. **Risks/Blockers**: Anything that needs immediate attention\n\n"
            "Keep it concise — this is read on mobile."
        )
    elif kind == "noon":
        parts.append(
            "Generate a midday briefing. Include:\n"
            "1. **Morning Activity**: What happened since the morning briefing\n"
            "2. **New Items**: Any new tickets, PRs, or discussions needing attention\n"
            "3. **Blockers**: Anything stalled or waiting\n\n"
            "Keep it brief — just the important changes since this morning."
        )
    else:
        parts.append(
            "Generate an evening digest for today. Include:\n"
            "1. **Day Summary**: What happened today across the org\n"
            "2. **Completed Work**: Significant PRs merged, tickets resolved\n"
            "3. **Open Items**: What carried over, what's still in progress\n"
            "4. **Tomorrow Preview**: What to expect tomorrow\n\n"
            "Keep it concise."
        )

    return "".join(parts)


def build_monitoring_prompt(
    slack_channels: list[str],
    jira_projects: list[str],
    github_repos: list[str],
) -> str:
    """Build prompt for monitoring cycle."""
    parts = [
        "Perform a monitoring check. Look for important, actionable items only.",
        "Skip routine activity — only surface things that need attention.",
        "",
    ]

    if slack_channels:
        channels = ", ".join(f"#{c}" for c in slack_channels)
        parts.append(f"**Slack**: Check channels {channels} for important messages, "
                     "escalations, or decisions in the last 30 minutes.")

    if jira_projects:
        projects = ", ".join(jira_projects)
        parts.append(f"**Jira**: Check projects {projects} for blocked tickets, "
                     "status changes on critical items, or sprint health issues.")

    if github_repos:
        repos = ", ".join(github_repos)
        parts.append(f"**GitHub**: Check repos {repos} for PRs needing review, "
                     "failed CI, or significant merges.")

    parts.extend([
        "",
        "For each important finding, provide:",
        "- Source (Slack/Jira/GitHub)",
        "- Brief summary (one sentence)",
        "- Why it matters / recommended action",
        "",
        "If nothing important is happening, respond with exactly: NO_INSIGHTS",
    ])

    return "\n".join(parts)
