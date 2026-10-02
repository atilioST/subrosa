"""Context assembly — prompt building, memory retrieval, formatting."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .prompt import SYSTEM_PROMPT
from .store import Store

if TYPE_CHECKING:
    from .procedures import ProcedureManager

# Primitives that should always surface (high signal-to-noise)
_HIGH_SIGNAL_PRIMITIVES = {"risk", "decision", "commitment", "event"}

logger = logging.getLogger(__name__)


def clock_emoji(time_str: str) -> str:
    """Return the closest clock-face emoji for an HH:MM time string.

    Maps to Unicode clock faces (🕐–🕧). Rounds to nearest half-hour.
    Falls back to 🕓 if parsing fails.
    """
    try:
        h, m = map(int, time_str.split(":"))
    except (ValueError, AttributeError):
        return "🕓"

    # Round to nearest half-hour
    half = 1 if m >= 15 and m < 45 else (0 if m < 15 else 0)
    if m >= 45:
        h += 1
    h = h % 12  # 0-11

    # Unicode clock faces: 🕐 is U+1F550 (1 o'clock), 🕜 is U+1F55C (1:30)
    # On-the-hour: U+1F550 + (h - 1) % 12
    # Half-past:   U+1F55C + (h - 1) % 12
    if half:
        base = 0x1F55C + (h - 1) % 12
    else:
        base = 0x1F550 + (h - 1) % 12
    return chr(base)


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


# ── Knowledge retrieval (Open Brain) ────────────────────────────────────────

async def retrieve_relevant_knowledge(
    store: Store,
    query: str,
    limit: int = 8,
    threshold: float = 0.3,
    days: int | None = None,
) -> list[dict]:
    """Semantic search over the knowledge table for items relevant to query."""
    try:
        return await store.search_knowledge_semantic(
            query=query,
            limit=limit,
            threshold=threshold,
            days=days,
        )
    except Exception:
        logger.debug("Knowledge retrieval failed", exc_info=True)
        return []


async def retrieve_recent_knowledge(
    store: Store,
    hours: int = 24,
    limit: int = 20,
    primitives: set[str] | None = None,
) -> list[dict]:
    """Fetch recently updated knowledge items, optionally filtered by primitive type."""
    try:
        cursor = await store._db.execute(
            """SELECT k.*, ke.embedding IS NOT NULL AS has_embedding
               FROM knowledge k
               LEFT JOIN knowledge_embeddings ke ON ke.knowledge_id = k.id
               WHERE k.active = 1
                 AND k.updated_at >= datetime('now', ?)
               ORDER BY k.updated_at DESC
               LIMIT ?""",
            (f"-{hours} hours", limit),
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        if primitives:
            rows = [r for r in rows if r.get("primitive") in primitives]
        return rows
    except Exception:
        logger.debug("Recent knowledge retrieval failed", exc_info=True)
        return []


def format_knowledge_for_prompt(items: list[dict], heading: str = "## Brain Context\n") -> str:
    """Format knowledge items as a prompt section, grouped by domain."""
    if not items:
        return ""

    by_domain: dict[str, list[dict]] = {}
    for item in items:
        by_domain.setdefault(item.get("domain", "General"), []).append(item)

    parts = [heading]
    for domain, domain_items in by_domain.items():
        parts.append(f"### {domain}")
        for item in domain_items:
            primitive = item.get("primitive", "")
            entity = item.get("entity_name", "")
            summary = item.get("summary", "")
            updated = item.get("updated_at", "")[:10]
            label = f"[{primitive}]" if primitive else ""
            entity_part = f"**{entity}**" if entity and entity != "unknown" else ""
            line_parts = [label, entity_part, summary]
            line = " ".join(p for p in line_parts if p)
            if updated:
                line += f" _{updated}_"
            parts.append(f"- {line}")
        parts.append("")

    return "\n".join(parts)


# ── Procedure formatting ──────────────────────────────────────────────────

def format_procedures_for_prompt(procedures: list[dict]) -> str:
    """Format matched procedures as a prompt section."""
    if not procedures:
        return ""

    parts = ["## Relevant Procedures\n"]
    for proc in procedures:
        title = proc.get("title", "Untitled")
        count = proc.get("success_count", 0)
        body = proc.get("body", "")
        parts.append(f"### {title} (used {count}x)")
        if body:
            parts.append(body)
        parts.append("")

    return "\n".join(parts)


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
    procedure_manager: ProcedureManager | None = None,
) -> str:
    """Assemble user prompt with memory context, procedures, task context, and media."""
    parts = []

    # Structured knowledge (Open Brain) — semantic search, recent window
    knowledge = await retrieve_relevant_knowledge(store, user_message, limit=6, days=30)
    knowledge_section = format_knowledge_for_prompt(knowledge)
    if knowledge_section:
        parts.append(knowledge_section)

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

    # Procedures
    if procedure_manager:
        try:
            procedures = await procedure_manager.find_relevant(user_message)
            proc_section = format_procedures_for_prompt(procedures)
            if proc_section:
                parts.append(proc_section)
        except Exception:
            logger.warning("Procedure retrieval failed", exc_info=True)

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

    # Inject recent knowledge as pre-distilled context (reduces MCP tool calls)
    if store:
        hours = {"morning": 18, "noon": 6, "evening": 12}.get(kind, 12)
        recent = await retrieve_recent_knowledge(
            store, hours=hours, limit=25,
            primitives=_HIGH_SIGNAL_PRIMITIVES,
        )
        knowledge_section = format_knowledge_for_prompt(
            recent, heading=f"## Pre-distilled Activity (last {hours}h)\n"
        )
        if knowledge_section:
            parts.append(knowledge_section)
            parts.append("---\n")

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
                        lines.append(f"- 🕓 #{t['id']}: {t['title']} (due {t.get('due_date', 'N/A')[:10]})")
                if recurring:
                    lines.append("## Recurring Items")
                    for t in recurring:
                        lines.append(f"- 🕓 #{t['id']}: {t['title']}")
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


async def build_monitoring_prompt(
    slack_channels: list[str],
    jira_projects: list[str],
    github_repos: list[str],
    store: Store | None = None,
    monitoring_interval_minutes: int = 30,
) -> str:
    """Build prompt for monitoring cycle."""
    # Prepend recently distilled items so the agent skips what it already reported
    recently_reported: str = ""
    if store:
        recent = await retrieve_recent_knowledge(
            store,
            hours=max(1, monitoring_interval_minutes // 60 + 1),
            limit=10,
            primitives=_HIGH_SIGNAL_PRIMITIVES,
        )
        if recent:
            recently_reported = format_knowledge_for_prompt(
                recent, heading="## Already known (skip re-reporting these)\n"
            )

    parts = [
        "Perform a monitoring check. Surface only important, actionable items.",
        "",
        "## CRITICAL — HIGHEST PRIORITY (search these first)",
        "",
        "1. **@atilio mentions workspace-wide** — use slack_search_messages with"
        " query `@atilio`. ANY mention of Atilio anywhere is important.",
        "2. **Class 1/2 incident channels** — search `in:class1` and `in:class2`."
        " These are dynamically created incident channels. ANY activity matters.",
        "3. **Key people** — search messages from: brock, wthorn (Walter),"
        " john.leigh, jon.scharff.",
        "",
        "## RELEASE STATUS",
        "",
        "4. Search messages from Luke Chavez (lchavez) and Akanksha Shrivastava"
        " for release status posts.",
        "",
    ]

    if slack_channels:
        channels = ", ".join(f"#{c}" for c in slack_channels)
        parts.extend([
            "## STANDARD CHANNEL SCAN",
            "",
            f"5. Check channels {channels} for important messages,"
            " escalations, or decisions in the last 45 minutes.",
            "",
        ])

    if jira_projects:
        projects = ", ".join(jira_projects)
        parts.append(f"**Jira**: Check projects {projects} for new Class 1/2"
                     " issues or blocked critical items.")

    if github_repos:
        repos = ", ".join(github_repos)
        parts.append(f"**GitHub**: Check repos {repos} for PRs needing review, "
                     "failed CI, or significant merges.")

    parts.extend([
        "",
        "## SIGNAL FILTER — only surface if:",
        "",
        "- **P0**: Any @atilio mention in a class1 or class2 channel",
        "- Direct question or request aimed at Atilio",
        "- Decision announced that affects Scout org",
        "- Incident, outage, or customer escalation",
        "- Blocker called out by a team lead",
        "- Brock directive that requires action or response",
        "- New Class 1 or Class 2 issue mentioned",
        "- Release status updates from Luke or Akanksha",
        "",
        "## OUTPUT RULES",
        "",
        "- Class 1/2 mentions of Atilio get a ‼️ prefix",
        "- Lead with the most urgent item",
        "- Keep it under 500 characters",
        "- Include channel names and people",
        "- NEVER mention sprints — Scout uses Kanban",
        "- No filler, no pleasantries — be direct",
        "",
        "If nothing important is happening, respond with exactly: NO_INSIGHTS",
    ])

    result = "\n".join(parts)
    if recently_reported:
        result = recently_reported + "\n" + result
    return result


_KIND_LABELS = {
    "brock_dm": "DM from Brock",
    "dm": "unread DM",
    "mention": "unread @-mention",
    "jira_mention": "Jira @-mention",
    "brock_post": "Brock channel post",
    "red_alert": "#red_alert_scout_ai post",
    "error_post": "#eng-scout_errors post",
}


def _format_scan_item(n: int, item, tz) -> str:
    when = item.ts.astimezone(tz).strftime("%a %H:%M %Z")
    head = f"[{n}] {_KIND_LABELS.get(item.kind, item.kind)} | {item.where} | {item.author} | {when}"
    lines = [head]
    if item.note:
        lines.append(f"    note: {item.note}")
    if item.permalink:
        lines.append(f"    link: {item.permalink}")
    body = item.text.strip() or "(no text)"
    lines.extend("    > " + ln for ln in body.splitlines())
    return "\n".join(lines)


def build_hourly_scan_prompt(
    items: list,
    failures: dict[str, str],
    since_human: str,
    tz,
) -> str:
    """Build the judge/summarize prompt for the hourly alert scan.

    All fetching and mechanical filtering already happened in
    ``scan_fetch.prefetch`` (cutoff, unread vs ``last_read``, already-replied,
    self/bot drops, name resolution). The model only judges and writes — it
    has no tools. Emits ``NO_CHANGES`` when nothing is worth reporting so the
    scheduler stays silent.
    """
    parts = [
        f"Hourly alert scan (Slack + Jira) — new since {since_human}. This is an"
        " alert channel, not a digest — when in doubt, leave it out.",
        "",
        "The items below were fetched and filtered by code. They are already"
        " restricted to the cutoff; @-mentions and DMs are already confirmed"
        " UNREAD and NOT yet answered by Atilio; Atilio's own posts and"
        " bot/app messages are already removed where they should be; names are"
        " already resolved. Do not re-check any of that. You have no tools —"
        " work only from these items.",
        "",
        "## HOW TO JUDGE EACH KIND",
        "",
        "- **unread @-mention / unread DM / DM from Brock** — always report:"
        " who, where, and a one-line gist of what they want.",
        "- **Jira @-mention** — always report: issue key, who mentioned him,"
        " and what they want.",
        "- **Brock channel post** — report ONLY if (a) it asks for or clearly"
        " expects a response/action from Atilio or his team — a direct question,"
        " a request, a deadline, a decision he's waiting on — and the note says"
        " Atilio has not replied; or (b) it signals frustration, anger, or"
        " disappointment (sharp tone, escalation, 'why is this still…', public"
        " call-outs). Skip everything else, however important it sounds. Flag"
        " tone explicitly when (b) applies. Consecutive short posts in the same"
        " channel are one conversation — judge them together.",
        "- **#red_alert_scout_ai post** — ANYTHING here gets reported.",
        "- **#eng-scout_errors post** — give a short assessment: what errors"
        " occurred, new vs recurring, apparent severity and customer impact,"
        " and whether anything needs action. Name the actual error from the"
        " post text (bot alerts carry title, error message, culprit). Never"
        " describe a post as an 'automated alert' or say the content is in an"
        " attachment. Posts explicitly marked as tests can be one line.",
        "",
        "## OUTPUT RULES",
        "",
        "- Default: report findings as a plain, concise summary with NO header."
        " Ordinary mentions/DMs/Jira mentions, acknowledgments, FYIs and"
        " #red_alert_scout_ai activity are just summarized.",
        "- Use a `‼️ Needs attention` heading ONLY for: (a) any direct ask of"
        " Atilio — a question or request directed at him; (b) ANY DM from"
        " Brock (@mbrocklehurst), always. Brock's channel posts that need a"
        " response or show anger are still reported, but get the header only if"
        " they also meet (a). An acknowledgment or FYI with no ask (e.g. 'will"
        " put it on the roadmap') is not (a): summarize, no header.",
        "- Order: Needs-attention items first, then the plain summary items,"
        " then the error assessment. Omit the header if it has no items.",
        "- OMIT any section with nothing to report — no 'no activity' lines.",
        "- 1–3 concise bullets per item. Include channel/issue and who said it.",
        "- **People by real name, never by ID or handle.** Use the names as"
        " given in the items; never output a raw `U…` id or account id.",
        "- Keep it tight and mobile-friendly. No preamble, no pleasantries.",
        "- NEVER mention sprints — Scout uses Kanban.",
    ]

    if failures:
        parts += [
            "",
            "## SOURCES THAT COULD NOT BE CHECKED",
            "",
            *(f"- {label}: {reason}" for label, reason in failures.items()),
            "",
            "End the report with one line per failed source, e.g."
            " `⚠️ Couldn't check Jira mentions: <reason>`. This line is"
            " mandatory — a failed check must never be silent.",
        ]

    parts += ["", f"## ITEMS ({len(items)})", ""]
    parts += [_format_scan_item(i, it, tz) for i, it in enumerate(items, 1)]

    parts += [
        "",
        "## CRITICAL — DELTA SUPPRESSION",
        "",
        "If none of the items is worth reporting under the rules above"
        + (" (impossible here — a source failed, so report that)" if failures else "")
        + ", respond with exactly: NO_CHANGES",
        "(nothing else — no explanation).",
    ]
    return "\n".join(parts)
