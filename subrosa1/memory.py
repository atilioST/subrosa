"""Memory extraction — explicit requests and implicit extraction from conversations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .store import Store

logger = logging.getLogger(__name__)

# Patterns that signal explicit memory requests
_EXPLICIT_PATTERNS = [
    re.compile(r"^remember\s+that\s+(.+)", re.IGNORECASE),
    re.compile(r"^note\s+that\s+(.+)", re.IGNORECASE),
    re.compile(r"^keep\s+in\s+mind\s+(?:that\s+)?(.+)", re.IGNORECASE),
    re.compile(r"^fyi[,:\s]+(.+)", re.IGNORECASE),
]

# Heuristics for memory type detection
_PERSON_INDICATORS = re.compile(
    r"\b(is a|is the|works on|leads|manages|owns|reports to|role is|PM for|EM for|engineer on)\b",
    re.IGNORECASE,
)
_PREFERENCE_INDICATORS = re.compile(
    r"\b(i prefer|i like|i want|i need|i don't like|i hate|always|never|my .+ is)\b",
    re.IGNORECASE,
)
_PROJECT_NAMES = {"scout", "compass", "missions", "sdlc", "subrosa", "sitetracker"}
_KNOWN_PEOPLE = {"brock", "himanshu", "jared", "bailee", "walter", "atilio"}


def detect_explicit_memory_request(text: str) -> str | None:
    """Check if the message is an explicit memory request. Returns content or None."""
    text = text.strip()
    for pattern in _EXPLICIT_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip()
    return None


def _detect_type(content: str) -> str:
    cl = content.lower()
    has_person = any(n in cl for n in _KNOWN_PEOPLE)
    if has_person and _PERSON_INDICATORS.search(content):
        return "person"
    if any(p in cl for p in _PROJECT_NAMES) and not has_person:
        return "project"
    if _PREFERENCE_INDICATORS.search(content):
        return "preference"
    return "fact"


def _extract_subject(content: str, memory_type: str) -> str:
    cl = content.lower()
    if memory_type == "person":
        for name in _KNOWN_PEOPLE:
            if name in cl:
                return name.title()
    if memory_type == "project":
        for proj in _PROJECT_NAMES:
            if proj in cl:
                return proj.title()
    return " ".join(content.split()[:4])


def _extract_topics(content: str, known_topics: list[str]) -> list[str]:
    cl = content.lower()
    topics = set()
    for t in known_topics:
        if t.lower() in cl:
            topics.add(t.lower())
    for name in _KNOWN_PEOPLE:
        if name in cl:
            topics.add(name)
    return list(topics)


def _extract_attributes(content: str, memory_type: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if memory_type == "person":
        role_match = re.search(r"is (?:a |the )?(\w+(?:\s+\w+)?)\s+(?:for|on|of|at)\b", content, re.IGNORECASE)
        if role_match:
            attrs["role"] = role_match.group(1)
        for proj in _PROJECT_NAMES:
            if proj in content.lower():
                attrs["team"] = proj.title()
                break
    return attrs


async def create_explicit_memory(
    store: Store,
    raw_content: str,
    known_topics: list[str] | None = None,
) -> dict:
    """Create a memory from an explicit user request."""
    topics_list = known_topics or list(_PROJECT_NAMES)
    memory_type = _detect_type(raw_content)
    subject = _extract_subject(raw_content, memory_type)
    topics = _extract_topics(raw_content, topics_list)
    attributes = _extract_attributes(raw_content, memory_type)

    return await store.save_memory(
        subject=subject,
        content=raw_content,
        topic_type=memory_type,
        source="user",
        confidence=1.0,
        tags=topics,
        attributes=attributes,
    )


_EXTRACTION_PROMPT = """\
Analyze this conversation and extract any facts worth remembering long-term.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

Extract facts as a JSON array. Each fact should have:
- "type": one of "person", "project", "preference", "fact", "insight"
- "subject": the main entity (person name, project name, or short label)
- "content": the fact in a single clear sentence
- "topics": list of relevant topic tags (from: scout, compass, missions, sdlc, subrosa, or person names)
- "confidence": 0.0-1.0 how confident you are this is worth remembering

Only extract genuinely useful facts. Skip:
- Transient information (meeting times, temporary states)
- Information already obvious from context
- Trivial pleasantries
- Meta discussions about implementation, tools, or system architecture

Focus on domain knowledge: people, projects, business context, team structure, and user preferences.

If there are no facts worth extracting, return an empty array: []

Respond with ONLY the JSON array, no other text."""


async def extract_memories_from_conversation(
    store: Store,
    user_message: str,
    assistant_response: str,
    extraction_model: str = "haiku",
    known_topics: list[str] | None = None,
) -> list[dict]:
    """Extract and store memories from a conversation using Haiku.

    Designed to run as fire-and-forget via asyncio.create_task().
    """
    try:
        prompt = _EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        options = ClaudeAgentOptions(
            model=extraction_model,
            max_turns=1,
            system_prompt="You are a fact extraction assistant. Extract structured facts from conversations. Respond only with valid JSON.",
            permission_mode="bypassPermissions",
            cli_path="/home/ati/.local/bin/claude",
        )

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result_text = message.result or ""

        if not result_text:
            return []

        # Parse JSON (handle markdown code blocks)
        json_text = result_text.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"^```\w*\n?", "", json_text)
            json_text = re.sub(r"\n?```$", "", json_text)

        facts = json.loads(json_text)
        if not isinstance(facts, list):
            return []

        topics_list = known_topics or list(_PROJECT_NAMES)
        saved: list[dict] = []

        for fact in facts:
            if not isinstance(fact, dict):
                continue
            confidence = float(fact.get("confidence", 0.5))
            if confidence < 0.3:
                continue

            content = fact.get("content", "")
            if not content:
                continue

            topic_type = fact.get("type", "fact")
            if topic_type not in ("person", "project", "preference", "fact", "insight"):
                topic_type = "fact"

            subject = fact.get("subject") or _extract_subject(content, topic_type)
            topics = fact.get("topics", [])
            if not topics:
                topics = _extract_topics(content, topics_list)

            mem = await store.save_memory(
                subject=subject,
                content=content,
                topic_type=topic_type,
                source="extracted",
                confidence=confidence,
                tags=[t.lower() for t in topics if isinstance(t, str)],
            )
            saved.append(mem)

        if saved:
            logger.info("Extracted %d memories from conversation", len(saved))
        return saved

    except Exception:
        logger.warning("Failed to extract memories", exc_info=True)
        return []


class MemoryExtractor:
    """Manages background memory extraction tasks."""

    def __init__(self, store: Store, extraction_model: str = "haiku", known_topics: list[str] | None = None):
        self._store = store
        self._model = extraction_model
        self._known_topics = known_topics
        self._tasks: set[asyncio.Task] = set()

    def schedule_extraction(self, user_message: str, assistant_response: str) -> None:
        """Fire-and-forget memory extraction."""
        task = asyncio.create_task(
            extract_memories_from_conversation(
                self._store, user_message, assistant_response,
                self._model, self._known_topics,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
