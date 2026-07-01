"""Knowledge distillation pipeline.

Reads raw events (from the monitoring system or fixture data), calls Haiku to
extract structured knowledge items, and writes them to the knowledge table.

Architecture:
  - distill(events)  — pure extraction: takes event dicts, returns count written.
                        Used by the eval harness and by run().
  - run()            — production path: fetches unprocessed events from the DB,
                        calls distill(), updates last_distilled_at.
  - schedule()       — fire-and-forget: wraps run() as a background asyncio task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Store

logger = logging.getLogger(__name__)

_BATCH_SIZE = 15  # events per Haiku call

_SYSTEM_PROMPT = (
    "You are a knowledge extraction assistant for a VP of Engineering. "
    "Extract structured, actionable knowledge from engineering activity events. "
    "Respond only with valid JSON arrays. No prose, no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
Extract structured knowledge items from the engineering events below.

## Ontology

### Domains (which VP responsibility area this concerns)
{domains}

### Memory primitives (what kind of knowledge this is)
{primitives}

### Entity types (choose the most specific that applies)
- **product** — user-facing software: Scout, Sitetracker, mobile app, a named feature
- **system** — internal infrastructure or tooling: Bedrock, Sentry, a CI/CD pipeline, an npm package
- **person** — a named individual
- **team** — an engineering or business team
- **initiative** — a named project or program (a release like "Scout 0.4.0" or "Mobile 100.19.0" is an initiative)
- **process** — a recurring practice: incident response, upgrade restart, code review
- **vendor** — an external company or SaaS tool

## Extraction rules

For each event, extract 1–3 knowledge items that a VP of Engineering would want
to remember. A good item is specific, factual, and will still matter next week.

**Extract at least one item** from any event that contains:
- A bug, incident, or security issue (with ticket ID or description)
- A release, release decision, or milestone
- A customer escalation or customer impact signal
- A risk, blocker, or delay to any deliverable
- A dependency or architectural decision

**Skip entirely:**
- Channel join/leave, routine "standup done" updates, small talk
- Things that are transient and will be irrelevant in 48 hours
- Vague sentiment with no signal ("things are going well")

**For each item output a JSON object:**
{{
  "event_id": "<id from the event>",
  "domain": "<exact domain name from list above>",
  "primitive": "<exact primitive name from list above>",
  "entity_type": "<one of: product, system, person, team, initiative, process, vendor>",
  "entity_name": "<the specific name — for releases use 'Mobile 100.19.0' not 'mobile app'>",
  "summary": "<1–3 sentences, precise, factual, no hedging, include specifics like ticket numbers, dates, names>",
  "confidence": <0.0–1.0>,
  "expires_at": "<ISO date if time-bounded (e.g. current sprint state), null otherwise>",
  "source": "<slack|jira|github|manual>"
}}

Return a single JSON array of all items across all events. Return [] only if all events are noise (channel joins, small talk).

## Events to process

{events}
"""


def _format_domains(domains: list[dict]) -> str:
    return "\n".join(f"- **{d['name']}**: {d['description']}" for d in domains)


def _format_primitives(primitives: list[dict]) -> str:
    return "\n".join(f"- **{p['name']}**: {p['description']}" for p in primitives)


def _format_events(events: list[dict]) -> str:
    parts = []
    for e in events:
        eid = e.get("id", "?")
        source = e.get("source", "unknown")
        channel = e.get("channel", e.get("event_type", ""))
        ts = e.get("timestamp", e.get("ts", ""))
        summary = e.get("summary", e.get("content", ""))
        line = f"[{eid}] {source}/{channel} {ts}\n{summary}"
        parts.append(line)
    return "\n\n".join(parts)


def _parse_json_response(text: str) -> list[dict]:
    """Parse Haiku's response, handling markdown code fences."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        return []
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        # Try to extract a JSON array with a regex
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse distillation response as JSON")
        return []


_CLAUDE_CLI = "/home/ati/.local/bin/claude"


async def _call_haiku(prompt: str, model: str) -> list[dict]:
    """Call the Claude CLI via subprocess; the CLI handles rate limits natively."""
    cmd = [
        _CLAUDE_CLI,
        "--model", model,
        "--output-format", "text",
        "--no-session-persistence",
        "--system-prompt", _SYSTEM_PROMPT,
        "-p", prompt,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            logger.warning("Claude CLI exited %d: %s", proc.returncode, err[:200])
            return []
        return _parse_json_response(stdout.decode(errors="replace"))
    except Exception:
        logger.warning("Haiku call failed during distillation", exc_info=True)
        return []


_VALID_ENTITY_TYPES = {"person", "team", "product", "system", "initiative", "process", "vendor"}


def _validate_item(item: dict, known_domains: set[str], known_primitives: set[str]) -> bool:
    """Return True if item has all required fields with valid values."""
    required = ("domain", "primitive", "entity_type", "entity_name", "summary")
    if not all(item.get(f) for f in required):
        return False
    if item["domain"] not in known_domains:
        logger.debug("Unknown domain '%s' — skipping", item["domain"])
        return False
    if item["primitive"] not in known_primitives:
        logger.debug("Unknown primitive '%s' — skipping", item["primitive"])
        return False
    if item["entity_type"] not in _VALID_ENTITY_TYPES:
        logger.debug("Unknown entity_type '%s' — skipping", item["entity_type"])
        return False
    return True


class Distiller:
    """Distills raw events into structured knowledge items."""

    def __init__(self, store: Store, model: str = "haiku"):
        self._store = store
        self._model = model
        self._tasks: set[asyncio.Task] = set()

    async def distill(self, events: list[dict]) -> int:
        """
        Distill a list of event dicts into knowledge items.
        Returns the number of knowledge items written.
        Used directly by the eval harness.
        """
        if not events:
            return 0

        domains = await self._store.get_ontology_domains()
        primitives = await self._store.get_ontology_primitives()
        known_domains = {d["name"] for d in domains}
        known_primitives = {p["name"] for p in primitives}

        total = 0
        # Process in batches
        for i in range(0, len(events), _BATCH_SIZE):
            batch = events[i : i + _BATCH_SIZE]
            prompt = _PROMPT_TEMPLATE.format(
                domains=_format_domains(domains),
                primitives=_format_primitives(primitives),
                events=_format_events(batch),
            )

            items = await _call_haiku(prompt, self._model)
            logger.info("Batch %d/%d: %d events → %d items extracted",
                        i // _BATCH_SIZE + 1, -(-len(events) // _BATCH_SIZE),
                        len(batch), len(items))

            for item in items:
                if not _validate_item(item, known_domains, known_primitives):
                    continue
                confidence = float(item.get("confidence", 0.8))
                if confidence < 0.3:
                    continue
                try:
                    await self._store.insert_knowledge(
                        domain=item["domain"],
                        primitive=item["primitive"],
                        entity_type=item["entity_type"],
                        entity_name=item["entity_name"],
                        summary=item["summary"],
                        source=item.get("source", "distilled"),
                        source_ref=str(item.get("event_id", "")),
                        confidence=confidence,
                        expires_at=item.get("expires_at"),
                    )
                    total += 1
                except Exception:
                    logger.warning("Failed to write knowledge item", exc_info=True)

        logger.info("Distillation complete: %d events → %d knowledge items", len(events), total)
        return total

    async def run(self) -> int:
        """
        Production path: fetch unprocessed events, distill, update watermark.
        Returns the count of knowledge items written.
        """
        last_distilled = await self._store.get_meta("last_distilled_at")
        events = await self._store.get_events_since(last_distilled)

        if not events:
            logger.debug("Distiller: no new events since %s", last_distilled)
            return 0

        logger.info("Distiller: processing %d new event(s) since %s", len(events), last_distilled)
        count = await self.distill(events)
        await self._store.set_meta("last_distilled_at", datetime.now(UTC).isoformat())
        return count

    def schedule(self) -> None:
        """Fire-and-forget: run distillation as a background task."""
        task = asyncio.create_task(self._run_safe())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_safe(self) -> None:
        try:
            await self.run()
        except Exception:
            logger.warning("Background distillation failed", exc_info=True)
