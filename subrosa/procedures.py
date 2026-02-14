"""Procedural memory — learn reusable approaches from complex tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .store import EmbeddingService, Store

logger = logging.getLogger(__name__)

_PROCEDURES_DIR = Path("~/.subrosa/procedures").expanduser()

_REFLECTION_PROMPT = """\
The user has asked to save a reusable procedure from this completed task.
Analyze the work that was done and extract the OPTIMAL path — if there were
errors, retries, or corrections along the way, distill what the ideal approach
would be if doing it again from scratch.

USER REQUEST:
{user_request}

TOOLS USED (in order):
{tools_used}

AGENT NARRATION (reasoning during work):
{narration}

FINAL RESPONSE:
{final_response}

Extract a reusable procedure as JSON:
{{
  "title": "Short descriptive title",
  "trigger_examples": ["3-5 example phrases that would trigger this procedure"],
  "steps": ["Step 1: ...", "Step 2: ..."],
  "what_works": ["Key insight 1", "Key insight 2"],
  "what_doesnt_work": ["Anti-pattern 1"],
  "output_format": "Brief description of ideal output structure"
}}

Focus on the SHORTEST CORRECT PATH. Omit dead ends, retries, and mistakes —
capture only the steps that actually contributed to the final result.

Respond with ONLY valid JSON, no other text."""


_PROCEDURE_TRIGGERS = [
    re.compile(r"remember how (?:we|you|i) did (?:this|that)", re.IGNORECASE),
    re.compile(r"save (?:this|that) (?:as a )?procedure", re.IGNORECASE),
    re.compile(r"remember (?:this|that) (?:approach|workflow|process|procedure|mission)", re.IGNORECASE),
]


def detect_procedure_request(text: str) -> bool:
    """Check if the message is an explicit request to save a procedure."""
    return any(p.search(text) for p in _PROCEDURE_TRIGGERS)


class ProcedureManager:
    """Manages procedural memory — reflection, storage, and retrieval."""

    def __init__(
        self,
        store: Store,
        reflection_model: str = "haiku",
        min_tool_calls: int = 3,
        similarity_threshold: float = 0.5,
        update_threshold: float = 0.7,
        max_per_prompt: int = 2,
    ):
        self._store = store
        self._reflection_model = reflection_model
        self._min_tool_calls = min_tool_calls
        self._similarity_threshold = similarity_threshold
        self._update_threshold = update_threshold
        self._max_per_prompt = max_per_prompt
        self._embeddings = EmbeddingService()
        self._tasks: set[asyncio.Task] = set()
        self._last_exchange: dict | None = None

    def record_exchange(
        self,
        user_request: str,
        tools_used: list[str],
        narration: str,
        final_response: str,
    ) -> None:
        """Store the last exchange so reflection can be triggered later."""
        self._last_exchange = {
            "user_request": user_request,
            "tools_used": tools_used,
            "narration": narration,
            "final_response": final_response,
        }

    def schedule_reflection(self) -> bool:
        """Trigger reflection on the last recorded exchange. Returns False if nothing to reflect on."""
        if not self._last_exchange:
            return False
        exchange = self._last_exchange
        self._last_exchange = None
        task = asyncio.create_task(
            self._reflect(
                exchange["user_request"],
                exchange["tools_used"],
                exchange["narration"],
                exchange["final_response"],
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _reflect(
        self,
        user_request: str,
        tools_used: list[str],
        narration: str,
        final_response: str,
    ) -> None:
        try:
            # Check for existing similar procedure
            existing = await self._find_similar_for_update(user_request)

            prompt = _REFLECTION_PROMPT.format(
                user_request=user_request,
                tools_used=", ".join(tools_used),
                narration=narration[:3000],
                final_response=final_response[:2000],
            )

            options = ClaudeAgentOptions(
                model=self._reflection_model,
                max_turns=1,
                system_prompt="You are a procedure extraction assistant. Analyze completed tasks and extract reusable procedures. Respond only with valid JSON.",
                permission_mode="bypassPermissions",
                cli_path="/home/ati/.local/bin/claude",
            )

            result_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_text = message.result or ""

            if not result_text:
                return

            # Parse JSON
            json_text = result_text.strip()
            if json_text.startswith("```"):
                json_text = re.sub(r"^```\w*\n?", "", json_text)
                json_text = re.sub(r"\n?```$", "", json_text)

            data = json.loads(json_text)

            if data.get("skip"):
                logger.debug("Reflection skipped for: %s", user_request[:60])
                return

            if existing:
                path = await self._update_procedure(existing["path"], data)
                logger.info("Updated procedure: %s", path.name)
            else:
                path = await self._save_procedure(data)
                logger.info("Created procedure: %s", path.name)

        except Exception:
            logger.warning("Procedure reflection failed", exc_info=True)

    async def find_relevant(
        self, query_text: str, top_k: int | None = None, threshold: float | None = None
    ) -> list[dict]:
        """Semantic search for matching procedures."""
        top_k = top_k or self._max_per_prompt
        threshold = threshold or self._similarity_threshold

        try:
            query_blob = self._embeddings.embed(query_text)
            query_vec = EmbeddingService.deserialize(query_blob)

            rows = await self._store.get_procedure_embeddings()
            if not rows:
                return []

            results = []
            for row in rows:
                proc_vec = EmbeddingService.deserialize(row["embedding"])
                sim = EmbeddingService.cosine_similarity(query_vec, proc_vec)
                if sim >= threshold:
                    proc = self._load_procedure(Path(row["file_path"]))
                    if proc:
                        proc["score"] = sim
                        results.append(proc)

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception:
            logger.warning("Procedure search failed", exc_info=True)
            return []

    async def _find_similar_for_update(self, query_text: str) -> dict | None:
        """Find an existing procedure similar enough to update instead of creating new."""
        matches = await self.find_relevant(
            query_text, top_k=1, threshold=self._update_threshold
        )
        return matches[0] if matches else None

    def _load_procedure(self, path: Path) -> dict | None:
        """Parse YAML frontmatter + markdown body from a procedure file."""
        if not path.exists():
            return None
        try:
            text = path.read_text()
            if not text.startswith("---"):
                return None

            parts = text.split("---", 2)
            if len(parts) < 3:
                return None

            meta = yaml.safe_load(parts[1])
            body = parts[2].strip()

            return {
                "path": path,
                "title": meta.get("title", path.stem),
                "created": meta.get("created", ""),
                "updated": meta.get("updated", ""),
                "trigger_examples": meta.get("trigger_examples", []),
                "success_count": meta.get("success_count", 0),
                "last_used": meta.get("last_used", ""),
                "body": body,
            }
        except Exception:
            logger.warning("Failed to load procedure: %s", path, exc_info=True)
            return None

    async def _save_procedure(self, data: dict) -> Path:
        """Write a new procedure file and store its embedding."""
        _PROCEDURES_DIR.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC)
        slug = re.sub(r"[^a-z0-9]+", "-", data["title"].lower()).strip("-")[:50]
        filename = f"{now.strftime('%Y%m%d')}-{slug}.md"
        path = _PROCEDURES_DIR / filename

        body = self._format_body(data)
        meta = {
            "title": data["title"],
            "created": now.isoformat(),
            "updated": now.isoformat(),
            "trigger_examples": data.get("trigger_examples", []),
            "success_count": 1,
            "last_used": now.isoformat(),
        }

        content = f"---\n{yaml.dump(meta, default_flow_style=False, sort_keys=False)}---\n\n{body}"
        path.write_text(content)

        # Store embedding
        embed_text = meta["title"] + " " + " ".join(meta["trigger_examples"])
        embedding = self._embeddings.embed(embed_text)
        await self._store.save_procedure_embedding(str(path), embedding)

        return path

    async def _update_procedure(self, path: Path, data: dict) -> Path:
        """Update an existing procedure: increment success_count, merge data."""
        existing = self._load_procedure(path)
        if not existing:
            return await self._save_procedure(data)

        now = datetime.now(UTC)

        # Merge trigger examples
        existing_triggers = set(existing.get("trigger_examples", []))
        new_triggers = set(data.get("trigger_examples", []))
        merged_triggers = list(existing_triggers | new_triggers)[:10]

        meta = {
            "title": existing.get("title", data["title"]),
            "created": existing.get("created", now.isoformat()),
            "updated": now.isoformat(),
            "trigger_examples": merged_triggers,
            "success_count": existing.get("success_count", 0) + 1,
            "last_used": now.isoformat(),
        }

        body = self._format_body(data)
        content = f"---\n{yaml.dump(meta, default_flow_style=False, sort_keys=False)}---\n\n{body}"
        path.write_text(content)

        # Update embedding
        embed_text = meta["title"] + " " + " ".join(merged_triggers)
        embedding = self._embeddings.embed(embed_text)
        await self._store.save_procedure_embedding(str(path), embedding)

        return path

    @staticmethod
    def _format_body(data: dict) -> str:
        """Format procedure body from structured data."""
        sections = []

        steps = data.get("steps", [])
        if steps:
            sections.append("## Steps")
            for i, step in enumerate(steps, 1):
                line = step if step.startswith(f"{i}.") else f"{i}. {step}"
                sections.append(line)
            sections.append("")

        works = data.get("what_works", [])
        if works:
            sections.append("## What works")
            for item in works:
                sections.append(f"- {item}")
            sections.append("")

        doesnt = data.get("what_doesnt_work", [])
        if doesnt:
            sections.append("## What doesn't work")
            for item in doesnt:
                sections.append(f"- {item}")
            sections.append("")

        fmt = data.get("output_format")
        if fmt:
            sections.append("## Output format")
            sections.append(fmt)
            sections.append("")

        return "\n".join(sections)
