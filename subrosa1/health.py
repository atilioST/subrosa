"""Health tracking + Langfuse observability."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Health tracking ─────────────────────────────────────────────────────────

class Health:
    """Tracks three timestamps to determine if the bot is alive."""

    def __init__(self):
        self.last_response_at: float = 0.0
        self.last_agent_at: float = 0.0
        self.last_poll_at: float = 0.0
        self._start_time: float = time.monotonic()

    def record_response(self) -> None:
        self.last_response_at = time.monotonic()

    def record_agent(self) -> None:
        self.last_agent_at = time.monotonic()

    def record_poll(self) -> None:
        self.last_poll_at = time.monotonic()

    def is_healthy(self, max_stale_seconds: int = 300) -> bool:
        now = time.monotonic()
        # Healthy if we've polled recently
        if self.last_poll_at > 0 and (now - self.last_poll_at) > max_stale_seconds:
            return False
        return True

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time

    def status_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "uptime_seconds": round(self.uptime),
            "last_response_ago": round(now - self.last_response_at) if self.last_response_at else None,
            "last_agent_ago": round(now - self.last_agent_at) if self.last_agent_at else None,
            "last_poll_ago": round(now - self.last_poll_at) if self.last_poll_at else None,
            "healthy": self.is_healthy(),
        }


# ── Langfuse ────────────────────────────────────────────────────────────────

_langfuse = None


def init_langfuse(
    public_key: str = "",
    secret_key: str = "",
    host: str = "https://cloud.langfuse.com",
):
    """Initialize Langfuse. Returns the client or None."""
    global _langfuse

    if not public_key or not secret_key:
        logger.info("Langfuse not configured — tracing disabled")
        return None

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    if host:
        os.environ.setdefault("LANGFUSE_BASE_URL", host)

    try:
        from langfuse import get_client
        _langfuse = get_client()
        if _langfuse.auth_check():
            logger.info("Langfuse authenticated")
        else:
            logger.warning("Langfuse auth failed")
            _langfuse = None
    except Exception:
        logger.warning("Langfuse init failed", exc_info=True)
        _langfuse = None

    return _langfuse


def get_langfuse():
    return _langfuse


def trace_invocation(
    name: str,
    input_prompt: str,
    output_text: str,
    model: str,
    total_cost_usd: float | None = None,
    usage: dict[str, Any] | None = None,
    duration_ms: int = 0,
    session_id: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Record a Claude invocation as a Langfuse trace. Returns trace ID."""
    lf = get_langfuse()
    if not lf:
        return None

    try:
        usage_details = {}
        if usage:
            usage_details = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            }

        trace = lf.trace(
            name=name,
            input=input_prompt,
            output=output_text,
            metadata={
                "session_id": session_id,
                "tags": tags or [],
                **(metadata or {}),
            },
        )

        trace.generation(
            name="claude-agent-sdk",
            model=model,
            input=input_prompt,
            output=output_text,
            usage_details=usage_details if usage_details else None,
            metadata={
                "total_cost_usd": total_cost_usd,
                "duration_ms": duration_ms,
                "session_id": session_id,
                **(metadata or {}),
            },
        )

        return trace.id
    except Exception:
        logger.warning("Failed to record Langfuse trace", exc_info=True)
        return None


def shutdown_langfuse() -> None:
    lf = get_langfuse()
    if lf:
        lf.flush()
        lf.shutdown()
        logger.info("Langfuse shut down")
