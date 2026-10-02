"""Single path to Claude — one invoke() function, no persistent sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from .health import trace_invocation

logger = logging.getLogger(__name__)

CLI_PATH = "/home/ati/.local/bin/claude"

# Monkey-patch SDK message parser to handle unknown message types (e.g. rate_limit_event)
# instead of crashing the entire stream. Must patch both the module AND the client,
# because client.py does `from .message_parser import parse_message` (binds a local ref).
import claude_agent_sdk._internal.message_parser as _mp  # noqa: E402
import claude_agent_sdk._internal.client as _client  # noqa: E402

_original_parse_message = _mp.parse_message


@dataclass
class RateLimitEvent:
    """Surfaced to callers so they can inform the user."""
    retry_after_seconds: float = 0
    message: str = ""


def _tolerant_parse_message(data: dict):
    try:
        return _original_parse_message(data)
    except Exception:
        msg_type = data.get("type", "unknown")
        if msg_type == "rate_limit_event":
            retry = data.get("retry_after", data.get("retry_after_seconds", 0))
            msg = data.get("message", "Rate limited by API")
            logger.warning("Rate limited — retry after %ss: %s", retry, msg)
            return RateLimitEvent(retry_after_seconds=float(retry), message=msg)
        logger.warning("Skipping unrecognized SDK message type: %s", msg_type)
        return None


_mp.parse_message = _tolerant_parse_message
_client.parse_message = _tolerant_parse_message


@dataclass
class ToolCall:
    name: str
    detail: str = ""


class InvocationProgress:
    """Live record of an in-flight invocation.

    Written by Agent._do_invoke as SDK messages stream in; read by the
    WorkingIndicator for live status and by callers to salvage partial
    output when an invocation is cancelled or hits the hard ceiling.
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.tools: list[ToolCall] = []
        self.text_parts: list[str] = []
        self.continuations = 0

    def note_tool(self, name: str, tool_input: dict | None) -> None:
        self.tools.append(ToolCall(name=name, detail=_tool_detail(tool_input)))

    def note_text(self, text: str) -> None:
        self.text_parts.append(text)

    @property
    def last_tool(self) -> ToolCall | None:
        return self.tools[-1] if self.tools else None

    @property
    def partial_text(self) -> str:
        return "\n".join(self.text_parts)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


def _tool_detail(tool_input: dict | None) -> str:
    """Short human-readable hint of what a tool call is doing."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("query", "pattern", "url", "path", "file_path", "command", "prompt"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip().replace("\n", " ")
            return value[:60] + ("…" if len(value) > 60 else "")
    return ""


@dataclass
class AgentResponse:
    text: str = ""
    session_id: str = ""
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    duration_ms: int = 0
    num_turns: int = 0
    is_error: bool = False
    tools_used: list[str] = field(default_factory=list)
    narration: str = ""
    subtype: str = ""


class Agent:
    """The ONLY way Claude is invoked in the entire codebase."""

    def __init__(self, model: str = "sonnet", max_turns: int = 10, max_continuations: int = 2):
        self.model = model
        self.max_turns = max_turns
        self.max_continuations = max_continuations

    async def invoke(
        self,
        prompt: str,
        system_prompt: str,
        resume_session: str | None = None,
        trace_name: str = "",
        trace_tags: list[str] | None = None,
        max_turns: int | None = None,
        progress: InvocationProgress | None = None,
        model: str | None = None,
        no_tools: bool = False,
    ) -> AgentResponse:
        """Invoke Claude. Never raises (except CancelledError) — returns
        AgentResponse with is_error on failure. Auto-continues when the CLI
        stops on the turn cap mid-task. `model` overrides self.model for this
        call (scheduled jobs run a cheaper model than interactive questions).
        `no_tools` disables built-in tools and all MCP servers — for prompts
        whose data is already embedded (e.g. the prefetched alert scan)."""
        model = model or self.model
        try:
            response = await self._do_invoke(
                prompt, system_prompt, resume_session, trace_name, trace_tags,
                max_turns, progress, model, no_tools,
            )
        except Exception as e:
            if resume_session:
                logger.warning("Resume failed, retrying as one-shot")
                try:
                    response = await self._do_invoke(
                        prompt, system_prompt, None, trace_name, trace_tags,
                        max_turns, progress, model, no_tools,
                    )
                except Exception as e:
                    logger.exception("Agent invocation failed (one-shot retry)")
                    return AgentResponse(text=_error_text(e), is_error=True)
            else:
                logger.exception("Agent invocation failed")
                return AgentResponse(text=_error_text(e), is_error=True)

        for i in range(self.max_continuations):
            if response.subtype != "error_max_turns" or not response.session_id:
                break
            logger.info(
                "Turn cap hit — auto-continuing (%d/%d)", i + 1, self.max_continuations
            )
            if progress:
                progress.continuations = i + 1
            try:
                continued = await self._do_invoke(
                    "You stopped at the turn limit mid-task. Continue and finish the task.",
                    system_prompt, response.session_id, trace_name, trace_tags,
                    max_turns, progress, model, no_tools,
                )
            except Exception:
                logger.exception("Continuation failed — returning partial result")
                break
            response = _merge_responses(response, continued)

        return response

    async def _do_invoke(
        self,
        prompt: str,
        system_prompt: str,
        resume_session: str | None,
        trace_name: str,
        trace_tags: list[str] | None,
        max_turns: int | None = None,
        progress: InvocationProgress | None = None,
        model: str | None = None,
        no_tools: bool = False,
    ) -> AgentResponse:
        model = model or self.model
        try:
            return await self._run_query(
                prompt, system_prompt, resume_session, trace_name, trace_tags,
                max_turns, progress, model, no_tools=no_tools,
            )
        except _UnrecognizedModel:
            # A full model id (claude-opus-5-5) is rejected by a CLI older than
            # the model — seen right after a release, before the CLI updated.
            # The family alias always resolves on any CLI version.
            alias = _model_alias(model)
            if alias == model:
                raise
            logger.warning("CLI does not know %s — retrying with alias %r", model, alias)
            return await self._run_query(
                prompt, system_prompt, resume_session, trace_name, trace_tags,
                max_turns, progress, alias, no_tools=no_tools,
            )

    async def _run_query(
        self,
        prompt: str,
        system_prompt: str,
        resume_session: str | None,
        trace_name: str,
        trace_tags: list[str] | None,
        max_turns: int | None,
        progress: InvocationProgress | None,
        model: str,
        *,
        no_tools: bool = False,
    ) -> AgentResponse:
        stderr_lines: list[str] = []
        options = ClaudeAgentOptions(
            model=model,
            max_turns=max_turns or self.max_turns,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            cli_path=CLI_PATH,
            setting_sources=["user"],
            stderr=stderr_lines.append,
        )
        if no_tools:
            # No built-ins, and --strict-mcp-config with no --mcp-config loads
            # zero MCP servers (faster start, nothing for the model to call).
            options.tools = []
            options.extra_args = {"strict-mcp-config": None}

        if resume_session:
            options.resume = resume_session

        mode = "resume" if resume_session else "one-shot"
        preview = prompt.replace("\n", " ")[:120]
        logger.info("→ Agent (%s, %s): %s", mode, model, preview)

        text_parts: list[str] = []
        tools_used: list[str] = []
        result: ResultMessage | None = None
        rate_limited = False
        api_error: str | None = None

        try:
            async for message in query(prompt=prompt, options=options):
                if message is None:
                    continue
                if isinstance(message, RateLimitEvent):
                    rate_limited = True
                    continue
                if isinstance(message, AssistantMessage):
                    if message.error:
                        api_error = message.error
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                            if progress:
                                progress.note_text(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tools_used.append(block.name)
                            if progress:
                                progress.note_tool(block.name, block.input)
                            # Log tool name + input for debugging
                            input_preview = str(block.input)[:200]
                            logger.info("  tool: %s → %s", block.name, input_preview)
                elif isinstance(message, ResultMessage):
                    result = message
        except Exception as e:
            if stderr_lines:
                logger.warning("CLI stderr:\n%s", "\n".join(stderr_lines[-20:]))
            if any("unrecognized_model" in line for line in stderr_lines):
                raise _UnrecognizedModel(model) from e
            # The CLI reports API failures (expired login, billing, …) as a
            # final assistant message, then exits 1 — the SDK only surfaces
            # "exit code 1". Re-raise with the CLI's own explanation.
            if api_error:
                raise CLIError(api_error, "\n".join(text_parts).strip()) from e
            raise

        full_text = "\n".join(text_parts) if text_parts else ""

        if result is None:
            if rate_limited:
                logger.warning("Invocation cut short by API rate limit")
                error_msg = "I got rate-limited by the API — try again in a minute."
            else:
                logger.error("No ResultMessage from Agent SDK")
                error_msg = full_text or "No response received."
            return AgentResponse(
                text=error_msg,
                is_error=True,
                tools_used=tools_used,
            )

        response_text = result.result or full_text

        # Trace to Langfuse
        trace_invocation(
            name=trace_name or "agent-invocation",
            input_prompt=prompt,
            output_text=response_text,
            model=model,
            total_cost_usd=result.total_cost_usd,
            usage=result.usage,
            duration_ms=result.duration_ms,
            session_id=result.session_id,
            tags=trace_tags,
        )

        tools_summary = f", tools: {', '.join(tools_used)}" if tools_used else ""
        logger.info(
            "← %d turns, $%.4f, %.1fs%s",
            result.num_turns,
            result.total_cost_usd or 0,
            result.duration_ms / 1000,
            tools_summary,
        )

        return AgentResponse(
            text=response_text,
            session_id=result.session_id,
            total_cost_usd=result.total_cost_usd,
            usage=result.usage,
            duration_ms=result.duration_ms,
            num_turns=result.num_turns,
            is_error=result.is_error,
            tools_used=tools_used,
            narration=full_text,
            subtype=result.subtype,
        )


class CLIError(Exception):
    """The CLI reported an API error (e.g. authentication_failed) and exited."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


def _error_text(exc: BaseException) -> str:
    """User-facing text for a failed invocation."""
    if isinstance(exc, CLIError):
        if exc.kind == "authentication_failed":
            return (
                "Claude login has expired — run `claude /login` on SubrosaBox."
                f"\n({exc.detail or exc.kind})"
            )
        return f"Agent error ({exc}) — please try again."
    return "Agent error — please try again."


class _UnrecognizedModel(Exception):
    """The Claude CLI rejected the model id (it predates that model)."""


def _model_alias(model: str) -> str:
    """claude-opus-5-5 → opus, claude-sonnet-5-5 → sonnet; aliases unchanged."""
    for family in ("opus", "sonnet", "haiku"):
        if model.startswith(f"claude-{family}-"):
            return family
    return model


def _merge_responses(prior: AgentResponse, continued: AgentResponse) -> AgentResponse:
    """Combine a turn-capped response with its continuation."""
    if continued.is_error and continued.subtype not in ("error_max_turns",):
        # Continuation itself failed — keep what the first pass produced.
        text = prior.text
    else:
        text = "\n".join(t for t in (prior.text, continued.text) if t)
    return AgentResponse(
        text=text,
        session_id=continued.session_id or prior.session_id,
        total_cost_usd=(prior.total_cost_usd or 0) + (continued.total_cost_usd or 0),
        usage=continued.usage,
        duration_ms=prior.duration_ms + continued.duration_ms,
        num_turns=prior.num_turns + continued.num_turns,
        is_error=continued.is_error,
        tools_used=prior.tools_used + continued.tools_used,
        narration="\n".join(n for n in (prior.narration, continued.narration) if n),
        subtype=continued.subtype,
    )
