"""Single path to Claude — one invoke() function, no persistent sessions."""

from __future__ import annotations

import logging
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


class Agent:
    """The ONLY way Claude is invoked in the entire codebase."""

    def __init__(self, model: str = "sonnet", max_turns: int = 10):
        self.model = model
        self.max_turns = max_turns

    async def invoke(
        self,
        prompt: str,
        system_prompt: str,
        resume_session: str | None = None,
        trace_name: str = "",
        trace_tags: list[str] | None = None,
    ) -> AgentResponse:
        """Invoke Claude. Never raises — returns AgentResponse with is_error on failure."""
        try:
            return await self._do_invoke(
                prompt, system_prompt, resume_session, trace_name, trace_tags
            )
        except Exception:
            if resume_session:
                logger.warning("Resume failed, retrying as one-shot")
                try:
                    return await self._do_invoke(
                        prompt, system_prompt, None, trace_name, trace_tags
                    )
                except Exception:
                    logger.exception("Agent invocation failed (one-shot retry)")
            else:
                logger.exception("Agent invocation failed")
            return AgentResponse(text="Agent error — please try again.", is_error=True)

    async def _do_invoke(
        self,
        prompt: str,
        system_prompt: str,
        resume_session: str | None,
        trace_name: str,
        trace_tags: list[str] | None,
    ) -> AgentResponse:
        options = ClaudeAgentOptions(
            model=self.model,
            max_turns=self.max_turns,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            cli_path=CLI_PATH,
            setting_sources=["user"],
        )

        if resume_session:
            options.resume = resume_session

        mode = "resume" if resume_session else "one-shot"
        preview = prompt.replace("\n", " ")[:120]
        logger.info("→ Agent (%s): %s", mode, preview)

        text_parts: list[str] = []
        tools_used: list[str] = []
        result: ResultMessage | None = None

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tools_used.append(block.name)
                        # Log tool name + input for debugging
                        input_preview = str(block.input)[:200]
                        logger.info("  tool: %s → %s", block.name, input_preview)
            elif isinstance(message, ResultMessage):
                result = message

        full_text = "\n".join(text_parts) if text_parts else ""

        if result is None:
            logger.error("No ResultMessage from Agent SDK")
            return AgentResponse(
                text=full_text or "No response received.",
                is_error=True,
                tools_used=tools_used,
            )

        response_text = result.result or full_text

        # Trace to Langfuse
        trace_invocation(
            name=trace_name or "agent-invocation",
            input_prompt=prompt,
            output_text=response_text,
            model=self.model,
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
        )
