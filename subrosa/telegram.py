"""Telegram bot — handlers, WorkingIndicator, send, FIFO queue."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Callable

from telegram import Bot, Update
from telegram.constants import ChatAction, MessageLimit, ParseMode
from telegram.error import BadRequest, Forbidden, InvalidToken, RetryAfter
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

if TYPE_CHECKING:
    from telegram import Message

    from .agent import Agent
    from .config import Config
    from .health import Health
    from .memory import MemoryExtractor
    from .procedures import ProcedureManager
    from .store import Store

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 2.0
_MAX_QUEUE = 5
MAX_LENGTH = MessageLimit.MAX_TEXT_LENGTH  # 4096

# Heartbeat schedule: 30s, 2m, 10m, 30m
_UPDATE_SCHEDULE = [30, 120, 600, 1800]


# ── Send helpers ────────────────────────────────────────────────────────────

def _markdown_to_html(text: str) -> str:
    text = re.sub(r"```(\w*)\n(.*?)```", r"<pre>\2</pre>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    return text


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def split_message(text: str, max_length: int = MAX_LENGTH) -> list[str]:
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        chunk = text[:max_length]
        split_pos = chunk.rfind("\n")
        if split_pos == -1 or split_pos < max_length // 2:
            split_pos = chunk.rfind(". ")
            if split_pos != -1:
                split_pos += 1
        if split_pos == -1 or split_pos < max_length // 2:
            split_pos = chunk.rfind(" ")
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


async def _send_with_retry(
    bot: Bot, chat_id: int, text: str, parse_mode: str | None = None, max_retries: int = 3
) -> bool:
    for attempt in range(max_retries + 1):
        try:
            await asyncio.wait_for(
                bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode),
                timeout=30,
            )
            return True
        except (BadRequest, Forbidden, InvalidToken):
            if parse_mode:
                return False
            logger.warning("Permanent Telegram error sending to %s", chat_id, exc_info=True)
            return False
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except Exception:
            if attempt == max_retries:
                logger.warning("Failed to send after %d attempts", max_retries + 1, exc_info=True)
                return False
            await asyncio.sleep(1 * (2 ** attempt))
    return False


async def send_message(bot: Bot, chat_id: int, text: str) -> None:
    """Send message with HTML, chunking, and retry. Never raises."""
    if not text or not text.strip():
        return
    html = _markdown_to_html(text)
    for chunk in split_message(html):
        sent = await _send_with_retry(bot, chat_id, chunk, parse_mode=ParseMode.HTML)
        if not sent:
            await _send_with_retry(bot, chat_id, _strip_html(chunk))


# ── WorkingIndicator ────────────────────────────────────────────────────────

class WorkingIndicator:
    """'Working on it...' with exponential backoff heartbeat."""

    def __init__(self, bot: Bot, chat_id: int):
        self._bot = bot
        self._chat_id = chat_id
        self._message_id: int | None = None
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        try:
            msg = await self._bot.send_message(chat_id=self._chat_id, text="Working on it...")
            self._message_id = msg.message_id
        except Exception:
            logger.debug("Failed to send working indicator", exc_info=True)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        try:
            labels = ["30s", "2m", "10m", "30m"]
            for delay, label in zip(_UPDATE_SCHEDULE, labels):
                await asyncio.sleep(delay)
                await self._edit(f"Still working... ({label})")
            while True:
                await asyncio.sleep(1800)
                await self._edit("Still working... (30m+)")
        except asyncio.CancelledError:
            pass

    async def _edit(self, text: str) -> None:
        if not self._message_id:
            return
        try:
            await self._bot.edit_message_text(
                text=text, chat_id=self._chat_id, message_id=self._message_id
            )
        except Exception:
            logger.debug("Working indicator edit failed", exc_info=True)

    async def finalize(self, final_text: str) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._message_id:
            try:
                await asyncio.wait_for(
                    self._bot.delete_message(chat_id=self._chat_id, message_id=self._message_id),
                    timeout=10,
                )
            except Exception:
                logger.debug("Failed to delete working indicator", exc_info=True)
        await send_message(self._bot, self._chat_id, final_text)

    async def delete(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._message_id:
            try:
                await self._bot.delete_message(chat_id=self._chat_id, message_id=self._message_id)
            except Exception:
                pass
            self._message_id = None


# ── Bot ─────────────────────────────────────────────────────────────────────

class TelegramBot:
    def __init__(
        self,
        config: Config,
        agent: Agent,
        store: Store,
        health: Health,
        memory_extractor: MemoryExtractor | None = None,
        procedure_manager: ProcedureManager | None = None,
    ):
        self._config = config
        self._agent = agent
        self._store = store
        self._health = health
        self._memory_extractor = memory_extractor
        self._procedure_manager = procedure_manager
        self._app: Application | None = None

        self._pending: dict[int, tuple[list[str], asyncio.Task]] = {}
        self._current_task: str | None = None
        self._queued: list[tuple[int, str, list[dict] | None]] = []
        self._silence_until: float = 0.0

    @property
    def app(self) -> Application | None:
        return self._app

    def _restricted(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            if update.effective_chat.id != self._config.chat_id:
                logger.warning("Unauthorized: chat_id=%s", update.effective_chat.id)
                return
            return await func(update, context, *args, **kwargs)
        return wrapper

    # ── Message handling ────────────────────────────────────────────────

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._health.record_poll()
        chat_id = update.effective_chat.id
        text = update.message.text

        if chat_id in self._pending:
            texts, task = self._pending[chat_id]
            texts.append(text)
            task.cancel()
        else:
            self._pending[chat_id] = ([text], None)

        texts = self._pending[chat_id][0]

        async def dispatch():
            await asyncio.sleep(_DEBOUNCE_SECONDS)
            combined = "\n".join(texts)
            del self._pending[chat_id]
            await self._process_message(chat_id, combined)

        task = asyncio.create_task(dispatch())
        self._pending[chat_id] = (texts, task)

    async def _handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._health.record_poll()
        from .media import download_media
        chat_id = update.effective_chat.id
        try:
            media_files = await download_media(self._app.bot, update.message)
        except Exception:
            logger.exception("Failed to download media")
            await send_message(self._app.bot, chat_id, "Failed to download media.")
            return
        if not media_files:
            return
        text = update.message.caption or ""
        await self._process_message(chat_id, text, media_files=media_files)

    async def _process_message(self, chat_id: int, text: str, media_files: list[dict] | None = None) -> None:
        # Check for skill trigger
        if text.startswith("/") and not media_files:
            from .skills import find_skill_by_trigger
            skill = find_skill_by_trigger(text.split()[0])
            if skill:
                await self._execute_skill(chat_id, skill)
                return

        # Status bypass
        if self._is_status_check(text) and not media_files:
            await self._reply_inline_status(chat_id)
            return

        # Queue if busy
        if self._current_task is not None:
            if len(self._queued) >= _MAX_QUEUE:
                await send_message(self._app.bot, chat_id, "Too many queued messages. Try again later.")
                return
            self._queued.append((chat_id, text, media_files))
            pos = len(self._queued)
            await send_message(
                self._app.bot, chat_id,
                f"Working on: {self._current_task}\nYour message is queued ({pos} in queue).",
            )
            return

        await self._do_process(chat_id, text, media_files)
        await self._drain_queue()

    async def _execute_skill(self, chat_id: int, skill) -> None:
        """Execute a skill and send response."""
        indicator = WorkingIndicator(self._app.bot, chat_id)
        try:
            await indicator.start()
            from .context import build_system_prompt
            system_prompt = build_system_prompt(self._config.briefing_path)

            # Inject skill instruction as user prompt
            response = await asyncio.wait_for(
                self._agent.invoke(skill.instruction, system_prompt, trace_name=f"skill-{skill.name}"),
                timeout=self._config.briefing_timeout,
            )
            self._health.record_agent()
            await indicator.finalize(response.text)
        except asyncio.TimeoutError:
            await indicator.delete()
            await send_message(self._app.bot, chat_id, f"Skill '{skill.name}' timed out.")
        except Exception:
            logger.exception("Skill execution failed: %s", skill.name)
            await indicator.delete()
            await send_message(self._app.bot, chat_id, f"Failed to execute skill '{skill.name}'.")

    def _is_status_check(self, text: str) -> bool:
        n = text.strip().lower().rstrip("?!.")
        return n in {
            "status", "what are you doing", "what's going on", "whats going on",
            "busy", "are you busy", "you busy", "wyd", "sup", "what's up", "whats up",
        }

    async def _reply_inline_status(self, chat_id: int) -> None:
        msg = f"Working on: {self._current_task}" if self._current_task else "Idle — ready for messages."
        if self._queued:
            msg += f"\nQueue: {len(self._queued)} waiting"
        await send_message(self._app.bot, chat_id, msg)

    async def _drain_queue(self) -> None:
        while self._queued:
            chat_id, text, media = self._queued.pop(0)
            try:
                await self._do_process(chat_id, text, media)
            except Exception:
                logger.exception("Failed to process queued message")

    async def _do_process(self, chat_id: int, text: str, media_files: list[dict] | None = None) -> None:
        """The critical path: user message → agent → response."""
        self._current_task = text[:60] + ("..." if len(text) > 60 else "")
        indicator = WorkingIndicator(self._app.bot, chat_id)
        start = time.monotonic()
        try:
            await indicator.start()

            # Build prompts
            from .context import build_system_prompt, build_user_prompt
            system_prompt = build_system_prompt(self._config.briefing_path)

            # Check for explicit memory request
            from .memory import detect_explicit_memory_request, create_explicit_memory
            mem_content = detect_explicit_memory_request(text)
            if mem_content:
                mem = await create_explicit_memory(
                    self._store, mem_content, self._config.known_topics
                )
                await indicator.finalize(
                    f"Remembered: **{mem['subject']}** ({mem['topic_type']})\n_{mem['content']}_"
                )
                self._current_task = None
                return

            # Check for explicit procedure save request
            if self._procedure_manager:
                from .procedures import detect_procedure_request
                if detect_procedure_request(text):
                    if self._procedure_manager.schedule_reflection():
                        await indicator.finalize("Reflecting on our last interaction and saving a procedure...")
                    else:
                        await indicator.finalize("Nothing recent to save — try after a complex task.")
                    self._current_task = None
                    return

            # Get conversation for session resumption
            conv = await self._store.get_or_create_conversation(chat_id)
            resume_session = conv.get("agent_session_id") or None

            user_prompt = await build_user_prompt(
                text, self._store,
                known_topics=self._config.known_topics,
                max_memories=self._config.max_memories_per_prompt,
                max_memory_tokens=self._config.max_memory_tokens,
                media_files=media_files,
                procedure_manager=self._procedure_manager,
            )

            # Invoke agent with timeout
            response = await asyncio.wait_for(
                self._agent.invoke(
                    user_prompt, system_prompt,
                    resume_session=resume_session,
                    trace_name="interactive",
                ),
                timeout=self._config.agent_timeout,
            )

            self._health.record_agent()
            self._health.record_response()

            # Update conversation with session ID (or clear stale one)
            if response.session_id:
                await self._store.update_conversation(
                    conv["id"], agent_session_id=response.session_id
                )
            elif resume_session:
                # Resume failed and fell back to one-shot — clear stale session
                await self._store.update_conversation(
                    conv["id"], agent_session_id=""
                )

            # Log event
            await self._store.log_event(
                source="telegram", event_type="message",
                summary=text[:200],
                content=response.text[:1000],
                session_id=response.session_id,
            )

            # Add timing footer for long responses
            elapsed = time.monotonic() - start
            result = response.text
            if elapsed > 10:
                tools = f", {len(response.tools_used)} tools" if response.tools_used else ""
                result += f"\n\n— {elapsed:.0f}s{tools}"

            await indicator.finalize(result)

            # Schedule background memory extraction
            if self._memory_extractor and self._config.implicit_extraction:
                self._memory_extractor.schedule_extraction(text, response.text)

            # Record exchange for potential procedure reflection
            if self._procedure_manager and response.tools_used:
                self._procedure_manager.record_exchange(
                    user_request=text,
                    tools_used=response.tools_used,
                    narration=response.narration,
                    final_response=response.text,
                )

        except asyncio.TimeoutError:
            await indicator.delete()
            timeout_min = self._config.agent_timeout // 60
            await send_message(
                self._app.bot, chat_id,
                f"Agent timed out ({timeout_min}m). Try again or /stop.",
            )
        except Exception:
            logger.exception("Error processing message")
            await indicator.delete()
            await send_message(self._app.bot, chat_id, "Something went wrong.")
        finally:
            self._current_task = None

    # ── Commands ────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await send_message(self._app.bot, update.effective_chat.id, "Subrosa online. How can I help?")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        h = self._health.status_dict()
        lines = ["<b>Subrosa Status</b>\n"]

        if self._current_task:
            lines.append(f"Working on: {self._current_task}")
        else:
            lines.append("Idle — ready for messages")

        if self._queued:
            lines.append(f"Queue: {len(self._queued)} waiting")

        lines.append(f"Uptime: {h['uptime_seconds'] // 3600}h {(h['uptime_seconds'] % 3600) // 60}m")
        lines.append(f"Healthy: {'yes' if h['healthy'] else 'NO'}")

        if h.get("last_response_ago"):
            lines.append(f"Last response: {h['last_response_ago']}s ago")

        # Memory count
        try:
            count = await self._store.memory_count()
            lines.append(f"Memories: {count}")
        except Exception:
            pass

        # Recent diagnostics
        try:
            diags = await self._store.get_recent_diagnostics(hours=24)
            issues = [d for d in diags if d["level"] in ("warning", "error")]
            if issues:
                lines.append("\n<b>Recent issues</b>")
                now = datetime.now(UTC)
                for d in issues[:5]:
                    ts = datetime.fromisoformat(d["timestamp"]).replace(tzinfo=UTC)
                    delta = now - ts
                    ago = f"{delta.total_seconds() / 60:.0f}m" if delta.total_seconds() < 3600 else f"{delta.total_seconds() / 3600:.0f}h"
                    lines.append(f"  {ago} ago: {d['component']} — {d['event']}")
        except Exception:
            pass

        await send_message(self._app.bot, chat_id, "\n".join(lines))

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        kind = context.args[0] if context.args else "morning"
        indicator = WorkingIndicator(self._app.bot, chat_id)
        try:
            await indicator.start()
            from .context import build_system_prompt, build_briefing_prompt
            system_prompt = build_system_prompt(self._config.briefing_path)
            prompt = await build_briefing_prompt(kind, self._store)
            response = await asyncio.wait_for(
                self._agent.invoke(prompt, system_prompt, trace_name=f"briefing-{kind}"),
                timeout=self._config.briefing_timeout,
            )
            self._health.record_agent()
            await indicator.finalize(response.text)
        except asyncio.TimeoutError:
            await indicator.delete()
            await send_message(self._app.bot, chat_id, "Briefing timed out.")
        except Exception:
            logger.exception("Briefing failed")
            await indicator.delete()
            await send_message(self._app.bot, chat_id, "Failed to generate briefing.")

    async def _cmd_silence(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        duration = int(context.args[0]) if context.args else 60
        self._silence_until = time.monotonic() + duration * 60
        await send_message(
            self._app.bot, update.effective_chat.id,
            f"Monitoring silenced for {duration} minutes.",
        )

    async def _cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        text = " ".join(context.args) if context.args else ""
        if not text:
            await send_message(self._app.bot, chat_id, "<b>Usage:</b> /remember &lt;fact&gt;")
            return
        from .memory import create_explicit_memory
        mem = await create_explicit_memory(self._store, text, self._config.known_topics)
        await send_message(
            self._app.bot, chat_id,
            f"Remembered: <b>{mem['subject']}</b> ({mem['topic_type']})\n<i>{mem['content']}</i>",
        )

    async def _cmd_memories(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        arg = context.args[0] if context.args else ""

        if arg.startswith("--type="):
            type_str = arg.split("=", 1)[1]
            memories = await self._store.list_memories(topic_type=type_str, limit=20)
        elif arg:
            memories = await self._store.get_memories_by_topics([arg.lower()], limit=20)
        else:
            memories = await self._store.list_memories(limit=20)

        if not memories:
            await send_message(self._app.bot, chat_id, "No memories found.")
            return

        lines = [f"<b>Memories</b> ({len(memories)})\n"]
        for m in memories:
            tags = ", ".join(m.get("tags", []))
            line = f"#{m['id']} [{m['topic_type']}] <b>{m['subject']}</b>: {m['content']}"
            if tags:
                line += f" <i>({tags})</i>"
            lines.append(line)
        await send_message(self._app.bot, chat_id, "\n".join(lines))

    async def _cmd_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        if not context.args:
            await send_message(self._app.bot, chat_id, "<b>Usage:</b> /forget &lt;id&gt;")
            return
        try:
            memory_id = int(context.args[0])
        except ValueError:
            await send_message(self._app.bot, chat_id, "Invalid ID.")
            return
        mem = await self._store.get_memory(memory_id)
        if not mem or not mem.get("active"):
            await send_message(self._app.bot, chat_id, f"Memory #{memory_id} not found.")
            return
        await self._store.delete_memory(memory_id)
        await send_message(self._app.bot, chat_id, f"Forgotten: #{memory_id} <b>{mem['subject']}</b>")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        if self._queued:
            count = len(self._queued)
            self._queued.clear()
            await send_message(self._app.bot, chat_id, f"Cleared {count} queued message(s).")
        else:
            await send_message(self._app.bot, chat_id, "Nothing queued.")

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await send_message(self._app.bot, update.effective_chat.id, "Restarting...")
        pid = os.getpid()
        loop = asyncio.get_event_loop()
        loop.call_later(0.5, lambda: os.kill(pid, signal.SIGTERM))

    async def _cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        lines = [
            "<b>Schedule</b>",
            f"Morning: {self._config.morning_briefing}",
            f"Noon: {self._config.noon_briefing}",
            f"Evening: {self._config.evening_digest}",
            f"Monitoring: every {self._config.monitoring_interval_minutes}m",
            f"Work hours: {self._config.work_hours_start}–{self._config.work_hours_end}",
            f"Timezone: {self._config.timezone}",
        ]
        await send_message(self._app.bot, update.effective_chat.id, "\n".join(lines))

    # ── Build app ───────────────────────────────────────────────────────

    def build_app(self) -> Application:
        self._app = ApplicationBuilder().token(self._config.bot_token).build()

        commands = {
            "start": self._cmd_start,
            "status": self._cmd_status,
            "briefing": self._cmd_briefing,
            "silence": self._cmd_silence,
            "remember": self._cmd_remember,
            "memories": self._cmd_memories,
            "forget": self._cmd_forget,
            "stop": self._cmd_stop,
            "restart": self._cmd_restart,
            "schedule": self._cmd_schedule,
        }
        for name, handler in commands.items():
            self._app.add_handler(CommandHandler(name, self._restricted(handler)))

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._restricted(self._handle_text))
        )
        self._app.add_handler(
            MessageHandler(
                filters.PHOTO | filters.Document.ALL | filters.VIDEO | filters.AUDIO | filters.VOICE,
                self._restricted(self._handle_media),
            )
        )

        async def _error_handler(update, context):
            logger.error("Uncaught handler error: %s", context.error, exc_info=context.error)

        self._app.add_error_handler(_error_handler)
        return self._app

    async def send_to_chat(self, chat_id: int, text: str) -> None:
        if self._app:
            await send_message(self._app.bot, chat_id, text)

    @property
    def is_silenced(self) -> bool:
        return time.monotonic() < self._silence_until
