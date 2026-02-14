"""Subrosa entry point — loads config, wires subsystems, runs the bot."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .config import load_config
from .health import Health, init_langfuse, shutdown_langfuse
from .memory import MemoryExtractor
from .procedures import ProcedureManager
from .scheduler import Scheduler
from .store import Store
from .telegram import TelegramBot, send_message

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=log_level,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


async def _async_main() -> None:
    # Load .env
    env_path = Path("~/.subrosa/.env").expanduser()
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)

    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error(
            "Config file not found at ~/.subrosa/config.toml. "
            "Copy config/config.example.toml and configure it."
        )
        sys.exit(1)

    # Reconfigure logging from config
    _setup_logging(config.log_level)

    # Initialize Langfuse (optional)
    init_langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )

    # Initialize store
    store = Store()
    await store.initialize()

    # Clear stale session IDs — old CLI subprocesses are dead after restart
    cleared = await store.clear_all_sessions()
    if cleared:
        logger.info("Cleared %d stale agent session(s)", cleared)

    # Create core objects
    health = Health()
    agent = Agent(model=config.model, max_turns=config.max_turns)
    memory_extractor = MemoryExtractor(
        store=store,
        extraction_model=config.extraction_model,
        known_topics=config.known_topics,
    ) if config.memory_enabled and config.implicit_extraction else None

    procedure_manager = ProcedureManager(
        store=store,
        reflection_model=config.reflection_model,
        min_tool_calls=config.min_tool_calls,
        similarity_threshold=config.procedure_similarity_threshold,
        update_threshold=config.procedure_update_threshold,
        max_per_prompt=config.max_procedures_per_prompt,
    ) if config.procedures_enabled else None

    # Create Telegram bot
    bot = TelegramBot(
        config=config,
        agent=agent,
        store=store,
        health=health,
        memory_extractor=memory_extractor,
        procedure_manager=procedure_manager,
    )
    app = bot.build_app()

    # Create scheduler
    scheduler = Scheduler(
        config=config,
        agent=agent,
        bot=bot,
        store=store,
        health=health,
    )

    # Initialize Telegram (retry on network errors)
    for attempt in range(1, 6):
        try:
            await asyncio.wait_for(app.initialize(), timeout=10)
            break
        except Exception as e:
            if attempt == 5:
                logger.error("Failed to connect to Telegram after 5 attempts")
                raise
            wait = attempt * 10
            logger.warning(
                "Telegram connect failed (attempt %d/5): %s — retrying in %ds",
                attempt, e, wait,
            )
            await asyncio.sleep(wait)

    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    health.record_poll()

    # Start scheduler
    scheduler.start()

    # Send startup notification
    if config.chat_id:
        try:
            await asyncio.sleep(1)
            await send_message(app.bot, config.chat_id, "Subrosa online")
        except Exception:
            logger.warning("Failed to send startup notification", exc_info=True)

    await store.log_diagnostic("app", "startup complete")
    logger.info("Subrosa is online")

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    await store.log_diagnostic("app", "shutdown started")

    scheduler.shutdown(wait=False)

    if app.updater.running:
        await app.updater.stop()
    await app.stop()
    await app.shutdown()

    # Send offline notification (best-effort, 5s timeout)
    if config.chat_id:
        try:
            from telegram import Bot
            offline_bot = Bot(token=config.bot_token)
            await asyncio.wait_for(
                offline_bot.send_message(chat_id=config.chat_id, text="Subrosa offline"),
                timeout=5,
            )
        except Exception:
            pass

    await store.close()
    shutdown_langfuse()
    logger.info("Subrosa stopped")


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    main()
