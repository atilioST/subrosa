"""Terminal REPL for talking to Subrosa without Telegram."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .config import load_config
from .context import build_system_prompt, build_user_prompt, build_briefing_prompt
from .health import Health, init_langfuse, shutdown_langfuse
from .memory import (
    MemoryExtractor,
    create_explicit_memory,
    detect_explicit_memory_request,
)
from .store import Store

logger = logging.getLogger(__name__)

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _print(text: str) -> None:
    print(f"\n{_CYAN}{text}{_RESET}\n")


async def _repl() -> None:
    env_path = Path("~/.subrosa/.env").expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    try:
        config = load_config()
    except FileNotFoundError:
        print("Config not found at ~/.subrosa/config.toml", file=sys.stderr)
        sys.exit(1)

    init_langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )

    store = Store()
    await store.initialize()

    health = Health()
    agent = Agent(model=config.model, max_turns=config.max_turns)
    system_prompt = build_system_prompt(config.briefing_path)

    print(f"{_BOLD}Subrosa CLI{_RESET}")
    print(f"{_DIM}Commands: /briefing, /status, /remember, /memories, /forget, /tasks, /quit{_RESET}\n")

    session_id: str | None = None

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(f"{_BOLD}you>{_RESET} ")
            )
        except (EOFError, KeyboardInterrupt):
            break

        text = user_input.strip()
        if not text:
            continue

        if text in ("/quit", "/exit", "/q"):
            break

        if text == "/status":
            h = health.status_dict()
            count = await store.memory_count()
            print(f"\n{_BOLD}Status{_RESET}")
            print(f"  Uptime: {h['uptime_seconds'] // 60}m")
            print(f"  Memories: {count}")
            if session_id:
                print(f"  Session: {session_id[:12]}...")
            print()
            continue

        if text.startswith("/briefing"):
            parts = text.split(maxsplit=1)
            kind = parts[1] if len(parts) > 1 else "morning"
            print(f"{_DIM}Generating {kind} briefing...{_RESET}")
            prompt = await build_briefing_prompt(kind, store)
            response = await agent.invoke(prompt, system_prompt, trace_name=f"briefing-{kind}")
            _print(response.text)
            continue

        if text.startswith("/remember"):
            content = text[len("/remember"):].strip()
            if not content:
                print(f"{_DIM}Usage: /remember <fact>{_RESET}")
                continue
            mem = await create_explicit_memory(store, content, config.known_topics)
            print(f"\n{_CYAN}Remembered: {mem['subject']} ({mem['topic_type']}){_RESET}")
            print(f"{_DIM}{mem['content']}{_RESET}\n")
            continue

        if text.startswith("/memories"):
            arg = text[len("/memories"):].strip()
            if arg.startswith("--type="):
                type_str = arg.split("=", 1)[1]
                memories = await store.list_memories(topic_type=type_str, limit=20)
            elif arg:
                memories = await store.get_memories_by_topics([arg.lower()], limit=20)
            else:
                memories = await store.list_memories(limit=20)
            if not memories:
                print(f"{_DIM}No memories found.{_RESET}")
                continue
            print(f"\n{_BOLD}Memories ({len(memories)}){_RESET}")
            for m in memories:
                tags = ", ".join(m.get("tags", []))
                print(f"  #{m['id']} [{m['topic_type']}] {_BOLD}{m['subject']}{_RESET}: {m['content']}{_DIM} ({tags}){_RESET}")
            print()
            continue

        if text.startswith("/forget"):
            parts = text.split()
            if len(parts) < 2:
                print(f"{_DIM}Usage: /forget <id>{_RESET}")
                continue
            try:
                memory_id = int(parts[1])
            except ValueError:
                print(f"{_DIM}Invalid ID.{_RESET}")
                continue
            mem = await store.get_memory(memory_id)
            if not mem or not mem.get("active"):
                print(f"{_DIM}Memory #{memory_id} not found.{_RESET}")
                continue
            await store.delete_memory(memory_id)
            print(f"\n{_CYAN}Forgotten: #{memory_id} {mem['subject']}{_RESET}\n")
            continue

        if text == "/tasks":
            tasks = await store.get_active_tasks()
            if not tasks:
                print(f"\n{_DIM}No active tasks.{_RESET}\n")
                continue
            print(f"\n{_BOLD}Tasks ({len(tasks)}){_RESET}")
            for t in tasks:
                print(f"  #{t['id']} [{t['status']}] {t['title']}")
            print()
            continue

        # Check for explicit memory
        mem_content = detect_explicit_memory_request(text)
        if mem_content:
            mem = await create_explicit_memory(store, mem_content, config.known_topics)
            print(f"\n{_CYAN}Remembered: {mem['subject']} ({mem['topic_type']}){_RESET}")
            print(f"{_DIM}{mem['content']}{_RESET}\n")
            continue

        # Build prompt and invoke
        print(f"{_DIM}Thinking...{_RESET}")
        user_prompt = await build_user_prompt(
            text, store,
            known_topics=config.known_topics,
            max_memories=config.max_memories_per_prompt,
            max_memory_tokens=config.max_memory_tokens,
        )
        response = await agent.invoke(
            user_prompt, system_prompt,
            resume_session=session_id,
            trace_name="cli",
        )
        health.record_agent()
        health.record_response()

        if response.session_id:
            session_id = response.session_id

        _print(response.text)

    await store.close()
    shutdown_langfuse()
    print(f"\n{_DIM}Goodbye.{_RESET}")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=logging.WARNING,
    )
    logging.getLogger("subrosa.agent").setLevel(logging.INFO)

    try:
        asyncio.run(_repl())
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
