"""APScheduler — briefings and monitoring with direct agent calls."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from .agent import Agent
    from .config import Config
    from .health import Health
    from .store import Store
    from .telegram import TelegramBot

logger = logging.getLogger(__name__)


async def _briefing_job(
    agent: Agent,
    bot: TelegramBot,
    store: Store,
    health: Health,
    config: Config,
    kind: str = "morning",
) -> None:
    """Run a briefing and send to Telegram."""
    if bot.is_silenced:
        logger.info("Briefing skipped — silenced")
        return

    try:
        from .context import build_system_prompt, build_briefing_prompt
        system_prompt = build_system_prompt(config.briefing_path)
        prompt = await build_briefing_prompt(kind, store)

        response = await asyncio.wait_for(
            agent.invoke(prompt, system_prompt, trace_name=f"briefing-{kind}"),
            timeout=config.briefing_timeout,
        )
        health.record_agent()

        if response.is_error:
            await bot.send_to_chat(config.chat_id, f"Briefing ({kind}) failed: {response.text}")
        else:
            await bot.send_to_chat(config.chat_id, response.text)

        await store.log_event(
            source="scheduler", event_type=f"briefing_{kind}",
            summary=f"{kind} briefing",
            content=response.text[:500],
        )

    except asyncio.TimeoutError:
        logger.warning("Briefing %s timed out", kind)
        await bot.send_to_chat(config.chat_id, f"Briefing ({kind}) timed out.")
        await store.log_diagnostic("scheduler", f"briefing_{kind} timeout", level="warning")
    except Exception:
        logger.exception("Briefing %s failed", kind)
        await bot.send_to_chat(config.chat_id, f"Briefing ({kind}) failed.")
        await store.log_diagnostic("scheduler", f"briefing_{kind} error", level="error")


async def _skill_job(
    agent: Agent,
    bot: TelegramBot,
    store: Store,
    health: Health,
    config: Config,
    skill_name: str,
) -> None:
    """Execute a scheduled skill."""
    if bot.is_silenced:
        logger.info("Skill %s skipped — silenced", skill_name)
        return

    try:
        from .skills import load_skill
        skill = load_skill(skill_name)

        if not skill:
            logger.error("Skill not found: %s", skill_name)
            return

        from .context import build_system_prompt
        system_prompt = build_system_prompt(config.briefing_path)

        response = await asyncio.wait_for(
            agent.invoke(skill.instruction, system_prompt, trace_name=f"skill-{skill_name}"),
            timeout=config.briefing_timeout,
        )
        health.record_agent()

        if not response.is_error:
            await bot.send_to_chat(config.chat_id, response.text)

        await store.log_event(
            source="scheduler", event_type=f"skill_{skill_name}",
            summary=f"{skill_name} skill",
            content=response.text[:500],
        )

    except asyncio.TimeoutError:
        logger.warning("Skill %s timed out", skill_name)
        await bot.send_to_chat(config.chat_id, f"Skill '{skill_name}' timed out.")
        await store.log_diagnostic("scheduler", f"skill_{skill_name} timeout", level="warning")
    except Exception:
        logger.exception("Skill %s failed", skill_name)
        await bot.send_to_chat(config.chat_id, f"Skill '{skill_name}' failed.")
        await store.log_diagnostic("scheduler", f"skill_{skill_name} error", level="error")


async def _monitoring_job(
    agent: Agent,
    bot: TelegramBot,
    store: Store,
    health: Health,
    config: Config,
) -> None:
    """Run a monitoring cycle."""
    if bot.is_silenced:
        logger.info("Monitoring skipped — silenced")
        return

    try:
        from .context import build_system_prompt, build_monitoring_prompt
        system_prompt = build_system_prompt(config.briefing_path)
        prompt = build_monitoring_prompt(
            config.slack_channels, config.jira_projects, config.github_repos
        )

        response = await asyncio.wait_for(
            agent.invoke(prompt, system_prompt, trace_name="monitoring"),
            timeout=config.monitoring_timeout,
        )
        health.record_agent()

        # Only send if there are insights
        if response.text.strip() != "NO_INSIGHTS" and not response.is_error:
            await bot.send_to_chat(config.chat_id, response.text)
            await store.log_event(
                source="scheduler", event_type="monitoring",
                summary="monitoring insights",
                content=response.text[:500],
            )

    except asyncio.TimeoutError:
        logger.warning("Monitoring timed out")
        await store.log_diagnostic("scheduler", "monitoring timeout", level="warning")
    except Exception:
        logger.exception("Monitoring failed")
        await store.log_diagnostic("scheduler", "monitoring error", level="error")


class Scheduler:
    """APScheduler wrapper for all scheduled jobs."""

    def __init__(
        self,
        config: Config,
        agent: Agent,
        bot: TelegramBot,
        store: Store,
        health: Health,
    ):
        self._config = config
        self._agent = agent
        self._bot = bot
        self._store = store
        self._health = health
        self._scheduler = AsyncIOScheduler(timezone=config.timezone)
        self._apply_schedule()

    def _apply_schedule(self) -> None:
        c = self._config
        tz = c.timezone
        common = {
            "agent": self._agent,
            "bot": self._bot,
            "store": self._store,
            "health": self._health,
            "config": c,
        }

        # Morning briefing
        h, m = map(int, c.morning_briefing.split(":"))
        self._scheduler.add_job(
            _briefing_job,
            CronTrigger(hour=h, minute=m, timezone=tz),
            kwargs={**common, "kind": "morning"},
            id="morning_briefing", replace_existing=True,
        )

        # Noon briefing
        h, m = map(int, c.noon_briefing.split(":"))
        self._scheduler.add_job(
            _briefing_job,
            CronTrigger(hour=h, minute=m, timezone=tz),
            kwargs={**common, "kind": "noon"},
            id="noon_briefing", replace_existing=True,
        )

        # Evening digest
        h, m = map(int, c.evening_digest.split(":"))
        self._scheduler.add_job(
            _briefing_job,
            CronTrigger(hour=h, minute=m, timezone=tz),
            kwargs={**common, "kind": "evening"},
            id="evening_digest", replace_existing=True,
        )

        # Monitoring poll (during work hours)
        interval = c.monitoring_interval_minutes
        if interval > 0:
            work_start_h = int(c.work_hours_start.split(":")[0])
            work_end_h = int(c.work_hours_end.split(":")[0])
            self._scheduler.add_job(
                _monitoring_job,
                CronTrigger(
                    minute=f"*/{interval}",
                    hour=f"{work_start_h}-{work_end_h}",
                    timezone=tz,
                ),
                kwargs=common,
                id="monitoring_poll", replace_existing=True,
            )

        # Skill schedules
        from .skills import get_skill_schedules
        skill_schedules = get_skill_schedules()

        for skill_name, schedules in skill_schedules.items():
            for label, cron_expr in schedules.items():
                job_id = f"skill_{skill_name}_{label}"
                self._scheduler.add_job(
                    _skill_job,
                    CronTrigger.from_crontab(cron_expr, timezone=tz),
                    kwargs={**common, "skill_name": skill_name},
                    id=job_id, replace_existing=True,
                )
                logger.info("Scheduled skill: %s (%s) — %s", skill_name, label, cron_expr)

        logger.info(
            "Schedule: morning=%s, noon=%s, evening=%s, monitoring=%s (%s-%s %s)",
            c.morning_briefing, c.noon_briefing, c.evening_digest,
            f"every {interval}m" if interval > 0 else "disabled",
            c.work_hours_start, c.work_hours_end, tz,
        )

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = False) -> None:
        self._scheduler.shutdown(wait=wait)
        logger.info("Scheduler stopped")
