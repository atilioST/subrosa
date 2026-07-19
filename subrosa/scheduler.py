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
    from .distiller import Distiller
    from .health import Health
    from .store import Store
    from .telegram import TelegramBot

logger = logging.getLogger(__name__)

# Watermark: ISO-8601 UTC timestamp of the last scan alert we actually SENT.
# We only report deltas since this point, and only advance it when we send —
# so a quiet scan never skips activity and never produces a message.
_SCAN_WATERMARK_KEY = "hourly_scan_last_sent"


async def _distill_job(distiller: Distiller, store: Store) -> None:
    """Scheduled distillation — drain new events into the knowledge table."""
    try:
        await distiller.run()
    except Exception:
        logger.exception("Scheduled distillation failed")
        await store.log_diagnostic("distiller", "scheduled run failed", level="error")


async def _hourly_scan_job(
    agent: Agent,
    bot: TelegramBot,
    store: Store,
    health: Health,
    config: Config,
    distiller: Distiller | None = None,
) -> None:
    """Hourly Slack alert scan — sends only when something meets the criteria."""
    if bot.is_silenced:
        logger.info("Hourly scan skipped — silenced")
        return

    from datetime import UTC, datetime, timedelta
    from zoneinfo import ZoneInfo

    now = datetime.now(UTC)
    last = await store.get_meta(_SCAN_WATERMARK_KEY)
    since_dt = now - timedelta(minutes=config.hourly_scan_interval_minutes)
    if last:
        try:
            since_dt = datetime.fromisoformat(last)
        except ValueError:
            logger.warning("Bad scan watermark %r — falling back to interval", last)

    tz = ZoneInfo(config.timezone)
    since_local = since_dt.astimezone(tz)

    try:
        from .context import build_system_prompt, build_hourly_scan_prompt
        system_prompt = build_system_prompt(config.briefing_path)
        prompt = await build_hourly_scan_prompt(
            since_human=since_local.strftime("%Y-%m-%d %H:%M %Z"),
            since_date=since_local.strftime("%Y-%m-%d"),
        )

        response = await asyncio.wait_for(
            agent.invoke(
                prompt, system_prompt,
                trace_name="hourly-scan",
                max_turns=config.briefing_max_turns,
            ),
            timeout=config.briefing_timeout,
        )
        health.record_agent()

        text = response.text.strip()

        if response.is_error:
            logger.warning("Hourly scan agent error: %s", text[:200])
            await store.log_diagnostic("scheduler", "hourly_scan error", level="error")
            return  # do not advance watermark — retry the same window next hour

        if not text or text == "NO_CHANGES":
            logger.info("Hourly scan — no changes since %s", since_dt.isoformat())
            return  # stay silent; watermark unchanged so nothing gets skipped

        await bot.send_to_chat(config.chat_id, response.text)
        await store.set_meta(_SCAN_WATERMARK_KEY, now.isoformat())
        await store.log_event(
            source="scheduler", event_type="hourly_scan",
            summary="hourly Slack alert scan",
            content=response.text[:4000],
        )

    except asyncio.TimeoutError:
        logger.warning("Hourly scan timed out")
        await store.log_diagnostic("scheduler", "hourly_scan timeout", level="warning")
    except Exception:
        logger.exception("Hourly scan failed")
        await store.log_diagnostic("scheduler", "hourly_scan error", level="error")

    # Distill freshly logged events right away (no-op if nothing new)
    if distiller is not None:
        distiller.schedule()


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
            agent.invoke(
                prompt, system_prompt,
                trace_name=f"briefing-{kind}",
                max_turns=config.briefing_max_turns,
            ),
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
            content=response.text[:4000],
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
            content=response.text[:4000],
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
    distiller: Distiller | None = None,
) -> None:
    """Run a monitoring cycle, then schedule distillation of new events."""
    if bot.is_silenced:
        logger.info("Monitoring skipped — silenced")
        return

    try:
        from .context import build_system_prompt, build_monitoring_prompt
        system_prompt = build_system_prompt(config.briefing_path)
        prompt = await build_monitoring_prompt(
            config.slack_channels, config.jira_projects, config.github_repos,
            store=store,
            monitoring_interval_minutes=config.monitoring_interval_minutes,
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
                content=response.text[:4000],
            )

    except asyncio.TimeoutError:
        logger.warning("Monitoring timed out")
        await store.log_diagnostic("scheduler", "monitoring timeout", level="warning")
    except Exception:
        logger.exception("Monitoring failed")
        await store.log_diagnostic("scheduler", "monitoring error", level="error")

    # Fire-and-forget distillation after each monitoring poll
    if distiller is not None:
        distiller.schedule()


class Scheduler:
    """APScheduler wrapper for all scheduled jobs."""

    def __init__(
        self,
        config: Config,
        agent: Agent,
        bot: TelegramBot,
        store: Store,
        health: Health,
        distiller: Distiller | None = None,
    ):
        self._config = config
        self._agent = agent
        self._bot = bot
        self._store = store
        self._health = health
        self._distiller = distiller
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

        # Day-of-week filter (mon-fri if weekdays_only)
        dow = "mon-fri" if c.weekdays_only else None

        # Legacy fixed-time briefings — disabled by default, kept for opt-in.
        if c.briefings_enabled:
            # Morning briefing
            h, m = map(int, c.morning_briefing.split(":"))
            self._scheduler.add_job(
                _briefing_job,
                CronTrigger(hour=h, minute=m, day_of_week=dow, timezone=tz),
                kwargs={**common, "kind": "morning"},
                id="morning_briefing", replace_existing=True,
            )

            # Noon briefing
            h, m = map(int, c.noon_briefing.split(":"))
            self._scheduler.add_job(
                _briefing_job,
                CronTrigger(hour=h, minute=m, day_of_week=dow, timezone=tz),
                kwargs={**common, "kind": "noon"},
                id="noon_briefing", replace_existing=True,
            )

            # Evening digest
            h, m = map(int, c.evening_digest.split(":"))
            self._scheduler.add_job(
                _briefing_job,
                CronTrigger(hour=h, minute=m, day_of_week=dow, timezone=tz),
                kwargs={**common, "kind": "evening"},
                id="evening_digest", replace_existing=True,
            )

        # Hourly Slack alert scan (delta-only), on the hour with jitter,
        # work hours, weekdays.
        if c.hourly_scan_enabled:
            start_h = int(c.hourly_scan_start.split(":")[0])
            end_h = int(c.hourly_scan_end.split(":")[0])
            scan_interval = c.hourly_scan_interval_minutes
            minute_expr = f"*/{scan_interval}" if 0 < scan_interval < 60 else "0"
            self._scheduler.add_job(
                _hourly_scan_job,
                CronTrigger(
                    minute=minute_expr,
                    hour=f"{start_h}-{end_h}",
                    day_of_week="mon-fri",
                    timezone=tz,
                    jitter=c.hourly_scan_jitter_seconds,
                ),
                kwargs={**common, "distiller": self._distiller},
                id="hourly_scan", replace_existing=True,
            )
            logger.info(
                "Scheduled hourly scan: %02d:00-%02d:00 every %dm (±%ds jitter) mon-fri",
                start_h, end_h, scan_interval, c.hourly_scan_jitter_seconds,
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
                kwargs={**common, "distiller": self._distiller},
                id="monitoring_poll", replace_existing=True,
            )

        # Scheduled distillation: events → knowledge table. Runs during work
        # hours, offset to :30 so it never races the top-of-hour digest.
        di = c.brain_distill_interval_minutes
        if c.brain_enabled and di > 0 and self._distiller is not None:
            work_start_h = int(c.work_hours_start.split(":")[0])
            work_end_h = int(c.work_hours_end.split(":")[0])
            minute_expr = f"*/{di}" if 0 < di < 60 else "30"
            self._scheduler.add_job(
                _distill_job,
                CronTrigger(
                    minute=minute_expr,
                    hour=f"{work_start_h}-{work_end_h}",
                    day_of_week=dow,
                    timezone=tz,
                ),
                kwargs={"distiller": self._distiller, "store": self._store},
                id="distill", replace_existing=True,
            )
            logger.info(
                "Scheduled distillation: every %dm, %02d:00-%02d:00 %s",
                di if di < 60 else 60, work_start_h, work_end_h, tz,
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
            "Schedule: briefings=%s, hourly_scan=%s, monitoring=%s (%s-%s %s) weekdays_only=%s",
            "on" if c.briefings_enabled else "off",
            "on" if c.hourly_scan_enabled else "off",
            f"every {interval}m" if interval > 0 else "disabled",
            c.work_hours_start, c.work_hours_end, tz, c.weekdays_only,
        )

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = False) -> None:
        self._scheduler.shutdown(wait=wait)
        logger.info("Scheduler stopped")
