"""Configuration loading from TOML + environment variables."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path("~/.subrosa/config.toml").expanduser()


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _resolve_deep(obj: object) -> object:
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_deep(v) for v in obj]
    return obj


@dataclass
class Config:
    # Telegram
    bot_token: str = ""
    chat_id: int = 0

    # Agent
    model: str = "sonnet"
    max_turns: int = 10
    briefing_max_turns: int = 25
    agent_timeout: int = 300  # soft deadline: notify the user, keep working
    agent_hard_timeout: int = 1800  # hard ceiling: cancel and return partial output
    briefing_timeout: int = 300
    monitoring_timeout: int = 180

    # Schedule
    briefings_enabled: bool = False
    morning_briefing: str = "07:30"
    noon_briefing: str = "12:00"
    evening_digest: str = "18:00"
    monitoring_interval_minutes: int = 30
    work_hours_start: str = "07:00"
    work_hours_end: str = "19:00"
    timezone: str = "America/Denver"
    weekdays_only: bool = False

    # Hourly Slack alert scan (delta-only): mentions, key-people criticals,
    # red-alert channel, error assessment. interval_minutes also sets the
    # fallback look-back window when no watermark exists yet.
    hourly_scan_enabled: bool = True
    hourly_scan_interval_minutes: int = 60
    hourly_scan_start: str = "08:00"
    hourly_scan_end: str = "18:00"
    hourly_scan_jitter_seconds: int = 120

    # Monitoring scope
    slack_channels: list[str] = field(default_factory=list)
    jira_projects: list[str] = field(default_factory=list)
    github_repos: list[str] = field(default_factory=list)

    # Memory
    memory_enabled: bool = True
    implicit_extraction: bool = True
    extraction_model: str = "haiku"
    max_memories_per_prompt: int = 10
    max_memory_tokens: int = 2000
    known_topics: list[str] = field(default_factory=lambda: [
        "scout", "compass", "missions", "sdlc", "subrosa",
    ])

    # Procedures
    procedures_enabled: bool = True
    reflection_model: str = "haiku"
    min_tool_calls: int = 3
    procedure_similarity_threshold: float = 0.5
    procedure_update_threshold: float = 0.7
    max_procedures_per_prompt: int = 2

    # Context
    max_recent_messages: int = 5

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Briefing
    briefing_path: str = "~/.subrosa/briefing.md"

    # Brain server
    brain_enabled: bool = False
    brain_host: str = "127.0.0.1"
    brain_port: int = 7771
    brain_token: str = ""
    brain_distill_interval_minutes: int = 60  # 0 disables scheduled distillation

    # Logging
    log_level: str = "INFO"


def load_config(path: Path | None = None) -> Config:
    """Load configuration from TOML file with env var substitution."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    raw = _resolve_deep(raw)

    tg = raw.get("telegram", {})
    sched = raw.get("schedule", {})
    work_hours = sched.get("work_hours", {})
    hourly_scan = sched.get("hourly_scan", sched.get("person_digest", {}))
    mon = raw.get("monitoring", {})
    ag = raw.get("agent", {})
    lf = raw.get("langfuse", {})
    br = raw.get("briefing", {})
    mem = raw.get("memory", {})
    proc = raw.get("procedures", {})
    ctx = raw.get("context", {})
    brain = raw.get("brain", {})
    log = raw.get("logging", {})

    allowed = tg.get("allowed_chat_ids", [])

    return Config(
        bot_token=tg.get("bot_token", ""),
        chat_id=allowed[0] if allowed else 0,
        model=ag.get("model", "sonnet"),
        max_turns=ag.get("max_turns", 10),
        briefing_max_turns=ag.get("briefing_max_turns", 25),
        agent_timeout=ag.get("agent_timeout", 300),
        agent_hard_timeout=ag.get("agent_hard_timeout", 1800),
        briefing_timeout=ag.get("briefing_timeout", 300),
        monitoring_timeout=ag.get("monitoring_timeout", 180),
        briefings_enabled=sched.get("briefings_enabled", False),
        morning_briefing=sched.get("morning_briefing", "07:30"),
        noon_briefing=sched.get("noon_briefing", "12:00"),
        evening_digest=sched.get("evening_digest", "18:00"),
        monitoring_interval_minutes=sched.get("monitoring_interval_minutes", 30),
        work_hours_start=work_hours.get("start", "07:00"),
        work_hours_end=work_hours.get("end", "19:00"),
        timezone=sched.get("timezone", "America/Denver"),
        weekdays_only=sched.get("weekdays_only", False),
        hourly_scan_enabled=hourly_scan.get("enabled", True),
        hourly_scan_interval_minutes=hourly_scan.get("interval_minutes", 60),
        hourly_scan_start=hourly_scan.get("start", "08:00"),
        hourly_scan_end=hourly_scan.get("end", "18:00"),
        hourly_scan_jitter_seconds=hourly_scan.get("jitter_seconds", 120),
        slack_channels=mon.get("slack_channels", []),
        jira_projects=mon.get("jira_projects", []),
        github_repos=mon.get("github_repos", []),
        memory_enabled=mem.get("enabled", True),
        implicit_extraction=mem.get("implicit_extraction", True),
        extraction_model=mem.get("extraction_model", "haiku"),
        max_memories_per_prompt=mem.get("max_memories_per_prompt", 10),
        max_memory_tokens=mem.get("max_memory_tokens", 2000),
        known_topics=mem.get("known_topics", [
            "scout", "compass", "missions", "sdlc", "subrosa",
        ]),
        procedures_enabled=proc.get("enabled", True),
        reflection_model=proc.get("reflection_model", "haiku"),
        min_tool_calls=proc.get("min_tool_calls", 3),
        procedure_similarity_threshold=proc.get("similarity_threshold", 0.5),
        procedure_update_threshold=proc.get("update_threshold", 0.7),
        max_procedures_per_prompt=proc.get("max_per_prompt", 2),
        max_recent_messages=ctx.get("max_recent_messages", 5),
        langfuse_public_key=lf.get("public_key", ""),
        langfuse_secret_key=lf.get("secret_key", ""),
        langfuse_host=lf.get("host", "https://cloud.langfuse.com"),
        briefing_path=br.get("path", "~/.subrosa/briefing.md"),
        brain_enabled=brain.get("enabled", False),
        brain_host=brain.get("host", "127.0.0.1"),
        brain_port=brain.get("port", 7771),
        brain_token=brain.get("token", ""),
        brain_distill_interval_minutes=brain.get("distill_interval_minutes", 60),
        log_level=log.get("level", "INFO"),
    )
