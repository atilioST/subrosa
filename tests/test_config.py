"""Tests for config loading."""

import os
import tempfile
from pathlib import Path

from subrosa1.config import Config, load_config, _resolve_env_vars


def test_resolve_env_vars():
    os.environ["TEST_TOKEN"] = "abc123"
    assert _resolve_env_vars("${TEST_TOKEN}") == "abc123"
    assert _resolve_env_vars("no-vars-here") == "no-vars-here"
    assert _resolve_env_vars("${NONEXISTENT_VAR_XYZ}") == "${NONEXISTENT_VAR_XYZ}"
    del os.environ["TEST_TOKEN"]


def test_load_config_from_toml():
    toml_content = """
[telegram]
bot_token = "test-token-123"
allowed_chat_ids = [12345]

[agent]
model = "sonnet"
max_turns = 5
agent_timeout = 120

[schedule]
morning_briefing = "08:00"
timezone = "UTC"

[schedule.work_hours]
start = "09:00"
end = "17:00"

[monitoring]
slack_channels = ["general", "dev"]
jira_projects = ["PROJ"]

[memory]
enabled = true
known_topics = ["test"]

[langfuse]
public_key = "pk-test"

[logging]
level = "DEBUG"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    assert config.bot_token == "test-token-123"
    assert config.chat_id == 12345
    assert config.model == "sonnet"
    assert config.max_turns == 5
    assert config.agent_timeout == 120
    assert config.morning_briefing == "08:00"
    assert config.timezone == "UTC"
    assert config.work_hours_start == "09:00"
    assert config.slack_channels == ["general", "dev"]
    assert config.jira_projects == ["PROJ"]
    assert config.memory_enabled is True
    assert config.known_topics == ["test"]
    assert config.langfuse_public_key == "pk-test"
    assert config.log_level == "DEBUG"
    os.unlink(f.name)


def test_defaults():
    config = Config()
    assert config.agent_timeout == 300
    assert config.briefing_timeout == 300
    assert config.monitoring_timeout == 180
    assert config.model == "sonnet"
    assert config.max_turns == 10


def test_load_real_config():
    """Verify we can load the actual subrosa config."""
    path = Path("~/.subrosa/config.toml").expanduser()
    if path.exists():
        config = load_config(path)
        assert config.bot_token != ""
        assert config.chat_id > 0
