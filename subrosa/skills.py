"""Skill loading and management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base paths
_SKILLS_DIR = Path(__file__).parent.parent / "skills"
_CONFIG_DIR = Path(__file__).parent.parent / "config"


class Skill:
    """Represents a loaded skill."""

    def __init__(self, name: str, config: dict[str, Any], instruction: str):
        self.name = name
        self.config = config
        self.instruction = instruction
        self.triggers = config.get("triggers", [])
        self.description = config.get("description", "")

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, triggers={self.triggers})"


def load_skill(name: str) -> Skill | None:
    """Load a skill by name from the skills directory.

    Args:
        name: Skill name (directory name in skills/)

    Returns:
        Skill object if found, None otherwise
    """
    skill_dir = _SKILLS_DIR / name

    if not skill_dir.exists():
        logger.warning("Skill directory not found: %s", skill_dir)
        return None

    # Load config
    config_path = skill_dir / "config.json"
    if not config_path.exists():
        logger.warning("Skill config not found: %s", config_path)
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        logger.error("Failed to load skill config %s: %s", config_path, e)
        return None

    # Load skill instruction
    skill_path = skill_dir / "skill.md"
    if not skill_path.exists():
        logger.warning("Skill instruction not found: %s", skill_path)
        return None

    instruction_parts = [skill_path.read_text()]

    # Load additional context files
    for filename in config.get("context_files", []):
        if filename == "skill.md":
            continue  # Already loaded

        file_path = skill_dir / filename
        if file_path.exists():
            instruction_parts.append(file_path.read_text())
        else:
            logger.warning("Context file not found: %s", file_path)

    instruction = "\n\n".join(instruction_parts)
    logger.info("Loaded skill: %s (%d chars)", name, len(instruction))

    return Skill(name, config, instruction)


def load_all_enabled_skills() -> list[Skill]:
    """Load all enabled skills from config/skills.json.

    Returns:
        List of loaded Skill objects
    """
    skills_config_path = _CONFIG_DIR / "skills.json"

    if not skills_config_path.exists():
        logger.warning("No skills config found at %s", skills_config_path)
        return []

    try:
        skills_config = json.loads(skills_config_path.read_text())
    except Exception as e:
        logger.error("Failed to load skills config: %s", e)
        return []

    enabled = skills_config.get("enabled", [])
    skills = []

    for name in enabled:
        skill = load_skill(name)
        if skill:
            skills.append(skill)

    logger.info("Loaded %d enabled skills: %s", len(skills), [s.name for s in skills])
    return skills


def find_skill_by_trigger(trigger: str) -> Skill | None:
    """Find a skill matching the given trigger (slash command).

    Args:
        trigger: Command trigger (e.g., "/briefing")

    Returns:
        Matching Skill or None
    """
    # Normalize trigger
    if not trigger.startswith("/"):
        trigger = f"/{trigger}"

    for skill in load_all_enabled_skills():
        if trigger in skill.triggers:
            return skill

    return None


def get_skill_schedules() -> dict[str, dict[str, str]]:
    """Get scheduled execution times for skills.

    Returns:
        Dict mapping skill names to schedule configs
        Example: {"briefing": {"morning": "0 7 * * *", "noon": "0 12 * * *"}}
    """
    skills_config_path = _CONFIG_DIR / "skills.json"

    if not skills_config_path.exists():
        return {}

    try:
        skills_config = json.loads(skills_config_path.read_text())
        return skills_config.get("schedules", {})
    except Exception as e:
        logger.error("Failed to load skill schedules: %s", e)
        return {}
