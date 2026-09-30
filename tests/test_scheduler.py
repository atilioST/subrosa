"""Scheduler helpers — hourly scan suppression."""

import pytest

from subrosa.scheduler import is_no_changes


@pytest.mark.parametrize("text", [
    "NO_CHANGES",
    "  NO_CHANGES  ",
    "no_changes",
    "",
    "   ",
    "\n\n",
    # Regression: model prefixed a line of reasoning before the sentinel.
    # Strict equality let these through and paged the user (events 1325, 1326).
    "All matches predate the 15:01 MDT cutoff.\n\nNO_CHANGES",
    "All results fall before the 10:01 MDT cutoff (latest was 09:48).\n\nNO_CHANGES",
    "Checked all channels.\nNothing met the criteria.\n**NO_CHANGES**",
    "Quiet window.\n\n`NO_CHANGES`",
])
def test_suppressed(text):
    assert is_no_changes(text) is True


@pytest.mark.parametrize("text", [
    "**Brock — critical**\n\n- Direct ask on SCOUT-16653.",
    "#eng-scout_errors — 1 new alert (16:23 MDT)",
    # Mentions the sentinel mid-body but ends with real content: must send.
    "Prior scan returned NO_CHANGES, but this one found a Class 2 page.",
    "NO_CHANGES\n\nCorrection: Walter escalated in #scout-guild.",
])
def test_sent(text):
    assert is_no_changes(text) is False

