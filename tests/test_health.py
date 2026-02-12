"""Tests for health tracking."""

import time
from subrosa1.health import Health


def test_initial_state():
    h = Health()
    assert h.is_healthy()
    assert h.last_response_at == 0.0


def test_record_and_status():
    h = Health()
    h.record_response()
    h.record_agent()
    h.record_poll()

    status = h.status_dict()
    assert status["healthy"] is True
    assert status["last_response_ago"] is not None
    assert status["last_agent_ago"] is not None
    assert status["last_poll_ago"] is not None
    assert status["uptime_seconds"] >= 0


def test_stale_poll():
    h = Health()
    h.last_poll_at = time.monotonic() - 600  # 10 minutes ago
    assert h.is_healthy(max_stale_seconds=300) is False
