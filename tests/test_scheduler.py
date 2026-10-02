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


def _item(kind="mention", text="can you review?", **kw):
    from datetime import UTC, datetime
    from subrosa.scan_fetch import ScanItem
    base = dict(kind=kind, source="slack", where="#general", author="Alice Ng",
                ts=datetime(2026, 9, 30, 3, 5, tzinfo=UTC), text=text,
                permalink="https://x.slack.com/archives/C1/p1")
    base.update(kw)
    return ScanItem(**base)


def test_hourly_scan_prompt_scope():
    from zoneinfo import ZoneInfo
    from subrosa.context import build_hourly_scan_prompt

    items = [
        _item(),
        _item("brock_post", "why is this still broken", author="Matthew Brocklehurst",
              note="Atilio has not replied"),
        _item("error_post", "NullPointerException in Foo", where="#eng-scout_errors"),
    ]
    prompt = build_hourly_scan_prompt(
        items, {}, since_human="2026-09-29 21:00 MDT", tz=ZoneInfo("America/Denver"),
    )
    # Items are embedded with real names, kinds, notes and local times.
    assert "unread @-mention | #general | Alice Ng | Tue 21:05 MDT" in prompt
    assert "> can you review?" in prompt
    assert "note: Atilio has not replied" in prompt
    assert "## ITEMS (3)" in prompt
    # Mechanical work is done in code — the model must not redo it.
    assert "You have no tools" in prompt
    assert "slack_search_messages" not in prompt and "last_read" not in prompt
    # Brock judgment, error assessment.
    assert "frustration, anger" in prompt
    assert "NEVER a reason to report" in prompt
    assert "thinking out loud" in prompt and "yolo approach" in prompt
    assert "new vs recurring" in prompt
    # Header is reserved for direct asks and Brock DMs.
    assert "ONLY for" in prompt
    assert "ANY DM from Brock" in prompt
    assert "NO header" in prompt
    assert "never by ID" in prompt
    assert "NO_CHANGES" in prompt
    assert "COULD NOT BE CHECKED" not in prompt


def test_hourly_scan_prompt_reports_failed_sources():
    from zoneinfo import ZoneInfo
    from subrosa.context import build_hourly_scan_prompt

    prompt = build_hourly_scan_prompt(
        [_item()], {"Jira mentions": "Jira auth failed (HTTP 401)"},
        since_human="x", tz=ZoneInfo("UTC"),
    )
    assert "- Jira mentions: Jira auth failed (HTTP 401)" in prompt
    assert "must never be silent" in prompt


# ── run_alert_scan / _hourly_scan_job ────────────────────────────────────

class _Cfg:
    timezone = "America/Denver"
    briefing_path = None
    scheduled_model = "sonnet"
    hourly_scan_interval_minutes = 60
    briefing_timeout = 30
    chat_id = 1


class _Agent:
    def __init__(self, text="‼️ Needs attention\n- Alice asks for a review", is_error=False):
        from subrosa.agent import AgentResponse
        self.calls = []
        self.response = AgentResponse(text=text, is_error=is_error)

    async def invoke(self, prompt, system_prompt, **kw):
        self.calls.append((prompt, kw))
        return self.response


class _Store:
    def __init__(self):
        self.meta = {}
        self.events = []

    async def get_meta(self, k):
        return self.meta.get(k)

    async def set_meta(self, k, v):
        self.meta[k] = v

    async def log_event(self, **kw):
        self.events.append(kw)

    async def log_diagnostic(self, *a, **kw):
        pass


class _Bot:
    is_silenced = False

    def __init__(self):
        self.sent = []

    async def send_to_chat(self, chat_id, text):
        self.sent.append(text)


class _Health:
    def __init__(self):
        self.agent_runs = 0

    def record_agent(self):
        self.agent_runs += 1


def _patch_prefetch(monkeypatch, items=(), failures=None):
    from subrosa import scan_fetch
    from subrosa.scan_fetch import ScanResult

    async def fake(since, now=None, **kw):
        return ScanResult(items=list(items), failures=dict(failures or {}))
    monkeypatch.setattr(scan_fetch, "prefetch", fake)
    monkeypatch.setattr("subrosa.context.build_system_prompt", lambda p=None: "sys")


async def _run_job(agent):
    from subrosa.scheduler import _hourly_scan_job
    store, bot, health = _Store(), _Bot(), _Health()
    await _hourly_scan_job(agent, bot, store, health, _Cfg())
    return store, bot, health


async def test_zero_items_skips_llm_and_keeps_watermark(monkeypatch):
    _patch_prefetch(monkeypatch)
    agent = _Agent()
    store, bot, health = await _run_job(agent)
    assert agent.calls == []
    assert bot.sent == [] and store.meta == {} and health.agent_runs == 0


async def test_items_invoke_llm_without_tools_and_advance(monkeypatch):
    _patch_prefetch(monkeypatch, items=[_item()])
    agent = _Agent()
    store, bot, health = await _run_job(agent)
    prompt, kw = agent.calls[0]
    assert kw["no_tools"] is True and kw["max_turns"] <= 3
    assert "can you review?" in prompt
    assert bot.sent == [agent.response.text]
    assert "hourly_scan_last_sent" in store.meta


async def test_llm_no_changes_stays_silent(monkeypatch):
    _patch_prefetch(monkeypatch, items=[_item("brock_post", "Huge")])
    agent = _Agent(text="Nothing qualifies.\n\nNO_CHANGES")
    store, bot, _ = await _run_job(agent)
    assert bot.sent == [] and store.meta == {}


async def test_failure_only_sends_notice_without_llm_or_advance(monkeypatch):
    _patch_prefetch(monkeypatch, failures={"Jira mentions": "Jira auth failed (HTTP 401)"})
    agent = _Agent()
    store, bot, _ = await _run_job(agent)
    assert agent.calls == []
    assert bot.sent == ["⚠️ Couldn't check Jira mentions: Jira auth failed (HTTP 401)"]
    assert store.meta == {}


async def test_failure_survives_llm_no_changes(monkeypatch):
    _patch_prefetch(monkeypatch, items=[_item("brock_post", "Huge")],
                    failures={"Slack": "Slack auth failed (invalid_auth)"})
    store, bot, _ = await _run_job(_Agent(text="NO_CHANGES"))
    assert bot.sent == ["⚠️ Couldn't check Slack: Slack auth failed (invalid_auth)"]
    assert store.meta == {}


async def test_agent_error_does_not_send_or_advance(monkeypatch):
    _patch_prefetch(monkeypatch, items=[_item()])
    store, bot, _ = await _run_job(_Agent(text="Agent error", is_error=True))
    assert bot.sent == [] and store.meta == {}
