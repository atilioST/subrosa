"""Hourly scan prefetch — filtering logic against a mocked Slack/Jira API."""

import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from subrosa import scan_fetch as sf

NOW = datetime(2026, 10, 1, 18, 0, tzinfo=UTC)
SINCE = NOW - timedelta(hours=1)
ME = sf.SLACK_SELF_ID


def ts(minutes_after_since: float) -> str:
    return f"{(SINCE + timedelta(minutes=minutes_after_since)).timestamp():.6f}"


def ch(cid: str, name: str = "", im: bool = False, mpim: bool = False) -> dict:
    return {"id": cid, "name": name or cid, "is_im": im, "is_mpim": mpim}


def msg(user: str, t: str, channel: dict, text: str = "hi", thread_ts: str | None = None, **kw) -> dict:
    link = f"https://x.slack.com/archives/{channel['id']}/p{t.replace('.', '')}"
    if thread_ts:
        link += f"?thread_ts={thread_ts}"
    return {"user": user, "username": user.lower(), "ts": t, "channel": channel,
            "text": text, "permalink": link, **kw}


USERS = {
    "UALICE": {"real_name": "Alice Ng"},
    "UBOB": {"real_name": "Bob Ray"},
    "UBOT": {"real_name": "Deploy Bot", "is_bot": True},
    sf.BROCK_ID: {"real_name": "Matthew Brocklehurst"},
    ME: {"real_name": "Atilio Jobson"},
}


class FakeAPI:
    """Routes Slack + Jira calls; per-query search results are configurable."""

    def __init__(self):
        self.search: dict[str, list[dict]] = {}
        self.last_read: dict[str, str] = {}
        self.replies: dict[tuple[str, str], list[dict]] = {}
        self.history: dict[str, list[dict]] = {}
        self.slack_error: str | None = None
        self.jira_status = 200
        self.jira_issues: list[dict] = []
        self.jira_comments: dict[str, list[dict]] = {}
        self.refreshed = False
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append(url.path)
        if url.host == "slack.com":
            return self._slack(url.path.rsplit("/", 1)[-1], dict(url.params))
        if url.host == "auth.atlassian.com":
            self.refreshed = True
            return httpx.Response(200, json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
        return self._jira(url.path, dict(url.params))

    def _slack(self, method: str, p: dict) -> httpx.Response:
        if self.slack_error:
            return httpx.Response(200, json={"ok": False, "error": self.slack_error})
        if method == "search.messages":
            query = p["query"].rsplit(" after:", 1)[0]
            matches = sorted(self.search.get(query, []), key=lambda m: -float(m["ts"]))
            return httpx.Response(200, json={"ok": True, "messages": {"matches": matches, "paging": {"pages": 1}}})
        if method == "users.info":
            return httpx.Response(200, json={"ok": True, "user": USERS.get(p["user"], {})})
        if method == "conversations.info":
            return httpx.Response(200, json={"ok": True, "channel": {"last_read": self.last_read.get(p["channel"], "0")}})
        if method == "conversations.replies":
            return httpx.Response(200, json={"ok": True, "messages": self.replies.get((p["channel"], p["ts"]), [])})
        if method == "conversations.history":
            return httpx.Response(200, json={"ok": True, "messages": self.history.get(p["channel"], [])})
        if method == "conversations.members":
            return httpx.Response(200, json={"ok": True, "members": [ME, "UALICE", "UBOB"]})
        return httpx.Response(404)

    def _jira(self, path: str, p: dict) -> httpx.Response:
        if self.jira_status != 200:
            return httpx.Response(self.jira_status)
        if path.endswith("/search/jql"):
            self.jql = p.get("jql")
            return httpx.Response(200, json={"issues": self.jira_issues})
        if path.endswith("/comment"):
            key = path.split("/")[-2]
            return httpx.Response(200, json={"comments": self.jira_comments.get(key, [])})
        if path.endswith("/rest/api/2/user"):
            return httpx.Response(200, json={"displayName": "Carol Diaz"})
        return httpx.Response(404)


@pytest.fixture(autouse=True)
def _clear_caches():
    sf._slack_users.clear()
    sf._jira_users.clear()


@pytest.fixture
def api():
    return FakeAPI()


@pytest.fixture
def tokens(tmp_path):
    slack = tmp_path / "slack.json"
    slack.write_text(json.dumps({"ok": True, "authed_user": {"id": ME, "access_token": "xoxp-test"}}))
    jira = tmp_path / "jira.json"
    jira.write_text(json.dumps({
        "access_token": "a", "refresh_token": "r", "expires_in": 3600,
        "obtained_at": int(time.time()), "cloud_id": "cloud", "site_url": "https://site.example",
    }))
    return slack, jira


async def run(api, tokens):
    slack, jira = tokens
    return await sf.prefetch(
        SINCE, NOW, transport=httpx.MockTransport(api.handler),
        slack_token_path=slack, jira_token_path=jira,
    )


def kinds(result):
    return sorted((i.kind, i.text) for i in result.items)


# ── Slack: unread / replied / self / bot ────────────────────────────────

async def test_unread_filter_uses_channel_last_read(api, tokens):
    c = ch("C1", "general")
    api.search[f"<@{ME}>"] = [
        msg("UALICE", ts(10), c, "read one"),
        msg("UBOB", ts(30), c, f"unread one <@{ME}>"),
    ]
    api.last_read["C1"] = ts(20)
    r = await run(api, tokens)
    assert kinds(r) == [("mention", "unread one @Atilio Jobson")]
    assert r.items[0].author == "Bob Ray"
    assert r.items[0].where == "#general"


async def test_cutoff_drops_older_messages(api, tokens):
    c = ch("C1", "general")
    api.search[f"<@{ME}>"] = [msg("UALICE", ts(-5), c, "before cutoff")]
    assert (await run(api, tokens)).items == []


async def test_drops_mention_answered_in_thread(api, tokens):
    c = ch("C1", "general")
    root = ts(1)
    m = msg("UALICE", ts(5), c, "q?", thread_ts=root)
    api.search[f"<@{ME}>"] = [m]
    api.replies[("C1", root)] = [{"user": ME, "ts": ts(6)}]
    assert (await run(api, tokens)).items == []


async def test_drops_top_level_answered_in_conversation(api, tokens):
    d = ch("D1", im=True)
    api.search["to:me"] = [msg("UALICE", ts(5), d, "ping")]
    api.history["D1"] = [{"user": ME, "ts": ts(7)}]
    assert (await run(api, tokens)).items == []


async def test_earlier_self_reply_does_not_count(api, tokens):
    d = ch("D1", im=True)
    api.search["to:me"] = [msg("UALICE", ts(5), d, "ping")]
    api.history["D1"] = [{"user": ME, "ts": ts(3)}]  # before the message
    r = await run(api, tokens)
    assert kinds(r) == [("dm", "ping")]
    assert r.items[0].where == "DM"


async def test_drops_self_and_bots(api, tokens):
    d = ch("D1", im=True)
    api.search["to:me"] = [
        msg(ME, ts(5), d, "mine"),
        msg("UBOT", ts(6), d, "bot user"),
        msg("UALICE", ts(7), d, "bot id", bot_id="B123"),
        msg("USLACKBOT", ts(8), d, "slackbot"),
        msg("UALICE", ts(9), d, "human"),
    ]
    assert kinds(await run(api, tokens)) == [("dm", "human")]


async def test_brock_dm_and_dedup(api, tokens):
    d = ch("D9", im=True)
    m = msg(sf.BROCK_ID, ts(5), d, f"<@{ME}> call me")
    api.search["to:me"] = [m]
    api.search[f"<@{ME}>"] = [m]
    api.search[f"from:@{sf.BROCK_HANDLE}"] = [m]  # DM — not a channel post
    r = await run(api, tokens)
    assert [(i.kind, i.author) for i in r.items] == [("brock_dm", "Matthew Brocklehurst")]


async def test_group_dm_named_by_members(api, tokens):
    g = ch("G1", "mpdm-ajobson--alice--bob-1", mpim=True)
    api.search["to:me"] = [msg("UALICE", ts(5), g, "hey all")]
    r = await run(api, tokens)
    assert r.items[0].where == "group DM (Alice Ng, Bob Ray)"


async def test_brock_channel_post_carries_reply_note(api, tokens):
    c = ch("C2", "leadership")
    api.search[f"from:@{sf.BROCK_HANDLE}"] = [msg(sf.BROCK_ID, ts(5), c, "why is this still broken")]
    r = await run(api, tokens)
    assert r.items[0].kind == "brock_post"
    assert "not replied" in r.items[0].note


async def test_watched_channels_keep_bots_and_read_attachments(api, tokens):
    red = ch("C3", sf.RED_ALERT_CHANNEL)
    err = ch("C4", sf.ERRORS_CHANNEL)
    api.search[f"in:{sf.RED_ALERT_CHANNEL}"] = [msg("UBOB", ts(5), red, "prod down")]
    api.search[f"in:{sf.ERRORS_CHANNEL}"] = [
        msg("UBOT", ts(6), err, "", attachments=[{"title": "NullPointerException", "text": "in Foo.bar"}]),
        msg(ME, ts(7), err, "my own note"),
    ]
    r = await run(api, tokens)
    assert kinds(r) == [("error_post", "NullPointerException\nin Foo.bar"), ("red_alert", "prod down")]


# ── Jira ─────────────────────────────────────────────────────────────────

def jcomment(cid, minutes, author_id, body, name="Carol Diaz"):
    created = (SINCE + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    return {"id": cid, "created": created, "body": body,
            "author": {"accountId": author_id, "displayName": name}}


async def test_jira_comment_filtering(api, tokens):
    tag = f"[~accountid:{sf.JIRA_SELF_ID}]"
    api.jira_issues = [{"key": "SCOUT-1", "fields": {"summary": "Fix it"}}]
    api.jira_comments["SCOUT-1"] = [
        jcomment("1", 10, "acct:carol", f"{tag} please review"),        # keep
        jcomment("2", -10, "acct:carol", f"{tag} old"),                  # before cutoff
        jcomment("3", 12, sf.JIRA_SELF_ID, f"{tag} talking to myself"),  # self
        jcomment("4", 15, "acct:carol", "no mention here"),              # no tag
    ]
    r = await run(api, tokens)
    assert [(i.kind, i.where, i.author) for i in r.items] == [("jira_mention", "SCOUT-1 — Fix it", "Carol Diaz")]
    assert r.items[0].text == "@Carol Diaz please review"
    assert r.items[0].permalink == "https://site.example/browse/SCOUT-1?focusedCommentId=1"


async def test_jira_lookback_window_padded(api, tokens):
    await run(api, tokens)
    # lookback (60m) + 15m padding; precise filter is on comment `created`
    assert f'comment ~ "{sf.JIRA_SELF_ID}" AND updated >= -75m' in api.jql


async def test_jira_refreshes_stale_token(api, tokens, monkeypatch):
    _, jira = tokens
    data = json.loads(jira.read_text())
    data["obtained_at"] = int(time.time()) - 4000
    jira.write_text(json.dumps(data))
    monkeypatch.setenv("JIRA_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("JIRA_OAUTH_CLIENT_SECRET", "sec")
    r = await run(api, tokens)
    assert api.refreshed and r.failures == {}
    saved = json.loads(jira.read_text())
    assert saved["access_token"] == "new" and saved["cloud_id"] == "cloud"


# ── Per-source failure ───────────────────────────────────────────────────

async def test_jira_auth_failure_does_not_break_slack(api, tokens, monkeypatch):
    monkeypatch.setenv("JIRA_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("JIRA_OAUTH_CLIENT_SECRET", "sec")
    api.jira_status = 401  # still 401 after the forced refresh
    api.search[f"in:{sf.RED_ALERT_CHANNEL}"] = [msg("UBOB", ts(5), ch("C3", sf.RED_ALERT_CHANNEL), "fire")]
    r = await run(api, tokens)
    assert kinds(r) == [("red_alert", "fire")]
    assert api.refreshed
    assert r.failures == {"Jira mentions": "Jira auth failed (HTTP 401)"}


async def test_jira_expired_without_credentials_fails_cleanly(api, tokens, monkeypatch):
    _, jira = tokens
    data = json.loads(jira.read_text())
    data["obtained_at"] = int(time.time()) - 4000
    jira.write_text(json.dumps(data))
    monkeypatch.delenv("JIRA_OAUTH_CLIENT_ID", raising=False)
    r = await run(api, tokens)
    assert "Jira mentions" in r.failures


async def test_slack_auth_failure_collapses_and_keeps_jira(api, tokens):
    api.slack_error = "invalid_auth"
    tag = f"[~accountid:{sf.JIRA_SELF_ID}]"
    api.jira_issues = [{"key": "SCOUT-2", "fields": {"summary": "S"}}]
    api.jira_comments["SCOUT-2"] = [jcomment("9", 5, "acct:carol", f"{tag} hi")]
    r = await run(api, tokens)
    assert [i.kind for i in r.items] == ["jira_mention"]
    assert r.failures == {"Slack": "Slack auth failed (invalid_auth)"}


async def test_missing_slack_token_file(api, tokens, tmp_path):
    _, jira = tokens
    r = await sf.prefetch(
        SINCE, NOW, transport=httpx.MockTransport(api.handler),
        slack_token_path=tmp_path / "nope.json", jira_token_path=jira,
    )
    assert "Slack" in r.failures
    assert "xoxp" not in json.dumps(r.failures)
