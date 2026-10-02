"""Deterministic prefetch for the hourly alert scan.

Python does every mechanical step — Slack/Jira searches, cutoff filtering,
unread (`last_read`) and already-replied checks, self/bot drops, user-name
resolution — and hands the LLM only the surviving items to judge and summarize.

Credentials are reused from the MCP servers' token stores (read directly; the
MCP packages are not imported):
- Slack: static xoxp user token, nested somewhere in ~/.slack-mcp-tokens.json.
- Jira: Atlassian OAuth in ~/.jira-mcp-tokens.json (1h access token). We
  refresh it ourselves and write it back in the same shape the MCP expects,
  using JIRA_OAUTH_CLIENT_ID / JIRA_OAUTH_CLIENT_SECRET from the environment.

Each check fails independently: a broken source is recorded in
``ScanResult.failures`` and the others still run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

# Atilio's identities. Slack search does NOT match a plain "@atilio" (handle is
# ajobson) — mentions are searched in user-id form.
SLACK_SELF_ID = "U06C0AXSZ45"
SLACK_SELF_HANDLE = "ajobson"
JIRA_SELF_ID = "712020:f7c4a338-b0a9-49ed-b319-92042a637410"
BROCK_ID = "U04H81X0M1R"
BROCK_HANDLE = "mbrocklehurst"
RED_ALERT_CHANNEL = "red_alert_scout_ai"
ERRORS_CHANNEL = "eng-scout_errors"

SLACK_TOKEN_FILE = Path.home() / ".slack-mcp-tokens.json"
JIRA_TOKEN_FILE = Path.home() / ".jira-mcp-tokens.json"
JIRA_TOKEN_URL = "https://auth.atlassian.com/oauth/token"

# Kinds, in dedup priority order (a message matched by several checks keeps
# the first kind that survived its filters).
KINDS = ("brock_dm", "dm", "mention", "jira_mention", "brock_post", "red_alert", "error_post")

_TEXT_LIMIT = 1200
_ERROR_TEXT_LIMIT = 1800
_SEARCH_PAGES = 5  # x100 matches, newest first — stops early at the cutoff


@dataclass
class ScanItem:
    kind: str
    source: str  # "slack" | "jira"
    where: str  # "#channel", "DM", "group DM (…)", or "SCOUT-123 — summary"
    author: str  # real name
    ts: datetime  # UTC
    text: str
    permalink: str
    note: str = ""  # extra fact for the judge, e.g. "Atilio already replied"


@dataclass
class ScanResult:
    items: list[ScanItem] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)  # check label → reason

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in KINDS}
        for it in self.items:
            out[it.kind] = out.get(it.kind, 0) + 1
        return out


class SourceError(Exception):
    """A source could not be checked. Message is short and secret-free."""


# ── Slack ────────────────────────────────────────────────────────────────

def _find_xoxp(obj) -> str | None:
    if isinstance(obj, str):
        return obj if obj.startswith("xoxp-") else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            if found := _find_xoxp(v):
                return found
    return None


def load_slack_token(path: Path = SLACK_TOKEN_FILE) -> str:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SourceError(f"Slack token file unreadable ({type(e).__name__})") from None
    token = _find_xoxp(data)
    if not token:
        raise SourceError("no Slack user token found — re-authorize the Slack MCP")
    return token


# Module-level caches: names rarely change, so they live for the process.
_slack_users: dict[str, dict] = {}
_jira_users: dict[str, str] = {}


class SlackClient:
    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(
            base_url="https://slack.com/api/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30, transport=transport,
        )
        self._last_read: dict[str, float] = {}
        self._mpim_names: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(self, method: str, **params) -> dict:
        for attempt in range(2):
            try:
                r = await self._http.get(method, params=params)
            except httpx.HTTPError as e:
                raise SourceError(f"Slack network error ({type(e).__name__})") from None
            if r.status_code == 429 and attempt == 0:
                await asyncio.sleep(min(float(r.headers.get("Retry-After", 2)), 10))
                continue
            if r.status_code != 200:
                raise SourceError(f"Slack HTTP {r.status_code} on {method}")
            data = r.json()
            if not data.get("ok"):
                err = data.get("error", "unknown")
                if err in ("invalid_auth", "token_revoked", "not_authed", "account_inactive"):
                    raise SourceError(f"Slack auth failed ({err})")
                raise SourceError(f"Slack {method} failed ({err})")
            return data
        raise SourceError(f"Slack rate-limited on {method}")

    async def search(self, query: str, since: datetime) -> list[dict]:
        """All matches for `query` at/after `since`, newest first."""
        # `after:` is date-granular and exclusive; widen by two days so any
        # timezone interpretation still covers the cutoff, then filter on ts.
        after = (since.astimezone(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
        cutoff = since.timestamp()
        out: list[dict] = []
        for page in range(1, _SEARCH_PAGES + 1):
            data = await self.call(
                "search.messages", query=f"{query} after:{after}",
                count=100, page=page, sort="timestamp", sort_dir="desc",
            )
            msgs = data.get("messages", {})
            matches = msgs.get("matches", [])
            older = False
            for m in matches:
                if float(m.get("ts", 0)) >= cutoff:
                    out.append(m)
                else:
                    older = True
            pages = msgs.get("paging", {}).get("pages", 1)
            if older or not matches or page >= pages:
                break
        return out

    async def user(self, uid: str) -> dict:
        if uid not in _slack_users:
            try:
                u = (await self.call("users.info", user=uid)).get("user", {})
            except SourceError:
                u = {}
            prof = u.get("profile") or {}
            _slack_users[uid] = {
                "name": u.get("real_name") or prof.get("real_name")
                or prof.get("display_name") or "unknown user",
                "bot": bool(u.get("is_bot") or u.get("is_app_user")) or uid == "USLACKBOT",
            }
        return _slack_users[uid]

    async def last_read(self, channel: str) -> float:
        if channel not in self._last_read:
            info = (await self.call("conversations.info", channel=channel)).get("channel", {})
            self._last_read[channel] = float(info.get("last_read") or 0)
        return self._last_read[channel]

    async def replied_after(self, channel: str, ts: str, thread_ts: str | None) -> bool:
        """Has Atilio posted after `ts` in the same thread or conversation?"""
        after = float(ts)

        def mine(msgs: list[dict]) -> bool:
            return any(
                m.get("user") == SLACK_SELF_ID and float(m.get("ts", 0)) > after
                for m in msgs
            )

        try:
            data = await self.call(
                "conversations.replies", channel=channel,
                ts=thread_ts or ts, oldest=ts, limit=200,
            )
            if mine(data.get("messages", [])):
                return True
            if not thread_ts:  # top-level message: also check the conversation
                data = await self.call(
                    "conversations.history", channel=channel, oldest=ts, limit=200,
                )
                return mine(data.get("messages", []))
        except SourceError as e:
            logger.debug("reply check failed for %s/%s: %s", channel, ts, e)
        return False

    async def where(self, ch: dict) -> str:
        if ch.get("is_im"):
            return "DM"
        if ch.get("is_mpim"):
            cid = ch.get("id", "")
            if cid not in self._mpim_names:
                try:
                    members = (await self.call("conversations.members", channel=cid)).get("members", [])
                    names = [(await self.user(u))["name"] for u in members if u != SLACK_SELF_ID]
                    self._mpim_names[cid] = f"group DM ({', '.join(names)})"
                except SourceError:
                    self._mpim_names[cid] = "group DM"
            return self._mpim_names[cid]
        return f"#{ch.get('name', '?')}"

    async def render_text(self, m: dict, limit: int) -> str:
        return _truncate(await self._resolve_mrkdwn(_message_text(m)), limit)

    async def _resolve_mrkdwn(self, text: str) -> str:
        for uid in set(re.findall(r"<@([UW][A-Z0-9]+)(?:\|[^>]*)?>", text)):
            name = (await self.user(uid))["name"]
            text = re.sub(rf"<@{uid}(?:\|[^>]*)?>", f"@{name}", text)
        text = re.sub(r"<#C[A-Z0-9]+\|([^>]*)>", r"#\1", text)
        text = re.sub(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>", lambda mm: mm.group(1) or "@group", text)
        text = re.sub(r"<!(here|channel|everyone)(?:\|[^>]*)?>", r"@\1", text)
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
        return text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")


def _message_text(m: dict) -> str:
    """Body text plus the parts bots hide in blocks/attachments."""
    parts: list[str] = []

    def add(s) -> None:
        if isinstance(s, str) and (s := s.strip()) and s not in parts:
            parts.append(s)

    add(m.get("text"))
    for b in m.get("blocks") or []:
        if not isinstance(b, dict) or b.get("type") == "rich_text":
            continue  # rich_text duplicates `text`
        if isinstance(b.get("text"), dict):
            add(b["text"].get("text"))
        for f in b.get("fields") or []:
            if isinstance(f, dict):
                add(f.get("text"))
        for e in b.get("elements") or []:
            if isinstance(e, dict) and b.get("type") == "context":
                add(e.get("text"))
    for a in m.get("attachments") or []:
        if not isinstance(a, dict):
            continue
        add(a.get("pretext"))
        add(a.get("title"))
        add(a.get("text"))
        for f in a.get("fields") or []:
            if isinstance(f, dict):
                add(f"{f.get('title', '')}: {f.get('value', '')}".strip(": "))
        if not (a.get("title") or a.get("text")):
            add(a.get("fallback"))
    for f in m.get("files") or []:
        if isinstance(f, dict):
            add(f"[file: {f.get('title') or f.get('name') or 'attachment'}]")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _thread_ts(m: dict) -> str | None:
    q = parse_qs(urlparse(m.get("permalink", "")).query)
    tts = (q.get("thread_ts") or [None])[0]
    return tts if tts and tts != m.get("ts") else None


def _is_bot(m: dict) -> bool:
    return bool(m.get("bot_id")) or m.get("subtype") in ("bot_message", "workflow_message")


def _is_self(m: dict) -> bool:
    return m.get("user") == SLACK_SELF_ID or m.get("username") == SLACK_SELF_HANDLE


def _ts_dt(ts: str) -> datetime:
    return datetime.fromtimestamp(float(ts), UTC)


async def _slack_item(sc: SlackClient, m: dict, kind: str, note: str = "") -> ScanItem:
    uid = m.get("user") or ""
    author = (await sc.user(uid))["name"] if uid else (m.get("username") or "unknown user")
    limit = _ERROR_TEXT_LIMIT if kind == "error_post" else _TEXT_LIMIT
    return ScanItem(
        kind=kind, source="slack", where=await sc.where(m.get("channel") or {}),
        author=author, ts=_ts_dt(m["ts"]), text=await sc.render_text(m, limit),
        permalink=m.get("permalink", ""), note=note,
    )


async def _human_other(sc: SlackClient, m: dict) -> bool:
    """Not Atilio, not a bot/app/workflow."""
    if _is_self(m) or _is_bot(m):
        return False
    uid = m.get("user") or ""
    if not uid or uid.startswith("B"):
        return False
    return not (await sc.user(uid))["bot"]


async def _unread_unanswered(sc: SlackClient, m: dict) -> bool:
    ch = (m.get("channel") or {}).get("id", "")
    if float(m["ts"]) <= await sc.last_read(ch):
        return False
    return not await sc.replied_after(ch, m["ts"], _thread_ts(m))


async def slack_mentions_and_dms(sc: SlackClient, since: datetime) -> list[ScanItem]:
    """Criterion 1: unread @-mentions and DMs nobody (Atilio) has answered."""
    mentions = await sc.search(f"<@{SLACK_SELF_ID}>", since)
    dms = await sc.search("to:me", since)
    out: list[ScanItem] = []
    for kind_default, matches in (("dm", dms), ("mention", mentions)):
        for m in matches:
            ch = m.get("channel") or {}
            is_dm = bool(ch.get("is_im") or ch.get("is_mpim"))
            if kind_default == "dm" and not is_dm:
                continue
            if not await _human_other(sc, m) or not await _unread_unanswered(sc, m):
                continue
            if is_dm and m.get("user") == BROCK_ID:
                kind = "brock_dm"
            else:
                kind = "dm" if is_dm else "mention"
            out.append(await _slack_item(sc, m, kind))
    return out


async def slack_brock_posts(sc: SlackClient, since: datetime) -> list[ScanItem]:
    """Criterion 3: Brock's channel posts — the LLM judges ask/anger."""
    out: list[ScanItem] = []
    for m in await sc.search(f"from:@{BROCK_HANDLE}", since):
        ch = m.get("channel") or {}
        if ch.get("is_im") or ch.get("is_mpim"):
            continue  # DMs are criterion 1
        replied = await sc.replied_after(ch.get("id", ""), m["ts"], _thread_ts(m))
        out.append(await _slack_item(
            sc, m, "brock_post",
            note="Atilio has already replied after this" if replied else "Atilio has not replied",
        ))
    return out


async def slack_channel_posts(sc: SlackClient, since: datetime, channel: str, kind: str) -> list[ScanItem]:
    """Criteria 4 & 5: every post in a watched channel (bots included)."""
    out: list[ScanItem] = []
    for m in await sc.search(f"in:{channel}", since):
        if _is_self(m):
            continue
        out.append(await _slack_item(sc, m, kind))
    return out


# ── Jira ─────────────────────────────────────────────────────────────────

_jira_lock = asyncio.Lock()


def _save_json_atomic(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp = Path(f.name)
    tmp.chmod(0o600)
    tmp.rename(path)


async def jira_access(
    http: httpx.AsyncClient, path: Path = JIRA_TOKEN_FILE, force: bool = False,
) -> dict:
    """Current Jira tokens, refreshed (and written back for the MCP) if stale."""
    async with _jira_lock:
        try:
            tokens = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise SourceError(f"Jira token file unreadable ({type(e).__name__})") from None
        if not tokens.get("access_token") or not tokens.get("cloud_id"):
            raise SourceError("Jira not authorized — re-authorize the Jira MCP")

        age = time.time() - tokens.get("obtained_at", 0)
        if not force and age < tokens.get("expires_in", 3600) * 0.8:
            return tokens

        cid = os.environ.get("JIRA_OAUTH_CLIENT_ID")
        secret = os.environ.get("JIRA_OAUTH_CLIENT_SECRET")
        if not (cid and secret and tokens.get("refresh_token")):
            raise SourceError("Jira auth expired and cannot refresh (JIRA_OAUTH_CLIENT_ID/SECRET not set)")
        try:
            r = await http.post(JIRA_TOKEN_URL, data={
                "grant_type": "refresh_token", "client_id": cid,
                "client_secret": secret, "refresh_token": tokens["refresh_token"],
            })
        except httpx.HTTPError as e:
            raise SourceError(f"Jira token refresh network error ({type(e).__name__})") from None
        if r.status_code != 200:
            raise SourceError(f"Jira auth failed (token refresh HTTP {r.status_code}) — re-authorize the Jira MCP")
        new = r.json()
        for k in ("cloud_id", "site_url", "sites"):
            new[k] = tokens.get(k)
        new["obtained_at"] = int(time.time())
        _save_json_atomic(path, new)
        logger.info("Jira access token refreshed")
        return new


async def jira_mentions(
    since: datetime, now: datetime,
    transport: httpx.AsyncBaseTransport | None = None,
    token_path: Path = JIRA_TOKEN_FILE,
) -> list[ScanItem]:
    """Criterion 2: Jira comments @-mentioning Atilio since the cutoff."""
    async with httpx.AsyncClient(timeout=30, transport=transport) as http:
        tokens = await jira_access(http, token_path)

        async def get(p: str, **params) -> dict:
            nonlocal tokens
            for attempt in range(2):
                url = f"https://api.atlassian.com/ex/jira/{tokens['cloud_id']}{p}"
                try:
                    r = await http.get(url, params=params, headers={
                        "Authorization": f"Bearer {tokens['access_token']}",
                        "Accept": "application/json",
                    })
                except httpx.HTTPError as e:
                    raise SourceError(f"Jira network error ({type(e).__name__})") from None
                if r.status_code == 401 and attempt == 0:
                    tokens = await jira_access(http, token_path, force=True)
                    continue
                if r.status_code in (401, 403):
                    raise SourceError(f"Jira auth failed (HTTP {r.status_code})")
                if r.status_code != 200:
                    raise SourceError(f"Jira HTTP {r.status_code}")
                return r.json()
            raise SourceError("Jira auth failed")

        async def display_name(account_id: str) -> str:
            if account_id not in _jira_users:
                try:
                    _jira_users[account_id] = (await get("/rest/api/2/user", accountId=account_id)).get(
                        "displayName") or "unknown user"
                except SourceError:
                    _jira_users[account_id] = "unknown user"
            return _jira_users[account_id]

        lookback = max(int((now - since).total_seconds() // 60), 1) + 15
        jql = f'comment ~ "{JIRA_SELF_ID}" AND updated >= -{lookback}m ORDER BY updated DESC'
        issues = (await get("/rest/api/2/search/jql", jql=jql, fields="summary", maxResults=50)).get("issues", [])
        site = (tokens.get("site_url") or "").rstrip("/")
        tag = f"[~accountid:{JIRA_SELF_ID}]"
        out: list[ScanItem] = []
        for issue in issues:
            key = issue["key"]
            summary = (issue.get("fields") or {}).get("summary", "")
            data = await get(f"/rest/api/2/issue/{key}/comment", orderBy="-created", maxResults=50)
            for c in data.get("comments", []):
                created = _parse_jira_time(c.get("created", ""))
                if created is None or created < since:
                    continue
                author = c.get("author") or {}
                body = c.get("body") or ""
                if author.get("accountId") == JIRA_SELF_ID or tag not in body:
                    continue
                for aid in set(re.findall(r"\[~accountid:([^\]]+)\]", body)):
                    body = body.replace(f"[~accountid:{aid}]", f"@{await display_name(aid)}")
                out.append(ScanItem(
                    kind="jira_mention", source="jira", where=f"{key} — {summary}",
                    author=author.get("displayName") or "unknown user",
                    ts=created.astimezone(UTC), text=_truncate(body, _TEXT_LIMIT),
                    permalink=f"{site}/browse/{key}?focusedCommentId={c.get('id')}" if site else key,
                ))
        return out


def _parse_jira_time(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ── Orchestration ────────────────────────────────────────────────────────

async def prefetch(
    since: datetime,
    now: datetime | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    slack_token_path: Path = SLACK_TOKEN_FILE,
    jira_token_path: Path = JIRA_TOKEN_FILE,
) -> ScanResult:
    """Run every check; never raises for a single source failing."""
    now = now or datetime.now(UTC)
    result = ScanResult()
    checks: dict[str, object] = {}
    sc: SlackClient | None = None

    try:
        sc = SlackClient(load_slack_token(slack_token_path), transport=transport)
    except SourceError as e:
        result.failures["Slack"] = str(e)

    if sc is not None:
        checks["Slack mentions & DMs"] = slack_mentions_and_dms(sc, since)
        checks["Slack (Brock)"] = slack_brock_posts(sc, since)
        checks[f"#{RED_ALERT_CHANNEL}"] = slack_channel_posts(sc, since, RED_ALERT_CHANNEL, "red_alert")
        checks[f"#{ERRORS_CHANNEL}"] = slack_channel_posts(sc, since, ERRORS_CHANNEL, "error_post")
    checks["Jira mentions"] = jira_mentions(since, now, transport=transport, token_path=jira_token_path)

    try:
        outcomes = await asyncio.gather(*checks.values(), return_exceptions=True)
    finally:
        if sc is not None:
            await sc.aclose()

    collected: list[ScanItem] = []
    for label, res in zip(checks, outcomes):
        if isinstance(res, BaseException):
            reason = str(res) if isinstance(res, SourceError) else f"unexpected {type(res).__name__}"
            if not isinstance(res, SourceError):
                logger.warning("Scan check %s failed", label, exc_info=res)
            result.failures[label] = reason
        else:
            collected.extend(res)

    # Collapse identical Slack failures (e.g. auth) into one line.
    slack_fails = {k: v for k, v in result.failures.items() if k != "Jira mentions"}
    if len(slack_fails) > 1 and len(set(slack_fails.values())) == 1:
        for k in slack_fails:
            del result.failures[k]
        result.failures["Slack"] = next(iter(slack_fails.values()))

    # Dedup by message, keeping the highest-priority kind.
    rank = {k: i for i, k in enumerate(KINDS)}
    best: dict[str, ScanItem] = {}
    for it in collected:
        key = (it.permalink.split("?")[0] if it.source == "slack" else it.permalink) \
            or f"{it.where}|{it.ts.timestamp()}"
        if key not in best or rank[it.kind] < rank[best[key].kind]:
            best[key] = it
    result.items = sorted(best.values(), key=lambda i: (rank[i.kind], i.ts))
    logger.info("Scan prefetch: %s, failures=%s", result.counts(), list(result.failures))
    return result
