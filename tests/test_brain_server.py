"""Tests for the brain server's ASGI auth layer and health endpoint."""

import json
import pickle

import numpy as np
import pytest
import pytest_asyncio

import subrosa.brain_server as bs
from subrosa.brain_server import BearerAuthASGI, _healthz, _send_plain
from subrosa.store import Store


class Recorder:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        return self.messages[0]["status"]

    @property
    def body(self):
        return b"".join(m.get("body", b"") for m in self.messages[1:])


async def ok_app(scope, receive, send):
    await _send_plain(send, 200, b"inner")


def http_scope(path="/mcp", headers=None):
    return {
        "type": "http",
        "path": path,
        "headers": headers or [],
    }


async def test_auth_accepts_valid_token():
    app = BearerAuthASGI(ok_app, "secret")
    rec = Recorder()
    await app(http_scope(headers=[(b"authorization", b"Bearer secret")]), None, rec)
    assert rec.status == 200
    assert rec.body == b"inner"


async def test_auth_rejects_missing_and_wrong_token():
    app = BearerAuthASGI(ok_app, "secret")
    for headers in ([], [(b"authorization", b"Bearer wrong")], [(b"authorization", b"secret")]):
        rec = Recorder()
        await app(http_scope(headers=headers), None, rec)
        assert rec.status == 401, f"headers={headers}"


async def test_auth_exempts_healthz():
    app = BearerAuthASGI(ok_app, "secret", exempt_paths=frozenset({"/healthz"}))
    rec = Recorder()
    await app(http_scope(path="/healthz"), None, rec)
    assert rec.status == 200


async def test_auth_passes_non_http_scopes():
    called = []

    async def lifespan_app(scope, receive, send):
        called.append(scope["type"])

    app = BearerAuthASGI(lifespan_app, "secret")
    await app({"type": "lifespan"}, None, None)
    assert called == ["lifespan"]


async def test_auth_disabled_without_token():
    app = BearerAuthASGI(ok_app, "")
    rec = Recorder()
    await app(http_scope(), None, rec)
    # Empty token means auth layer is a passthrough (start() warns loudly)
    assert rec.status == 200


@pytest_asyncio.fixture
async def store():
    s = Store(db_path=":memory:")
    await s.initialize()
    s._embeddings.embed = lambda text: pickle.dumps(np.zeros(3, dtype=np.float32))
    yield s
    await s.close()


async def test_healthz_reports_freshness(store, monkeypatch):
    monkeypatch.setattr(bs, "_store", store)
    await store.insert_knowledge(
        domain="Strategic", primitive="reference", entity_type="product",
        entity_name="Scout", summary="test item",
    )
    await store.set_meta("last_distilled_at", "2026-07-08T15:00:00+00:00")

    rec = Recorder()
    await _healthz(http_scope(path="/healthz"), None, rec)
    assert rec.status == 200
    body = json.loads(rec.body)
    assert body["status"] == "ok"
    assert body["knowledge_count"] == 1
    assert body["last_distilled_at"] == "2026-07-08T15:00:00+00:00"
    assert "uptime_seconds" in body
