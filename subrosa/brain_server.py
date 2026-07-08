"""Open Brain MCP server.

Exposes the knowledge store as an MCP server over streamable HTTP (`/mcp`),
with the legacy SSE transport (`/sse`) kept alive for a transition window.
Runs stateless so a Subrosa restart never invalidates client sessions.

Tools:
  search_knowledge   — semantic search over knowledge_embeddings
  query_entity       — all knowledge about a named entity
  get_domain_brief   — structured summary for a VP domain
  capture_thought    — write a knowledge item from any session
  add_domain         — extend the ontology with a new domain
  add_primitive      — extend the ontology with a new primitive

Endpoints:
  /mcp      — streamable HTTP MCP transport (preferred)
  /sse      — legacy SSE transport (deprecated, kept for old clients)
  /healthz  — unauthenticated JSON: knowledge count, distillation freshness

Auth: every request except /healthz must carry Authorization: Bearer <brain_token>.
The auth layer is pure ASGI — Starlette's BaseHTTPMiddleware is incompatible
with streaming responses (it asserts on client disconnect mid-stream).

Architecture note: the MCP server runs as a background asyncio task inside
the Subrosa process, sharing the Store instance (same SQLite connection,
same embedding model singleton). No second process, no second DB connection.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from .config import Config
    from .store import Store

logger = logging.getLogger(__name__)

# Module-level references set by start()
_store: Store | None = None
_started_at: float = time.monotonic()


# ── ASGI helpers ─────────────────────────────────────────────────────────────

async def _send_plain(send, status: int, body: bytes, content_type: bytes = b"text/plain") -> None:
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", content_type),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


class BearerAuthASGI:
    """Pure ASGI bearer-token auth (no BaseHTTPMiddleware — see module docstring)."""

    def __init__(self, app, token: str, exempt_paths: frozenset[str] = frozenset()):
        self._app = app
        self._token = token.encode()
        self._exempt = exempt_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self._token and scope.get("path") not in self._exempt:
            auth = b""
            for key, value in scope.get("headers", []):
                if key == b"authorization":
                    auth = value
                    break
            if not hmac.compare_digest(auth, b"Bearer " + self._token):
                await _send_plain(send, 401, b"Unauthorized")
                return
        await self._app(scope, receive, send)


async def _healthz(scope, receive, send) -> None:
    """Unauthenticated liveness + freshness probe."""
    body: dict[str, Any] = {"status": "ok"}
    try:
        if _store is not None:
            body["knowledge_count"] = await _store.knowledge_count()
            body["last_distilled_at"] = await _store.get_meta("last_distilled_at")
        body["uptime_seconds"] = int(time.monotonic() - _started_at)
    except Exception:
        logger.debug("healthz store query failed", exc_info=True)
        body = {"status": "degraded"}
    await _send_plain(send, 200, json.dumps(body).encode(), b"application/json")


# ── FastMCP server ───────────────────────────────────────────────────────────

mcp = FastMCP(
    name="subrosa-brain",
    instructions=(
        "Subrosa Open Brain — structured knowledge distilled from engineering activity. "
        "Use search_knowledge for semantic queries, query_entity for entity-specific lookups, "
        "get_domain_brief for VP-domain summaries, and capture_thought to write new knowledge."
    ),
    # Stateless: each request is self-contained, so server restarts never
    # leave clients holding dead session ids.
    stateless_http=True,
)


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_knowledge(
    query: str,
    domain: str | None = None,
    primitive: str | None = None,
    entity_type: str | None = None,
    entity_name: str | None = None,
    days: int | None = None,
    limit: int = 10,
) -> str:
    """Semantic search over structured engineering knowledge.

    Use this as the primary query tool. Returns knowledge items ranked by
    relevance to `query`, optionally filtered.

    Args:
        query: Natural language query (e.g. "supply chain attack risk", "mobile release status")
        domain: Restrict to one VP domain (e.g. "Delivery & Execution", "Product Outcome Ownership")
        primitive: Restrict to one primitive type (e.g. "risk", "decision", "state", "event")
        entity_type: Restrict to entity type (product | system | person | team | initiative | process | vendor)
        entity_name: Restrict to a specific named entity
        days: Only return items updated in the last N days
        limit: Max results (default 10)
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})
    results = await _store.search_knowledge_semantic(
        query=query,
        domain=domain,
        primitive=primitive,
        entity_type=entity_type,
        entity_name=entity_name,
        days=days,
        limit=limit,
    )
    # Drop embedding blob from output
    clean = [{k: v for k, v in r.items() if k != "embedding"} for r in results]
    return json.dumps(clean, indent=2, default=str)


@mcp.tool()
async def query_entity(
    entity_name: str,
    entity_type: str | None = None,
    domain: str | None = None,
    days: int | None = None,
    limit: int = 20,
) -> str:
    """All knowledge about a specific entity (person, team, product, system, etc.).

    Use this to get a complete picture of what we know about a named entity.
    Returns items newest-first.

    Args:
        entity_name: The entity to look up (e.g. "Scout", "Himanshu", "Mobile 100.19.0")
        entity_type: Optional filter (person | team | product | system | initiative | process | vendor)
        domain: Optional VP domain filter
        days: Only return items updated in the last N days
        limit: Max results (default 20)
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})
    results = await _store.query_entity(
        entity_name=entity_name,
        entity_type=entity_type,
        domain=domain,
        days=days,
        limit=limit,
    )
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
async def get_domain_brief(
    domain: str,
    days: int = 14,
) -> str:
    """Structured knowledge summary for a VP domain, grouped by primitive type.

    Returns all knowledge for the domain over the time window, organized as:
    decisions made, risks identified, current states, events, tasks, commitments, etc.

    Args:
        domain: VP domain name (e.g. "Delivery & Execution", "Technical & Quality Leadership")
        days: Time window in days (default 14)
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})
    result = await _store.get_domain_brief(domain=domain, days=days)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def capture_thought(
    text: str,
    domain: str | None = None,
    primitive: str | None = None,
    entity_name: str | None = None,
    entity_type: str | None = None,
    source: str = "manual",
) -> str:
    """Write a knowledge item directly to the brain.

    Use this to capture decisions, risks, commitments, or observations from any
    session — a meeting, a code review, a conversation. If domain/primitive/
    entity are omitted, Haiku classifies them automatically.

    Args:
        text: The knowledge item to capture (1-3 sentences, precise and factual)
        domain: VP domain (optional — will be inferred if omitted)
        primitive: Memory primitive: reference|state|event|decision|commitment|risk|task|principle
        entity_name: The primary entity this concerns (optional)
        entity_type: product|system|person|team|initiative|process|vendor
        source: Where this came from (default "manual")
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})

    # If classification fields are missing, call Haiku to classify
    if not domain or not primitive or not entity_name or not entity_type:
        classified = await _classify_thought(text, domain, primitive, entity_name, entity_type)
        domain = classified.get("domain", domain or "Strategic")
        primitive = classified.get("primitive", primitive or "reference")
        entity_name = classified.get("entity_name", entity_name or "unknown")
        entity_type = classified.get("entity_type", entity_type or "product")

    kid = await _store.insert_knowledge(
        domain=domain,
        primitive=primitive,
        entity_type=entity_type,
        entity_name=entity_name,
        summary=text,
        source=source,
        confidence=0.9,
    )
    return json.dumps({"status": "ok", "id": kid, "domain": domain, "primitive": primitive,
                       "entity_name": entity_name})


@mcp.tool()
async def add_domain(name: str, description: str) -> str:
    """Add a new VP domain to the ontology.

    The next distillation cycle will include the new domain in its prompt.

    Args:
        name: Short domain name (e.g. "Security & Compliance")
        description: What responsibility area this covers (1-2 sentences)
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})
    await _store.upsert_ontology_domain(name=name, description=description)
    return json.dumps({"status": "ok", "name": name})


@mcp.tool()
async def add_primitive(name: str, description: str, retrieval_behavior: str) -> str:
    """Add a new memory primitive type to the ontology.

    Args:
        name: Short primitive name (e.g. "observation")
        description: What kind of knowledge this captures (1 sentence)
        retrieval_behavior: When/how to surface this primitive in queries (1 sentence)
    """
    if _store is None:
        return json.dumps({"error": "store not initialized"})
    await _store.upsert_ontology_primitive(
        name=name, description=description, retrieval_behavior=retrieval_behavior
    )
    return json.dumps({"status": "ok", "name": name})


# ── Haiku classifier for capture_thought ────────────────────────────────────

_CLASSIFY_SYSTEM = (
    "You are a knowledge classifier. Given a text snippet, output a single JSON object "
    "with keys: domain, primitive, entity_name, entity_type. Use the ontology provided."
)

_CLASSIFY_CLI = "/home/ati/.local/bin/claude"


async def _classify_thought(
    text: str,
    domain: str | None,
    primitive: str | None,
    entity_name: str | None,
    entity_type: str | None,
) -> dict[str, str]:
    """Use Haiku to classify a thought that's missing ontology fields."""
    if _store is None:
        return {}
    domains = await _store.get_ontology_domains()
    primitives = await _store.get_ontology_primitives()
    domain_list = ", ".join(d["name"] for d in domains)
    primitive_list = ", ".join(p["name"] for p in primitives)
    prompt = (
        f"Domains: {domain_list}\n"
        f"Primitives: {primitive_list}\n"
        f"Entity types: person, team, product, system, initiative, process, vendor\n\n"
        f"Text: {text}\n\n"
        f"Classify this. Return only JSON with keys: domain, primitive, entity_name, entity_type."
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            _CLASSIFY_CLI,
            "--model", "claude-haiku-4-5-20251001",
            "--output-format", "text",
            "--no-session-persistence",
            "--system-prompt", _CLASSIFY_SYSTEM,
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        text_out = stdout.decode(errors="replace").strip()
        import re
        text_out = re.sub(r"^```(?:json)?\s*", "", text_out)
        text_out = re.sub(r"\s*```$", "", text_out).strip()
        return json.loads(text_out)
    except Exception:
        logger.warning("Haiku classification failed for capture_thought", exc_info=True)
        return {}


# ── Lifecycle ────────────────────────────────────────────────────────────────

def _build_asgi(config: Config):
    """Root ASGI app: /healthz + streamable HTTP (/mcp) + legacy SSE (/sse)."""
    sse_app = mcp.sse_app()          # serves /sse (GET) + /messages/ (POST)
    http_app = mcp.streamable_http_app()  # serves /mcp

    async def root(scope, receive, send):
        if scope["type"] == "lifespan":
            # The streamable transport's session manager lives in http_app's
            # lifespan; SSE has no lifespan requirements.
            await http_app(scope, receive, send)
            return
        path = scope.get("path", "")
        if scope["type"] == "http" and path == "/healthz":
            await _healthz(scope, receive, send)
            return
        if path == "/sse" or path.startswith("/messages"):
            await sse_app(scope, receive, send)
            return
        await http_app(scope, receive, send)

    if not config.brain_token:
        logger.warning("Brain server: no token configured — running without auth")
        return root
    return BearerAuthASGI(root, config.brain_token, exempt_paths=frozenset({"/healthz"}))


async def _serve(config: Config) -> None:
    """Run the MCP server as a long-running async task."""
    import uvicorn

    cfg = uvicorn.Config(
        _build_asgi(config),
        host=config.brain_host,
        port=config.brain_port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(cfg)
    logger.info(
        "Brain server starting on %s:%d (/mcp streamable, /sse legacy, /healthz)",
        config.brain_host, config.brain_port,
    )
    await server.serve()


def start(store: Store, config: Config) -> asyncio.Task:
    """
    Start the brain MCP server as a background asyncio task.
    Returns the task so the caller can track it.
    """
    global _store, _started_at
    _store = store
    _started_at = time.monotonic()

    task = asyncio.create_task(_serve(config))
    task.add_done_callback(_on_done)
    return task


def _on_done(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.info("Brain server task cancelled")
    elif task.exception():
        logger.error("Brain server task failed: %s", task.exception())
    else:
        logger.info("Brain server task finished")
