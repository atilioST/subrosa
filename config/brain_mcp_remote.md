# Subrosa Brain — Remote MCP Client Setup

The Open Brain MCP server runs inside the Subrosa process on SubrosaBox
(`192.168.0.154:7771`) and exposes 6 tools over HTTP SSE.

## Prerequisites

- Port 7771 must be open on SubrosaBox's firewall for `192.168.0.0/24`
  (see Firewall section below)
- Subrosa must be running with `[brain] enabled = true` in config.toml

## Claude Code — add as MCP server

Add to `~/.claude.json` on the remote machine (under `"mcpServers"`):

```json
"subrosa-brain": {
  "type": "sse",
  "url": "http://192.168.0.154:7771/sse",
  "headers": {
    "Authorization": "Bearer <SUBROSA_BRAIN_TOKEN>"
  }
}
```

Replace `<SUBROSA_BRAIN_TOKEN>` with the value from `~/.subrosa/.env` on
SubrosaBox.

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) under `"mcpServers"`:

```json
"subrosa-brain": {
  "type": "sse",
  "url": "http://192.168.0.154:7771/sse",
  "headers": {
    "Authorization": "Bearer <SUBROSA_BRAIN_TOKEN>"
  }
}
```

## Available tools

| Tool | Description |
|---|---|
| `search_knowledge` | Semantic search, optionally filtered by domain/primitive/entity/days |
| `query_entity` | All knowledge about a named entity (person, product, system…) |
| `get_domain_brief` | Summary for a VP domain over a time window |
| `capture_thought` | Write a knowledge item; Haiku auto-classifies if fields omitted |
| `add_domain` | Extend the ontology with a new VP domain |
| `add_primitive` | Extend the ontology with a new memory primitive type |

## Firewall — SubrosaBox

Run on SubrosaBox as root to open port 7771 for LAN clients:

```
# Add to /etc/nftables.conf inside chain input, before the rate-limit line:
ip saddr 192.168.0.0/24 tcp dport 7771 accept comment "subrosa brain MCP (LAN only)"

# Then reload:
systemctl reload nftables
```

Or as a one-shot live rule (no reboot persistence):

```
nft add rule inet filter input ip saddr 192.168.0.0/24 tcp dport 7771 accept
```

## Verify connectivity

From a remote machine:

```bash
curl -H "Authorization: Bearer <token>" \
     http://192.168.0.154:7771/sse \
     --max-time 3 -v 2>&1 | grep -E "< HTTP|Connected"
```

Expected: `< HTTP/1.1 200 OK`
