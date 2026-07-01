# Subrosa Web Interface — Implementation Plan

## Decisions
- **Repo**: Inside subrosa, new `frontend/` directory
- **Telegram**: Keep as secondary channel (remote/mobile access when off local network)
- **Web UI**: Primary interface on local network
- **Stack**: React + Vite + Tailwind (Bun for packages), Python backend gets HTTP routes
- **Streaming**: SSE (Server-Sent Events) for real-time agent responses
- **Auth**: None (single user, local network)

## Architecture

```
subrosa/
├── frontend/                  # NEW — React + Vite + Tailwind
│   ├── src/
│   │   ├── App.tsx            # State-driven routing (Reduction pattern)
│   │   ├── components/
│   │   │   ├── Chat.tsx       # Main chat interface
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageInput.tsx
│   │   │   ├── MemoryBrowser.tsx
│   │   │   ├── SkillBar.tsx
│   │   │   └── StatusBar.tsx
│   │   ├── services/
│   │   │   └── api.ts         # Typed fetch wrapper + SSE client
│   │   ├── types/
│   │   │   └── index.ts
│   │   └── index.css          # Tailwind + custom theme
│   ├── vite.config.ts         # Proxy /api → Python backend
│   ├── tailwind.config.js
│   └── package.json
├── subrosa/
│   ├── app.py                 # MODIFIED — start HTTP server alongside Telegram
│   ├── web.py                 # NEW — FastAPI app with routes
│   ├── agent.py               # UNCHANGED — core agent invocation
│   ├── context.py             # UNCHANGED — prompt assembly
│   ├── memory.py              # UNCHANGED
│   ├── store.py               # UNCHANGED
│   ├── skills.py              # UNCHANGED
│   ├── procedures.py          # UNCHANGED
│   ├── scheduler.py           # MODIFIED — dual delivery: Telegram + inbox table
│   ├── config.py              # MODIFIED — add web_port, keep telegram settings
│   ├── telegram.py            # KEPT — secondary channel for remote access
│   └── media.py               # KEPT — Telegram media handling stays
```

## Message Flow (New)

```
Browser POST /api/chat { message, conversation_id? }
    → web.py handler
    → build_system_prompt() + build_user_prompt()  (unchanged)
    → Agent.invoke(user_prompt, system_prompt, resume_session)
    → SSE stream: each text chunk → event: chunk
    → SSE stream: final event: done { session_id, tools_used }
    → Background: memory extraction (unchanged)
    → Background: procedure reflection (unchanged)
```

## API Endpoints

### Chat
- `POST /api/chat` — Send message, returns SSE stream
  - Body: `{ message: string, conversation_id?: string, files?: File[] }`
  - Response: `text/event-stream`
  - Events: `chunk` (text delta), `status` (tool use indicator), `done` (final metadata)

### Conversations
- `GET /api/conversations` — List conversations
- `GET /api/conversations/:id/messages` — Get message history
- `DELETE /api/conversations/:id` — Clear conversation (equivalent to /clear)

### Memory
- `GET /api/memories` — List all memories (with search param)
- `GET /api/memories/:id` — Single memory detail

### Skills
- `GET /api/skills` — List available skills
- `POST /api/skills/:name/run` — Trigger a skill (e.g., briefing)

### Health
- `GET /api/health` — System status, uptime, last invocation time

### Scheduled Messages
- `GET /api/inbox` — Scheduled briefings and notifications stored for pickup
  - Scheduler writes to DB instead of Telegram; frontend polls or uses SSE

## Phases

### Phase 1: Backend HTTP Layer
**Files**: `web.py` (new), `app.py` (modify), `config.py` (modify)

1. Add FastAPI + uvicorn to dependencies
2. Create `web.py` with:
   - FastAPI app instance
   - `POST /api/chat` endpoint with SSE streaming
   - `GET /api/health` endpoint
   - Static file serving (production: serve `frontend/dist/`)
3. Modify `app.py`:
   - Start both uvicorn (web) and Telegram bot concurrently
   - Both share the same agent, store, memory, skills instances
   - Keep scheduler, store, memory, skills initialization unchanged
   - Pass shared state to web routes
4. Modify `config.py`:
   - Add `web_port` (default 8080), `web_host` (default 0.0.0.0 for LAN access)
   - Keep all telegram fields as-is
5. **Test gate**: `curl -X POST localhost:8080/api/chat` returns SSE stream with agent response; Telegram still works

### Phase 2: Frontend Foundation
**Files**: All new in `frontend/`

1. Initialize Bun project with React, Vite, Tailwind
2. Vite config: proxy `/api/*` → `http://localhost:8080`
3. Create typed API client (`services/api.ts`)
4. Create SSE streaming client for chat
5. Root layout: header (⛩️ Subrosa), main content area, input bar
6. Basic theme: dark, calm, minimal — temple keeper aesthetic
7. **Test gate**: Frontend loads, can send a message and see streamed response

### Phase 3: Chat Interface
**Files**: Components in `frontend/src/components/`

1. `Chat.tsx` — Main view, manages message state
2. `MessageList.tsx` — Scrollable message history with auto-scroll
3. `MessageInput.tsx` — Text input with Enter-to-send, Shift+Enter for newline
4. `MessageBubble.tsx` — Renders markdown (agent) or plain text (user)
5. `StreamingIndicator.tsx` — Shows "thinking..." with tool use status events
6. Conversation persistence: store messages in localStorage + fetch from API
7. **Test gate**: Full chat flow works — send message, see streaming response, history persists

### Phase 4: Subrosa Features
**Files**: New components + API endpoints

1. `SkillBar.tsx` — Row of skill buttons (briefing, etc.) above input
2. `POST /api/skills/:name/run` backend endpoint
3. `MemoryBrowser.tsx` — Searchable list of memories (sidebar or separate view)
4. `GET /api/memories` backend endpoint
5. `StatusBar.tsx` — Bottom bar showing connection status, last response time
6. `GET /api/inbox` — Scheduled message pickup (briefings land here too)
7. Modify `scheduler.py` for dual delivery: Telegram send (as before) + write to inbox table (for web pickup)
8. **Test gate**: Can trigger briefing from UI, see it stream in; can browse memories; Telegram still receives scheduled briefings

### Phase 5: Polish & Production
1. Add file upload support to `POST /api/chat`
2. Production build: `bun run --cwd frontend build`
3. FastAPI serves `frontend/dist/` as static files
4. Single command to start: `python -m subrosa` → serves web + Telegram on port 8080
5. Concurrency guard: requests from web and Telegram share a lock so agent isn't invoked twice simultaneously
6. **Test gate**: Full app works from single process; both web and Telegram functional

## Dependencies to Add
- **Python**: `fastapi`, `uvicorn`, `sse-starlette` (for SSE support)
- **Frontend**: `react`, `react-dom`, `vite`, `tailwindcss`, `@tailwindcss/typography`, `date-fns`

## Key Design Decisions

### Why SSE over WebSocket
- Agent responses are unidirectional (server → client)
- SSE is simpler — plain HTTP, auto-reconnect, no protocol upgrade
- Client sends via POST, receives via SSE — clean separation
- Subrosa's Agent.invoke() already yields chunks via async generator — maps directly to SSE events

### Why FastAPI
- Already async (matches subrosa's asyncio architecture)
- SSE support via sse-starlette
- Automatic OpenAPI docs at /docs (useful for debugging)
- Static file serving built in
- Minimal boilerplate

### Dual-Channel Scheduler
- Scheduler delivers to both channels: Telegram send (existing) + inbox table write (new)
- Frontend polls `GET /api/inbox` or receives via persistent SSE connection
- Unread indicator in web UI when new inbox items arrive
- Telegram keeps working as-is for mobile/remote access
- Briefings become searchable via web while still pushing to your phone

### Concurrency
- Both Telegram and web share a single agent instance
- asyncio Lock prevents simultaneous invocations (agent uses CLI subprocess)
- If one channel is processing, the other queues (same pattern Telegram already uses)

### Theme Direction
- Dark background (temple at night)
- Warm accent (amber/gold — torii gate color)
- Clean typography (sans-serif for readability)
- ⛩️ as logo mark
- Minimal chrome — the conversation is the interface
