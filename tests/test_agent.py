"""Tests for agent progress tracking and response merging."""

from subrosa.agent import (
    AgentResponse,
    InvocationProgress,
    _merge_responses,
    _tool_detail,
)


def test_tool_detail_extracts_query():
    assert _tool_detail({"query": "from:@brock after:2026-07-06"}) == "from:@brock after:2026-07-06"


def test_tool_detail_truncates():
    detail = _tool_detail({"query": "x" * 100})
    assert len(detail) == 61
    assert detail.endswith("…")


def test_tool_detail_handles_junk():
    assert _tool_detail(None) == ""
    assert _tool_detail({}) == ""
    assert _tool_detail({"count": 30}) == ""


def test_progress_tracks_tools_and_text():
    p = InvocationProgress()
    assert p.last_tool is None
    p.note_tool("mcp__slack__slack_search_messages", {"query": "from:@brock"})
    p.note_text("Looking at Slack...")
    assert p.last_tool.name == "mcp__slack__slack_search_messages"
    assert p.last_tool.detail == "from:@brock"
    assert p.partial_text == "Looking at Slack..."
    assert len(p.tools) == 1


def test_merge_responses_joins_text_and_sums():
    prior = AgentResponse(
        text="part one", session_id="s1", total_cost_usd=0.1,
        num_turns=10, duration_ms=1000, tools_used=["a"],
        is_error=True, subtype="error_max_turns",
    )
    continued = AgentResponse(
        text="part two", session_id="s2", total_cost_usd=0.2,
        num_turns=5, duration_ms=500, tools_used=["b"],
        is_error=False, subtype="success",
    )
    merged = _merge_responses(prior, continued)
    assert merged.text == "part one\npart two"
    assert merged.session_id == "s2"
    assert abs(merged.total_cost_usd - 0.3) < 1e-9
    assert merged.num_turns == 15
    assert merged.tools_used == ["a", "b"]
    assert not merged.is_error
    assert merged.subtype == "success"


def test_merge_responses_failed_continuation_keeps_prior_text():
    prior = AgentResponse(text="partial work", subtype="error_max_turns", is_error=True)
    continued = AgentResponse(text="Agent error — please try again.", is_error=True, subtype="")
    merged = _merge_responses(prior, continued)
    assert merged.text == "partial work"


def test_merge_responses_chained_max_turns_joins():
    prior = AgentResponse(text="one", subtype="error_max_turns", is_error=True, session_id="s1")
    continued = AgentResponse(text="two", subtype="error_max_turns", is_error=True, session_id="s2")
    merged = _merge_responses(prior, continued)
    assert merged.text == "one\ntwo"
    assert merged.subtype == "error_max_turns"


async def test_invoke_auto_continues_on_max_turns(monkeypatch):
    from subrosa.agent import Agent

    calls = []

    async def fake_do_invoke(self, prompt, system_prompt, resume_session, *args, **kwargs):
        calls.append((prompt, resume_session))
        if len(calls) == 1:
            return AgentResponse(
                text="half done", session_id="s1",
                is_error=True, subtype="error_max_turns", num_turns=10,
            )
        return AgentResponse(
            text="finished", session_id="s2",
            is_error=False, subtype="success", num_turns=3,
        )

    monkeypatch.setattr(Agent, "_do_invoke", fake_do_invoke)
    agent = Agent(model="sonnet", max_turns=10)
    response = await agent.invoke("do a big task", "system")

    assert len(calls) == 2
    assert calls[1][1] == "s1"  # continuation resumed the capped session
    assert response.text == "half done\nfinished"
    assert response.subtype == "success"
    assert not response.is_error


async def test_invoke_stops_continuing_at_limit(monkeypatch):
    from subrosa.agent import Agent

    calls = []

    async def always_capped(self, prompt, system_prompt, resume_session, *args, **kwargs):
        calls.append(prompt)
        return AgentResponse(
            text=f"chunk{len(calls)}", session_id=f"s{len(calls)}",
            is_error=True, subtype="error_max_turns", num_turns=10,
        )

    monkeypatch.setattr(Agent, "_do_invoke", always_capped)
    agent = Agent(model="sonnet", max_turns=10, max_continuations=2)
    response = await agent.invoke("big task", "system")

    assert len(calls) == 3  # initial + 2 continuations
    assert response.subtype == "error_max_turns"
    assert response.text == "chunk1\nchunk2\nchunk3"
