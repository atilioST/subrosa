"""Tests for telegram helpers."""

from subrosa.agent import InvocationProgress
from subrosa.telegram import (
    WorkingIndicator,
    _fmt_elapsed,
    _markdown_to_html,
    _short_tool_name,
    _strip_html,
    split_message,
)


def test_split_short_message():
    chunks = split_message("hello world")
    assert chunks == ["hello world"]


def test_split_long_message():
    text = "word " * 1000  # ~5000 chars
    chunks = split_message(text, max_length=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_markdown_to_html():
    assert "<b>bold</b>" in _markdown_to_html("**bold**")
    assert "<i>italic</i>" in _markdown_to_html("*italic*")
    assert "<code>code</code>" in _markdown_to_html("`code`")
    assert "<s>strike</s>" in _markdown_to_html("~~strike~~")


def test_strip_html():
    assert _strip_html("<b>bold</b> text") == "bold text"
    assert _strip_html("no tags") == "no tags"


def test_fmt_elapsed():
    assert _fmt_elapsed(45) == "45s"
    assert _fmt_elapsed(80) == "1m 20s"
    assert _fmt_elapsed(600) == "10m 00s"


def test_short_tool_name():
    assert _short_tool_name("mcp__slack__slack_search_messages") == "slack_search_messages"
    assert _short_tool_name("ToolSearch") == "ToolSearch"


def test_indicator_status_without_progress():
    ind = WorkingIndicator(bot=None, chat_id=1)
    assert ind._status_text().startswith("Working on it...")


def test_indicator_status_shows_tool_activity():
    progress = InvocationProgress()
    progress.note_tool("mcp__slack__slack_search_messages", {"query": "from:@brock"})
    progress.note_tool("mcp__slack__slack_search_messages", {"query": "in:class1"})
    ind = WorkingIndicator(bot=None, chat_id=1, progress=progress)
    status = ind._status_text()
    assert "slack_search_messages" in status
    assert "(in:class1)" in status
    assert "2 tools" in status


def test_indicator_status_shows_continuation_round():
    progress = InvocationProgress()
    progress.note_tool("Read", {"file_path": "/tmp/x"})
    progress.continuations = 1
    ind = WorkingIndicator(bot=None, chat_id=1, progress=progress)
    assert "round 2" in ind._status_text()
