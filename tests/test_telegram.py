"""Tests for telegram helpers."""

from subrosa.telegram import split_message, _markdown_to_html, _strip_html, WorkingIndicator


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
