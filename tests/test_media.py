"""Tests for media formatting."""

from subrosa1.media import format_media_for_prompt


def test_format_empty():
    assert format_media_for_prompt([]) == ""


def test_format_photo():
    result = format_media_for_prompt([
        {"type": "photo", "path": "/tmp/photo.jpg", "file_size": 1048576, "caption": "Screenshot"},
    ])
    assert "Photo" in result
    assert "/tmp/photo.jpg" in result
    assert "1.00MB" in result
    assert "Screenshot" in result


def test_format_document():
    result = format_media_for_prompt([
        {"type": "document", "path": "/tmp/doc.pdf", "file_size": 512000, "file_name": "report.pdf"},
    ])
    assert "Document" in result
    assert "report.pdf" in result


def test_format_multiple():
    result = format_media_for_prompt([
        {"type": "photo", "path": "/tmp/a.jpg", "file_size": 100000},
        {"type": "voice", "path": "/tmp/b.ogg", "file_size": 50000},
    ])
    assert "Photo" in result
    assert "Voice" in result
