"""Tests for app module."""

import importlib


def test_app_module_imports():
    """Verify app.py imports without errors."""
    mod = importlib.import_module("subrosa1.app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_async_main")
    assert hasattr(mod, "_setup_logging")


def test_setup_logging():
    """Verify logging setup doesn't crash."""
    from subrosa1.app import _setup_logging
    _setup_logging("DEBUG")
    _setup_logging("INFO")
    _setup_logging("WARNING")
