from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TEST_ROOT = Path(tempfile.mkdtemp(prefix="video-precheck-test-"))
os.environ.setdefault("VIDEO_PRECHECK_DATA_DIR", str(TEST_ROOT / "data"))
os.environ.setdefault("VIDEO_PRECHECK_DB", str(TEST_ROOT / "data" / "app.db"))
os.environ.setdefault("SEMANTIC_ENABLED", "false")
os.environ.setdefault("VISUAL_ENABLED", "false")
os.environ.setdefault("REWRITE_ENABLED", "false")
os.environ.setdefault("AUDIT_FEED_ENABLED", "false")


@pytest.fixture(autouse=True)
def _no_external_media_probe(monkeypatch):
    import app.api as api_module

    monkeypatch.setattr(api_module, "probe_duration_ms", lambda _: 0)
