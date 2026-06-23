"""Pytest fixtures shared across the ICSearcher test suite.

Stage 1 only tested pure functions that do not require a live SITL simulator.
Stage 2 generalized this: tests build a fresh ``ToolConfig`` per firmware mode
and repoint the package-level singleton at it, because the production code reads
the global ``toolConfig`` (mode is frozen at load time, so we cannot mutate it).
"""
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `from icsearcher.config import toolConfig`
# works regardless of pytest's CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(params=["Ardupilot", "PX4"])
def mode(monkeypatch, request):
    """Run the test once for each firmware mode.

    Because ``toolConfig`` is frozen at load time (stage 2), this fixture builds
    a fresh per-mode instance and repoints the ``icsearcher.config.toolConfig``
    singleton at it. Modules that captured the old singleton via
    ``from icsearcher.config import toolConfig`` keep their own reference, so
    tests that exercise such modules must import the symbol lazily inside the
    test body (which most already do).
    """
    import icsearcher.config as cfg

    fresh = cfg.ToolConfig(mode=request.param)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    yield request.param
