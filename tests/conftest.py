"""Pytest fixtures shared across the ICSearcher test suite.

Stage 1 only tests pure functions that do not require a live SITL simulator.
"""
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `from Cptool.config import toolConfig` works
# regardless of pytest's CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(params=["Ardupilot", "PX4"])
def mode(request):
    """Run the test once for each firmware mode, re-selecting the config."""
    from Cptool.config import toolConfig

    toolConfig.select_mode(request.param)
    yield request.param
