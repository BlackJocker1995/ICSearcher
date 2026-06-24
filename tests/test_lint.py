"""Static lint tests: scan the source tree for regressions of the stage-1 fixes.

These tests have no heavy dependencies and run in every environment, so they
guard against reintroducing the deprecated APIs that crashed on modern numpy.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories that are NOT our source: venvs, the cloned firmware under sims/,
# generated artifacts, and the nyctea/ reference project (a sibling codebase
# used as the multi-instance design reference, not ICSearcher source). Walking
# into these would (a) be slow and (b) hit non-UTF-8 files (e.g. vendored
# stubs) and crash the read, or flag code that isn't ours to lint.
_SKIP_DIRS = {".git", ".venv", "sims", "model", "result", "fig",
              "__pycache__", "nyctea"}


def _source_files():
    """Yield our own .py files, skipping vendored/generated trees."""
    for py in REPO.rglob("*.py"):
        parts = set(py.parts)
        if parts & _SKIP_DIRS:
            continue
        yield py


def _read(py: Path) -> str:
    """Read a source file as text, tolerating any stray non-UTF-8 bytes."""
    return py.read_bytes().decode("utf-8", errors="replace")


def test_no_deprecated_numpy_aliases():
    """No source file references removed numpy aliases / numpy.dual."""
    patterns = [
        re.compile(r"\bnp\.float\b(?!\d)"),
        re.compile(r"\bnp\.int\b(?!\d)"),
        re.compile(r"\bnp\.bool\b(?!\d)"),
        re.compile(r"numpy\.dual"),
        re.compile(r"\.fillna\(method=['\"](?:ffill|bfill)['\"]"),
    ]
    offenders = []
    for py in _source_files():
        if "tests" in py.parts:  # test docstrings mention these tokens
            continue
        text = _read(py)
        for pat in patterns:
            if pat.search(text):
                offenders.append(f"{py}: {pat.pattern}")
    assert not offenders, "deprecated tokens remain: " + ", ".join(offenders)


def test_no_deprecated_stubs_referenced():
    """The removed dead stubs must not be referenced anywhere."""
    removed = ("LogHandler", "ParameterManager", "MavlinkManager",
               "GAOptimizerOld", "ProblemGAOld")
    offenders = []
    for py in _source_files():
        if "tests" in py.parts:
            continue
        text = _read(py)
        for sym in removed:
            if sym in text:
                offenders.append(f"{py}: {sym}")
    assert not offenders, "removed stubs still referenced: " + ", ".join(offenders)


def test_no_hardcoded_repo_relative_mission_paths():
    """Mission/fit-collection paths must go through toolConfig, not be hardcoded."""
    offenders = []
    for py in _source_files():
        if "tests" in py.parts or py.name == "config.py":
            continue
        text = _read(py)
        # Both the legacy Cptool/ layout and the new data/ layout.
        if re.search(r"['\"](?:Cptool|data)/fitCollection(_px4)?\.txt['\"]", text):
            offenders.append(str(py))
    assert not offenders, "hardcoded mission paths should use toolConfig.mission_file(): " + ", ".join(offenders)
