"""Static lint tests: scan the source tree for regressions of the stage-1 fixes.

These tests have no heavy dependencies and run in every environment, so they
guard against reintroducing the deprecated APIs that crashed on modern numpy.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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
    for py in REPO.rglob("*.py"):
        # Skip tests (their docstrings mention these tokens) and git metadata.
        if ".git" in py.parts or "tests" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        for pat in patterns:
            if pat.search(text):
                offenders.append(f"{py}: {pat.pattern}")
    assert not offenders, "deprecated tokens remain: " + ", ".join(offenders)


def test_no_deprecated_stubs_referenced():
    """The removed dead stubs must not be referenced anywhere."""
    removed = ("LogHandler", "ParameterManager", "MavlinkManager",
               "GAOptimizerOld", "ProblemGAOld")
    offenders = []
    for py in REPO.rglob("*.py"):
        if ".git" in py.parts or "tests" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        for sym in removed:
            if sym in text:
                offenders.append(f"{py}: {sym}")
    assert not offenders, "removed stubs still referenced: " + ", ".join(offenders)


def test_no_hardcoded_repo_relative_mission_paths():
    """Mission/fit-collection paths must go through toolConfig, not be hardcoded."""
    offenders = []
    for py in REPO.rglob("*.py"):
        if ".git" in py.parts or "tests" in py.parts or py.name == "config.py":
            continue
        text = py.read_text(encoding="utf-8")
        if re.search(r"['\"]Cptool/fitCollection(_px4)?\.txt['\"]", text):
            offenders.append(str(py))
    assert not offenders, "hardcoded mission paths should use toolConfig.mission_file(): " + ", ".join(offenders)
