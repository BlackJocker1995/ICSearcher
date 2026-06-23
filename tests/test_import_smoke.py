"""Import-smoke tests: every module that previously crashed on import must now
import cleanly. These directly verify the stage-1 bug fixes:

* uavga/searcher.py imported a non-existent `Problem` name  -> 3.lgfuzzer crashed
* uavga/fuzzer.py   imported a non-existent `SearchOptimizer` name
* range/rangesearcher.py imported a non-existent `RangeProblem` name -> 5.range crashed
* deprecated numpy.dual / np.float would raise on modern numpy

If any of these regress, this test fails before the pipeline ever runs.
"""
import importlib

import pytest

# The heavy dependencies behind these modules are only present once the full
# Poetry environment is installed (`poetry install`). Skip gracefully in a
# minimal environment rather than reporting false failures.
pytest.importorskip("geatpy")
pytest.importorskip("pymavlink")


def _import(modname):
    return importlib.import_module(modname)


def test_import_uavga_searcher():
    """The buggy `from uavga.problem import Problem` import was removed."""
    mod = _import("uavga.searcher")
    assert hasattr(mod, "GAOptimizer")


def test_import_uavga_fuzzer():
    """The buggy `SearchOptimizer` import was removed; run_fuzzing is present."""
    mod = _import("uavga.fuzzer")
    assert hasattr(mod, "run_fuzzing")
    assert hasattr(mod, "split_segment")


def test_import_uavga_problem():
    mod = _import("uavga.problem")
    assert hasattr(mod, "ProblemGA")


def test_import_range_rangesearcher():
    """The buggy `RangeProblem` import was removed; GARangeOptimizer present."""
    mod = _import("range.rangesearcher")
    assert hasattr(mod, "GARangeOptimizer")


def test_import_range_rangeproblem():
    mod = _import("range.rangeproblem")
    assert hasattr(mod, "GARangeProblem")


def test_import_cptool_modules():
    """Cptool.* modules import without the removed dead stubs."""
    _import("Cptool.config")
    _import("Cptool.mavtool")
    _import("Cptool.logging_config")
    # gaMavlink / gaSimManager pull pymavlink + pexpect; importing them proves
    # the stdlib-logging migration left no NameError behind.
    _import("Cptool.gaMavlink")
    _import("Cptool.gaSimManager")


def test_no_deprecated_numpy_aliases():
    """None of our source files reference removed numpy aliases / numpy.dual."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    # Word-boundary patterns so np.float32 does not match np.float.
    patterns = [
        re.compile(r"\bnp\.float\b(?!\d)"),
        re.compile(r"\bnp\.int\b(?!\d)"),
        re.compile(r"\bnp\.bool\b(?!\d)"),
        re.compile(r"numpy\.dual"),
    ]
    offenders = []
    for py in repo.rglob("*.py"):
        # Skip the tests themselves (their docstrings mention these tokens) and
        # the git metadata.
        if ".git" in py.parts or "tests" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        for pat in patterns:
            if pat.search(text):
                offenders.append(f"{py}: {pat.pattern}")
    assert not offenders, "deprecated numpy tokens remain: " + ", ".join(offenders)
