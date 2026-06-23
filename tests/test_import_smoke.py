"""Import-smoke tests: every module that previously crashed on import must now
import cleanly. These directly verify the stage-1 import fixes, now against the
stage-2 package layout (icsearcher.*).
"""
import importlib

import pytest

# Heavy deps behind these modules are only present after the full Poetry
# environment is installed. Skip gracefully in a minimal environment.
pytest.importorskip("geatpy")
pytest.importorskip("pymavlink")


def _import(modname):
    return importlib.import_module(modname)


def test_import_search_searcher():
    mod = _import("icsearcher.search.searcher")
    assert hasattr(mod, "GAOptimizer")


def test_import_search_fuzzer():
    mod = _import("icsearcher.search.fuzzer")
    assert hasattr(mod, "run_fuzzing")
    assert hasattr(mod, "split_segment")


def test_import_search_problem():
    mod = _import("icsearcher.search.problem")
    assert hasattr(mod, "ProblemGA")


def test_import_search_io():
    mod = _import("icsearcher.search.io")
    assert hasattr(mod, "CandidateSet")
    assert hasattr(mod, "read_candidates")
    assert hasattr(mod, "write_candidates")


def test_import_range_searcher():
    mod = _import("icsearcher.range.searcher")
    assert hasattr(mod, "GARangeOptimizer")


def test_import_range_problem():
    mod = _import("icsearcher.range.problem")
    assert hasattr(mod, "GARangeProblem")


def test_import_icsearcher_modules():
    """Core icsearcher modules import without the removed dead stubs."""
    _import("icsearcher.config")
    _import("icsearcher.params")
    _import("icsearcher.logging_config")
    # comms / sim pull pymavlink + pexpect; importing proves the stdlib-logging
    # migration left no NameError behind.
    _import("icsearcher.comms")
    _import("icsearcher.sim")
