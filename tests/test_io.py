"""Tests for the inter-stage artifact read/write (icsearcher.search.io).

Specifically guards the populations/candidates **file-split** fix: stage 3
(fuzz) writes populations to ``pop{EXE}.npz`` and stage 4 (``pre``) writes
candidates to ``cand{EXE}.npz``. Before the split they shared one file, so the
first ``pre`` run silently corrupted the populations and a re-run crashed
(``read_populations`` found candidate keys ``obj``/``var`` instead of
``n_pops``).
"""
import numpy as np
import pytest

pymavlink = pytest.importorskip("pymavlink")

import icsearcher.config as cfg  # noqa: E402
from icsearcher.search import io as io_mod  # noqa: E402
from icsearcher.search.searcher import PopulationResult  # noqa: E402


@pytest.fixture
def isolated_io(monkeypatch, tmp_path):
    """Redirect io's artifact paths into a throwaway tmp dir.

    toolConfig is frozen at load time (by design), so rather than mutate the
    singleton we monkeypatch the two private path resolvers that every read/
    write goes through. This isolates each test from any real result/ dir.
    """
    from pathlib import Path
    result_dir = tmp_path / "result" / "Ardupilot"
    result_dir.mkdir(parents=True)

    def fake_pop_path():
        return Path(result_dir / "pop.npz")

    def fake_cand_path():
        return Path(result_dir / "cand.npz")

    monkeypatch.setattr(io_mod, "_pop_path", fake_pop_path)
    monkeypatch.setattr(io_mod, "_candidates_path", fake_cand_path)
    return result_dir


def _fake_pops(n=2, m=8, d=4):
    return [PopulationResult(Phen=np.random.rand(m, d), ObjV=np.random.rand(m, 1))
            for _ in range(n)]


def test_populations_and_candidates_use_separate_files(isolated_io):
    """write_populations + write_candidates produce two distinct files."""
    pops = _fake_pops()
    io_mod.write_populations(pops)
    io_mod.write_candidates(np.random.rand(5, 4), np.random.rand(5, 1))

    pop_path = io_mod._pop_path()
    cand_path = io_mod._candidates_path()
    assert pop_path != cand_path, "populations and candidates must not share a file"
    assert pop_path.exists() and cand_path.exists()


def test_pre_no_longer_corrupts_populations(isolated_io):
    """Running 'pre' (write_candidates) must leave read_populations intact.

    This is the regression: before the split, write_candidates overwrote the
    populations file, so a subsequent read_populations crashed with a numpy
    KeyError on the missing 'n_pops' key.
    """
    original = _fake_pops(n=3)
    io_mod.write_populations(original)

    # Simulate stage 4 'pre' writing its candidate output.
    io_mod.write_candidates(np.random.rand(5, 4), np.random.rand(5, 1))

    # read_populations must still work — the populations file is untouched.
    reloaded = io_mod.read_populations()
    assert len(reloaded) == 3
    np.testing.assert_array_equal(reloaded[0].Phen, original[0].Phen)
    np.testing.assert_array_equal(reloaded[0].ObjV, original[0].ObjV)


def test_roundtrip_populations(isolated_io):
    pops = _fake_pops(n=2, m=10, d=4)
    io_mod.write_populations(pops)
    reloaded = io_mod.read_populations()
    assert len(reloaded) == 2
    assert reloaded[0].Phen.shape == (10, 4)
    assert reloaded[0].ObjV.shape == (10, 1)


def test_roundtrip_candidates(isolated_io):
    obj = np.random.rand(7, 4)
    var = np.random.rand(7, 1)
    io_mod.write_candidates(obj, var)
    cands = io_mod.read_candidates()
    np.testing.assert_array_equal(np.asarray(cands.obj), obj)
    np.testing.assert_array_equal(np.asarray(cands.var), var)


def test_candidates_legacy_fallback_from_pop_npz(monkeypatch, tmp_path):
    """A pre-split run left candidate data in pop{EXE}.npz; read_candidates
    should still find it (backward compatibility) instead of crashing.

    This test must NOT monkeypatch ``_candidates_path`` — the legacy fallback
    logic lives *inside* the real resolver (it inspects pop.npz's keys). We
    chdir into tmp so the real resolvers (which build ``result/{MODE}/...``
    relative to CWD) read/write there.
    """
    (tmp_path / "result" / "Ardupilot").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    obj = np.random.rand(5, 4)
    var = np.random.rand(5, 1)
    # Write candidate-format keys directly into pop.npz (the bug scenario).
    np.savez_compressed(io_mod._pop_path(), obj=obj, var=var)

    cands = io_mod.read_candidates()
    np.testing.assert_array_equal(np.asarray(cands.obj), obj)
    np.testing.assert_array_equal(np.asarray(cands.var), var)
