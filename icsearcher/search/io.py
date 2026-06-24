"""Typed read/write for the inter-stage artifacts.

Two distinct artifacts live under ``result/{MODE}/``, kept in **separate
files** so that stage 4's output never overwrites stage 3's output:

- **Populations** (``pop{EXE}.npz``) — stage 3 (fuzz) writes the per-context GA
  populations; stage 4 ``pre`` reads them via :func:`read_populations`.
  Keys: ``phen_0``, ``objv_0``, …, ``n_pops``.
- **Candidates** (``cand{EXE}.npz``) — stage 4 ``pre`` writes the clustered
  candidate set; ``validate`` reads it via :func:`read_candidates`.
  Keys: ``obj`` (param vectors), ``var`` (objective scores).

These were previously the same file, which silently corrupted the populations
on the first ``pre`` run and then crashed on re-run (the candidate-format file
has no ``n_pops`` key). They are now separate.

Uses NumPy's ``.npz`` (compressed binary) instead of pickle — faster to
read/write for numpy arrays, smaller on disk, and no arbitrary code execution
risk. Backward-compatible with legacy ``.pkl`` files if present.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from icsearcher.config import toolConfig


@dataclass
class CandidateSet:
    """A flat set of fuzzing candidates selected for validation.

    Note on naming: despite the field names, ``obj`` holds the **parameter
    vectors** (the configs to validate) and ``var`` holds the **objective
    scores** (the predicted deviation). This odd naming is inherited from the
    legacy ``[obj, var]`` pickle order; it is consistent end-to-end because both
    the writer (``4_validate.pre``) and reader (``4_validate.validate``) use the
    same convention: ``candidates.obj`` is iterated as the param config
    (``value_vector``) and ``candidates.var`` as the score (``vars[0]``).

    Attributes:
        obj: parameter vectors, shape (n_candidates, n_params) — original units.
        var: objective scores, shape (n_candidates, 1) — negated deviation.
    """
    obj: np.ndarray
    var: np.ndarray


def _resolve_npz_or_pkl(npz: Path, pkl: Path) -> Path:
    """Prefer ``.npz``; fall back to legacy ``.pkl`` if only that exists."""
    if npz.exists():
        return npz
    if pkl.exists():
        return pkl
    return npz  # default for writing


def _pop_path() -> Path:
    """Path to the *populations* artifact (stage 3 output / stage 4 ``pre`` input)."""
    return _resolve_npz_or_pkl(
        Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.npz"),
        Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl"),
    )


def _candidates_path() -> Path:
    """Path to the *candidates* artifact (stage 4 ``pre`` output / ``validate`` input).

    Before the populations/candidates files were split, ``pre`` wrote the
    candidate set to ``pop{EXE}.npz`` (overwriting the populations). For
    backward compatibility, if neither ``cand{EXE}.npz`` nor its legacy ``.pkl``
    exists but a ``pop{EXE}.npz`` holding candidate keys (``obj``/``var``) does,
    that misplaced file is used. This lets in-flight pipelines finish after the
    rename without a manual migration step.
    """
    cand_npz = Path(f"result/{toolConfig.MODE}/cand{toolConfig.EXE}.npz")
    cand_pkl = Path(f"result/{toolConfig.MODE}/cand{toolConfig.EXE}.pkl")
    resolved = _resolve_npz_or_pkl(cand_npz, cand_pkl)
    if resolved.exists():
        return resolved
    # Back-compat: a pre-split run left the candidate set inside pop{EXE}.npz.
    legacy_pop_npz = Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.npz")
    if legacy_pop_npz.exists():
        try:
            keys = set(np.load(legacy_pop_npz).files)
        except Exception:
            keys = set()
        if {"obj", "var"} <= keys:
            return legacy_pop_npz
    return cand_npz  # default for writing


def write_candidates(obj: np.ndarray, var: np.ndarray) -> None:
    """Persist a CandidateSet as cand{EXE}.npz (compressed numpy format)."""
    path = _candidates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obj=np.asarray(obj), var=np.asarray(var))


def read_candidates() -> CandidateSet:
    """Load cand{EXE}.npz (or legacy .pkl).

    ``.npz`` format reads ``obj`` and ``var`` arrays directly.
    Legacy ``.pkl`` accepted: both CandidateSet containers and ``[obj, var]``
    lists produced by older pipeline runs.
    """
    path = _candidates_path()
    if path.suffix == ".npz":
        data = np.load(path)
        return CandidateSet(obj=data["obj"], var=data["var"])
    # Legacy .pkl back-compat
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, CandidateSet):
        return data
    a, b = np.asarray(data[0]), np.asarray(data[1])
    if a.ndim == 1 or (a.ndim == 2 and a.shape[1] <= 2 and b.shape[1] > a.shape[1]):
        return CandidateSet(obj=a, var=b)
    return CandidateSet(obj=b, var=a)


def write_populations(populations: list) -> None:
    """Write per-context GA populations (list of PopulationResult) as pop{EXE}.npz."""
    path = _pop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, pop in enumerate(populations):
        arrays[f"phen_{i}"] = np.asarray(pop.Phen)
        arrays[f"objv_{i}"] = np.asarray(pop.ObjV)
    arrays["n_pops"] = np.array(len(populations))
    np.savez_compressed(path, **arrays)


def read_populations() -> list:
    """Read per-context GA populations from pop{EXE}.npz back into PopulationResult."""
    from icsearcher.search.searcher import PopulationResult
    path = _pop_path()
    if path.suffix != ".npz" or not path.exists():
        # Legacy .pkl fallback
        pkl = path.with_suffix(".pkl")
        with open(pkl, "rb") as f:
            return pickle.load(f)
    data = np.load(path)
    n = int(data["n_pops"])
    pops = []
    for i in range(n):
        pops.append(PopulationResult(
            Phen=data[f"phen_{i}"],
            ObjV=data[f"objv_{i}"],
        ))
    return pops
