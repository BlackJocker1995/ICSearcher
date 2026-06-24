"""Typed read/write for the inter-stage artifacts.

Stage 3 (fuzz) writes per-context GA populations to ``pop{EXE}.npz``.
Stage 4 (pre-validate) rewrites that file as a flat candidate set.

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


def _pop_path() -> Path:
    """Return the path to the population/candidate file.

    Prefers ``.npz`` (new format). Falls back to ``.pkl`` (legacy) if the
    ``.npz`` does not exist but the ``.pkl`` does.
    """
    npz = Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.npz")
    pkl = Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl")
    if npz.exists():
        return npz
    if pkl.exists():
        return pkl
    return npz  # default for writing


def write_candidates(obj: np.ndarray, var: np.ndarray) -> None:
    """Persist a CandidateSet as pop{EXE}.npz (compressed numpy format)."""
    path = _pop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obj=np.asarray(obj), var=np.asarray(var))


def read_candidates() -> CandidateSet:
    """Load pop{EXE}.npz (or legacy .pkl).

    ``.npz`` format reads ``obj`` and ``var`` arrays directly.
    Legacy ``.pkl`` accepted: both CandidateSet containers and ``[obj, var]``
    lists produced by older pipeline runs.
    """
    path = _pop_path()
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
    """Write per-context GA populations (list of PopulationResult) as .npz."""
    path = Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, pop in enumerate(populations):
        arrays[f"phen_{i}"] = np.asarray(pop.Phen)
        arrays[f"objv_{i}"] = np.asarray(pop.ObjV)
    arrays["n_pops"] = np.array(len(populations))
    np.savez_compressed(path, **arrays)


def read_populations() -> list:
    """Read per-context GA populations from .npz back into PopulationResult."""
    from icsearcher.search.searcher import PopulationResult
    path = Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.npz")
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
