"""Typed read/write for the inter-stage pickle artifacts.

Stage 3 (fuzz) writes per-context GA populations to ``pop{EXE}.pkl``.
Stage 4 (pre-validate) rewrites that file as a flat candidate set. Historically
these two writers used *different* list orders (``[obj, var]`` vs
``[candidate_obj, candidate_var]``), so every reader had to know which writer
ran last. This module pins a single, documented container and centralizes the
serialization so the order can never drift again.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    return Path(f"result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl")


def write_candidates(obj: np.ndarray, var: np.ndarray) -> None:
    """Persist a CandidateSet as pop{EXE}.pkl."""
    path = _pop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(CandidateSet(obj=np.asarray(obj), var=np.asarray(var)), f)


def read_candidates() -> CandidateSet:
    """Load pop{EXE}.pkl.

    Accepts both the new CandidateSet container and the legacy two-element list
    form (``[obj, var]``) produced by older pipeline runs, so existing result
    files keep loading after the refactor.
    """
    with open(_pop_path(), "rb") as f:
        data = pickle.load(f)
    if isinstance(data, CandidateSet):
        return data
    # Legacy [obj, var] list. Be lenient about which side is which by length:
    # var is the wider array (n_params columns); obj is the scalar score.
    a, b = data
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1 or (a.ndim == 2 and a.shape[1] <= 2 and b.shape[1] > a.shape[1]):
        return CandidateSet(obj=a, var=b)
    return CandidateSet(obj=b, var=a)
