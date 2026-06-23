"""Tests for the pymoo-based GA search (problem + selectors).

Gated on pymoo being importable (the full Poetry environment). In a minimal
environment the whole module is skipped, so the test suite stays green.
"""
import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")

import icsearcher.config as cfg  # noqa: E402
from icsearcher.params import load_param, select_sub_dict, read_range_from_dict, read_unit_from_dict  # noqa: E402
from icsearcher.search.problem import ProblemGA, reasonable_range_static  # noqa: E402


class _FakePredictor:
    """Stand-in surrogate exposing the methods ProblemGA._evaluate calls."""

    def data_split_3d(self, merge_data):
        # merge_data shape: (n_params, n_windows, DATA_LEN)
        # produce a 3D input/output pair of matching window count.
        n_windows = merge_data.shape[1]
        feature_x = merge_data[:, :n_windows, :]
        feature_y = merge_data[:, :n_windows, :cfg.toolConfig.STATUS_LEN]
        return feature_x, feature_y

    def predict_feature(self, feature_x):
        # Return predictions of the same trailing dim as STATUS_LEN so reshape works.
        return np.zeros((feature_x.shape[0], feature_x.shape[1], cfg.toolConfig.STATUS_LEN))

    def cal_patch_deviation(self, predicted, actual):
        # Mean deviation per candidate; shape (n_candidates,).
        return np.linalg.norm(predicted - actual, axis=(1, 2))


def _make_problem():
    para = select_sub_dict(load_param(), cfg.toolConfig.PARAM_PART)
    bounds = read_range_from_dict(para)
    step = read_unit_from_dict(para)
    lb = (bounds[:, 0] // step).astype(float)
    ub = (bounds[:, 1] // step).astype(float)
    return ProblemGA(lb=lb, ub=ub, step=step, predictor=_FakePredictor())


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_problem_bounds_shape(monkeypatch, mode_name):
    """The pymoo problem declares n_var == len(PARAM_PART) and valid bounds."""
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    prob = _make_problem()
    assert prob.n_var == len(fresh.PARAM_PART)
    assert np.all(np.asarray(prob.xl) <= np.asarray(prob.xu))


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_problem_evaluate_finite(monkeypatch, mode_name):
    """_evaluate returns finite objective values for a small batch."""
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    prob = _make_problem()
    # Build a status context of SEGMENT_LEN+1 rows, INPUT_LEN windows supervised.
    seg = fresh.SEGMENT_LEN + 1
    status = np.random.default_rng(0).uniform(-1, 1, size=(seg, fresh.INPUT_LEN, fresh.DATA_LEN))
    prob.init_status(status)
    X = np.random.default_rng(1).uniform(np.asarray(prob.xl), np.asarray(prob.xu),
                                         size=(6, prob.n_var))
    out = {}
    prob._evaluate(X, out)
    F = out["F"]
    assert F.shape == (6, 1)
    assert np.all(np.isfinite(F))


def test_reasonable_range_static_restores_units():
    """reasonable_range_static multiplies by the step units."""
    x = np.ones((3, len(cfg.toolConfig.PARAM_PART)))
    restored = reasonable_range_static(x)
    assert restored.shape == (3, len(cfg.toolConfig.PARAM_PART))
    assert np.all(np.isfinite(restored))
