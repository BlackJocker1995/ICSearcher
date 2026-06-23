"""Tests for icsearcher.params parameter helpers (pure functions, no SITL).

Covers param loading, min-max scaling into [0,1], and the default-value padding
used to expand a partial param vector back to full length. Run for both modes
via the `mode` fixture in conftest.py.
"""
import numpy as np
import pytest

import icsearcher.config as cfg
from icsearcher.params import (
    get_default_values,
    load_param,
    load_sub_param,
    min_max_scaler_param,
    pad_configuration_default_value,
    read_range_from_dict,
    read_unit_from_dict,
    select_sub_dict,
)


@pytest.fixture(autouse=True)
def _bind_mode(mode, monkeypatch):
    """The mode fixture already repoints cfg.toolConfig; nothing else needed."""
    yield


def test_load_param_matches_mode(mode):
    """load_param returns a DataFrame whose columns equal config PARAM."""
    df = load_param()
    assert list(df.columns) == cfg.toolConfig.PARAM
    assert {"range", "step", "default"}.issubset(set(df.index))


def test_load_sub_param_is_subset(mode):
    """load_sub_param returns only the fuzzed subset."""
    df = load_sub_param()
    assert list(df.columns) == cfg.toolConfig.PARAM_PART


def test_min_max_scaler_param_in_unit_interval(mode):
    """Scaling must map every in-range value into [0,1]."""
    rng = np.random.default_rng(0)
    para = load_param()
    bounds = read_range_from_dict(para)  # (n_params, 2): [lb, ub]
    t = rng.uniform(0.1, 0.9, size=(5, bounds.shape[0]))
    sample = bounds[:, 0] + t * (bounds[:, 1] - bounds[:, 0])  # strictly in-range
    scaled = min_max_scaler_param(sample)
    assert np.all(np.isfinite(scaled))
    assert scaled.min() >= -1e-9
    assert scaled.max() <= 1.0 + 1e-9


def test_pad_configuration_default_value_shape(mode):
    """Padding a partial vector with defaults yields full PARAM width."""
    sub = select_sub_dict(load_param(), cfg.toolConfig.PARAM_PART)
    defaults = get_default_values(sub).values.astype(float)
    partial = np.repeat(defaults, 2, axis=0)
    padded = pad_configuration_default_value(partial)
    assert padded.shape == (2, len(cfg.toolConfig.PARAM))
    cols = list(cfg.toolConfig.PARAM)
    idx = [cols.index(p) for p in cfg.toolConfig.PARAM_PART]
    np.testing.assert_allclose(padded[:, idx], partial)


def test_range_and_unit_shapes(mode):
    """read_range_from_dict / read_unit_from_dict shapes match PARAM_PART."""
    sub = select_sub_dict(load_param(), cfg.toolConfig.PARAM_PART)
    assert read_range_from_dict(sub).shape == (len(cfg.toolConfig.PARAM_PART), 2)
    assert read_unit_from_dict(sub).shape == (len(cfg.toolConfig.PARAM_PART),)
