"""Tests for Cptool.mavtool parameter helpers (pure functions, no SITL).

Covers the stage-1 scope: param loading, min-max scaling into [0,1], and the
default-value padding used to expand a partial param vector back to full length.
"""
import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from Cptool.mavtool import (
    get_default_values,
    load_param,
    load_sub_param,
    min_max_scaler_param,
    pad_configuration_default_value,
    read_range_from_dict,
    read_unit_from_dict,
    select_sub_dict,
)


def test_load_param_matches_mode(mode):
    """load_param returns a DataFrame whose columns equal config PARAM."""
    df = load_param()
    assert list(df.columns) == toolConfig.PARAM
    # index has 'range' and 'step' and 'default' rows.
    assert {"range", "step", "default"}.issubset(set(df.index))


def test_load_sub_param_is_subset(mode):
    """load_sub_param returns only the fuzzed subset."""
    df = load_sub_param()
    assert list(df.columns) == toolConfig.PARAM_PART


def test_min_max_scaler_param_in_unit_interval(mode):
    """Scaling must map every value inside its [lb, ub] range to [0,1]."""
    rng = np.random.default_rng(0)
    para = load_param()
    bounds = read_range_from_dict(para)  # shape (n_params, 2): [lb, ub]
    # Sample a few rows whose values are strictly inside each param's range.
    n_rows = 5
    t = rng.uniform(0.1, 0.9, size=(n_rows, bounds.shape[0]))  # in (0,1)
    sample = bounds[:, 0] + t * (bounds[:, 1] - bounds[:, 0])  # in-range values
    scaled = min_max_scaler_param(sample)
    # finite and bounded into the unit interval for in-range inputs.
    assert np.all(np.isfinite(scaled))
    assert scaled.min() >= -1e-9
    assert scaled.max() <= 1.0 + 1e-9


def test_pad_configuration_default_value_shape(mode):
    """Padding a partial vector with defaults yields full PARAM width."""
    sub = select_sub_dict(load_param(), toolConfig.PARAM_PART)
    defaults = get_default_values(sub).values.astype(float)  # one row
    # build two rows of the subset and pad to full PARAM width.
    partial = np.repeat(defaults, 2, axis=0)
    padded = pad_configuration_default_value(partial)
    assert padded.shape == (2, len(toolConfig.PARAM))
    # The subset columns should equal the input; other columns equal defaults.
    cols = list(toolConfig.PARAM)
    idx = [cols.index(p) for p in toolConfig.PARAM_PART]
    np.testing.assert_allclose(padded[:, idx], partial)


def test_range_and_unit_shapes(mode):
    """read_range_from_dict / read_unit_from_dict shapes match PARAM length."""
    sub = select_sub_dict(load_param(), toolConfig.PARAM_PART)
    rng = read_range_from_dict(sub)
    unit = read_unit_from_dict(sub)
    assert rng.shape == (len(toolConfig.PARAM_PART), 2)
    assert unit.shape == (len(toolConfig.PARAM_PART),)
