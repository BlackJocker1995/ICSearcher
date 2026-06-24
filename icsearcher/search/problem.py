# -*- coding: utf-8 -*-
"""Surrogate-guided GA fuzzing problem (pymoo).

The decision variables are the fuzzed parameter subset (``PARAM_PART``),
encoded as integer multiples of each parameter's step unit. The objective is the
**predicted flight-status deviation** produced by the LSTM surrogate: the
fuzzing wants to *maximize* that deviation (configs that destabilize the drone),
so we expose it to pymoo as a minimization of ``-deviation``.

This replaces the geatpy ``ProblemGA`` from stage 1/2. The numerical pipeline
(scale params, tile them across the status context, run the predictor, compute
patch deviation) is preserved verbatim so the search stays logically equivalent.
"""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from icsearcher import config as _config_mod
from icsearcher.params import (
    load_param,
    min_max_scaler_param,
    pad_configuration_default_value,
    read_unit_from_dict,
    select_sub_dict,
)


def _config():
    """Lazy config access so per-mode monkeypatching of the singleton works.

    Importing ``toolConfig`` at module top-level binds the singleton captured at
    first import, which defeats per-mode monkeypatching (the same reason
    ``icsearcher.params`` reads it through a function).
    """
    return _config_mod.toolConfig


def reasonable_range_static(param):
    """Restore integer-encoded params to their original (step-scaled) units.

    Mirrors the legacy ``ProblemGA.reasonable_range_static`` so the candidate
    selectors in fuzzer.py keep working unchanged.
    """
    cfg = _config()
    para_dict = load_param()
    param_choice_dict = select_sub_dict(para_dict, cfg.PARAM_PART)
    step_unit = read_unit_from_dict(param_choice_dict)
    return param * step_unit


class ProblemGA(Problem):
    """Single-objective pymoo problem: minimize negative predicted deviation.

    Args:
        lb / ub: integer lower/upper bounds (already divided by step unit).
        step: per-parameter step unit (original units), used to restore scale.
        status_data: the supervised-learning context for the current flight
            segment, set per-context via :meth:`init_status`.
        predictor: the loaded LSTM surrogate.
    """

    def __init__(self, lb, ub, step, predictor=None):
        n_var = len(lb)
        # Variables are real-coded (pymoo's DE mutates in float space); they are
        # rounded to integer step-multiples inside _evaluate before the scale is
        # restored. This matches the legacy geatpy discrete encoding.
        super().__init__(n_var=n_var, n_obj=1, n_constr=0,
                         xl=np.asarray(lb, dtype=float),
                         xu=np.asarray(ub, dtype=float))
        self.step = np.asarray(step)
        self.predictor = predictor
        self.status_data = None

    # -- per-context state ------------------------------------------------
    def init_status(self, status_data):
        """Bind the flight-context segment this problem instance searches over."""
        self.status_data = status_data

    def set_predictor(self, predictor):
        self.predictor = predictor

    # -- pymoo entry point ------------------------------------------------
    def _evaluate(self, X, out, *args, **kwargs):
        x = self._preprocess(X)
        predicted_feature, feature_y = self._get_predictions(x)
        # Maximize deviation -> minimize its negation.
        out["F"] = -self._calculate_loss(predicted_feature, feature_y).reshape((-1, 1))

    # -- numerical pipeline (unchanged from geatpy version) ---------------
    def _preprocess(self, X):
        # Round the real-coded search variables to integer step-multiples, then
        # restore the original (step-scaled) units.
        x = self.reasonable_range(np.round(X)).to_numpy()
        if x.shape[1] != load_param().shape[1]:
            x = pad_configuration_default_value(x)
        return x

    def _get_predictions(self, x):
        cfg = _config()
        param = min_max_scaler_param(x)
        merge_data = self._prepare_merge_data(param)
        feature_x, feature_y = self.predictor.data_split_3d(merge_data)
        predicted_feature = self.predictor.predict_feature(feature_x)
        dims = (x.shape[0], -1, predicted_feature.shape[-1])
        return predicted_feature.reshape(dims), feature_y.reshape(dims)

    def _prepare_merge_data(self, param):
        cfg = _config()
        status = self.status_data.reshape((1, self.status_data.shape[0], -1, cfg.DATA_LEN))
        status = status[:, :, :, :cfg.STATUS_LEN]
        param = param.reshape((param.shape[0], 1, 1, -1))
        repeat_status = np.repeat(status, param.shape[0], axis=0)
        repeat_param = np.repeat(param, repeat_status.shape[2], axis=2)
        repeat_param = np.repeat(repeat_param, repeat_status.shape[1], axis=1)
        merge_data = np.c_[repeat_status, repeat_param]
        return merge_data.reshape((merge_data.shape[0], merge_data.shape[1], -1)).astype(float)

    def _calculate_loss(self, predicted, actual):
        return self.predictor.cal_patch_deviation(predicted, actual)

    def reasonable_range(self, param):
        """Restore integer-encoded params to step-scaled original units."""
        np_config = param * self.step
        return pd.DataFrame(np_config, columns=_config().PARAM_PART)
