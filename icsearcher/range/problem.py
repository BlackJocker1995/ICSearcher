# -*- coding: utf-8 -*-
"""Safe-range derivation problem (pymoo, multi-objective).

Each parameter contributes a ``(low, high)`` pair to the decision vector; the
two objectives are (a) the number of validated configurations that fall inside
the proposed ranges and (b) the pass-rate among them. Both are maximized, which
we fold into pymoo's minimization by negating them.

Replaces the geatpy ``GARangeProblem``. The ``satisfy_range`` counting logic is
preserved verbatim.
"""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from icsearcher.config import toolConfig
from icsearcher.params import read_unit_from_dict


class GARangeProblem(Problem):
    """2-objective problem: maximize (#configs-in-range, pass-rate)."""

    def __init__(self, lb, ub, step, result_data):
        # Real-coded (NSGA-II mutates in float space); rounded to integer
        # step-multiples in _evaluate before scale is restored.
        super().__init__(
            n_var=len(lb),
            n_obj=2,
            n_constr=0,
            xl=np.asarray(lb, dtype=float),
            xu=np.asarray(ub, dtype=float),
        )
        self.step = np.asarray(step)
        self.data = result_data

    def init_bounds_and_step(self, param_bounds, step):
        """Kept for call-site compatibility (set_bounds() in the optimizer)."""
        self.step = np.asarray(step)
        # Bounds were already provided at construction; nothing else to do.

    def _evaluate(self, X, out, *args, **kwargs):
        x = self.reasonable_range(np.round(X))
        bottom, top = x[:, ::2], x[:, 1::2]
        score_rate, score_len = self._calculate_scores(top, bottom)
        # Maximize both -> minimize the negations.
        out["F"] = np.column_stack([-score_len, -score_rate])

    def _calculate_scores(self, top, bottom):
        score_rate = np.zeros(top.shape[0])
        score_len = np.zeros(top.shape[0])
        for i, (t, b) in enumerate(zip(top, bottom)):
            rate, length = self.satisfy_range(t, b)
            score_rate[i] = rate
            score_len[i] = length
        return score_rate, score_len

    def reasonable_range(self, param):
        """Restore integer-encoded (low, high) pairs to step-scaled units."""
        return param * np.repeat(self.step, 2)

    def satisfy_range(self, top, button):
        """Count validated configs inside [button, top] and their pass-rate."""
        values = self.data.values[:, :-1]
        to_top = (top - values).min(axis=1)
        to_button = (values - button).min(axis=1)
        index = np.where((to_top >= 0) & (to_button >= 0))[0]
        if len(index) == 0:
            return 0, len(index)
        satisfy_value = self.data.iloc[index]
        pass_index = satisfy_value.result == 'pass'
        pass_rate = pass_index.values.sum() / satisfy_value.shape[0]
        print(f'include num: {len(index)}   rate: {pass_rate}')
        return pass_rate, len(index)

    @staticmethod
    def reasonable_range_static(param):
        from icsearcher.params import load_param
        para_dict = load_param()
        step_unit = read_unit_from_dict(para_dict)
        return param * np.repeat(step_unit, 2)
