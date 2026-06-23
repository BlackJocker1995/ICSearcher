# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
import geatpy as ea
import pandas as pd

from icsearcher.config import toolConfig
from icsearcher.params import load_param, read_unit_from_dict


class BaseRangeProblem(ABC):
    def __init__(self):
        self.param_bounds = None
        self.step = None

    @abstractmethod
    def evaluate(self, population):
        pass

    def init_bounds_and_step(self, param_bounds, step):
        """Initialize parameter bounds and step size"""
        self.param_bounds = param_bounds
        self.step = step

    def param_value2step(self, configuration):
        """Convert parameter values to step-aligned values"""
        np_config = np.ceil(configuration / self.step) * self.step
        return pd.DataFrame([np_config.tolist()], 
                          columns=toolConfig.PARAM).iloc[0].to_dict()


class GARangeProblem(BaseRangeProblem, ea.Problem):
    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin, result_data):
        BaseRangeProblem.__init__(self)
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                          varTypes, lb, ub, lbin, ubin)
        self.data = result_data

    def aimFunc(self, pop):
        """Calculate objective functions for the population"""
        x = self._preprocess_population(pop)
        bottom, top = self._split_bounds(x)
        score_rate, score_len = self._calculate_scores(top, bottom)
        pop.ObjV = self._combine_objectives(score_rate, score_len)

    def _preprocess_population(self, pop):
        """Preprocess population data"""
        return self.reasonable_range(pop.Phen)

    def _split_bounds(self, x):
        """Split chromosome into bottom and top bounds"""
        return x[:, ::2], x[:, 1::2]

    def _calculate_scores(self, top, bottom):
        """Calculate satisfaction scores for the bounds"""
        score_rate = np.zeros(top.shape[0])
        score_len = np.zeros(top.shape[0])
        
        for i, (t, b) in enumerate(zip(top, bottom)):
            rate, length = self.satisfy_range(t, b)
            score_rate[i] = rate
            score_len[i] = length
            
        return score_rate, score_len

    def _combine_objectives(self, score_rate, score_len):
        """Combine objectives into final fitness values"""
        return np.hstack([
            score_len.reshape((-1, 1)),
            score_rate.reshape((-1, 1))
        ])

    def reasonable_range(self, param):
        return param * np.repeat(self.step, 2)

    def satisfy_range(self, top, button):
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
        para_dict = load_param()
        step_unit = read_unit_from_dict(para_dict)
        return param * np.repeat(step_unit, 2)