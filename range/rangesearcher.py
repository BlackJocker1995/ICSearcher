import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, get_default_values, read_range_from_dict
from range.rangeproblem import GARangeProblem


class BaseRangeOptimizer(ABC):
    def __init__(self):
        self.problem = None
        self.start_value = None
        self._setup_params()
        
    def _setup_params(self):
        """Initialize parameter settings"""
        self.participle_param = toolConfig.PARAM
        para_dict = load_param()
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = get_default_values(self.param_choice_dict)
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)

    def set_bounds(self):
        """Set problem bounds and step size"""
        self.problem.init_bounds_and_step(self.sub_value_range, self.step_unit)

class GARangeOptimizer(BaseRangeOptimizer):
    def __init__(self, result_data):
        super().__init__()
        self._init_ga_problem(result_data)
        self.NDSet = None
        self.population = None
        self.algorithm = None

    def _init_ga_problem(self, result_data):
        """Initialize GA problem parameters"""
        Dim = self.sub_value_range.shape[0] * 2
        varTypes = [1] * Dim  # 0: continuous, 1: discrete
        
        # Initialize bounds
        lb = np.repeat(self.sub_value_range[:, 0] / self.step_unit, 2)
        lb[1::2] = self.default_pop // self.step_unit
        ub = np.repeat(self.sub_value_range[:, 1] / self.step_unit, 2)
        ub[::2] = self.default_pop // self.step_unit
        
        # Boundary inclusion flags
        lbin = [1] * Dim  # Include lower bounds
        ubin = [1] * Dim  # Include upper bounds

        self.problem = GARangeProblem(
            name='ANAProblem',
            M=2,  # Number of objectives
            maxormins=[-1, -1],  # Maximize both objectives
            Dim=self.sub_value_range.shape[0],
            varTypes=varTypes,
            lb=lb, ub=ub,
            lbin=lbin, ubin=ubin,
            result_data=result_data
        )

    def run(self):
        """Execute optimization process"""
        population = self._init_population()
        self.algorithm = self._setup_algorithm(population)
        self.NDSet, self.population = self.algorithm.run()
        self._save_results()
        self._calculate_metrics()

    def _init_population(self, size=3000):
        """Initialize population"""
        Field = ea.crtfld('RI', self.problem.varTypes, 
                         self.problem.ranges,
                         self.problem.borders)
        return ea.Population('RI', Field, size)

    def _setup_algorithm(self, population):
        """Configure NSGA-II algorithm parameters"""
        algorithm = ea.moea_NSGA2_templet(self.problem, population)
        algorithm.MAXGEN = 300
        algorithm.mutOper.Pm = 0.5
        algorithm.recOper.XOVR = 0.9
        algorithm.maxTrappedCount = 10
        algorithm.drawing = 1
        return algorithm

    def _save_results(self):
        """Save optimization results"""
        with open('NDSetnew.pkl', 'wb') as f:
            pickle.dump(self.NDSet, f)
        self.NDSet.save()  # Save non-dominated set to file
        ea.moeaplot(self.NDSet.ObjV, xyzLabel=['No. of Validated Configuration', 'Incorrect/Validated Ratio'])
        print('用时：%s 秒' % (self.algorithm.passTime))
        print('非支配个体数：%s 个' % (self.NDSet.sizes))
        print('单位时间找到帕累托前沿点个数：%s 个' % (int(self.NDSet.sizes // self.algorithm.passTime)))

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        PF = self.problem.getReferObjV()  # Get true Pareto front
        if PF is not None and self.NDSet.sizes != 0:
            GD = ea.indicator.GD(self.NDSet.ObjV, PF)  # Calculate GD metric
            IGD = ea.indicator.IGD(self.NDSet.ObjV, PF)  # Calculate IGD metric
            HV = ea.indicator.HV(self.NDSet.ObjV, PF)  # Calculate HV metric
            Spacing = ea.indicator.Spacing(self.NDSet.ObjV)  # Calculate Spacing metric
            print('GD', GD)
            print('IGD', IGD)
            print('HV', HV)
            print('Spacing', Spacing)
        if PF is not None:
            metricName = [['IGD'], ['HV']]
            [NDSet_trace, Metrics] = ea.indicator.moea_tracking(self.algorithm.pop_trace, PF, metricName,
                                                                self.problem.maxormins)
            ea.trcplot(Metrics, labels=metricName, titles=metricName)

    def return_best_n_gen(self, n=1):
        if (self.algorithm is None) or (self.obj_trace is None) or (self.var_trace is None):
            raise ValueError('Please run() at first')

        candidate = self.problem.maxormins * self.obj_trace[:, 1]

        if n == 0:
            top_gen = np.zeros(len(candidate), dtype=int)
            for i in range(len(candidate)):
                min_index = np.argmin(candidate)
                top_gen[i] = min_index  # 记录最优种群个体是在哪一代
                candidate[min_index] = 0
        else:
            top_gen = np.zeros(n, dtype=int)
            for i in range(n):
                min_index = np.argmin(candidate)
                top_gen[i] = min_index  # 记录最优种群个体是在哪一代
                candidate[min_index] = 0
        params = self.var_trace[top_gen, :]
        return self.problem.reasonable_range(params)


