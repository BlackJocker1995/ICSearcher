import geatpy as ea
import numpy as np
from abc import ABC, abstractmethod

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.mavtool import load_param, select_sub_dict, get_default_values, read_unit_from_dict, read_range_from_dict
from ModelFit.approximate import CyLSTM
from uavga.problem import ProblemGA


class BaseSearchOptimizer(ABC):
    def __init__(self):
        self.predictor = None
        self.problem = None
        self.param_bounds = None
        self.step_unit = None
        self._setup_params()
        
    def _setup_params(self):
        """Initialize parameter settings"""
        self.participle_param = toolConfig.PARAM_PART
        para_dict = load_param()
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = get_default_values(self.param_choice_dict)
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)

    @abstractmethod 
    def start_optimize(self):
        pass

    @abstractmethod
    def return_best_n_gen(self, n=1):
        pass

class GAOptimizer(BaseSearchOptimizer):
    def __init__(self):
        super().__init__()
        self._init_ga_problem()
        self.NDSet = None
        self.population = None 
        self.algorithm = None

    def _init_ga_problem(self):
        """Initialize GA problem parameters"""
        Dim = self.sub_value_range.shape[0]
        varTypes = [1] * Dim  # 0: continuous variable; 1: discrete variable
        lb = self.sub_value_range[:, 0] // self.step_unit  # Lower bound
        ub = self.sub_value_range[:, 1] // self.step_unit  # Upper bound
        lbin = [0] * Dim  # Whether to include lower bound (0: no, 1: yes)
        ubin = [1] * Dim  # Whether to include upper bound (0: no, 1: yes)

        # Initialize problem instance
        self.problem = ProblemGA(
            name='UAVProblem',
            M=1,  # Number of objectives
            maxormins=[-1],  # -1: maximize, 1: minimize 
            Dim=Dim,
            varTypes=varTypes,
            lb=lb, ub=ub,
            lbin=lbin, ubin=ubin
        )

    def start_optimize(self):
        """Execute optimization process"""
        population = self._init_population()
        self.algorithm = self._setup_algorithm(population)
        prophet_pop = self._create_prophet_population()
        self.algorithm.call_aimFunc(prophet_pop)
        self.NDSet, self.population = self.algorithm.run(prophet_pop)

    def _init_population(self, size=500):
        """Initialize population"""
        Field = ea.crtfld('RI', self.problem.varTypes, 
                         self.problem.ranges,
                         self.problem.borders)
        return ea.Population('RI', Field, size)

    def _setup_algorithm(self, population):
        """Configure algorithm parameters"""
        algorithm = ea.soea_DE_currentToBest_1_bin_templet(self.problem, population)
        algorithm.MAXGEN = 50
        algorithm.mutOper.F = 0.7
        algorithm.recOper.XOVR = 0.7 
        algorithm.trappedValue = 0.1
        algorithm.maxTrappedCount = 10
        algorithm.drawing = 0
        return algorithm

    def _create_prophet_population(self):
        """Create prophet population based on default values"""
        prophet_chrom = np.array(self.default_pop / self.step_unit, dtype=int)
        Field = ea.crtfld('RI', self.problem.varTypes,
                         self.problem.ranges,
                         self.problem.borders)
        return ea.Population('RI', Field, 1, prophet_chrom)

    def return_best_n_gen(self, n=1):
        """Return best n generations"""
        if not hasattr(self, 'BestIndi') or self.population is None:
            raise ValueError('Please run start_optimize() first')
            
        obj_trace = self.population.Phen 
        var_trace = self.population.ObjV

        # Get unique candidates
        unique_indices = np.unique(var_trace, axis=0, return_index=True)[1]
        candidate_var = var_trace[unique_indices].reshape(-1)
        candidate_obj = obj_trace[unique_indices]

        # Sort candidates
        sort_indices = np.argsort(self.problem.maxormins * candidate_var)
        candidate_obj = candidate_obj[sort_indices]

        return self.problem.reasonable_range(candidate_obj)[:n]
