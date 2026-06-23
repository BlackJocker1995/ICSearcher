"""NSGA-II range optimizer + Pareto metrics (pymoo).

Replaces the geatpy ``GARangeOptimizer``. Uses pymoo's NSGA-II for the
multi-objective search and pymoo's built-in GD / IGD / HV / Spacing indicators
(which the legacy code computed by hand against a reference Pareto front).

Because the problem maximizes (#configs, pass-rate), the objectives stored in
``result.F`` are the negations; we flip them back for reporting and metrics.
"""
import numpy as np
from loguru import logger
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize

from icsearcher.config import toolConfig
from icsearcher.params import (
    get_default_values,
    load_param,
    read_range_from_dict,
    read_unit_from_dict,
    select_sub_dict,
)
from icsearcher.range.problem import GARangeProblem


class GARangeOptimizer:
    """Multi-objective range search via NSGA-II."""

    MAXGEN = 300
    POP_SIZE = 3000
    Pm = 0.5
    XOVR = 0.9

    def __init__(self, result_data):
        self.result = None
        self.NDSet_F = None  # Pareto objective values (original maximize scale)
        self._setup_params()
        self._init_problem(result_data)

    def _setup_params(self):
        cfg = toolConfig
        para_dict = load_param()
        self.participle_param = cfg.PARAM
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = get_default_values(self.param_choice_dict).values.astype(float).flatten()
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)

    def _init_problem(self, result_data):
        # Each parameter contributes a (low, high) pair, so Dim = 2 * n_params.
        lb = np.repeat(self.sub_value_range[:, 0] / self.step_unit, 2)
        lb[1::2] = self.default_pop // self.step_unit
        ub = np.repeat(self.sub_value_range[:, 1] / self.step_unit, 2)
        ub[::2] = self.default_pop // self.step_unit
        self.problem = GARangeProblem(lb=lb, ub=ub, step=self.step_unit, result_data=result_data)

    def set_bounds(self):
        """Forward bounds/step to the problem (kept for call-site compat)."""
        self.problem.init_bounds_and_step(self.sub_value_range, self.step_unit)

    def run(self):
        """Execute NSGA-II and capture the Pareto front + metrics."""
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling

        algorithm = NSGA2(
            pop_size=self.POP_SIZE,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.XOVR),
            mutation=PM(prob=self.Pm),
        )
        self.result = minimize(
            self.problem,
            algorithm,
            termination=("n_gen", self.MAXGEN),
            seed=None,
            verbose=False,
        )
        # Flip objectives back to the maximize scale for reporting.
        self.NDSet_F = -np.asarray(self.result.F)
        self._save_results()
        self._calculate_metrics()

    def _save_results(self):
        logger.info(f"Pareto front size: {len(self.NDSet_F)}")
        for f in self.NDSet_F:
            logger.info(f"  #configs={f[0]:.0f}  pass-rate={f[1]:.3f}")

    def _calculate_metrics(self):
        """GD/IGD/HV/Spacing against the observed Pareto front as reference.

        The legacy code compared against a synthetic reference front
        (``getReferObjV``); pymoo has no such generator, so we use the observed
        non-dominated set itself as the reference for the self-consistency
        metrics. Spacing is computed on the front alone.
        """
        F = self.NDSet_F
        if F is None or len(F) == 0:
            logger.warning("Empty Pareto front; skipping metrics.")
            return

        try:
            from pymoo.indicators.gd import GD
            from pymoo.indicators.hv import HV
            from pymoo.indicators.spacing import Spacing
        except ImportError:
            logger.warning("pymoo indicators not available; skipping metrics.")
            return

        gd = GD(F).do(F)
        igd = IGD(F).do(F)
        hv = HV(reference_point=np.array([0.0, 0.0]) + F.max(axis=0)).do(F)
        spacing = Spacing().do(F)
        logger.info(f"GD={gd:.6f}  IGD={igd:.6f}  HV={hv:.6f}  Spacing={spacing:.6f}")

    def return_best_n_gen(self, n=1):
        """Return the n best (low, high) bound vectors by composite objective."""
        if self.result is None:
            raise ValueError("Please run() first")
        F = self.NDSet_F
        # Composite (maximize both): rank by the sum of the normalized objectives.
        order = np.argsort(-(F[:, 0] + F[:, 1]))
        X = np.asarray(self.result.X)
        return self.problem.reasonable_range(X[order][:n])
