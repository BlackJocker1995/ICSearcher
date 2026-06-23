"""DE-based optimizer for the surrogate fuzzing problem (pymoo).

Replaces the geatpy ``GAOptimizer``. The decision variables are integer
multiples of each parameter's step unit; the objective (maximize predicted
deviation) is folded into the problem as ``minimize(-deviation)``. A "prophet"
individual built from the firmware-default parameter values seeds the initial
population, mirroring the legacy behavior.

The pymoo result is exposed through :attr:`population` as a small container with
``X`` (decision variables) and ``F`` (objective) so the candidate selectors in
fuzzer.py can read it without depending on pymoo's Result/Population types.
"""
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.operators.sampling.base import Sampling

from icsearcher.config import toolConfig
from icsearcher.params import (
    get_default_values,
    read_range_from_dict,
    read_unit_from_dict,
    select_sub_dict,
    load_param,
)
from icsearcher.search.problem import ProblemGA


@dataclass
class PopulationResult:
    """Lightweight stand-in for a geatpy Population, as read by fuzzer.py.

    Attributes:
        Phen: decision variables (integer-encoded params), shape (n, n_params).
        ObjV: objective values, shape (n, 1) — note: this is the *negated*
            deviation as returned by pymoo (minimize). Selectors that compare
            ``ObjV`` keep the same ordering because negation is monotonic.
    """
    Phen: np.ndarray
    ObjV: np.ndarray


class _ProphetSampling(Sampling):
    """Seed the DE population with the firmware-default parameter vector.

    The first individual is the default config (a known-good starting point);
    the rest are drawn uniformly at random within bounds. Variables are
    real-coded (the problem rounds them to integer step-multiples in _evaluate).
    """

    def __init__(self, default_encoded, size):
        super().__init__()
        self.default_encoded = np.asarray(default_encoded, dtype=float)
        self.size = size

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng()
        lo = np.asarray(problem.xl, dtype=float)
        hi = np.asarray(problem.xu, dtype=float)
        pop = rng.uniform(low=lo, high=hi, size=(n_samples, problem.n_var))
        pop[0] = self.default_encoded
        return pop


class GAOptimizer:
    """Drives the DE search over a single flight context."""

    # Default DE hyperparameters. pymoo's DE requires F in [0.4, 0.7] and
    # CR in [0.2, 0.8]; we pick interior values to stay within those bounds.
    MAXGEN = 50
    POP_SIZE = 500
    F = 0.5       # differential weight
    CR = 0.7      # crossover rate

    def __init__(self):
        self.predictor = None
        self.problem = None
        self.population = None  # PopulationResult after start_optimize
        self.result = None      # raw pymoo Result
        self._setup_params()
        self._init_problem()

    def _setup_params(self):
        cfg = toolConfig
        para_dict = load_param()
        self.participle_param = cfg.PARAM_PART
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = get_default_values(self.param_choice_dict).values.astype(float).flatten()
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)

    def _init_problem(self):
        lb = (self.sub_value_range[:, 0] // self.step_unit).astype(float)
        ub = (self.sub_value_range[:, 1] // self.step_unit).astype(float)
        self.problem = ProblemGA(lb=lb, ub=ub, step=self.step_unit, predictor=self.predictor)

    # -- API used by run_fuzzing -----------------------------------------
    def set_bounds(self):
        """No-op: bounds are set in __init__. Kept for call-site compatibility."""
        # The legacy fuzzer called set_bounds() before set_predictor(); bounds
        # already live on self.problem, so nothing to do here.
        return self

    def set_predictor(self, predictor):
        self.predictor = predictor
        self.problem.set_predictor(predictor)
        return self

    def start_optimize(self):
        """Run DE on the currently-bound flight context.

        The full final population (not just the best individual) is captured so
        the candidate selectors in fuzzer.py can cluster across hundreds of
        individuals per context, matching the legacy geatpy behavior.
        """
        prophet = np.floor(self.default_pop / self.step_unit).astype(float)
        sampling = _ProphetSampling(prophet, self.POP_SIZE)
        algorithm = DE(
            pop_size=self.POP_SIZE,
            sampling=sampling,
            F=self.F,
            CR=self.CR,
        )
        self.result = minimize(
            self.problem,
            algorithm,
            termination=("n_gen", self.MAXGEN),
            seed=None,
            verbose=False,
        )
        # Capture the whole final population (pymoo exposes it on result.pop).
        phen = np.asarray(self.result.pop.get("X"))
        objv = np.asarray(self.result.pop.get("F"))
        if phen.ndim == 1:
            phen = phen.reshape(1, -1)
            objv = objv.reshape(1, -1)
        self.population = PopulationResult(Phen=phen, ObjV=objv)

    def return_best_n_gen(self, n=1):
        """Return the n best decision variables from the last optimization."""
        if self.population is None:
            raise ValueError("Please run start_optimize() first")
        var = self.population.ObjV.reshape(-1)
        x = self.population.Phen
        order = np.argsort(var)
        return self.problem.reasonable_range(x[order][:n])
