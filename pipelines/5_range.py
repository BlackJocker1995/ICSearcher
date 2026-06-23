"""Stage 5 — derive safe parameter ranges via NSGA-II.

The firmware is chosen by ``data/config.yaml``'s ``mode`` field (overridable
via ``ICSEARCHER_MODE``).
"""
import pandas as pd

from icsearcher.config import toolConfig
from icsearcher.range.searcher import GARangeOptimizer


def main():
    # The parameters fuzzed must correspond to those the predictor was trained on.
    result_data = pd.read_csv(f'result/{toolConfig.MODE}/params{toolConfig.EXE}.csv',
                              header=0).drop(columns="score")
    ga = GARangeOptimizer(result_data)
    ga.set_bounds()
    ga.run()


if __name__ == '__main__':
    main()
