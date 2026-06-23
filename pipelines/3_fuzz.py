"""Stage 3 — surrogate-guided fuzzing.

Loads the held-out raw test segments produced by stage 2 and runs the GA
search, dumping per-context populations to result/{MODE}/pop{EXE}.pkl.

The firmware is chosen by ``data/config.yaml``'s ``mode`` field (overridable
via ``ICSEARCHER_MODE``).
"""
import pickle

from icsearcher.config import toolConfig
from icsearcher.search.fuzzer import run_fuzzing


def main():
    with open(f"model/{toolConfig.MODE}/raw_test.pkl", 'rb') as f:
        np_data = pickle.load(f)
    run_fuzzing(np_data)


if __name__ == '__main__':
    main()
