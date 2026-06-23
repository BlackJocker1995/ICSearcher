"""Stage 4 — candidate selection and SITL validation.

Unified over ArduPilot / PX4. The firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable via ``ICSEARCHER_MODE``).

Usage:
    python pipelines/4_validate.py pre              # cluster-select candidates
    python pipelines/4_validate.py validate         # validate in SITL
    python pipelines/4_validate.py validate --device 1   # specific SITL instance
"""
import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
from loguru import logger

from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4
from icsearcher.search.fuzzer import return_cluster_thres_gen
from icsearcher.search.io import read_candidates, write_candidates
from icsearcher.sim import GaSimManager


def _params_csv():
    return f'result/{toolConfig.MODE}/params{toolConfig.EXE}.csv'


def _mavlink_cls():
    return GaMavlinkPX4 if toolConfig.MODE == "PX4" else GaMavlinkAPM


def pre():
    """Cluster the fuzzing populations into a diverse candidate set."""
    candidate_var, candidate_obj = return_cluster_thres_gen(0.5)
    candidate_obj = np.array(candidate_obj, dtype=float).round(8)
    candidate_var = np.array(candidate_var, dtype=float).round(8)
    write_candidates(candidate_obj, candidate_var)


def _ensure_csv_header():
    """Create the params CSV with the correct header if it does not exist.

    Uses PARAM_PART (the subset being fuzzed) plus the score/result columns.
    """
    path = _params_csv()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns=(toolConfig.PARAM_PART + ['score', 'result'])).to_csv(path, index=False)


def validate(device=0):
    candidates = read_candidates()
    candidate_obj, candidate_var = candidates.obj, candidates.var

    manager = GaSimManager(debug=toolConfig.DEBUG)
    csv_path = _params_csv()
    mavlink_cls = _mavlink_cls()
    mission = toolConfig.mission_file()

    # Validate candidates in random order.
    rand_index = np.arange(candidate_obj.shape[0])
    np.random.shuffle(rand_index)
    candidate_obj = candidate_obj[rand_index]
    candidate_var = candidate_var[rand_index]

    for index, vars, value_vector in zip(np.arange(candidate_obj.shape[0]), candidate_var, candidate_obj):
        print(f'======================={index} / {candidate_obj.shape[0]} ==========================')

        # Skip configs already validated.
        if os.path.exists(csv_path):
            while not os.access(csv_path, os.R_OK):
                time.sleep(0.1)
            data = pd.read_csv(csv_path)
            exit_data = data.drop(['score', 'result'], axis=1, inplace=False)
            if ((exit_data - value_vector).sum(axis=1).abs() < 0.00001).sum() > 0:
                continue

        configuration = pd.Series(value_vector, index=toolConfig.PARAM_PART).to_dict()

        # Start a (per-firmware) SITL instance.
        if toolConfig.MODE == "PX4":
            manager.start_sitl()
        else:
            manager.start_multiple_sitl(device)
        manager.mav_monitor_init(mavlink_cls, device)

        if not manager.mav_monitor_connect():
            manager.stop_sitl()
            continue

        manager.mav_monitor.set_mission(mission, israndom=False)
        manager.mav_monitor.set_params(configuration)
        if toolConfig.MODE == "PX4":
            time.sleep(2)
        manager.mav_monitor.start_mission()
        result = manager.mav_monitor_error()
        logger.info(f"Validated result: {result}")

        # Persist the row.
        _ensure_csv_header()
        while not os.access(csv_path, os.W_OK):
            time.sleep(0.1)
        tmp_row = value_vector.tolist()
        tmp_row.append(vars[0])
        tmp_row.append(result)
        with open(csv_path, 'a+') as f:
            csv.writer(f).writerow(tmp_row)
            logger.debug(f"Write row to params{toolConfig.EXE}.csv.")

        manager.stop_sitl()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: candidate selection + validation")
    parser.add_argument("step", choices=["pre", "validate"], help="which sub-step to run")
    parser.add_argument('--device', dest='device', type=int, default=0,
                        help='SITL instance index for validate (multi-instance)')
    args = parser.parse_args()
    if args.step == "pre":
        pre()
    else:
        validate(device=args.device)


if __name__ == '__main__':
    main()
