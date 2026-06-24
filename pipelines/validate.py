"""Stage 4 — candidate selection and SITL validation.

Unified over ArduPilot / PX4. The firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable via ``ICSEARCHER_MODE``).

Usage:
    python pipelines/4_validate.py pre              # cluster-select candidates
    python pipelines/4_validate.py validate             # validate in SITL
    python pipelines/4_validate.py validate --instances 4  # 4 parallel SITLs
    python pipelines/4_validate.py validate --device 1   # legacy single-instance on device 1
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from loguru import logger

from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4
from icsearcher.concurrency import LockedCsv, MultiInstanceRunner, WorkerContext
from icsearcher.search.fuzzer import return_cluster_thres_gen
from icsearcher.search.io import read_candidates, write_candidates
from icsearcher.sim import GaSimManager


def _params_csv():
    return f'result/{toolConfig.MODE}/params{toolConfig.EXE}.csv'


def _mavlink_cls():
    return GaMavlinkPX4 if toolConfig.MODE == "PX4" else GaMavlinkAPM


def _csv_header():
    """Header for params{EXE}.csv: the fuzzed subset + score/result columns."""
    return list(toolConfig.PARAM_PART) + ['score', 'result']


def pre():
    """Cluster the fuzzing populations into a diverse candidate set."""
    candidate_var, candidate_obj = return_cluster_thres_gen(0.5)
    candidate_obj = np.array(candidate_obj, dtype=float).round(8)
    candidate_var = np.array(candidate_var, dtype=float).round(8)
    write_candidates(candidate_obj, candidate_var)


def _already_validated(locked_csv, value_vector):
    """True if ``value_vector`` already appears in the CSV (under lock)."""
    rows = locked_csv.read_rows()
    if len(rows) <= 1:
        return False
    data = pd.DataFrame(rows[1:], columns=_csv_header())
    # The param columns are everything except score/result.
    param_cols = list(toolConfig.PARAM_PART)
    try:
        existing = data[param_cols].astype(float).to_numpy()
    except (ValueError, KeyError):
        return False
    diff = (existing - np.asarray(value_vector, dtype=float)).sum(axis=1).abs()
    return bool((diff < 0.00001).sum() > 0)


def validate_one(ctx: WorkerContext) -> None:
    """Worker: pull candidates from the shared queue, validate each on its SITL.

    ``ctx.shared`` carries:
        queue: a ``multiprocessing.Queue`` of (value_vector, score) tuples —
            workers consume it concurrently (work-stealing), so a slow flight
            on one instance doesn't idle the others.
        csv: a :class:`LockedCsv` wrapping params{EXE}.csv (flock-protected).
    """
    instance_id = ctx.instance_id
    queue: "mp.Queue" = ctx.shared["queue"]
    locked_csv: LockedCsv = ctx.shared["csv"]
    mavlink_cls = _mavlink_cls()
    mission = toolConfig.mission_file()

    while True:
        try:
            item = queue.get_nowait()
        except Exception:
            logger.info(f"[{instance_id}] queue empty, worker done.")
            return
        value_vector, score = item

        # Skip configs already validated — under the lock so two workers don't
        # both pass the check and double-validate the same candidate.
        if _already_validated(locked_csv, value_vector):
            logger.info(f"[{instance_id}] candidate already validated, skipping.")
            continue

        configuration = pd.Series(value_vector, index=toolConfig.PARAM_PART).to_dict()

        manager = GaSimManager(debug=toolConfig.DEBUG)
        try:
            # Always use the multi-instance launcher: it derives the port from
            # the instance id, so the monitor below connects to the right SITL.
            # (The old PX4 branch called start_sitl() — single instance — while
            # the monitor listened on 14540+device, a dead-port mismatch.)
            manager.start_multiple_sitl(instance_id)
            manager.mav_monitor_init(mavlink_cls, instance_id)

            if not manager.mav_monitor_connect():
                logger.warning(f"[{instance_id}] connection failed, skipping candidate.")
                manager.stop_sitl()
                continue

            manager.mav_monitor.set_mission(mission, israndom=False)
            manager.mav_monitor.set_params(configuration)
            if toolConfig.MODE == "PX4":
                time.sleep(2)
            manager.mav_monitor.start_mission()
            result = manager.mav_monitor_error()
            logger.info(f"[{instance_id}] Validated result: {result}")

            # Persist the row under flock (replaces the racy os.access + bare
            # append that corrupted rows under parallel writers).
            row = list(value_vector) + [float(score), result]
            locked_csv.append_row(row)
            logger.debug(f"[{instance_id}] wrote row to {locked_csv.path}.")
        except Exception as e:
            logger.warning(f"[{instance_id}] validation error: {e}")
        finally:
            try:
                manager.stop_sitl()
            except Exception as e:
                logger.warning(f"[{instance_id}] stop_sitl failed: {e}")
            time.sleep(1)


def validate(instances=None, device=0):
    """Validate candidates, optionally across N parallel SITL instances.

    Args:
        instances: number of concurrent instances. ``None`` falls back to the
            legacy single-instance behaviour on ``device`` (back-compat with
            ``--device``).
        device: instance index when ``instances`` is None (legacy path).
    """
    candidates = read_candidates()
    candidate_obj, candidate_var = candidates.obj, candidates.var

    csv_path = _params_csv()
    locked_csv = LockedCsv(csv_path, header=_csv_header())
    locked_csv.ensure_created()

    # Validate candidates in random order for diversity across instances.
    rand_index = np.arange(candidate_obj.shape[0])
    np.random.shuffle(rand_index)
    candidate_obj = candidate_obj[rand_index]
    candidate_var = candidate_var[rand_index]

    if instances is None or instances <= 1:
        # Legacy single-instance path: feed all candidates to one worker bound
        # to ``device``. Preserves the pre-parallel --device behaviour.
        queue: "mp.Queue" = mp.Queue()
        for value_vector, score in zip(candidate_obj, candidate_var):
            queue.put((np.asarray(value_vector, dtype=float), float(score[0])))
        ctx = WorkerContext(instance_id=device,
                            shared={"queue": queue, "csv": locked_csv})
        validate_one(ctx)
        return

    logger.info(f"Validating {candidate_obj.shape[0]} candidates across "
                f"{instances} parallel instances [mode={toolConfig.MODE}]")

    # Work-stealing queue: each candidate is validated exactly once by
    # whichever worker is free, so stragglers don't leave instances idle.
    queue = mp.Queue()
    for value_vector, score in zip(candidate_obj, candidate_var):
        queue.put((np.asarray(value_vector, dtype=float), float(score[0])))

    runner = MultiInstanceRunner(
        n_instances=instances,
        worker_fn=validate_one,
        shared={"queue": queue, "csv": locked_csv},
        debug=toolConfig.DEBUG,
    )
    runner.run()


def main():
    parser = argparse.ArgumentParser(description="Stage 4: candidate selection + validation")
    parser.add_argument("step", choices=["pre", "validate"], help="which sub-step to run")
    parser.add_argument('--instances', dest='instances', type=int, default=None,
                        help='number of concurrent SITL instances for validate '
                             '(default: toolConfig.INSTANCES)')
    parser.add_argument('--device', dest='device', type=int, default=0,
                        help='SITL instance index when not using --instances '
                             '(legacy single-instance path)')
    args = parser.parse_args()
    if args.step == "pre":
        pre()
    else:
        instances = args.instances if args.instances is not None else toolConfig.INSTANCES
        validate(instances=instances, device=args.device)


if __name__ == '__main__':
    main()
