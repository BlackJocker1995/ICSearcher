"""Stage 0 — collect flight logs by repeatedly launching the SITL simulator.

Unified over ArduPilot / PX4: the firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable by the ``ICSEARCHER_MODE`` env var). Log rollback
on a failed flight differs per firmware.

**Multi-instance.** ``--instances N`` (or ``parallel.instances`` in config.yaml,
or the ``ICSEARCHER_INSTANCES`` env var) runs N SITL instances concurrently,
each on its own UDP port and in its own working directory (see
``toolConfig.mavlink_port`` / ``ardu_instance_path`` / ``px4_instance_path``).
A shared ``multiprocessing.Value`` counter coordinates how many logs have been
collected in total so the N workers stop together at ``TARGET_LOGS``.
"""
import argparse
import multiprocessing as mp
import os
import time
from datetime import datetime

from loguru import logger

from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4
from icsearcher.concurrency import MultiInstanceRunner, WorkerContext
from icsearcher.sim import GaSimManager

TARGET_LOGS = 500


def _count_logs():
    """Number of collected logs so far, per firmware (shared paths).

    Note: under multi-instance collection each worker writes to its own
    per-instance log dir, so this counts the *shared* directory only. The
    global target is tracked atomically by the shared counter instead.
    """
    if toolConfig.MODE == "PX4":
        log_path = toolConfig.PX4_LOG_PATH
        if not os.path.isdir(log_path):
            return 0
        return len([n for n in os.listdir(log_path) if n.endswith(".ulg")])
    # ArduPilot tracks the next index in logs/LASTLOG.TXT.
    log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
    if not os.path.exists(log_index):
        os.makedirs(os.path.dirname(log_index), exist_ok=True)
        with open(log_index, "w") as f:
            f.write('0')
    with open(log_index, 'r') as f:
        return int(f.readline())


def _mavlink_cls():
    return GaMavlinkPX4 if toolConfig.MODE == "PX4" else GaMavlinkAPM


def _rollback_failed_flight(instance_id):
    """Delete the most recent log for this instance on a failed flight."""
    if toolConfig.MODE == "PX4":
        GaMavlinkPX4.delete_current_log(instance_id)
    else:
        GaMavlinkAPM.delete_current_log(instance_id)


def collect_one(ctx: WorkerContext) -> None:
    """Worker: fly repeatedly on instance ``ctx.instance_id`` until target met.

    ``ctx.shared["counter"]`` is a shared ``multiprocessing.Value`` holding the
    running total of successfully collected logs across all workers. Each
    worker claims the next slot by reading-then-incrementing under the value's
    built-in lock, then flies one mission. On success it keeps the log (the
    increment already happened); on failure it rolls back and decrements.
    """
    instance_id = ctx.instance_id
    counter: "mp.Value" = ctx.shared["counter"]
    mission = toolConfig.mission_file()
    mavlink_cls = _mavlink_cls()

    while True:
        # Atomically check + claim the next slot.
        with counter.get_lock():
            collected = counter.value
            if collected >= TARGET_LOGS:
                logger.info(f"[{instance_id}] target reached ({collected}/{TARGET_LOGS}), exiting.")
                return
            counter.value += 1
            this_log = counter.value

        logger.info(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] instance {instance_id} "
            f"flying for log {this_log}/{TARGET_LOGS}"
        )

        manager = GaSimManager(debug=toolConfig.DEBUG)
        result = None
        try:
            logger.info(f"[{instance_id}] Starting SITL...")
            manager.start_multiple_sitl(instance_id)
            logger.info(f"[{instance_id}] Initializing MAVLink monitor...")
            manager.mav_monitor_init(mavlink_cls, instance_id)

            logger.info(f"[{instance_id}] Connecting to drone...")
            if not manager.mav_monitor_connect():
                logger.warning(f"[{instance_id}] Connection failed, restarting SITL.")
                manager.stop_sitl()
                _claim_failed(counter)
                continue

            logger.info(f"[{instance_id}] Uploading mission: {mission}")
            manager.mav_monitor.set_mission(mission, False)
            if toolConfig.MODE == "PX4":
                logger.debug(f"[{instance_id}] Waiting 2s for PX4 params to settle...")
                time.sleep(2)

            logger.info(f"[{instance_id}] Setting random params and arming...")
            manager.mav_monitor.set_random_param_and_start()

            logger.info(f"[{instance_id}] Waiting for flight to complete...")
            result = manager.mav_monitor.wait_complete()
            logger.info(f"[{instance_id}] Flight result: {result}")
        except Exception as e:
            logger.warning(f"[{instance_id}] Unexpected error: {e}")
        finally:
            logger.debug(f"[{instance_id}] Stopping SITL...")
            try:
                manager.stop_sitl()
            except Exception as e:
                logger.warning(f"[{instance_id}] stop_sitl failed: {e}")

        if not result:
            logger.info(f"[{instance_id}] Flight failed, rolling back log...")
            _rollback_failed_flight(instance_id)
            _claim_failed(counter)
        else:
            logger.info(f"[{instance_id}] Flight OK, log kept.")


def _claim_failed(counter: "mp.Value") -> None:
    """Undo a slot claim when a flight fails (so the target stays accurate)."""
    with counter.get_lock():
        # Never go negative if rollback races with a fresh claim.
        if counter.value > 0:
            counter.value -= 1


def main():
    parser = argparse.ArgumentParser(description="Stage 0: collect flight logs")
    parser.add_argument('--instances', dest='instances', type=int,
                        default=toolConfig.INSTANCES,
                        help='number of concurrent SITL instances '
                             '(default: toolConfig.INSTANCES / ICSEARCHER_INSTANCES)')
    args = parser.parse_args()

    if args.instances < 1:
        parser.error("--instances must be >= 1")

    logger.info(f"Collecting {TARGET_LOGS} logs with {args.instances} instance(s) "
                f"[mode={toolConfig.MODE}]")

    # Shared, atomic log counter so all workers converge on the same target.
    already = min(_count_logs(), TARGET_LOGS)
    counter = mp.Value("i", already)

    runner = MultiInstanceRunner(
        n_instances=args.instances,
        worker_fn=collect_one,
        shared={"counter": counter},
        debug=toolConfig.DEBUG,
    )
    runner.run()
    logger.info(f"Collection complete: {counter.value} logs total.")


if __name__ == '__main__':
    main()
