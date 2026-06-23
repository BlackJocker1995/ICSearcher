"""Stage 0 — collect flight logs by repeatedly launching the SITL simulator.

Unified over ArduPilot / PX4: the firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable via the ``ICSEARCHER_MODE`` env var). Log rollback
on a failed flight differs per firmware.
"""
import os
from time import sleep
from datetime import datetime

from loguru import logger

from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4
from icsearcher.sim import GaSimManager

TARGET_LOGS = 500


def _count_logs():
    """Number of collected logs so far, per firmware."""
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



    sleep(1)
    while _count_logs() < TARGET_LOGS:
        sleep(0.5)
        collected = _count_logs()
        progress = collected / TARGET_LOGS * 100
        logger.info(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"collecting {collected}/{TARGET_LOGS} ({progress:.1f}%)"
        )
        result = None
        logger.info("Starting SITL...")
        manager.start_sitl()
        try:
            logger.info("Initializing MAVLink monitor...")
            manager.mav_monitor_init(mavlink_cls, 0)

            logger.info("Connecting to drone...")
            if not manager.mav_monitor_connect():
                logger.warning("Connection failed, restarting SITL.")
                manager.stop_sitl()
                continue

            logger.info(f"Uploading mission: {mission}")
            manager.mav_monitor.set_mission(mission, False)
            if toolConfig.MODE == "PX4":
                logger.debug("Waiting 2s for PX4 params to settle...")
                sleep(2)

            logger.info("Setting random params and arming...")
            manager.mav_monitor.set_random_param_and_start()

            logger.info("Waiting for flight to complete...")
            result = manager.mav_monitor.wait_complete()
            logger.info(f"Flight result: {result}")
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
        finally:
            logger.debug("Stopping SITL...")
            manager.stop_sitl()

        if not result:
            logger.info("Flight failed, rolling back log...")
            if toolConfig.MODE == "PX4":
                GaMavlinkPX4.delete_current_log()
            else:
                GaMavlinkAPM.delete_current_log()
        else:
            logger.info("Flight OK, log kept.")