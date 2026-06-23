"""Stage 1 — convert raw flight logs (.BIN / .ulg) to CSV features.

Unified over ArduPilot / PX4. The firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable via ``ICSEARCHER_MODE``).
"""
from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4


def main():
    if toolConfig.MODE == "PX4":
        # PX4: read raw .ulg logs (written by collect into PX4_LOG_PATH) and
        # convert them in parallel (6 workers) into logs/ulg_changed/csv, which
        # is where the train stage reads from.
        GaMavlinkPX4.extract_log_path(
            toolConfig.PX4_LOG_PATH,
            f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/ulg_changed",
            skip=False, threat=6)
    else:
        # ArduPilot: read the raw .BIN log set (written by collect into logs/)
        # and convert it into logs/bin_regular/csv, which is where the train
        # stage reads from.
        GaMavlinkAPM.extract_log_path(
            f"{toolConfig.ARDUPILOT_LOG_PATH}/logs",
            f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_regular",
            skip=False)


if __name__ == '__main__':
    main()
