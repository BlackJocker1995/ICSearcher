"""Stage 1 — convert raw flight logs (.BIN / .ulg) to CSV features.

Unified over ArduPilot / PX4. The firmware is chosen by ``data/config.yaml``'s
``mode`` field (overridable via ``ICSEARCHER_MODE``).
"""
from icsearcher.config import toolConfig
from icsearcher.comms import GaMavlinkAPM, GaMavlinkPX4


def main():
    if toolConfig.MODE == "PX4":
        # PX4: parallel-convert the newest .ulg log set (6 workers).
        GaMavlinkPX4.extract_log_path("csv", threat=6)
    else:
        # ArduPilot: convert the .BIN log set into CSV.
        GaMavlinkAPM.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_ga", skip=False)


if __name__ == '__main__':
    main()
