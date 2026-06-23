from loguru import logger
import os
import time
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkPX4
from Cptool.gaSimManager import GaSimManager


def file_number(path):
    return len([name for name in os.listdir(path) if name.endswith(".ulg")])


if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    # Create txt if not exists
    manager = GaSimManager(debug=toolConfig.DEBUG)

    time.sleep(1)
    while file_number(f"{toolConfig.PX4_LOG_PATH}") < 500:
        try:
            time.sleep(0.5)

            print('--------------------------------------------------------------------------------------------------')
            print(
                f'--------- {datetime.now()} === lastindex: {file_number(f"{toolConfig.PX4_LOG_PATH}")}--------------')
            print('--------------------------------------------------------------------------------------------------')

            manager.start_sitl()

            manager.mav_monitor_init(GaMavlinkPX4, 0)

            manager.mav_monitor.set_mission(toolConfig.mission_file(), False)

            time.sleep(2)
            manager.mav_monitor.set_random_param_and_start()
            # manager.mav_monitor.start_mission()

            result = manager.mav_monitor.wait_complete()

            # manager.mav_monitor.reset_params()

            manager.stop_sitl()

            if not result:
                # Delete current log
                GaMavlinkPX4.delete_current_log()

        except Exception as e:
            logger.warning(e)
            continue
