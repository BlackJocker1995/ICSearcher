import glob
import json
import multiprocessing
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from pymavlink import mavutil, mavwp
from pymavlink.DFReader import DFMessage
from pymavlink.mavutil import mavserial
from pyulog import ULog
from tqdm import tqdm

from icsearcher.config import toolConfig
from icsearcher.params import load_param, read_path_specified_file, select_sub_dict


def _convert_log_chunk(args):
    """Module-level worker for parallel log conversion (ProcessPoolExecutor).

    Must be top-level (picklable) — it cannot be a bound method. Reads the
    firmware mode from the spawned process's config (driven by ICSEARCHER_MODE,
    since toolConfig is frozen at import). Each raw log in ``in_path`` is parsed
    by the per-firmware extract_log_file and written to ``out_path/csv/<name>.csv``.
    Input and output are separate because the raw-log dir (e.g. ``logs/``) and
    the converted-CSV dir (e.g. ``logs/bin_regular/``) live at different levels.
    """
    in_path, out_path, file_list, skip, mode = args
    # Re-derive the config for the worker process so it picks up ICSEARCHER_MODE.
    from icsearcher.config import ToolConfig
    cfg = ToolConfig(mode=mode)
    extractor = GaMavlinkPX4.extract_log_file if cfg.MODE == "PX4" else GaMavlinkAPM.extract_log_file
    for file in tqdm(file_list):
        name, _ = file.split('.')
        if skip and os.path.exists(f'{out_path}/csv/{name}.csv'):
            continue
        try:
            csv_data = extractor(in_path + f'/{file}')
            csv_data.to_csv(f'{out_path}/csv/{name}.csv', index=False)
        except Exception as e:
            logger.warning(f"Error processing {file} : {e}")
            continue


class DroneMavlink:
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(DroneMavlink, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._master: mavserial = None
        self._port = port
        self.takeoff = False

    # Mavlink common operation

    def connect(self):
        """Connect drone"""
        self._master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(self._port))
        logger.info(f"Listening for MAVLink heartbeat on UDP port {self._port} (timeout 30s)...")
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            logger.warning(f"No heartbeat on port {self._port} after 30s — is SITL running?")
            return False
        logger.info("Heartbeat from system (system %u component %u) from %u" % (
            self._master.target_system, self._master.target_component, self._port))
        return True
        logger.info("Heartbeat from system (system %u component %u) from %u" % (
            self._master.target_system, self._master.target_component, self._port))
        return True

    def ready2fly(self) -> bool:
        """wait for IMU can work"""
        attempt = 0
        while True:
            attempt += 1
            message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
            if message is None:
                if attempt % 2 == 0:  # every ~60s
                    logger.info(f"Still waiting for IMU ready... (attempt {attempt})")
                continue
            message = message.to_dict()["text"]
            if toolConfig.MODE == "Ardupilot" and "IMU0 is using GPS" in message:
                logger.debug("Ready to fly.")
                return True
            if toolConfig.MODE == "PX4" and "home set" in message:
                logger.debug("Ready to fly.")
                return True

    def set_mission(self, mission_file, israndom: bool = False, timeout=30) -> bool:
        """
        Set mission
        :param israndom: random mission order
        :param mission_file: mission file
        :param timeout:
        :return: success
        """
        if not self._master:
            logger.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')

        loader = mavwp.MAVWPLoader()
        loader.target_system = self._master.target_system
        loader.target_component = self._master.target_component
        loader.load(mission_file)
        logger.debug(f"Load mission file {mission_file}")

        # if px4, set home at first
        if toolConfig.MODE == "PX4":
            self.px4_set_home()

        if israndom:
            loader = self.random_mission(loader)
        # clear the waypoint
        self._master.waypoint_clear_all_send()
        # send the waypoint count
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            # looping to send each waypoint information
            # Ardupilot method
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logger.debug(f'Sending waypoint {msg.seq}')
            mission_ack_msg = self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=timeout)
            logger.info(f'Upload mission finish.')
        except TimeoutError:
            logger.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """
        Arm and start the flight
        :return:
        """
        if not self._master:
            logger.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        # self._master.set_mode_loiter()
        if toolConfig.MODE == "PX4":
            self._master.set_mode_auto()
            self._master.arducopter_arm()
            self._master.set_mode_auto()
        else:
            self._master.arducopter_arm()
            self._master.set_mode_auto()

        logger.info('Armed, mission started.')

    def set_param(self, param: str, value: float) -> None:
        """
        set a value of specific parameter
        :param param: name of the parameter
        :param value: float value want to set
        """
        if not self._master:
            raise ValueError('Connect at first!')

        self._master.param_set_send(param, value)
        self.get_param(param)

    def set_params(self, params_dict: dict) -> None:
        """
        set multiple parameter
        :param params_dict: a dict consist of {parameter:values}...
        """
        for param, value in params_dict.items():
            self.set_param(param, value)

    def reset_params(self):
        self.set_param("FORMAT_VERSION", 0)

    def get_param(self, param: str) -> float:
        """
        get current value of a parameter.
        :param param: name
        :return: value of parameter
        """
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logger.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_params(self, params: list) -> dict:
        """
        get current value of a parameters.
        :param params:
        :return: value of parameter
        """
        out_dict = {}
        for param in params:
            out_dict[param] = self.get_param(param)
        return out_dict

    def get_msg(self, msg_type, block=False):
        """
        receive the mavlink message
        :param msg_type:
        :param block:
        :return:
        """
        msg = self._master.recv_match(type=msg_type, blocking=block)
        return msg

    def set_mode(self, mode: str):
        """
        Set flight mode
        :param mode: string type of a mode, it will be convert to an int values.
        :return:
        """
        if not self._master:
            logger.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        mode_id = self._master.mode_mapping()[mode]

        self._master.mav.set_mode_send(self._master.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                       mode_id)
        while True:
            message = self._master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
            if message['command'] == mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
                logger.debug(f'Mode: {mode} Set successful')
                break

    # Special operation
    def set_random_param_and_start(self):
        param_configuration = self.create_random_params(toolConfig.PARAM)
        logger.info(f"Setting {len(param_configuration)} random params and arming...")
        self.set_params(param_configuration)
        # Unlock the uav
        self.start_mission()

    def px4_set_home(self):
        if toolConfig.HOME is None:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               -35.362758,
                                               149.165135,
                                               583.730592)
        else:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               40.072842,
                                               -105.230575,
                                               0.000000)
        msg = self._master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=30)
        logger.debug(f"Home set callback: {msg.command}")

    def gcs_msg_request(self):
        self._master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        # self._master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        #                                 mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def wait_complete(self):
        pass

    # Static method
    @staticmethod
    def create_random_params(param_choice):
        """
        create a configuration with random values
        :param param_choice: parameters you want to choose
        :return:
        """
        para_dict = load_param()

        param_choice_dict = select_sub_dict(para_dict, param_choice)

        out_dict = {}
        for key, param_range in param_choice_dict.items():
            value = round(random.uniform(param_range['range'][0], param_range['range'][1]) / param_range['step']) * \
                    param_range['step']
            out_dict[key] = value
        return out_dict

    @staticmethod
    def random_mission(loader):
        """
        create random order of a mission
        :param loader: waypoint loader
        :return:
        """
        index = random.sample(loader.wpoints[2:loader.count() - 1], loader.count() - 3)
        index = loader.wpoints[0:2] + index
        index.append(loader.wpoints[-1])
        for i, points in enumerate(index):
            points.seq = i
        loader.wpoints = index
        return loader

    @staticmethod
    def extract_log_path(in_path, out_path=None, skip=True, threat=None):
        """Extract and convert raw flight logs (.BIN / .ulg) to CSV.

        Args:
            in_path: directory holding the raw logs (read from here).
            out_path: directory whose ``csv/`` subdir receives the converted
                CSVs (defaults to ``in_path`` for backward compatibility).
                Kept separate from ``in_path`` because raw logs and converted
                CSVs live at different tree levels (e.g. ``logs/`` vs
                ``logs/bin_regular/``).
            skip: if True, skip logs whose CSV already exists.
            threat: number of parallel worker processes (None = serial).
        """
        if out_path is None:
            out_path = in_path
        if toolConfig.MODE == "PX4":
            file_list = read_path_specified_file(in_path, 'ulg')
        else:
            file_list = read_path_specified_file(in_path, 'BIN')
        os.makedirs(f"{out_path}/csv", exist_ok=True)

        if threat is not None:
            # Parallel conversion via ProcessPoolExecutor (replaces Ray).
            chunks = [c for c in np.array_split(file_list, threat) if len(c)]
            args = [(in_path, out_path, list(chunk), skip, toolConfig.MODE) for chunk in chunks]
            with ProcessPoolExecutor(max_workers=threat) as pool:
                list(pool.map(_convert_log_chunk, args))
        else:
            _convert_log_chunk((in_path, out_path, list(file_list), skip, toolConfig.MODE))


class GaMavlinkAPM(DroneMavlink, multiprocessing.Process):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(GaMavlinkAPM, self).__init__(port, recv_msg_queue, send_msg_queue)

    @staticmethod
    def log_extract_apm(msg: DFMessage):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg.get_type() == 'ATT':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    'Roll': msg.Roll,
                    'Pitch': msg.Pitch,
                    'Yaw': msg.Yaw,
                }
        elif msg.get_type() == 'RATE':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # deg to rad
                'RateRoll': msg.R,
                'RatePitch': msg.P,
                'RateYaw': msg.Y,
            }
        # elif msg.get_type() == 'POS':
        #     out = {
        #         'TimeS': msg.TimeUS / 1000000,
        #         # deglongtitude
        #         'Lat': msg.Lat,
        #         'Lng': msg.Lng,
        #         'Alt': msg.Alt,
        #     }
        elif msg.get_type() == 'IMU':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'AccX': msg.AccX,
                'AccY': msg.AccY,
                'AccZ': msg.AccZ,
                'GyrX': msg.GyrX,
                'GyrY': msg.GyrY,
                'GyrZ': msg.GyrZ,
            }
        elif msg.get_type() == 'VIBE':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # m/s^2
                'VibeX': msg.VibeX,
                'VibeY': msg.VibeY,
                'VibeZ': msg.VibeZ,
            }
        elif msg.get_type() == 'MAG':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'MagX': msg.MagX,
                'MagY': msg.MagY,
                'MagZ': msg.MagZ,
            }
        elif msg.get_type() == 'PARM':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                msg.Name: msg.Value
            }
        return out

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')

        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.ffill()
            group_item = group_item.bfill()
            df_array.loc[len(df_array.index)] = group_item.mean()
        # Drop nan
        df_array = df_array.ffill()
        df_array = df_array.dropna()

        # Sort
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        # Status value + Parameter name
        order_name.extend(param_name)
        df_array = df_array[order_name]
        return df_array

    @staticmethod
    def extract_log_file(log_file):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """
        accept_item = toolConfig.LOG_MAP

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []
        accpet_param = load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            if msg.get_type() in ['ATT', 'RATE']:
                out_data.append(GaMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() in ['IMU', 'MAG']:
                if hasattr(msg, "I") and msg.I == 0:
                    out_data.append(GaMavlinkAPM.log_extract_apm(msg))
                else:
                    out_data.append(GaMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() == 'VIBE':
                if hasattr(msg, "IMU") and msg.IMU == 0:
                    out_data.append(GaMavlinkAPM.log_extract_apm(msg))
                else:
                    out_data.append(GaMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() == 'PARM' and msg.Name in accpet_param:
                out_data.append(GaMavlinkAPM.log_extract_apm(msg))
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = GaMavlinkAPM.fill_and_process_pd_log(pd_array)
        return pd_array

    @staticmethod
    def extract_log_file_des_and_ach(log_file):
        """
        extract log message form a bin file with att desired and achieved
        :param log_file:
        :return:
        """

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []

        while True:
            msg = logs.recv_match(type=["ATT"])
            if msg is None:
                break
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Roll': msg.Roll,
                'DesRoll': msg.DesRoll,
                'Pitch': msg.Pitch,
                'DesPitch': msg.DesPitch,
                'Yaw': msg.Yaw,
                'DesYaw': msg.DesYaw
            }
            out_data.append(out)

        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        return pd_array

    # Special function
    @classmethod
    def random_param_value(cls, param_json: dict):
        """
        random create the value
        :param param_json:
        :return:
        """
        out = {}
        for name, item in param_json.items():
            range = item['range']
            step = item['step']
            random_sample = random.randrange(range[0], range[1], step)
            out[name] = random_sample
        return out

    @staticmethod
    def delete_current_log(instance_id=None):
        """Delete the most recent ArduPilot flight log on a failed flight.

        Args:
            instance_id: if given, operate on the per-instance directory
                (``ARDUPILOT_LOG_PATH/instance_{i}/logs``) so concurrent
                multi-instance collectors don't touch each other's logs. If
                ``None`` (the default), uses the shared ``ARDUPILOT_LOG_PATH``
                for backward compatibility with single-instance runs.
        """
        if instance_id is None:
            logs_dir = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs"
        else:
            logs_dir = toolConfig.ardu_instance_log_path(instance_id)
        log_index = f"{logs_dir}/LASTLOG.TXT"

        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())
        # To string
        num = f'{num}'
        log_file = f"{logs_dir}/{num.rjust(8, '0')}.BIN"
        # Remove file
        if os.path.exists(log_file):
            os.remove(log_file)
            # Fix last index number
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    def wait_complete(self, remain_fail=False, timeout=60 * 5):
        if not self._master:
            raise ValueError('Connect at first!')
        try:
            timeout_start = time.time()
            wait_cycle = 0
            while time.time() < timeout_start + timeout:
                wait_cycle += 1
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                if message is None:
                    if wait_cycle % 2 == 0:  # every ~60s
                        logger.info(f"Waiting for flight completion... ({wait_cycle * 30}s)")
                    continue
                # print(message)
                message = message.to_dict()
                out_msg = "None"
                line = message['text']
                if message["severity"] == 6:
                    if "Land" in line:
                        # if successful landed, break the loop and return true
                        logger.info(f"Successful break the loop.")
                        return True
                elif message["severity"] == 2 or message["severity"] == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground at" in line:
                        pass
                    elif "Potential Thrust Loss" in line:
                        pass
                    elif "Crash" in line:
                        pass
                    elif "PreArm" in line:
                        pass
                        # will not generate log file
                        logger.info(f"Get error with {message['text']}")
                        return True
                    logger.info(f"Get error with {message['text']}")
                    if remain_fail:
                        # Keep problem log
                        return True
                    else:
                        return False
        except TimeoutError:
            # Mission point time out, change other params
            logger.warning('Wp timeout!')
            return False
        except KeyboardInterrupt:
            logger.info('Key bordInterrupt! exit')
            return False
        return False

        # Ardupilot

    def run(self):
        """
        loop check
        :return:
        """

        while True:
            msg = self._master.recv_match(type=['STATUSTEXT'], blocking=False)
            if msg is not None:
                msg = msg.to_dict()
                # print(msg2)
                if msg['severity'] in [0, 2]:
                    # self.send_msg_queue.put('crash')
                    logger.info('ArduCopter detect Crash.')
                    self.send_msg_queue.put('error')
                    break


class GaMavlinkPX4(DroneMavlink, multiprocessing.Process):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(GaMavlinkPX4, self).__init__(port, recv_msg_queue, send_msg_queue)
        self.log_file = None

    def wait_complete(self, remain_fail=False, timeout=60 * 5):
        if not self._master:
            raise ValueError('Connect at first!')
        try:
            timeout_start = time.time()
            wait_cycle = 0
            while time.time() < timeout_start + timeout:
                wait_cycle += 1
                # PX4 needs manual send the heartbeat for GCS
                self.gcs_msg_request()
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=False, timeout=30)
                if message is None:
                    if wait_cycle % 2 == 0:
                        logger.info(f"Waiting for PX4 flight completion... ({wait_cycle * 30}s)")
                    continue
                message = message.to_dict()
                out_msg = "None"
                line = message['text']
                if message["severity"] == 6:
                    if "landed" in line:
                        # if successful landed, break the loop and return true
                        logger.info(f"Successful break the loop.")
                        return True
                elif message["severity"] == 2 or message["severity"] == 0:
                    # Appear error, break loop and return false
                    if "SIM Hit ground at" in line:
                        pass
                    elif "Potential Thrust Loss" in line:
                        pass
                    elif "Crash" in line:
                        pass
                    elif "PreArm" in line:
                        pass
                        # will not generate log file
                        logger.info(f"Get error with {message['text']}")
                        return True
                    logger.info(f"Get error with {message['text']}")
                    if remain_fail:
                        # Keep problem log
                        return True
                    else:
                        return False
            return False
        except TimeoutError:
            # Mission point time out, change other params
            logger.warning('Wp timeout!')
            return False
        except KeyboardInterrupt:
            logger.info('Key bordInterrupt! exit')
            return False

    def init_ulg_log_file(self, device_i=None):
        if device_i is None:
            log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"

            list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file = latest_file
            logger.info(f"Current log file: {latest_file}")
        else:
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d", now)
            # Per-instance log dir (matches start_multiple_sitl's instance_{i}).
            log_path = (f"{toolConfig.px4_instance_path(device_i)}/log/{now_time}/*.ulg")

            list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file = latest_file
            logger.info(f"Current log file: {latest_file}")

    @staticmethod
    def fill_and_process_pd_log(pd_array: pd.DataFrame):
        # Round TimesS
        pd_array["TimeS"] = pd_array["TimeS"] / 1000000
        pd_array['TimeS'] = pd_array['TimeS'].round(1)

        pd_array = pd_array.drop_duplicates(keep='first')

        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)

        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.ffill()
            group_item = group_item.bfill()
            df_array.loc[len(df_array.index)] = group_item.mean()
        # Drop nan
        df_array = df_array.ffill()
        df_array = df_array.dropna()

        return df_array

    @staticmethod
    def extract_log_file(log_file):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """

        ulog = ULog(log_file)

        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)[["timestamp",
                                                                                "roll_body", "pitch_body", "yaw_body"]]
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)[["timestamp",
                                                                              "roll", "pitch", "yaw"]]
        acc_gyr = pd.DataFrame(ulog.get_dataset('sensor_combined').data)[["timestamp",
                                                                          "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
                                                                          "accelerometer_m_s2[0]",
                                                                          "accelerometer_m_s2[1]",
                                                                          "accelerometer_m_s2[2]"]]
        mag = pd.DataFrame(ulog.get_dataset('sensor_mag').data)[["timestamp", "x", "y", "z"]]
        vibe = pd.DataFrame(ulog.get_dataset('sensor_accel').data)[["timestamp", "x", "y", "z"]]
        # Param
        param = pd.Series(ulog.initial_parameters)
        param = param[toolConfig.PARAM]
        # select parameters
        for t, name, value in ulog.changed_parameters:
            if name in toolConfig.PARAM:
                param[name] = round(value, 5)

        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        mag.columns = ["TimeS", "MagX", "MagY", "MagZ"]
        vibe.columns = ["TimeS", "VibeX", "VibeY", "VibeZ"]
        # Merge values
        pd_array = pd.concat([att, rate, acc_gyr, mag, vibe]).sort_values(by='TimeS')

        # Process
        df_array = GaMavlinkPX4.fill_and_process_pd_log(pd_array)
        # Add parameters
        param_values = np.tile(param.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        # Sort
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))

        return df_array

    @classmethod
    def delete_current_log(cls, instance_id=None):
        """Delete the most recent PX4 .ulg flight log on a failed flight.

        Args:
            instance_id: if given, operate on the per-instance log dir
                (``instance_{i}/log/<date>/``) so concurrent multi-instance
                collectors don't delete each other's logs. If ``None`` (the
                default), uses the shared ``PX4_LOG_PATH`` for backward
                compatibility with single-instance runs.
        """
        if instance_id is None:
            log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"
        else:
            now_time = time.strftime("%Y-%m-%d", time.localtime())
            log_path = f"{toolConfig.px4_instance_path(instance_id)}/log/{now_time}/*.ulg"

        list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
        if not list_of_files:
            return
        latest_file = max(list_of_files, key=os.path.getctime)
        # Remove file
        if os.path.exists(latest_file):
            os.remove(latest_file)
