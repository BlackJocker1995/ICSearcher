# coding:utf-8
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

# Absolute path to the repo root (the directory that contains Cptool/).
# Every relative path in the project is resolved against this so the pipeline
# no longer depends on the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Cptool"


class ToolConfig:
    class ConstError(PermissionError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __init__(self):
        # yaml_config is internal scratch state, not a frozen config constant,
        # so it bypasses the write-once __setattr__ guard via __dict__.
        self.__dict__["yaml_config"] = self._load_yaml_config()
        self._init_defaults()

    def _load_yaml_config(self):
        """Load YAML config with fallback to empty dict"""
        config_path = DATA_DIR / 'config.yaml'
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config.yaml ({str(e)}), using defaults")
            return {}

    def _get_yaml_value(self, *keys, default=None):
        """Safely get nested YAML config value with fallback"""
        config = self.yaml_config
        for key in keys:
            if not isinstance(config, dict):
                return default
            config = config.get(key, default)
        return config

    def _param_file(self, mode):
        """Resolve the parameter JSON path for the given mode.

        Reads ``param_files.<mode>`` from config.yaml (which is relative to the
        repo root by default) and falls back to ``Cptool/param_<mode>.json``.
        """
        key = 'ardupilot' if mode == 'Ardupilot' else 'px4'
        rel = self._get_yaml_value('param_files', key, default=f'Cptool/param_{key}.json')
        return str(REPO_ROOT / rel) if not os.path.isabs(rel) else rel

    def resolve(self, rel):
        """Resolve a repo-relative path to an absolute string.

        Absolute paths are returned unchanged so machine-specific simulator
        locations keep working. Use this for mission / fit-collection files
        and any other data file bundled in the repo.
        """
        if not rel:
            return rel
        return rel if os.path.isabs(rel) else str(REPO_ROOT / rel)

    def mission_file(self):
        """Absolute path to the fit-collection mission for the current mode."""
        name = 'Cptool/fitCollection_px4.txt' if self.__dict__['MODE'] == 'PX4' else 'Cptool/fitCollection.txt'
        rel = self._get_yaml_value('missions', 'fit_collection', self.__dict__['MODE'].lower(), default=name)
        return self.resolve(rel)

    def _init_defaults(self):
        """Initialize with YAML values or defaults.

        Note: the YAML nests ``debug``/``wind_range``/``window``/``altitude``
        under ``simulation:``, so they are read from there. The earlier code
        read them from the top level and silently ignored the YAML values.
        """
        sim = self._get_yaml_value('simulation', default={}) or {}

        self.__dict__["MODE"] = self._get_yaml_value('mode', default=None)
        self.__dict__["SPEED"] = sim.get('speed', 3)
        self.__dict__["HOME"] = sim.get('home', "AVC_plane")
        self.__dict__["DEBUG"] = sim.get('debug', True)
        self.__dict__["WIND_RANGE"] = sim.get('wind_range', [8, 10.7])

        window = sim.get('window', {}) or {}
        # Renamed from the misleading WEIGHT (the value is a render width).
        self.__dict__["WIDTH"] = window.get('width', 640)
        self.__dict__["HEIGHT"] = window.get('height', 480)

        altitude = sim.get('altitude', {}) or {}
        self.__dict__["LIMIT_H"] = altitude.get('limit_high', 50)
        self.__dict__["LIMIT_L"] = altitude.get('limit_low', 40)

        paths = self._get_yaml_value('paths', default={}) or {}
        self.__dict__["ARDUPILOT_LOG_PATH"] = paths.get('ardupilot_log', '/media/rain/data')
        self.__dict__["SITL_PATH"] = paths.get('sitl', "/home/rain/ardupilot/Tools/autotest/sim_vehicle.py")
        self.__dict__["AIRSIM_PATH"] = paths.get('airsim', "/media/rain/data/airsim/Africa_Savannah/LinuxNoEditor/Africa_001.sh")
        self.__dict__["PX4_RUN_PATH"] = paths.get('px4_run', '/home/rain/PX4-Autopilot')
        self.__dict__["JMAVSIM_PATH"] = paths.get('jmavsim', "/home/rain/PX4-Autopilot/Tools/jmavsim_run.sh")
        self.__dict__["MORSE_PATH"] = paths.get('morse', "/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py")

        model = self._get_yaml_value('model', default={}) or {}
        # CLUSTER_CHOICE_NUM lives at the top level of the YAML.
        self.__dict__["CLUSTER_CHOICE_NUM"] = self._get_yaml_value('cluster_choice_num', default=10)
        # Echo model.* into config so consumers can read them if desired.
        self.__dict__["INPUT_LEN"] = model.get('input_len', 4)
        self.__dict__["OUTPUT_LEN"] = model.get('output_len', 1)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value

    def __getattr__(self, item):
        if self.__dict__.get("MODE") is None:
            raise ValueError("Set config Mode at first!")
        raise AttributeError(item)

    def select_mode(self, mode):
        if mode not in ["Ardupilot", "PX4"]:
            raise ValueError("Bad mode")
        # Change Mode
        self.__dict__["MODE"] = mode

        if mode == "Ardupilot":
            # Simulation Type
            # Ardupilot : ['Airsim', 'Morse', 'Gazebo', 'SITL']
            self.__dict__["SIM"] = "SITL"  # "Jmavsim"

            # Mavlink Part
            self.__dict__["LOG_MAP"] = ['IMU', 'ATT', 'RATE', 'PARM', 'VIBE', "MAG"]  # "POS"
            # Online Mavlink Part
            self.__dict__["OL_LOG_MAP"] = ['ATTITUDE', 'RAW_IMU', 'VIBRATION']  # 'GLOBAL_POSITION_INT'
            # Status Order
            self.__dict__["STATUS_ORDER"] = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                                             # 'Lat', 'Lng', 'Alt',
                                             'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                             'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ']

            with open(self._param_file(mode), 'r') as f:
                param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
            self.__dict__["PARAM"] = param_name

            self.__dict__["PARAM_PART"] = [
                "PSC_VELXY_P",
                "PSC_VELXY_I",
                "PSC_VELXY_D",
                "PSC_ACCZ_P",
                "PSC_ACCZ_I",
                "ATC_ANG_RLL_P",
                "ATC_RAT_RLL_P",
                "ATC_RAT_RLL_I",
                "ATC_RAT_RLL_D",
                "ATC_ANG_PIT_P",
                "ATC_RAT_PIT_P",
                "ATC_RAT_PIT_I",
                "ATC_RAT_PIT_D",
                "ATC_ANG_YAW_P",
                "ATC_RAT_YAW_P",
                "ATC_RAT_YAW_I",
                "ATC_RAT_YAW_D",
                "WPNAV_SPEED",
                "WPNAV_ACCEL",
                "ANGLE_MAX"
            ]
        elif mode == "PX4":
            # PX4 : ['Jmavsim']
            self.__dict__["SIM"] = "Jmavsim"  # "Jmavsim"

            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d", now)
            # File path
            self.__dict__["PX4_LOG_PATH"] = f"{self.__dict__['PX4_RUN_PATH']}/build/px4_sitl_default/logs/{now_time}"
            # Status Order
            self.__dict__["STATUS_ORDER"] = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                                             'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                             'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ']

            with open(self._param_file(mode), 'r') as f:
                param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
            self.__dict__["PARAM"] = param_name

            self.__dict__["PARAM_PART"] = [
                "MC_ROLL_P",
                "MC_PITCH_P",
                "MC_YAW_P",
                "MC_YAW_WEIGHT",
                "MPC_XY_P",
                "MPC_Z_P",
                "MC_PITCHRATE_P",
                "MC_ROLLRATE_P",
                "MC_YAWRATE_P",
                "MPC_TILTMAX_AIR",
                "MIS_YAW_ERR",
                "MPC_Z_VEL_MAX_DN",
                "MPC_Z_VEL_MAX_UP",
                "MPC_TKO_SPEED"
            ]

        if len(self.__dict__["PARAM_PART"]) == len(self.__dict__["PARAM"]):
            self.__dict__["EXE"] = ""
        else:
            self.__dict__["EXE"] = len(self.__dict__["PARAM_PART"])

        ######################
        # Model Config       #
        ######################
        # Status length (exclude the leading TimeS column)
        self.__dict__["STATUS_LEN"] = len(self.__dict__["STATUS_ORDER"]) - 1

        # Parameter length
        self.__dict__["PARAM_LEN"] = len(self.__dict__["PARAM"])

        # input data entry length = status channels + all params
        self.__dict__["DATA_LEN"] = self.__dict__["STATUS_LEN"] + len(self.__dict__["PARAM"])

        # Whole predictor input length
        self.__dict__["INPUT_DATA_LEN"] = self.__dict__["DATA_LEN"] * self.__dict__["INPUT_LEN"]

        # Whole predictor output length
        self.__dict__["OUTPUT_DATA_LEN"] = self.__dict__["STATUS_LEN"] * self.__dict__["OUTPUT_LEN"]

        # Vector length of a segment
        self.__dict__["SEGMENT_LEN"] = 10 + self.__dict__["INPUT_LEN"]

        # transform values
        self.__dict__["RETRANS"] = self._get_yaml_value('model', 'retrans', default=True)

        # Validate now that mode-derived paths are known.
        self.validate_config()

    def get(self, key, default=None):
        """Safe config getter with default value"""
        return self.__dict__.get(key, default)

    def validate_config(self):
        """Validate critical configuration values"""
        required = ['MODE', 'SITL_PATH', 'PARAM']
        for key in required:
            if not self.__dict__.get(key):
                raise ValueError(f"Missing required config: {key}")

        if self.__dict__["MODE"] not in ["Ardupilot", "PX4"]:
            raise ValueError("Invalid MODE - must be 'Ardupilot' or 'PX4'")

        # Warn (do not fail) when an external simulator path is configured but
        # missing on this machine; the operator may be running a subset of the
        # pipeline that does not need it.
        paths = ['SITL_PATH', 'PX4_RUN_PATH', 'ARDUPILOT_LOG_PATH']
        for path_key in paths:
            path = self.__dict__.get(path_key)
            if path and not os.path.exists(path):
                print(f"Warning: Path does not exist: {path_key}={path}")


toolConfig = ToolConfig()
# Respect the mode declared in config.yaml instead of hardcoding ArduPilot.
# Pipeline scripts may still override with toolConfig.select_mode("PX4") /
# select_mode("Ardupilot") via the --mode flag (wired up in stage 2).
if toolConfig.MODE is None:
    toolConfig.select_mode("Ardupilot")
else:
    toolConfig.select_mode(toolConfig.MODE)
