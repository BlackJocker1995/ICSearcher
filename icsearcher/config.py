# coding:utf-8
"""Project configuration.

The :data:`toolConfig` singleton is constructed once at import time from
``data/config.yaml``. The mode (``PX4`` or ``Ardupilot``) is **frozen at load
time**: it comes from ``config.yaml``'s ``mode`` field and may be overridden by
the ``ICSEARCHER_MODE`` environment variable. All mode-derived constants
(``STATUS_ORDER``, ``PARAM``, ``PARAM_PART``, simulator paths, derived lengths)
are computed once during construction.

There is no runtime ``select_mode`` anymore — stage 2 removed the fragile
write-once ``__setattr__`` dance that let each ``_px4`` script mutate the
singleton after the fact. To run the pipeline in a different mode, set the
``mode`` field in ``data/config.yaml`` (or the env var) before importing
anything that reads ``toolConfig``.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

# Absolute path to the repo root. Every relative path in the project is
# resolved against this so the pipeline no longer depends on the CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent

# If the file-based resolution doesn't point at the project root (e.g. under
# some editable-install configurations where __file__ resolves to a synthetic
# path), fall back to the working directory. Both methods are validated by
# checking for a known marker (the icsearcher/config.py itself).
if not (REPO_ROOT / "icsearcher" / "config.py").is_file():
    cwd = Path.cwd()
    if (cwd / "icsearcher" / "config.py").is_file():
        REPO_ROOT = cwd
    else:
        raise RuntimeError(
            f"Cannot locate project root: {REPO_ROOT} (from __file__) "
            f"and {cwd} (from CWD) neither contains icsearcher/config.py. "
            "Run from the project directory."
        )

DATA_DIR = REPO_ROOT / "data"

VALID_MODES = ("Ardupilot", "PX4")

# Per-mode parameter subsets that participate in fuzzing.
PARAM_PART_ARDUPILOT = [
    "PSC_VELXY_P", "PSC_VELXY_I", "PSC_VELXY_D",
    "PSC_ACCZ_P", "PSC_ACCZ_I",
    "ATC_ANG_RLL_P", "ATC_RAT_RLL_P", "ATC_RAT_RLL_I", "ATC_RAT_RLL_D",
    "ATC_ANG_PIT_P", "ATC_RAT_PIT_P", "ATC_RAT_PIT_I", "ATC_RAT_PIT_D",
    "ATC_ANG_YAW_P", "ATC_RAT_YAW_P", "ATC_RAT_YAW_I", "ATC_RAT_YAW_D",
    "WPNAV_SPEED", "WPNAV_ACCEL", "ANGLE_MAX",
]
PARAM_PART_PX4 = [
    "MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P", "MC_YAW_WEIGHT",
    "MPC_XY_P", "MPC_Z_P",
    "MC_PITCHRATE_P", "MC_ROLLRATE_P", "MC_YAWRATE_P",
    "MPC_TILTMAX_AIR", "MIS_YAW_ERR",
    "MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP", "MPC_TKO_SPEED",
]

# Ordered status channels (the leading TimeS column is excluded from STATUS_LEN).
STATUS_ORDER_COMMON = [
    'TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
    'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
    'MagX', 'MagY', 'MagZ', 'VibeX', 'VibeY', 'VibeZ',
]


class ToolConfig:
    """Frozen-at-load config singleton.

    Construct with ``ToolConfig(mode=...)`` (mode defaults to the yaml/env
    value). After construction every attribute is read-only; attempts to set a
    new uppercase constant raise ``ConstError`` so accidental mutation fails
    loudly instead of silently corrupting a long fuzzing run.
    """

    class ConstError(PermissionError):
        pass

    def __init__(self, mode=None):
        self.__dict__["yaml_config"] = self._load_yaml_config()
        self._init_defaults()
        # If paths from yaml don't exist, try to auto-detect from sims/.
        self._detect_sims()
        # Resolve the definitive mode once: explicit arg > env var > yaml.
        resolved = mode or os.environ.get("ICSEARCHER_MODE") or self.__dict__["MODE"]
        if resolved not in VALID_MODES:
            raise ValueError(f"Invalid MODE {resolved!r}; expected one of {VALID_MODES}")
        self._apply_mode(resolved)

    # ------------------------------------------------------------------ loading
    def _load_yaml_config(self):
        """Load YAML config with fallback to empty dict.

        Priority:
          1. ``data/config.yaml`` (machine-specific, gitignored — generated
             by ``setup_sims.sh`` or copied from the .example template).
          2. ``data/config.yaml.example`` (committed template).
          3. Empty dict (all defaults used).
        """
        config_path = DATA_DIR / 'config.yaml'
        example_path = DATA_DIR / 'config.yaml.example'

        # If the machine-specific file doesn't exist yet, create it from the
        # example template so the user gets a ready-to-edit copy.
        if not config_path.exists() and example_path.exists():
            try:
                import shutil
                shutil.copy2(example_path, config_path)
                print(f"Created {config_path} from {example_path.name} — edit it if needed.")
            except OSError as e:
                print(f"Warning: could not copy {example_path} to {config_path}: {e}")

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config.yaml ({str(e)}), using defaults")
            return {}

    def _get_yaml_value(self, *keys, default=None):
        """Safely get nested YAML config value with fallback."""
        config = self.yaml_config
        for key in keys:
            if not isinstance(config, dict):
                return default
            config = config.get(key, default)
        return config

    def _param_file(self, mode):
        """Resolve the parameter JSON path for the given mode.

        Reads ``param_files.<mode>`` from config.yaml (relative to the repo
        root by default) and falls back to ``data/param_<mode>.json``.
        """
        key = 'ardupilot' if mode == 'Ardupilot' else 'px4'
        rel = self._get_yaml_value('param_files', key, default=f'data/param_{key}.json')
        return str(REPO_ROOT / rel) if not os.path.isabs(rel) else rel

    # ------------------------------------------------------------------ path API
    def resolve(self, rel):
        """Resolve a repo-relative path to an absolute string.

        Absolute paths are returned unchanged so machine-specific simulator
        locations keep working.
        """
        if not rel:
            return rel
        return rel if os.path.isabs(rel) else str(REPO_ROOT / rel)

    def mission_file(self):
        """Absolute path to the fit-collection mission for the current mode."""
        name = 'data/fitCollection_px4.txt' if self.MODE == 'PX4' else 'data/fitCollection.txt'
        rel = self._get_yaml_value('missions', 'fit_collection', self.MODE.lower(), default=name)
        return self.resolve(rel)

    # ------------------------------------------------------------------ multi-instance paths
    def _instance_subdir(self, i):
        """Render the per-instance subdirectory name from INSTANCE_DIR."""
        return self.INSTANCE_DIR.replace('{i}', str(int(i)))

    def ardu_instance_path(self, i):
        """Per-instance ArduPilot working directory.

        Each concurrent SITL instance gets its own directory under
        ``ARDUPILOT_LOG_PATH`` so its ``eeprom.bin`` / ``mav.parm`` / ``logs/``
        do not collide with sibling instances (the legacy multi-instance path
        shared one cwd, which raced when run in parallel). ``i`` is the
        0-based instance index.
        """
        return os.path.join(self.ARDUPILOT_LOG_PATH, self._instance_subdir(i))

    def ardu_instance_log_path(self, i):
        """Per-instance ArduPilot ``logs/`` directory (created lazily by callers)."""
        return os.path.join(self.ardu_instance_path(i), 'logs')

    def px4_instance_path(self, i):
        """Per-instance PX4 build directory (``instance_{i}`` under the build tree)."""
        return os.path.join(
            self.PX4_RUN_PATH, 'build', 'px4_sitl_default',
            self._instance_subdir(i),
        )

    # ------------------------------------------------------------------ multi-instance ports
    # The MAVLink GCS port for instance ``i`` is 14540+i. This is the single
    # source of truth: both the SITL launch (ArduPilot ``--out``, PX4's
    # sitl_multiple_run_single.sh wiring) and the MAVLink monitor must agree on
    # it, or a monitor connects to a dead port (the bug fixed by centralising
    # the derivation here).
    BASE_MAVLINK_PORT = 14540

    def mavlink_port(self, i):
        """MAVLink UDP port the GCS/monitor listens on for instance ``i``."""
        return self.BASE_MAVLINK_PORT + int(i)

    # ------------------------------------------------------------------ defaults
    def _init_defaults(self):
        """Initialize with YAML values or defaults.

        The YAML nests ``debug`` / ``wind_range`` / ``window`` / ``altitude``
        under ``simulation:``, so they are read from there.
        """
        sim = self._get_yaml_value('simulation', default={}) or {}
        self.__dict__["MODE"] = self._get_yaml_value('mode', default="Ardupilot")
        self.__dict__["SPEED"] = sim.get('speed', 3)
        self.__dict__["HOME"] = sim.get('home', "AVC_plane")
        self.__dict__["DEBUG"] = sim.get('debug', True)
        self.__dict__["WIND_RANGE"] = sim.get('wind_range', [8, 10.7])

        window = sim.get('window', {}) or {}
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
        self.__dict__["CLUSTER_CHOICE_NUM"] = self._get_yaml_value('cluster_choice_num', default=10)
        self.__dict__["INPUT_LEN"] = model.get('input_len', 4)
        self.__dict__["OUTPUT_LEN"] = model.get('output_len', 1)

        # Parallel multi-instance SITL. ``instances`` is how many simulator
        # processes run concurrently during collect/validate (default 1 keeps
        # the historical single-instance behaviour). ``instance_dir`` is the
        # template for the per-instance working directory; ``{i}`` is replaced
        # by the 0-based instance index. Override the count per-run with the
        # ``ICSEARCHER_INSTANCES`` env var (priority: env var > yaml).
        parallel = self._get_yaml_value('parallel', default={}) or {}
        env_instances = os.environ.get("ICSEARCHER_INSTANCES")
        if env_instances:
            try:
                instances = int(env_instances)
            except ValueError:
                raise ValueError(f"ICSEARCHER_INSTANCES must be an int, got {env_instances!r}")
        else:
            instances = int(parallel.get('instances', 1))
        if instances < 1:
            raise ValueError(f"parallel.instances must be >= 1, got {instances}")
        self.__dict__["INSTANCES"] = instances
        self.__dict__["INSTANCE_DIR"] = parallel.get('instance_dir', 'instance_{i}')

	# ------------------------------------------------------------------ sims auto-detect
    def _detect_sims(self):
        """Override non-existent paths with ones under ``sims/`` if available.

        If the user has run ``setup_sims.sh`` (or manually placed the simulators
        in ``sims/``), the default hardcoded paths in ``config.yaml`` likely don't
        exist — they point at home-directory locations from the template. This
        method checks each path and, if missing, looks for the equivalent under
        ``REPO_ROOT / sims / ...`` and uses that instead.
        """
        sims = REPO_ROOT / "sims"
        if not sims.is_dir():
            return  # nothing to auto-detect

        def _lookup(key, sims_rel):
            """If the current path for *key* doesn't exist, try sims/sims_rel."""
            cur = self.__dict__.get(key)
            if cur and os.path.exists(cur):
                return  # already valid
            candidate = (sims / sims_rel).resolve()
            if candidate.exists():
                self.__dict__[key] = str(candidate)
                print(f"  auto-detected {key} = {candidate}")

        _lookup("ARDUPILOT_LOG_PATH", "data")
        _lookup("SITL_PATH",          "ardupilot/Tools/autotest/sim_vehicle.py")
        _lookup("PX4_RUN_PATH",       "PX4-Autopilot")
        _lookup("JMAVSIM_PATH",       "PX4-Autopilot/Tools/jmavsim_run.sh")
        _lookup("MORSE_PATH",         "ardupilot/libraries/SITL/examples/Morse/quadcopter.py")

    # ------------------------------------------------------------------ mode
    def _apply_mode(self, mode):
        """Populate all mode-derived constants. Called once during __init__."""
        self.__dict__["MODE"] = mode

        if mode == "Ardupilot":
            self.__dict__["SIM"] = "SITL"
            self.__dict__["LOG_MAP"] = ['IMU', 'ATT', 'RATE', 'PARM', 'VIBE', "MAG"]
            self.__dict__["OL_LOG_MAP"] = ['ATTITUDE', 'RAW_IMU', 'VIBRATION']
            self.__dict__["STATUS_ORDER"] = list(STATUS_ORDER_COMMON)
            self.__dict__["PARAM_PART"] = list(PARAM_PART_ARDUPILOT)
        else:  # PX4
            self.__dict__["SIM"] = "Jmavsim"
            now_time = time.strftime("%Y-%m-%d", time.localtime())
            self.__dict__["PX4_LOG_PATH"] = f"{self.__dict__['PX4_RUN_PATH']}/build/px4_sitl_default/logs/{now_time}"
            self.__dict__["STATUS_ORDER"] = list(STATUS_ORDER_COMMON)
            self.__dict__["PARAM_PART"] = list(PARAM_PART_PX4)

        with open(self._param_file(mode), 'r') as f:
            param_name = pd.DataFrame(json.loads(f.read())).columns.tolist()
        self.__dict__["PARAM"] = param_name

        # EXE: '' when the fuzzed subset equals the full param set.
        self.__dict__["EXE"] = "" if len(self.PARAM_PART) == len(self.PARAM) else len(self.PARAM_PART)

        # ---- derived lengths ----
        self.__dict__["STATUS_LEN"] = len(self.STATUS_ORDER) - 1            # drop TimeS
        self.__dict__["PARAM_LEN"] = len(self.PARAM)
        self.__dict__["DATA_LEN"] = self.STATUS_LEN + len(self.PARAM)       # status + all params
        self.__dict__["INPUT_DATA_LEN"] = self.DATA_LEN * self.INPUT_LEN
        self.__dict__["OUTPUT_DATA_LEN"] = self.STATUS_LEN * self.OUTPUT_LEN
        self.__dict__["SEGMENT_LEN"] = 10 + self.INPUT_LEN
        self.__dict__["RETRANS"] = self._get_yaml_value('model', 'retrans', default=True)

        self.validate_config()

    # ------------------------------------------------------------------ helpers
    def get(self, key, default=None):
        """Safe config getter with a default value."""
        return self.__dict__.get(key, default)

    def validate_config(self):
        """Validate critical configuration values."""
        for key in ('MODE', 'SITL_PATH', 'PARAM'):
            if not self.__dict__.get(key):
                raise ValueError(f"Missing required config: {key}")
        if self.MODE not in VALID_MODES:
            raise ValueError(f"Invalid MODE {self.MODE!r}")

        # Warn (do not fail) when a configured simulator path is absent; the
        # operator may be running a pipeline stage that does not need it.
        for path_key in ('SITL_PATH', 'PX4_RUN_PATH', 'ARDUPILOT_LOG_PATH'):
            path = self.__dict__.get(path_key)
            if path and not os.path.exists(path):
                print(f"Warning: Path does not exist: {path_key}={path}")

    # ------------------------------------------------------------------ dunder
    def __setattr__(self, name, value):
        # After construction the config is effectively frozen.
        raise self.ConstError(
            f"toolConfig is frozen at load time; cannot set {name!r}. "
            "Change mode via data/config.yaml or the ICSEARCHER_MODE env var."
        )

    def __getattr__(self, item):
        # Only called when the attribute is genuinely missing.
        raise AttributeError(item)


# The singleton, constructed once from config.yaml (mode overridable via
# ICSEARCHER_MODE). Importing this module is the only way to read config.
toolConfig = ToolConfig()
