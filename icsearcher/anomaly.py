"""In-flight anomaly detection, decomposed from the legacy mav_monitor_error.

The original ``GaSimManager.mav_monitor_error`` was a 150-line loop that
interleaved four concerns: status-text classification, mission-segment
tracking, trajectory-deviation geometry, and timeout/stuck detection. This
module exposes those as a small, testable ``AnomalyDetector`` so the loop in
``sim.py`` stays readable and the geometry/classification can be unit-tested
without a live SITL.

The detection thresholds and string matches are preserved verbatim from the
stage-1/2 code so flight outcomes stay logically equivalent.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from loguru import logger
from pymavlink import mavwp

from icsearcher.config import toolConfig
from icsearcher.params import Location

# Outcome string constants (shared with the CSV writer in pipelines/4_validate).
PASS = "pass"
CRASH = "crash"
THRUST_LOSS = "Thrust Loss"
PREARM_FAILED = "PreArm Failed"
DEVIATION = "deviation"
TIMEOUT = "timeout"

# Detection thresholds (unchanged from the legacy code).
MISSION_TIMEOUT_S = 200
SMALL_MOVE_LIMIT = 10          # consecutive slow frames => timeout
DEVIATION_LIMIT = 15           # consecutive off-track frames => deviation
DEVIATION_DIST_M = 10          # point-to-line distance that counts as off-track
LOW_VELOCITY = 1.0             # m/s
LOW_ALT_CHANGE = 0.1           # m


@dataclass
class AnomalyDetector:
    """Stateful detector driven one telemetry frame at a time.

    Construct with the mission waypoints; call :meth:`on_status` /
    :meth:`on_mission_current` / :meth:`on_position` each loop iteration and
    read :attr:`result` (``None`` while the flight is ongoing, else an outcome
    string). :meth:`timed_out` checks the global wall-clock budget.
    """

    lpoint1: Location
    lpoint2: Location
    pre_location: Location = field(init=False)
    start_check: bool = False
    current_mission: int = 0
    pre_alt: float = 0.0
    small_move_num: int = 0
    deviation_num: int = 0
    start_time: float = field(default_factory=time.time)
    result: Optional[str] = None

    def __post_init__(self):
        # pre_location seeds from the first waypoint (matches the legacy init).
        self.pre_location = Location(self.lpoint1)

    # ------------------------------------------------------------------ status
    def on_status(self, status_message) -> None:
        """Classify a STATUSTEXT message; sets self.result if terminal."""
        if status_message is None or status_message.get_type() != "STATUSTEXT":
            return
        line = status_message.text
        logger.debug(f"Status message: {status_message}")
        sev = status_message.severity
        if sev == 6:
            if any(k in line for k in ("Disarming", "landed", "Landing", "Land")):
                logger.info("Successful break the loop.")
                self.result = PASS
            elif "preflight disarming" in line:
                self.result = PREARM_FAILED
        elif sev in (0, 2):
            if any(k in line for k in
                   ("SIM Hit ground", "Crash",
                    "Failsafe enabled: no global position", "failure detected")):
                self.result = CRASH
            elif "Potential Thrust Loss" in line:
                self.result = THRUST_LOSS
            elif "PreArm" in line:
                self.result = PREARM_FAILED

    # ----------------------------------------------------------- mission track
    def on_mission_current(self, position_msg) -> None:
        """Track mission-segment progress and gate the deviation check."""
        if position_msg is None or position_msg.get_type() != "MISSION_CURRENT":
            return
        seq = int(position_msg.seq)
        if seq > self.current_mission and seq != 6:
            logger.debug(f"Mission change {self.current_mission} -> {seq}")
            self.lpoint1 = self._waypoint(self.current_mission)
            self.lpoint2 = self._waypoint(seq)
            # Begin checking after the 2nd waypoint (PX4 after the 1st).
            if seq == 2:
                self.start_check = True
            if seq == 1 and toolConfig.MODE == "PX4":
                self.start_check = True
            self.current_mission = seq
            if toolConfig.MODE == "PX4" and seq == 5:
                self.start_check = False

    # ------------------------------------------------------------- deviation
    def on_position(self, position_msg) -> None:
        """Update position, then run the deviation / stuck-point heuristics."""
        if position_msg is None or position_msg.get_type() != "GLOBAL_POSITION_INT":
            return
        position_lat = position_msg.lat * 1.0e-7
        position_lon = position_msg.lon * 1.0e-7
        alt = position_msg.relative_alt / 1000
        time_usec = position_msg.time_boot_ms * 1e-6
        position = Location(position_lat, position_lon, time_usec)

        moving_dis = Location.distance(self.pre_location, position)
        time_step = position.timeS - self.pre_location.timeS
        alt_change = abs(self.pre_alt - alt)
        self.pre_location.x = position_lat
        self.pre_location.y = position_lon
        self.pre_alt = alt

        if not self.start_check:
            return

        velocity = moving_dis / time_step if time_step else 0.0
        if velocity < LOW_VELOCITY and alt_change < LOW_ALT_CHANGE:
            self.small_move_num += 1
            logger.debug(f"Small moving {self.small_move_num}, num now - {self.small_move_num}.")
        else:
            self.small_move_num = 0

        self.deviation_num = self._update_deviation(position, self.deviation_num)
        if self.deviation_num > DEVIATION_LIMIT:
            self.result = DEVIATION
        elif self.small_move_num > SMALL_MOVE_LIMIT:
            self.result = TIMEOUT

    def _update_deviation(self, position, deviation_num) -> int:
        dev = point_to_line_distance(position, self.lpoint1, self.lpoint2)
        if dev > DEVIATION_DIST_M:
            if dev % 5 == 0:
                logger.debug(f"Deviation {round(dev, 4)}, num now - {deviation_num}.")
            return deviation_num + 1
        return 0

    # ----------------------------------------------------------------- timeout
    def timed_out(self) -> bool:
        """Whether the global wall-clock budget is exhausted."""
        if (time.time() - self.start_time) > MISSION_TIMEOUT_S:
            self.result = TIMEOUT
            return True
        return False

    # ------------------------------------------------------------------ helper
    def _waypoint(self, index) -> Location:
        """Subclasses / callers bind this to the mission waypoint loader."""
        raise NotImplementedError("waypoint access is bound by the controller")


def point_to_line_distance(point: Location, a: Location, b: Location) -> float:
    """Perpendicular distance from ``point`` to segment ``a``-``b`` (Heron's).

    Matches the legacy formula exactly (the +0.01 under the sqrt guards against
    degenerate triangles near the endpoints).
    """
    side_a = Location.distance(point, a)
    side_b = Location.distance(point, b)
    side_c = Location.distance(a, b)
    if side_c == 0:
        return 0.0
    p = (side_a + side_b + side_c) / 2
    return 2 * math.sqrt(p * (p - side_a) * (p - side_b) * (p - side_c) + 0.01) / side_c


def build_detector(mission_file: str) -> AnomalyDetector:
    """Build an AnomalyDetector seeded from the first two mission waypoints.

    The detector needs on-demand waypoint lookup during flight; rather than hold
    the whole loader we capture the waypoints list and bind ``_waypoint`` to it.
    """
    loader = mavwp.MAVWPLoader()
    loader.load(mission_file)
    wpoints = loader.wpoints
    det = AnomalyDetector(
        lpoint1=Location(wpoints[0]),
        lpoint2=Location(wpoints[1]),
    )
    det._waypoint = lambda index: Location(wpoints[index])  # type: ignore[assignment]
    return det
