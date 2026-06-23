"""Tests for the decomposed anomaly detector (no SITL required).

The detector's geometry (point-to-line distance) and state transitions are pure
functions of telemetry-like message objects, so we can drive them with cheap
fakes and assert on the outcome strings. This locks in the stage-4b behavior
split from mav_monitor_error.
"""
import math

import pytest

# anomaly.py + params.py pull pymavlink; skip the whole module if it's absent.
pytest.importorskip("pymavlink")

import icsearcher.config as cfg
from icsearcher.anomaly import (
    AnomalyDetector,
    CRASH,
    DEVIATION,
    PASS,
    PREARM_FAILED,
    TIMEOUT,
    THRUST_LOSS,
    point_to_line_distance,
)
from icsearcher.params import Location


class _FakeMsg:
    """Minimal stand-in for a pymavlink message with the fields the detector reads."""

    def __init__(self, mtype, **fields):
        self._type = mtype
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self):
        return self._type


def _statustext(severity, text):
    return _FakeMsg("STATUSTEXT", severity=severity, text=text)


def _position(lat, lon, alt, time_boot_ms, timeS):
    return _FakeMsg("GLOBAL_POSITION_INT", lat=int(lat * 1e7), lon=int(lon * 1e7),
                    relative_alt=int(alt * 1000), time_boot_ms=time_boot_ms, timeS=timeS)


def _mission(seq):
    return _FakeMsg("MISSION_CURRENT", seq=seq)


# ---------------------------------------------------------------------- geometry
def test_point_to_line_distance_zero_on_line():
    """A point sitting on the segment has ~0 perpendicular distance."""
    a = Location(0.0, 0.0)
    b = Location(0.0, 0.01)  # ~1.1 km north
    mid = Location(0.0, 0.005)
    assert point_to_line_distance(mid, a, b) == 0.0 or point_to_line_distance(mid, a, b) < 1.0


def test_point_to_line_distance_degenerate_segment():
    """A zero-length segment returns 0 (no division by zero)."""
    a = Location(1.0, 1.0)
    b = Location(1.0, 1.0)
    assert point_to_line_distance(Location(2.0, 2.0), a, b) == 0.0


# ------------------------------------------------------------------- classification
@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_crash_on_sim_hit_ground(monkeypatch, mode_name):
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.01))
    det.on_status(_statustext(severity=2, text="SIM Hit ground at 10 m"))
    assert det.result == CRASH


def test_thrust_loss_and_prearm():
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.01))
    det.on_status(_statustext(severity=2, text="Potential Thrust Loss"))
    assert det.result == THRUST_LOSS

    det2 = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.01))
    det2.on_status(_statustext(severity=2, text="PreArm: GPS not ready"))
    assert det2.result == PREARM_FAILED


def test_pass_on_landed_severity_6():
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.01))
    det.on_status(_statustext(severity=6, text="Disarming"))
    assert det.result == PASS


def test_ignores_non_statustext_and_irrelevant_severity():
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.01))
    det.on_status(None)
    det.on_status(_FakeMsg("HEARTBEAT"))
    det.on_status(_statustext(severity=6, text="some benign info"))
    assert det.result is None


# -------------------------------------------------------------- mission gating
def test_deviation_requires_start_check(monkeypatch):
    """Before the 2nd waypoint is reached, off-track positions don't trip deviation."""
    monkeypatch.setattr(cfg, "toolConfig", cfg.ToolConfig(mode="Ardupilot"))
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.001))
    # start_check is False, so feeding far-away positions must not classify.
    far = _position(5.0, 5.0, 10, 1, 1.0)
    for _ in range(50):
        det.on_position(far)
    assert det.result is None


def test_deviation_triggers_after_threshold(monkeypatch):
    """Once start_check is on, sustained off-track *moving* positions => 'deviation'.

    Positions must keep moving (velocity >= 1) so the stuck/timeout heuristic
    does not trip first; the perpendicular distance to the (0,0)-(0,0.001)
    segment stays large, so deviation_num climbs past its limit.
    """
    monkeypatch.setattr(cfg, "toolConfig", cfg.ToolConfig(mode="Ardupilot"))
    det = AnomalyDetector(lpoint1=Location(0, 0), lpoint2=Location(0, 0.001))
    det.start_check = True
    det._waypoint = lambda i: Location(0, 0)  # type: ignore[assignment]
    # Each frame is far off-track (lat ~1 => ~111 km from the segment) and moves
    # north by ~0.001 deg between frames so velocity stays well above 1 m/s.
    for i in range(25):
        det.on_position(_position(1.0, 0.001 * i, 10, 1000 * i, float(i)))
        if det.result is not None:
            break
    assert det.result == DEVIATION
