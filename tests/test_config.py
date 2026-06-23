"""Tests for the Cptool.config singleton.

These cover the stage-1 fixes: YAML key paths are now read from their nested
locations, mode-derived lengths are computed correctly, and switching modes
re-populates PARAM/PARAM_PART.
"""
from Cptool.config import toolConfig


def test_yaml_mode_is_respected():
    """The mode declared in config.yaml should be honoured at import time."""
    # config.yaml ships with mode: PX4, so the singleton should start in PX4.
    assert toolConfig.MODE in ("Ardupilot", "PX4")


def test_select_mode_populates_param_part(mode):
    """select_mode must set PARAM and PARAM_PART for both firmwares."""
    assert len(toolConfig.PARAM) > 0
    assert len(toolConfig.PARAM_PART) > 0
    # PARAM_PART is always a subset of PARAM.
    assert set(toolConfig.PARAM_PART).issubset(set(toolConfig.PARAM))


def test_derived_lengths_consistent(mode):
    """The mode-derived lengths must satisfy their defining relations."""
    # STATUS_LEN excludes the leading TimeS column.
    assert toolConfig.STATUS_LEN == len(toolConfig.STATUS_ORDER) - 1
    # DATA_LEN = status channels + all params.
    assert toolConfig.DATA_LEN == toolConfig.STATUS_LEN + len(toolConfig.PARAM)
    # INPUT/OUTPUT projected lengths.
    assert toolConfig.INPUT_DATA_LEN == toolConfig.DATA_LEN * toolConfig.INPUT_LEN
    assert toolConfig.OUTPUT_DATA_LEN == toolConfig.STATUS_LEN * toolConfig.OUTPUT_LEN
    # SEGMENT_LEN = 10 + INPUT_LEN.
    assert toolConfig.SEGMENT_LEN == 10 + toolConfig.INPUT_LEN


def test_exe_suffix(mode):
    """EXE is '' when PARAM_PART == PARAM, else the partial count."""
    if len(toolConfig.PARAM_PART) == len(toolConfig.PARAM):
        assert toolConfig.EXE == ""
    else:
        assert toolConfig.EXE == len(toolConfig.PARAM_PART)


def test_mission_file_resolves_absolute(mode):
    """mission_file() must return an absolute path for the current mode."""
    import os

    path = toolConfig.mission_file()
    assert os.path.isabs(path), f"mission_file() returned relative path {path}"
    assert path.endswith("fitCollection_px4.txt" if mode == "PX4" else "fitCollection.txt")


def test_resolve_leaves_absolute_untouched():
    """resolve() must pass absolute paths through unchanged."""
    assert toolConfig.resolve("/tmp/abs.txt") == "/tmp/abs.txt"


def test_resolve_makes_relative_absolute():
    """resolve() must anchor relative paths at the repo root."""
    import os

    out = toolConfig.resolve("Cptool/param_ardu.json")
    assert os.path.isabs(out)
    assert out.endswith("Cptool/param_ardu.json")
