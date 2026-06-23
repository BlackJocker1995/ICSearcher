"""Tests for the icsearcher.config singleton.

Stage-2 scope: the config is frozen at load time, the mode comes from
config.yaml / ICSEARCHER_MODE / the ToolConfig(mode=...) ctor argument, and all
mode-derived constants are computed once. There is no runtime select_mode.
"""
import os

import pytest

from icsearcher.config import REPO_ROOT, ToolConfig, toolConfig


def test_singleton_mode_is_valid():
    """The module-level singleton must have constructed with a valid mode."""
    assert toolConfig.MODE in ("Ardupilot", "PX4")


def test_ctor_freezes_mode():
    """ToolConfig(mode=...) selects the mode at construction time."""
    for m in ("Ardupilot", "PX4"):
        cfg = ToolConfig(mode=m)
        assert cfg.MODE == m
        assert len(cfg.PARAM) > 0
        assert len(cfg.PARAM_PART) > 0


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        ToolConfig(mode="bogus")


def test_setattr_is_frozen():
    """After construction the singleton cannot be mutated."""
    with pytest.raises(ToolConfig.ConstError):
        toolConfig.MODE = "PX4"


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_param_part_is_subset(mode_name):
    """PARAM_PART is always a subset of PARAM for both firmwares."""
    cfg = ToolConfig(mode=mode_name)
    assert set(cfg.PARAM_PART).issubset(set(cfg.PARAM))


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_derived_lengths_consistent(mode_name):
    """The mode-derived lengths must satisfy their defining relations."""
    cfg = ToolConfig(mode=mode_name)
    assert cfg.STATUS_LEN == len(cfg.STATUS_ORDER) - 1               # drop TimeS
    assert cfg.DATA_LEN == cfg.STATUS_LEN + len(cfg.PARAM)
    assert cfg.INPUT_DATA_LEN == cfg.DATA_LEN * cfg.INPUT_LEN
    assert cfg.OUTPUT_DATA_LEN == cfg.STATUS_LEN * cfg.OUTPUT_LEN
    assert cfg.SEGMENT_LEN == 10 + cfg.INPUT_LEN


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_exe_suffix(mode_name):
    """EXE is '' when PARAM_PART == PARAM, else the partial count."""
    cfg = ToolConfig(mode=mode_name)
    if len(cfg.PARAM_PART) == len(cfg.PARAM):
        assert cfg.EXE == ""
    else:
        assert cfg.EXE == len(cfg.PARAM_PART)


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_mission_file_resolves_absolute(mode_name):
    """mission_file() must return an absolute path for the current mode."""
    cfg = ToolConfig(mode=mode_name)
    path = cfg.mission_file()
    assert os.path.isabs(path), f"mission_file() returned relative path {path}"
    assert path.endswith("fitCollection_px4.txt" if mode_name == "PX4" else "fitCollection.txt")


def test_resolve_leaves_absolute_untouched():
    """resolve() must pass absolute paths through unchanged."""
    assert toolConfig.resolve("/tmp/abs.txt") == "/tmp/abs.txt"


def test_resolve_makes_relative_absolute():
    """resolve() must anchor relative paths at the repo root."""
    out = toolConfig.resolve("data/param_ardu.json")
    assert os.path.isabs(out)
    assert out.endswith("data/param_ardu.json")
    assert os.path.exists(out)  # the data file really is there


def test_repo_root_points_at_repo():
    assert (REPO_ROOT / "icsearcher").is_dir()
    assert (REPO_ROOT / "data" / "config.yaml").is_file()
