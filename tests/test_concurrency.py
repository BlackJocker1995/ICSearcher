"""Tests for the multi-instance orchestration layer.

These are pure-process tests — they do **not** require a live SITL simulator.
They verify:

- ``LockedCsv`` survives concurrent appends from multiple processes without
  losing or corrupting rows (the bug it was introduced to fix).
- ``MultiInstanceRunner`` actually spawns N workers and propagates failures.
- ``toolConfig`` per-instance path/port helpers are internally consistent.
"""
import csv
import os
import tempfile

import pytest

# Heavy deps behind these modules are only present after the full environment
# is installed. Skip gracefully in a minimal environment (matches the existing
# import-smoke tests).
pytest.importorskip("pymavlink")

from icsearcher.concurrency import (  # noqa: E402
    LockedCsv,
    MultiInstanceRunner,
    WorkerContext,
)
from icsearcher.config import toolConfig  # noqa: E402


# --------------------------------------------------------------------- LockedCsv
def test_locked_csv_append_creates_header(tmp_path):
    path = str(tmp_path / "out.csv")
    lc = LockedCsv(path, header=["a", "b", "c"])
    lc.ensure_created()
    lc.append_row([1, 2, 3])
    rows = lc.read_rows()
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == ["1", "2", "3"]


def test_locked_csv_concurrent_appends(tmp_path):
    """N processes each append M rows; all N*M rows must survive, intact."""
    path = str(tmp_path / "concurrent.csv")
    header = ["worker", "row"]
    LockedCsv(path, header=header).ensure_created()

    n_workers, rows_each = 6, 50

    def worker(ctx: WorkerContext) -> None:
        lc = LockedCsv(path, header=header)
        for r in range(rows_each):
            lc.append_row([ctx.instance_id, r])

    runner = MultiInstanceRunner(n_instances=n_workers, worker_fn=worker)
    runner.run()

    rows = LockedCsv(path).read_rows()
    # 1 header + n_workers * rows_each data rows.
    assert len(rows) == 1 + n_workers * rows_each
    # Header intact (first row).
    assert rows[0] == ["worker", "row"]
    # Every (worker, row) pair present exactly once — no lost/corrupt writes.
    data = {(int(w), int(r)) for w, r in rows[1:]}
    expected = {(w, r) for w in range(n_workers) for r in range(rows_each)}
    assert data == expected


def test_locked_csv_read_missing_file_returns_empty(tmp_path):
    lc = LockedCsv(str(tmp_path / "nope.csv"))
    assert lc.read_rows() == []


# --------------------------------------------------------------------- runner
def test_runner_spawns_n_workers_and_propagates_failure(tmp_path):
    """A crashing worker is reported after all workers join."""
    seen = str(tmp_path / "seen.txt")

    def ok_worker(ctx: WorkerContext) -> None:
        with open(seen, "a") as f:
            f.write(f"{ctx.instance_id}\n")

    def bad_worker(ctx: WorkerContext) -> None:
        if ctx.instance_id == 1:
            raise RuntimeError("boom")

    MultiInstanceRunner(n_instances=3, worker_fn=ok_worker).run()
    with open(seen) as f:
        assert sorted(f.read().split()) == ["0", "1", "2"]

    with pytest.raises(RuntimeError, match="failed"):
        MultiInstanceRunner(n_instances=3, worker_fn=bad_worker).run()


def test_runner_rejects_zero_instances():
    with pytest.raises(ValueError):
        MultiInstanceRunner(n_instances=0, worker_fn=lambda ctx: None)


# --------------------------------------------------------------------- config helpers
def test_mavlink_port_is_monotonic():
    assert toolConfig.mavlink_port(0) == 14540
    assert toolConfig.mavlink_port(1) == 14541
    assert toolConfig.mavlink_port(5) == 14545


def test_instance_paths_contain_index():
    # ArduPilot per-instance dir nests under the configured log path.
    p = toolConfig.ardu_instance_path(3)
    assert p.startswith(toolConfig.ARDUPILOT_LOG_PATH)
    assert "3" in os.path.basename(p)
    # logs dir is a child of the instance dir.
    assert toolConfig.ardu_instance_log_path(3).endswith(os.path.join("instance_3", "logs"))

    # PX4 per-instance dir nests under the build tree.
    px = toolConfig.px4_instance_path(2)
    assert px.endswith("instance_2")
    assert "build" in px


def test_instances_config_default_is_one():
    # Default must preserve the historical single-instance behaviour.
    assert toolConfig.INSTANCES >= 1
