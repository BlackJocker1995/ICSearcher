"""Multi-instance SITL orchestration (stdlib ``multiprocessing``).

This is ICSearcher's analogue of nyctea's Ray-based multi-instance driver
(``Rl/learning_agent.py`` + ``0.train_ddpg_thread.py``): N worker processes,
each bound to its own simulator instance via a 0-based ``instance_id`` (which
maps to a distinct UDP port and a private working directory — see
``toolConfig.mavlink_port`` / ``ardu_instance_path`` / ``px4_instance_path``).

Differences from the nyctea model, by design:

- **No Ray.** ICSearcher deliberately dropped Ray (the log-conversion stage
  already uses stdlib ``ProcessPoolExecutor``); we keep that direction and use
  plain ``multiprocessing.Process``.
- **File coordination via ``fcntl.flock``, not a shared-memory buffer.** nyctea
  pools RL transitions in a ``shared_memory.ShareableList``; ICSearcher's
  stages instead append rows to a shared CSV, so we expose :class:`LockedCsv`
  — the same flock mechanism nyctea uses around its ``model/*.pth`` and
  ``buffer.pkl`` files — to make those appends safe under concurrency.

The :class:`MultiInstanceRunner` is stage-agnostic: it just fans out N
processes each running a user-supplied ``worker_fn(ctx)``. The stage-specific
work (collect vs validate) lives in the pipeline modules.
"""
import csv
import fcntl
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from loguru import logger


# --------------------------------------------------------------------- LockedCsv
class LockedCsv:
    """A CSV file whose reads/writes are guarded by an advisory flock.

    Replaces the racy ``os.access(...)`` busy-wait polling that the legacy
    ``validate.py`` used to "wait until the CSV is writable". That polling did
    not actually prevent two processes from interleaving a read-modify-write
    on the header/skip-check, so parallel validators could corrupt rows.
    ``fcntl.flock(LOCK_EX)`` is the same mechanism nyctea wraps around its
    ``model/*.pth`` and ``buffer.pkl`` for the same reason.

    Each public method opens, locks, operates, and releases in one call, so it
    is safe to share a single :class:`LockedCsv` across processes (the lock is
    per-fd, and each call opens its own fd). All processes must use this class
    to access the file — flock is advisory, so a bare ``open()`` elsewhere
    would still race.
    """

    def __init__(self, path: str, header: Optional[Sequence[str]] = None):
        self.path = path
        self.header = list(header) if header is not None else None

    def _ensure_header(self, fp):
        """Create the file with ``self.header`` if it is empty/missing."""
        if self.header is None:
            return
        if os.path.getsize(self.path) == 0:
            csv.writer(fp).writerow(self.header)
            fp.flush()

    def ensure_created(self) -> None:
        """Create the file (and header) if it does not exist. Idempotent."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # 'a+' creates if missing; we lock to avoid two writers racing on the
        # header write.
        with open(self.path, "a+", newline="") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                self._ensure_header(fp)
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

    def append_row(self, row: Sequence[Any]) -> None:
        """Append a single row under an exclusive lock."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a+", newline="") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            try:
                self._ensure_header(fp)
                csv.writer(fp).writerow(row)
                fp.flush()
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

    def read_rows(self) -> List[List[str]]:
        """Read all rows under a shared lock (consistent with concurrent writers)."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", newline="") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_SH)
            try:
                return list(csv.reader(fp))
            finally:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------- runner ctx
@dataclass
class WorkerContext:
    """Per-worker context passed into a ``worker_fn``.

    Attributes:
        instance_id: 0-based simulator index → UDP port + working dir.
        shared: a dict of process-shared objects the stage may set up before
            launch (e.g. a ``multiprocessing.Queue`` of candidates, a
            ``multiprocessing.Value`` log counter). These must be
            multiprocessing-safe primitives, not plain Python objects.
    """
    instance_id: int
    shared: dict = field(default_factory=dict)


WorkerFn = Callable[[WorkerContext], None]


# --------------------------------------------------------------------- the runner
class MultiInstanceRunner:
    """Fan out N worker processes, each bound to its own simulator instance.

    Usage::

        runner = MultiInstanceRunner(n_instances=4, worker_fn=my_worker)
        runner.run()   # blocks until all workers exit

    ``worker_fn(ctx)`` is called once per process and receives a
    :class:`WorkerContext` whose ``instance_id`` is the worker's simulator
    index. The worker is responsible for building its own ``GaSimManager``
    (MAVLink sockets and pexpect handles cannot be shared across processes)
    and for cleaning up its SITL on exit.

    Args:
        n_instances: number of concurrent simulator instances / worker procs.
        worker_fn: top-level (picklable) callable run by each worker.
        shared: optional dict of multiprocessing primitives to hand to every
            worker via ``ctx.shared``. Must contain only picklable /
            multiprocessing-safe objects (Queue, Value, Lock, ...).
        debug: forwarded to ``setup_logging`` inside each worker (loguru's sink
            is not inherited across fork, so workers reconfigure logging).
    """

    def __init__(self, n_instances: int, worker_fn: WorkerFn,
                 shared: Optional[dict] = None, debug: bool = False):
        if n_instances < 1:
            raise ValueError(f"n_instances must be >= 1, got {n_instances}")
        self.n_instances = n_instances
        self.worker_fn = worker_fn
        self.shared = dict(shared or {})
        self.debug = debug

    def _worker_entry(self, instance_id: int) -> None:
        """Process target: re-init logging, build ctx, call worker_fn.

        Top-level method bound to ``self`` is picklable for ``spawn``-style
        start methods because ``self`` is reconstructable in the child (the
        runner holds only plain data + a top-level function reference).
        """
        # Re-configure logging per process: loguru's default sink lives in the
        # parent's memory and is not inherited cleanly by the child.
        try:
            from icsearcher.logging_config import setup_logging
            setup_logging(debug=self.debug)
        except Exception as e:  # pragma: no cover - logging must never kill a worker
            logger.warning(f"worker {instance_id}: logging setup failed: {e}")

        ctx = WorkerContext(instance_id=instance_id, shared=self.shared)
        logger.info(f"worker {instance_id}: starting")
        self.worker_fn(ctx)

    def run(self) -> None:
        """Spawn all workers, block until they finish, surface failures.

        If any worker exits with a non-zero code, a ``RuntimeError`` listing
        the failed instance ids is raised after all workers have joined (so a
        single crash does not orphan the others).
        """
        procs = []
        for i in range(self.n_instances):
            p = mp.Process(target=self._worker_entry, args=(i,),
                           name=f"ics-worker-{i}")
            p.start()
            procs.append(p)
            logger.info(f"launched worker {i} (pid {p.pid})")

        failed: List[int] = []
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                failed.append(i)
                logger.error(f"worker {i} exited with code {p.exitcode}")

        if failed:
            raise RuntimeError(
                f"{len(failed)} worker(s) failed: instances {failed}"
            )


__all__ = ["LockedCsv", "WorkerContext", "MultiInstanceRunner"]
