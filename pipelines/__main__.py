"""Dispatcher for the ICSearcher pipeline stages.

Exposes one function per stage for the ``[tool.poetry.scripts]`` console entry
points, and a ``__main__`` argv dispatcher so the whole pipeline can be driven
with ``uv run python -m pipelines <stage> [args...]``.

The stage order is documented by the module list; running them out of order will
fail because each stage consumes the previous stage's output artifacts.
"""
import sys

# Stage order (for help text only). Each name maps to a module with main().
STAGES = ["collect", "convert", "train", "fuzz", "validate", "range"]


# --- console-script entry points (uv run icsearcher-<stage>) ---
def collect():
    from pipelines.collect import main
    main()


def convert():
    from pipelines.convert import main
    main()


def train():
    from pipelines.train import main
    main()


def fuzz():
    from pipelines.fuzz import main
    main()


def validate():
    from pipelines.validate import main
    main()


def range_stage():
    from pipelines.range import main
    main()


# --- argv dispatcher (uv run python -m pipelines <stage> ...) ---
_DISPATCH = {
    "collect": collect,
    "convert": convert,
    "train": train,
    "fuzz": fuzz,
    "validate": validate,
    "range": range_stage,
}


def _help():
    print("Usage: uv run python -m pipelines <stage> [args...]\n")
    print("Stages (run in this order):")
    for i, s in enumerate(STAGES):
        print(f"  {i}  {s}")
    print("\nExamples:")
    print("  uv run python -m pipelines collect")
    print("  uv run python -m pipelines train extract")
    print("  uv run python -m pipelines validate validate --device 1")
    print("\nOr via the console scripts (no 'python'):")
    print("  uv run icsearcher-collect")
    print("  uv run icsearcher-train extract")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _help()
        sys.exit(0)
    stage = sys.argv[1]
    if stage not in _DISPATCH:
        print(f"error: unknown stage {stage!r}; expected one of {STAGES}", file=sys.stderr)
        sys.exit(2)
    # Hand the remaining argv to the stage so its argparse sees its own args.
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    _DISPATCH[stage]()
