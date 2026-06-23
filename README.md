# ICSearcher

Surrogate-guided fuzzing for UAV autopilot controller parameters. ICSearcher
trains an LSTM surrogate that predicts flight-status deviation, uses a genetic
algorithm to search the controller-parameter space for configurations likely to
destabilize the drone, validates the candidates in a real SITL simulator, and
finally derives safe parameter ranges. It supports both **ArduPilot** and
**PX4** firmware.

ICSearcher is an improved version of
[LGDFuzzer](https://dl.acm.org/doi/10.1145/3510003.3510084) (ICSE 2022). The
original LGDFuzzer source lives in the
[`lgdfuzzer`](https://github.com/BlackJoker1995/uavga/tree/lgdfuzzer) branch.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Deployment walkthrough](#deployment-walkthrough)
   - [Step 1 — Install the Python environment](#step-1--install-the-python-environment)
   - [Step 2 — Provision the simulators](#step-2--provision-the-simulators)
   - [Step 3 — Configure the run](#step-3--configure-the-run)
   - [Step 4 — Run the pipeline](#step-4--run-the-pipeline)
5. [Configuration reference](#configuration-reference)
6. [Testing](#testing)
7. [Notes & troubleshooting](#notes--troubleshooting)

---

## How it works

ICSearcher is a six-stage pipeline. Each stage is a standalone script in
`pipelines/`; the firmware (ArduPilot / PX4) is selected once in
`data/config.yaml` and every stage branches on it.

```
Stage 0  collect      Launch SITL repeatedly with random params, collect logs
Stage 1  convert      Parse .BIN / .ulg flight logs into CSV feature rows
Stage 2  train        Build supervised features, split, train the LSTM surrogate
Stage 3  fuzz         GA search (differential evolution) guided by the surrogate
Stage 4  validate     Select diverse candidates, validate each in real SITL
Stage 5  range        Derive safe parameter ranges via NSGA-II
```

The surrogate model is a PyTorch LSTM; the GA engine is
[pymoo](https://pymoo.org) (differential evolution for fuzzing, NSGA-II for
range derivation, with built-in GD/IGD/HV/Spacing indicators).

---

## Repository layout

```
icsearcher/               Core package
  config.py               Frozen toolConfig singleton (loaded from data/config.yaml)
  logging_config.py       Unified loguru setup
  params.py               Parameter loading / scaling / Location geometry
  comms.py                MAVLink comms + log parsing (DroneMavlink, APM/PX4)
  sim.py                  Simulator lifecycle (SimManager / GaSimManager)
  anomaly.py              In-flight anomaly detector (decomposed from the monitor)
  model.py                LSTM / TCN surrogate model (PyTorch)
  search/                 GA fuzzing engine (problem, searcher, fuzzer, io)
  range/                  NSGA-II range derivation (problem, searcher)
data/                     config.yaml, param_*.json, mission*.txt, fitCollection*.txt
pipelines/                The six stage entry points (collect, convert, train, fuzz, validate, range)
scripts/setup_sims.sh     Bootstrap: clone & build ArduPilot SITL and PX4 + JMavSim
tests/                    Pure-function unit tests (no SITL required)
pyproject.toml            Project manifest (deps + icsearcher-* console entry points)
```

---

## Requirements

- **OS:** Ubuntu 20.04 or 22.04 (recommended). The simulators and their build
  toolchains are Linux-centric.
- **Python:** 3.9 – 3.12.
- **GPU:** optional but recommended for training. The project supports both a
  **CPU** and a **CUDA 12.1** PyTorch build — you pick one at install time
  (see Step 1).
- **Simulators:** ArduPilot SITL and/or PX4-Autopilot with JMavSim. The
  bootstrap script builds them for you (see Step 2).

---

## Deployment walkthrough

### Step 1 — Install the Python environment

All Python dependencies are declared in `pyproject.toml`. The project works with
either [uv](https://docs.astral.sh/uv/) (recommended, faster) or
[Poetry](https://python-poetry.org). Below is the **uv** flow.

```bash
# 1a. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1b. From the repository root, sync the project + a PyTorch backend (pick ONE)
cd ICSearcher
uv sync --group cpu     # CPU build, or
uv sync --group cuda    # CUDA 12.1 build
```

`uv sync` resolves the dependency tree, creates `.venv`, installs the project
(plus its `icsearcher-*` console commands), and writes `uv.lock`. This brings in
the scientific stack (numpy, pandas, scipy, scikit-learn), the drone-comms stack
(pymavlink, pyulog, pexpect), the GA engine (pymoo), and dev tools (pytest).
**PyTorch is pulled in by exactly one of the two groups — they are mutually
exclusive:**

```bash
# Option A — CPU build (no GPU required)
uv sync --group cpu

# Option B — CUDA 12.1 build (NVIDIA GPU required)
uv sync --group cuda
```

The surrogate model auto-selects the device at runtime (CUDA when available,
otherwise CPU), so the rest of the pipeline is identical for both.

> **Different CUDA toolkit?** `pyproject.toml` pins the CUDA index to `cu121` via
> an explicit `[[tool.poetry.source]]`. For a different toolkit, edit that URL's
> suffix (`cu118` / `cu124`).
>
> **CUDA install trouble?** Fall back to installing the CUDA wheel manually after
> a CPU sync:
> ```bash
> uv sync --group cpu
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
>
> **Prefer Poetry?** The same groups work: `poetry install --with cpu` /
> `poetry install --with cuda`. Then prefix commands with `poetry run`.

> **Optional TCN backend.** A TCN surrogate (`CyTCN`) is available but no longer
> needs an external package — it ships as a built-in `Conv1d` head in
> `icsearcher/model.py`.

### Step 2 — Provision the simulators

`scripts/setup_sims.sh` clones and builds the upstream simulators, installs
their system build dependencies, and provisions the PX4 multi-instance helper
the project relies on.

```bash
# Build both simulators into the default locations
# (~/ardupilot and ~/PX4-Autopilot — matching data/config.yaml)
./scripts/setup_sims.sh

# Or build only one
./scripts/setup_sims.sh --ardupilot
./scripts/setup_sims.sh --px4
```

Override the install locations with environment variables (absolute paths):

```bash
ARDUPILOT_DIR=/opt/ardupilot PX4_DIR=/opt/PX4-Autopilot ./scripts/setup_sims.sh
```

What the script does, per simulator:

- **ArduPilot** — clones `ArduPilot/ardupilot` at a stable `Copter-*` tag,
  refreshes submodules, runs `install-prereqs-ubuntu.sh`, installs
  `pymavlink`/`MAVProxy`/`dronekit-sitl`, and builds the ArduCopter SITL once.
- **PX4** — clones `PX4/PX4-Autopilot` at a stable release, runs `ubuntu.sh`,
  builds `px4_sitl jmavsim`, and writes `Tools/sitl_multiple_run_single.sh`
  (used by multi-instance validation).

Run it once from a sudo-capable account; the first build is slow (it fetches a
toolchain). After it finishes, the binaries remain and later pipeline runs are
fast.

### Step 3 — Configure the run

All configuration lives in **`data/config.yaml`**. The most important field is
`mode`, which selects the firmware and is **frozen at load time** (there is no
runtime switching):

```yaml
mode: PX4          # or "Ardupilot"
```

Then point the `paths:` block at your simulator locations (the defaults match
what `setup_sims.sh` produces):

```yaml
paths:
  ardupilot_log: /media/rain/data                              # ArduPilot log dir
  sitl: /home/rain/ardupilot/Tools/autotest/sim_vehicle.py     # ArduPilot SITL launcher
  px4_run: /home/rain/PX4-Autopilot                            # PX4 source root
  jmavsim: /home/rain/PX4-Autopilot/Tools/jmavsim_run.sh       # JMavSim launcher
```

> **Quick mode switch without editing the file:** set the `ICSEARCHER_MODE`
> environment variable before running any stage:
> ```bash
> ICSEARCHER_MODE=Ardupilot uv run icsearcher-collect
> ```
> Priority is: env var > `data/config.yaml`'s `mode` field.

See [Configuration reference](#configuration-reference) for every field.

### Step 4 — Run the pipeline

Run the stages in order. Each stage is a console command — no `python` or path
needed (the `icsearcher-*` entry points are installed by `uv sync`):

```bash
# Stage 0 — collect flight logs (launches SITL ~500 times)
uv run icsearcher-collect

# Stage 1 — convert raw logs (.BIN / .ulg) to CSV
uv run icsearcher-convert

# Stage 2 — feature engineering + training (run each sub-step in order)
uv run icsearcher-train extract      # build features + fit the scaler
uv run icsearcher-train split        # split features into train/test
uv run icsearcher-train raw_split    # carve held-out raw test segments
uv run icsearcher-train train        # train the LSTM surrogate

# Stage 3 — surrogate-guided fuzzing
uv run icsearcher-fuzz

# Stage 4 — select candidates, then validate each in real SITL
uv run icsearcher-validate pre                  # cluster-select candidates
uv run icsearcher-validate validate             # validate in SITL
uv run icsearcher-validate validate --device 1  # use a specific SITL instance

# Stage 5 — derive safe parameter ranges via NSGA-II
uv run icsearcher-range
```

Prefer the module dispatcher? `uv run python -m pipelines <stage> [args...]`
works too (e.g. `uv run python -m pipelines train extract`). With Poetry, prefix
the console commands with `poetry run` instead of `uv run`.

**Outputs** (git-ignored):

- `model/{MODE}/` — trained surrogate (`lstm.pt`), the fitted scaler
  (`trans.pkl`), and feature CSVs.
- `result/{MODE}/` — fuzzing populations (`pop{EXE}.pkl`), validated candidates
  (`params{EXE}.csv`).

---

## Configuration reference

`data/config.yaml` is the single source of configuration.

| Field | Description |
|-------|-------------|
| `mode` | `PX4` or `Ardupilot`. Frozen at load time; override per-run with `ICSEARCHER_MODE`. |
| `simulation.speed` | SITL simulation speed factor. |
| `simulation.home` | ArduPilot `--location` / PX4 home region tag. |
| `simulation.debug` | Verbose logging when true. |
| `simulation.wind_range` | Wind speed range for sampling. |
| `simulation.window.{width,height}` | Render resolution (AirSim only). |
| `simulation.altitude.{limit_high,limit_low}` | Altitude bounds. |
| `paths.ardupilot_log` | ArduPilot log directory. **Must contain** `logs/LASTLOG.TXT` (run one sim flight there first to auto-generate it). |
| `paths.sitl` | Path to ArduPilot's `sim_vehicle.py`. |
| `paths.px4_run` | PX4-Autopilot source root. The PX4 log path is derived from this automatically. |
| `paths.jmavsim` | Path to PX4's `jmavsim_run.sh`. |
| `paths.{airsim,morse}` | Optional alternate simulator launchers. |
| `model.{input_len,output_len,segment_len,retrans}` | Surrogate model hyperparameters. |
| `cluster_choice_num` | Candidates sampled per cluster during candidate selection. |
| `param_files.{ardupilot,px4}` | Parameter-definition JSONs (default `data/param_*.json`). |
| `missions.fit_collection.{ardupilot,px4}` | Mission files used for fitness collection. |

---

## Testing

Pure-function unit tests that do **not** require a live SITL simulator:

```bash
uv run pytest        # or: poetry run pytest
```

Coverage spans the config loader, parameter scaling, the pymoo problem shapes,
the PyTorch model forward/train path, and the decomposed anomaly detector
(geometry + classification + timeout). Tests requiring heavy backends (pymoo,
torch, pymavlink) self-skip when the backend is absent, so the suite is always
green in any environment.

---

## Notes & troubleshooting

**No GUI needed for unattended fuzzing.** PX4 SITL launches with `HEADLESS=1`
(no JMavSim 3D window); ArduPilot SITL never opens a GUI either. The anomaly
detector reads flight telemetry over MAVLink, not the 3D view. To see the
JMavSim window for debugging, remove `HEADLESS=1` from the PX4 branch of
`icsearcher/sim.py:start_sitl`.

**The lockfile is not committed.** Generate it locally with `uv lock`
(or `poetry lock`); the TensorFlow-free dependency graph resolves quickly.
Commit it if you want reproducible installs across machines.

**Retrain after upgrading past stage 4.** The surrogate artifact changed from
`lstm.h5` (Keras) to `lstm.pt` (PyTorch state-dict). Old Keras models are not
loadable — rerun `uv run icsearcher-train train`.

**Multi-instance validation.** `uv run icsearcher-validate validate --device N`
runs validation against SITL instance `N` (UDP port 14540+N). The legacy
`gnome-terminal --tab` multi-tab launcher was removed (its `-e` syntax is
rejected by modern gnome-terminal); run multiple `--device` instances yourself
if you want parallel validation.

**Logging.** Every module uses [loguru](https://github.com/Delgan/loguru)
through `icsearcher/logging_config.py`. `setup_logging(debug=...)` configures
one unified stderr sink and bridges any remaining stdlib `logging` calls into
it, so nothing is silenced.
