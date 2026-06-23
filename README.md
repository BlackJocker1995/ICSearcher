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
- **Python:** 3.9 – 3.11. Python 3.12 has known compatibility issues with
  ArduPilot's build system (the `imp` module was removed) and is not supported.
- **Simulators:** ArduPilot SITL and/or PX4-Autopilot with JMavSim. The
  bootstrap script builds them for you (see Step 2).

### Prerequisites (system packages — install once, needs sudo)

Before anything else, install the system build tools. This is the **only** step
that needs `sudo`; the project scripts themselves run as your normal user (the
firmware's own setup scripts, invoked later, will prompt for sudo internally).

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git \
    python3 python3-pip python3-dev python3-venv \
    build-essential \
    ccache \
    wget curl
```

That's the minimal set to clone repos, run the firmware setup scripts, and build
any Python source wheels. The firmware installers pull in the rest (cmake,
ninja, simulator libraries, …) themselves.

---

## Deployment walkthrough

### Step 1 — Install the Python environment

All Python dependencies are declared in `pyproject.toml` and managed with
[uv](https://docs.astral.sh/uv/). One command resolves the tree, creates
`.venv`, installs the project (plus its `icsearcher-*` console commands), and
writes `uv.lock`:

```bash
# 1a. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1b. From the repository root, sync everything
cd ICSearcher
uv sync

# 1c. (ArduPilot only) add the firmware build/runtime tools the SITL needs
uv sync --group ardupilot
```

This brings in the scientific stack (numpy, pandas, scipy, scikit-learn), the
drone-comms stack (pymavlink, pyulog, pexpect), the GA engine (pymoo), the
surrogate model backend (PyTorch), and dev tools (pytest). **PX4-only users
can stop at `uv sync`.** ArduPilot users add the `ardupilot` group (MAVProxy,
dronekit-sitl, etc.) so `sim_vehicle.py` runs inside the project venv — no
separate system `pip install` needed.

> **Optional TCN backend.** A TCN surrogate (`CyTCN`) is available but no longer
> needs an external package — it ships as a built-in `Conv1d` head in
> `icsearcher/model.py`.

### Step 2 — Provision the simulators

ICSearcher fuzzes real firmware, so you need the ArduPilot and/or PX4
**SITL** (Software-In-The-Loop) simulators built from source. The
`scripts/setup_sims.sh` helper does all of it and is written as a teaching
script — it prints what it's doing at each step.

```bash
# Make the script executable (one-time — needed after a fresh clone)
chmod +x scripts/setup_sims.sh

# Clone + build both simulators. Everything lands under ./sims/ inside the
# repo, so the whole install is self-contained and removed by `rm -rf sims`.
./scripts/setup_sims.sh

# Or build only one firmware
./scripts/setup_sims.sh --ardupilot
./scripts/setup_sims.sh --px4
```

**What it clones + builds** (all under the repository, ~10 GB):

```
ICSearcher/
├── sims/                         created by the script
│   ├── ardupilot/                cloned from github.com/ardupilot/ardupilot
│   ├── PX4-Autopilot/            cloned from github.com/PX4/PX4-Autopilot
│   └── data/                     flight logs (.BIN / .ulg) live here
├── data/config.yaml              the script rewrites 'paths:' to match
└── ...
```

Run it as your **normal user** (the same one that ran the Prerequisites
`sudo apt-get`). The script itself does not use `sudo` — but the firmware
repos' own setup scripts (ArduPilot's `install-prereqs-ubuntu.sh` and PX4's
`ubuntu.sh`) do, and will prompt for your password when they run. Do **not**
wrap the whole script in `sudo` (`sudo ./scripts/...`): it would run `uv`/builds
as root and break the project venv. The first build downloads a compiler
toolchain and is slow (20–60 min); later pipeline runs reuse the binaries.

> **Custom locations?** Override with env vars (absolute paths):
> ```bash
> SIM_ROOT=/opt/sims DATA_DIR=/var/lib/icsearcher ./scripts/setup_sims.sh
> ```
> **Different firmware version?**
> ```bash
> ARDUPILOT_BRANCH=Copter-4.5.2 PX4_BRANCH=v1.14.0 ./scripts/setup_sims.sh
> ```
> **Uninstall:** `rm -rf sims` removes everything the script created.

### Step 3 — Choose the firmware (paths are auto-configured)

If you ran `setup_sims.sh`, the `paths:` block in `data/config.yaml` is already
pointed at `sims/ardupilot`, `sims/PX4-Autopilot`, and `sims/data` — no manual
editing needed. The one thing you must still choose is the **firmware mode**,
which selects which simulator the pipeline drives and is **frozen at load
time** (there is no runtime switching):

```yaml
mode: PX4          # or "Ardupilot"
```

> **Quick mode switch without editing the file:** set the `ICSEARCHER_MODE`
> environment variable before running any stage:
> ```bash
> ICSEARCHER_MODE=Ardupilot uv run icsearcher-collect
> ```
> Priority is: env var > `data/config.yaml`'s `mode` field.

> **Installed only one firmware?** Set `mode` to whichever you built
> (`--ardupilot` → `Ardupilot`, `--px4` → `PX4`). The other firmware's paths in
> `config.yaml` simply stay unused.

See [Configuration reference](#configuration-reference) for every field.

### Step 4 — Run the pipeline

**Before running, make sure you've cloned + built the simulators first**
(`./scripts/setup_sims.sh`, see Step 2 above). The pipeline launches SITL
simulators — without them it cannot collect or validate flights.

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
works too (e.g. `uv run python -m pipelines train extract`).

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
uv run pytest
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
(the TensorFlow-free dependency graph resolves in a couple of seconds).
Commit `uv.lock` if you want reproducible installs across machines.

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
