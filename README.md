# ICSearcher and LGDFuzzer
This is an approach source code of ICSearcher.

The original code of the [paper(LGDFuzzer)](https://dl.acm.org/doi/10.1145/3510003.3510084) is in branch [lgdfuzzer](https://github.com/BlackJocker1995/uavga/tree/lgdfuzzer)

ICSearcher is an improved version of LGDFuzzer.

# Log
- Update: 22-07-15, support px4
- Stage-1 refactor: Poetry dependency management, unified loguru logging,
  crash-on-import fixes, config driven by `data/config.yaml`, upstream simulator
  setup script.
- Stage-2 refactor: packages renamed to `icsearcher/`, the `_px4` script twins
  merged into 6 unified `pipelines/` scripts, `toolConfig` frozen at load time
  (mode from yaml / `ICSEARCHER_MODE` env var; no runtime `select_mode`).
- Stage-3 refactor: GA engine migrated from geatpy to pymoo (differential
  evolution for fuzzing, NSGA-II for range derivation, built-in GD/IGD/HV/Spacing
  indicators). geatpy removed.
- Stage-4 refactor: surrogate model migrated from Keras/TensorFlow to PyTorch
  (CUDA); the parallel log converter moved from Ray to stdlib
  `concurrent.futures`. tensorflow/keras/keras-tcn/ray removed. Model artifact
  changed from `lstm.h5` to `lstm.pt` — retrain via `2_train.py train`.

## Requirement
OS: Ubuntu 20.04 / 22.04 (recommend). Python >= 3.9.

Dependencies are managed with [Poetry](https://python-poetry.org/) and declared
in `pyproject.toml` (pinned in `poetry.lock`). Install everything (including a
CUDA build of PyTorch) with:

```bash
poetry install
```

> CUDA: `pyproject.toml` pins PyTorch to the `cu121` wheel index. If your driver
> needs a different CUDA toolkit, edit the `[[tool.poetry.source]]` block
> (`cu118` / `cu124`).

Python packages used: numpy, pandas, scipy, scikit-learn, pymavlink, pyulog,
pymoo (GA), torch (surrogate model, CUDA), loguru, pyyaml, tqdm.

## Upstream simulators
Run the bootstrap script to clone & build ArduPilot SITL and PX4-Autopilot +
JMavSim, install their build deps, and provision the PX4 multi-instance helper:

```bash
scripts/setup_sims.sh                 # both simulators, default paths
scripts/setup_sims.sh --ardupilot     # only ArduPilot
scripts/setup_sims.sh --px4           # only PX4
```

Override the install locations with env vars (defaults match `data/config.yaml`):

```bash
ARDUPILOT_DIR=/opt/ardupilot PX4_DIR=/opt/PX4-Autopilot scripts/setup_sims.sh
```

GUI note: PX4 SITL is launched with `HEADLESS=1` (no JMavSim 3D window) so
unattended fuzzing does not need a display — the anomaly detector reads flight
telemetry over MAVLink. ArduPilot SITL never opens a GUI either. Remove
`HEADLESS=1` in `icsearcher/sim.py:start_sitl` (PX4 branch) if you want the 3D
view for debugging.

## Package layout (stage 2)
```
icsearcher/
  config.py          frozen toolConfig singleton (loaded from data/config.yaml)
  logging_config.py  unified loguru setup
  params.py          parameter loading / scaling / Location geometry
  comms.py           MAVLink comms + log parsing (DroneMavlink et al.)
  sim.py             simulator lifecycle + anomaly detector
  model.py           LSTM/TCN surrogate model
  search/            GA fuzzing engine (problem / searcher / fuzzer / io)
  range/             NSGA-II range derivation (problem / searcher)
data/                config.yaml, param_*.json, mission*.txt, fitCollection*.txt
pipelines/           6 unified stage scripts (0_collect .. 5_range)
scripts/             setup_sims.sh
tests/               pure-function unit tests
```

## Configuration
All configuration lives in `data/config.yaml`. The `mode:` field (`PX4` or
`Ardupilot`) is **authoritative and frozen at load time**: it is read once and
all mode-derived constants (`STATUS_ORDER`, `PARAM`, `PARAM_PART`, paths) are
computed from it. To run a stage in a different mode, either change `mode:` in
`data/config.yaml` or set the `ICSEARCHER_MODE` env var before running — there
is no runtime `select_mode` anymore. Point the `paths:` block at your simulator
locations.

Key fields:
* `mode` — `PX4` or `Ardupilot`.
* `paths.{sitl,px4_run,jmavsim,ardupilot_log,...}` — simulator executables / log dirs.
* `simulation.{speed,home,wind_range,window,altitude}` — sim parameters.
* `param_files.{ardupilot,px4}` — parameter JSON files (default `data/param_*.json`).

`ARDUPILOT_LOG_PATH` must contain a flag file `logs/LASTLOG.TXT` (run one sim
flight there first to auto-generate it). The PX4 log path is derived from
`px4_run` automatically.

## Pipeline (unified over ArduPilot / PX4)
Each stage is one script; the firmware is chosen by `mode` in `data/config.yaml`.

```bash
python pipelines/0_collect.py
python pipelines/1_convert.py
python pipelines/2_train.py extract      # build features + fit scaler
python pipelines/2_train.py split        # split features into train/test
python pipelines/2_train.py raw_split    # carve held-out raw test segments
python pipelines/2_train.py train        # train the LSTM
python pipelines/3_fuzz.py               # surrogate-guided fuzzing
python pipelines/4_validate.py pre       # cluster-select candidates
python pipelines/4_validate.py validate            # validate in SITL
python pipelines/4_validate.py validate --device 1 # a specific SITL instance
python pipelines/5_range.py              # derive safe-range guidelines
```

## Tests
Pure-function unit tests (no SITL required):

```bash
poetry run pytest
```

## Logging
All modules use [loguru](https://github.com/Delgan/loguru) via
`icsearcher/logging_config.py`. `setup_logging(debug=...)` configures one unified
stderr sink and bridges any remaining stdlib `logging` calls into it, so nothing
is silenced.
