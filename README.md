# ICSearcher and LGDFuzzer
This is an approach source code of ICSearcher.

The original code of the [paper(LGDFuzzer)](https://dl.acm.org/doi/10.1145/3510003.3510084) is in branch [lgdfuzzer](https://github.com/BlackJocker1995/uavga/tree/lgdfuzzer)

ICSearcher is an improved version of LGDFuzzer.

# Log
Update: 22-07-15, support px4
Update: stage-1 refactor — Poetry dependency management, unified loguru logging,
crash-on-import fixes, config driven by `config.yaml`, upstream simulator setup script.

## Requirement
OS: Ubuntu 20.04 / 22.04 (recommend). Python >= 3.9.

Dependencies are now managed with [Poetry](https://python-poetry.org/) and pinned
in `pyproject.toml` / `poetry.lock`. Install everything (including a CUDA build of
PyTorch) with:

```bash
poetry install
```

> CUDA: `pyproject.toml` pins PyTorch to the `cu121` wheel index. If your driver
> needs a different CUDA toolkit, edit the `[[tool.poetry.source]]` block
> (`cu118` / `cu124`).

Python packages used: numpy, pandas, scipy, scikit-learn, pymavlink, pyulog,
keras/tensorflow, geatpy (GA), loguru, pyyaml, tqdm, and torch (CUDA).

## Upstream simulators
Run the bootstrap script to clone & build ArduPilot SITL and PX4-Autopilot +
JMavSim, install their build deps, and provision the PX4 multi-instance helper:

```bash
scripts/setup_sims.sh                 # both simulators, default paths
scripts/setup_sims.sh --ardupilot     # only ArduPilot
scripts/setup_sims.sh --px4           # only PX4
```

Override the install locations with env vars (defaults match `config.yaml`):

```bash
ARDUPILOT_DIR=/opt/ardupilot PX4_DIR=/opt/PX4-Autopilot scripts/setup_sims.sh
```

GUI note: PX4 SITL is launched with `HEADLESS=1` (no JMavSim 3D window) so
unattended fuzzing does not need a display — the anomaly detector reads flight
telemetry over MAVLink. ArduPilot SITL never opens a GUI either. Remove
`HEADLESS=1` in `Cptool/gaSimManager.py:start_sitl` (PX4 branch) if you want the
3D view for debugging.

## Configuration
All configuration lives in `Cptool/config.yaml`. The `mode:` field
(`PX4` or `Ardupilot`) is authoritative: it is read once at startup and all
mode-derived constants (`STATUS_ORDER`, `PARAM`, `PARAM_PART`, paths) are computed
from it. Point the `paths:` block at your simulator locations.

Key fields:
* `mode` — `PX4` or `Ardupilot`.
* `paths.{sitl,px4_run,jmavsim,ardupilot_log,...}` — simulator executables / log dirs.
* `simulation.{speed,home,wind_range,window,altitude}` — sim parameters.
* `param_files.{ardupilot,px4}` — parameter JSON files.

`ARDUPILOT_LOG_PATH` must contain a flag file `logs/LASTLOG.TXT` (run one sim
flight there first to auto-generate it). PX4 log path is derived from
`px4_run` automatically.

## Pipeline
`0.collect.py` / `0.collect_px4.py` — start simulation to collect flight logs.
`1.trans_bin2csv.py` / `1.trans_bin2csv_px4.py` — transform the bin/ulg files to csv.
`2.extract_feature.py` / `2.extract_feature_px4.py` — extract features from csv.
`2.raw_split.py` / `2.raw_split_px4.py` — split test features for the searcher.
`2.feature_split.py` / `2.feature_split_px4.py` — split csv into train/test.
`2.train_lstm.py` / `2.train_lstm_px4.py` — train the LSTM predictor.
`3.lgfuzzer.py` / `3.lgfuzzer_px4.py` — start the surrogate-guided fuzzing.
`4.pre_validate.py` / `4.pre_validate_px4.py` — select candidates.
`4.validate.py` / `4.validate_px4.py` — validate configurations in the simulator
(use `--device {n}` to run a specific SITL instance in parallel).
`5.range.py` / `5.range_px4.py` — derive safe-range guidelines from validated results.

## Tests
Pure-function unit tests (no SITL required):

```bash
poetry run pytest
```

## Logging
All modules use [loguru](https://github.com/Delgan/loguru) via
`Cptool/logging_config.py`. `setup_logging(debug=...)` configures one unified
stderr sink and bridges any remaining stdlib `logging` calls into it, so nothing
is silenced.
