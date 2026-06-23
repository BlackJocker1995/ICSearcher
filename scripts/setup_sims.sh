#!/usr/bin/env bash
#
# setup_sims.sh — install the drone simulators ICSearcher fuzzes against.
#
# This is a teaching script: it explains every step as it goes, clones the two
# upstream firmware repositories (ArduPilot and PX4) into the repo, builds their
# SITL (Software-In-The-Loop) simulators, and wires up a dedicated directory
# where simulated flight data (logs) is stored. At the end it rewrites
# data/config.yaml so the project points at what was just installed.
#
# WHAT GETS INSTALLED (all under the repository, so uninstall = delete the dir):
#
#   ICSearcher/
#   ├── sims/                         SIM_ROOT (created by this script)
#   │   ├── ardupilot/                ArduPilot SITL source + build
#   │   ├── PX4-Autopilot/            PX4 source + build + JMavSim
#   │   └── data/                     DATA_DIR — flight logs live here
#   ├── data/config.yaml              rewritten with the paths above
#   └── ...
#
# USAGE
#   ./scripts/setup_sims.sh             # install BOTH simulators (default)
#   ./scripts/setup_sims.sh --ardupilot # only ArduPilot
#   ./scripts/setup_sims.sh --px4       # only PX4
#
#   # Override where things go (absolute paths recommended):
#   SIM_ROOT=/opt/sims DATA_DIR=/var/lib/icsearcher ./scripts/setup_sims.sh
#
#   # Pin a different firmware version (defaults are stable tags):
#   ARDUPILOT_BRANCH=Copter-4.5.2 PX4_BRANCH=v1.14.0 ./scripts/setup_sims.sh
#
# REQUIREMENTS
#   Ubuntu 20.04 or 22.04, ~10 GB free disk, internet.
#   System packages (git, build-essential, ...) installed beforehand — see README.
#   The firmware's own setup scripts (invoked below) use sudo and will prompt.
#   The first build downloads a compiler toolchain and is slow (20-60 min).
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (every path is overridable via an env var)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SIM_ROOT: where the firmware repos live. Default: inside the repo (sims/), so
# the whole install is self-contained and removed by `rm -rf sims`.
SIM_ROOT="${SIM_ROOT:-$REPO_ROOT/sims}"
ARDUPILOT_DIR="$SIM_ROOT/ardupilot"
PX4_DIR="$SIM_ROOT/PX4-Autopilot"

# DATA_DIR: where simulated flight logs are written. Default: sims/data.
# This becomes data/config.yaml's `paths.ardupilot_log` and the parent of the
# PX4 log path.
DATA_DIR="${DATA_DIR:-$SIM_ROOT/data}"

# Firmware versions (stable tags). Override with env vars if you need another.
ARDUPILOT_BRANCH="${ARDUPILOT_BRANCH:-Copter-4.5.2}"
PX4_BRANCH="${PX4_BRANCH:-v1.14.0}"
NJOBS="${NJOBS:-$(nproc)}"

# Upstream URLs (the two repos this script downloads).
ARDUPILOT_URL="https://github.com/ardupilot/ardupilot"
PX4_URL="https://github.com/PX4/PX4-Autopilot.git"

# Which simulators to install. Default: both.
INSTALL_ARDUPILOT=1
INSTALL_PX4=1
if [[ "${1:-}" == "--ardupilot" ]]; then INSTALL_PX4=0; fi
if [[ "${1:-}" == "--px4" ]]; then INSTALL_ARDUPILOT=0; fi

# Pretty logging.
log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
info() { printf '  \033[0;37m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '  \033[1;33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 0 — verify system build dependencies (installed by the user, see README)
# ---------------------------------------------------------------------------
install_system_deps() {
    log "Step 0 — checking system build dependencies"
    info "This script does NOT use sudo. Install the system packages yourself"
    info "first (see README §Prerequisites), then run this script as your normal user."
    # The firmware repos' own setup scripts (install-prereqs-ubuntu.sh /
    # ubuntu.sh) DO use sudo internally — they will prompt for your password
    # when they run. This is expected and the only place sudo appears.
    local missing=()
    for cmd in git python3 pip3 wget curl ccache make; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Missing commands: ${missing[*]}.
  Install them (one-time, needs sudo) — see README §Prerequisites:
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \\
        git python3 python3-pip python3-dev python3-venv \\
        build-essential ccache wget curl"
    fi
    ok "system build tools present"
}

# ---------------------------------------------------------------------------
# Step 1 — ArduPilot SITL
# ---------------------------------------------------------------------------
install_ardupilot() {
    log "Step A — ArduPilot SITL  →  $ARDUPILOT_DIR"
    info "Cloning $ARDUPILOT_URL (branch $ARDUPILOT_BRANCH)"

    if [[ ! -d "$ARDUPILOT_DIR/.git" ]]; then
        git clone --recurse-submodules "$ARDUPILOT_URL" "$ARDUPILOT_DIR"
    else
        warn "$ARDUPILOT_DIR already exists; reusing and refreshing submodules."
    fi
    git -C "$ARDUPILOT_DIR" checkout "$ARDUPILOT_BRANCH"
    git -C "$ARDUPILOT_DIR" submodule update --init --recursive

    info "Running ArduPilot's own prerequisite installer (toolchain, etc.)"
    info "(install-prereqs-ubuntu.sh uses sudo internally and will prompt for your password)"
    if [[ -f "$ARDUPILOT_DIR/Tools/environment_install/install-prereqs-ubuntu.sh" ]]; then
        bash "$ARDUPILOT_DIR/Tools/environment_install/install-prereqs-ubuntu.sh" -y || true
    fi

    info "Building ArduCopter SITL — first build downloads a toolchain (slow)"
    info "(ArduPilot's Python tools — MAVProxy, dronekit-sitl, pexpect — come from"
    info " the project venv via 'uv sync --group ardupilot'; no system pip install.)"
    # Run sim_vehicle.py inside the project venv so it sees pexpect / pymavlink /
    # MAVProxy installed there. uv run handles the interpreter + PYTHONPATH.
    ( cd "$ARDUPILOT_DIR" && \
      uv run --project "$REPO_ROOT" python "Tools/autotest/sim_vehicle.py" \
          -v ArduCopter -w --no-mission -j"$NJOBS" )

    ok "ArduPilot SITL ready: $ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py"
}

# ---------------------------------------------------------------------------
# Step 2 — PX4-Autopilot + JMavSim
# ---------------------------------------------------------------------------
install_px4() {
    log "Step B — PX4-Autopilot + JMavSim  →  $PX4_DIR"
    info "Cloning $PX4_URL (tag $PX4_BRANCH)"

    if [[ ! -d "$PX4_DIR/.git" ]]; then
        git clone --recursive "$PX4_URL" "$PX4_DIR"
    else
        warn "$PX4_DIR already exists; reusing and refreshing submodules."
    fi
    git -C "$PX4_DIR" checkout "$PX4_BRANCH"
    git -C "$PX4_DIR" submodule update --init --recursive

    info "Running PX4's ubuntu.sh dependency installer"
    info "(ubuntu.sh uses sudo internally and will prompt for your password)"
    bash "$PX4_DIR/Tools/setup/ubuntu.sh" || warn "ubuntu.sh reported issues; continuing."

    info "Pre-building PX4 SITL + JMavSim so the first fuzzing run is fast"
    ( cd "$PX4_DIR" && HEADLESS=1 make px4_sitl jmavsim ) || true
    # The build spawns a jmavsim process; stop it, the binaries are already built.
    pkill -f "jmavsim_run.sh" 2>/dev/null || true
    pkill -f "px4 -i" 2>/dev/null || true

    # PX4 needs a per-instance launcher for multi-SITL validation.
    provision_px4_multi_instance_helper

    ok "PX4 SITL ready: $PX4_DIR (JMavSim: $PX4_DIR/Tools/jmavsim_run.sh)"
}

provision_px4_multi_instance_helper() {
    # ICSearcher's multi-instance validation calls Tools/sitl_multiple_run_single.sh,
    # which upstream no longer ships. Recreate it so `--device N` works out of the box.
    local target="$PX4_DIR/Tools/sitl_multiple_run_single.sh"
    info "Writing PX4 multi-instance launcher: $target"
    cat > "$target" <<'HELPER'
#!/bin/bash
# Launch a single PX4 SITL instance by index. Created by ICSearcher's setup_sims.sh.
sitl_num=0
[ -n "$1" ] && sitl_num="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
build_path="${SCRIPT_DIR}/../build/px4_sitl_default"
pkill -f "px4 -i $sitl_num" 2>/dev/null || true
sleep 1
export PX4_SIM_MODEL=iris
working_dir="$build_path/instance_$sitl_num"
mkdir -p "$working_dir"
cd "$working_dir"
echo "starting PX4 SITL instance $sitl_num in $(pwd)"
"$build_path/bin/px4" -i "$sitl_num" -d "$build_path/etc" -s etc/init.d-posix/rcS
HELPER
    chmod +x "$target"
}

# ---------------------------------------------------------------------------
# Step 3 — data storage directory + wire up config.yaml
# ---------------------------------------------------------------------------
setup_data_dir() {
    log "Step C — flight-log storage  →  $DATA_DIR"
    info "All simulated flight logs (.BIN / .ulg) and the ArduPilot LASTLOG.TXT"
    info "index live here. Keeping them under SIM_ROOT makes cleanup trivial."
    mkdir -p "$DATA_DIR/logs"

    # ArduPilot tracks the next log number in logs/LASTLOG.TXT; create it so the
    # collect stage works on the very first run.
    if [[ ! -f "$DATA_DIR/logs/LASTLOG.TXT" ]]; then
        echo '0' > "$DATA_DIR/logs/LASTLOG.TXT"
    fi
    ok "data directory ready: $DATA_DIR"
}

update_config() {
    # Rewrite data/config.yaml's paths: block so the project points at what we
    # just installed. Only the four path keys we own are touched.
    local cfg="$REPO_ROOT/data/config.yaml"
    [[ -f "$cfg" ]] || { warn "$cfg not found; skipping config update."; return; }

    log "Step D — wiring data/config.yaml to the installed paths"
    python3 - "$cfg" "$ARDUPILOT_DIR" "$PX4_DIR" "$DATA_DIR" <<'PY'
import re, sys
cfg, ardupilot_dir, px4_dir, data_dir = sys.argv[1:5]
sitl = f"{ardupilot_dir}/Tools/autotest/sim_vehicle.py"
jmavsim = f"{px4_dir}/Tools/jmavsim_run.sh"
morse = f"{ardupilot_dir}/libraries/SITL/examples/Morse/quadcopter.py"
airsim = f"{data_dir}/airsim/Africa_Savannah/LinuxNoEditor/Africa_001.sh"

text = open(cfg).read()
repls = {
    r"(?m)^(\s*ardupilot_log:\s*).*":      rf"\g<1>{data_dir}",
    r"(?m)^(\s*sitl:\s*).*":               rf"\g<1>{sitl}",
    r"(?m)^(\s*airsim:\s*).*":             rf"\g<1>{airsim}",
    r"(?m)^(\s*px4_run:\s*).*":            rf"\g<1>{px4_dir}",
    r"(?m)^(\s*jmavsim:\s*).*":            rf"\g<1>{jmavsim}",
    r"(?m)^(\s*morse:\s*).*":              rf"\g<1>{morse}",
}
for pat, rep in repls.items():
    text = re.sub(pat, rep, text)
open(cfg, "w").write(text)
print(f"  updated {cfg}")
PY
    ok "config.yaml paths updated"
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
main() {
    log "ICSearcher simulator installer"
    info "SIM_ROOT     = $SIM_ROOT   (firmware repos go here)"
    info "DATA_DIR     = $DATA_DIR   (flight logs go here)"
    info "ArduPilot    = $ARDUPILOT_URL @ $ARDUPILOT_BRANCH"
    info "PX4          = $PX4_URL @ $PX4_BRANCH"
    echo

    mkdir -p "$SIM_ROOT"
    install_system_deps

    (( INSTALL_ARDUPILOT )) && install_ardupilot
    (( INSTALL_PX4 ))       && install_px4

    setup_data_dir
    update_config

    cat <<SUMMARY

$(printf '\033[1;32m✓ Installation complete.\033[0m')

What was installed (all under $SIM_ROOT — remove with 'rm -rf $SIM_ROOT'):
  ArduPilot SITL : $ARDUPILOT_DIR
  PX4 + JMavSim  : $PX4_DIR
  Flight logs    : $DATA_DIR            (also written into data/config.yaml)

Your data/config.yaml 'paths:' block now points at these locations, so you can
run the pipeline immediately. To switch firmware, edit 'mode:' in
data/config.yaml (PX4 or Ardupilot) or set ICSEARCHER_MODE.

Next:
  uv sync --group cuda        # or --group cpu
  uv run icsearcher-collect   # stage 0 — start collecting flight logs
SUMMARY
}

main "$@"
