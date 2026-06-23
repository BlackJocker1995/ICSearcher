#!/usr/bin/env bash
#
# Bootstrap script: clone & build the upstream drone simulators needed by
# ICSearcher (ArduPilot SITL and PX4-Autopilot + JMavSim), install their Python
# dependencies, and provision the PX4 multi-instance helper that the project
# relies on.
#
# Usage:
#   scripts/setup_sims.sh                  # installs both (default paths)
#   ARDUPILOT_DIR=/opt/ardupilot PX4_DIR=/opt/PX4-Autopilot scripts/setup_sims.sh
#   scripts/setup_sims.sh --ardupilot      # only ArduPilot
#   scripts/setup_sims.sh --px4            # only PX4
#
# After it finishes, point Cptool/config.yaml at the chosen paths
# (paths.sitl / paths.px4_run / paths.jmavsim) — they already match the
# defaults below.
#
# GUI note: PX4 SITL is launched with HEADLESS=1 by ICSearcher (no 3D window)
# which is what you want for unattended fuzzing — the anomaly detector reads
# flight telemetry over MAVLink, not the JMavSim window. ArduPilot SITL never
# spawns a GUI either. If you do want the JMavSim 3D view for debugging, remove
# HEADLESS=1 from Cptool/gaSimManager.py:start_sitl (PX4 branch).
#
# Tested on Ubuntu 20.04 / 22.04. Run from a sudo-capable account.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
ARDUPILOT_BRANCH="${ARDUPILOT_BRANCH:-Copter-4.3.4}"   # a stable ArduCopter tag
PX4_BRANCH="${PX4_BRANCH:-v1.13.3}"                    # a stable PX4 release
NJOBS="${NJOBS:-$(nproc)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# What to install. Default: both. Flags narrow it.
INSTALL_ARDUPILOT=1
INSTALL_PX4=1
if [[ "${1:-}" == "--ardupilot" ]]; then INSTALL_PX4=0; fi
if [[ "${1:-}" == "--px4" ]]; then INSTALL_ARDUPILOT=0; fi

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
die()  { printf '\033[1;31m[err]\033[0m %s\n'   "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# System packages (needed by both simulators' build)
# ---------------------------------------------------------------------------
install_system_deps() {
    log "Installing system build dependencies (sudo required)..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-dev python3-venv \
        build-essential ccache g++ gcc make cmake ninja-build \
        genromfs dosfstools ncurses-dev libncurses-dev \
        libtool libxml2-dev libxslt1-dev zip unzip \
        wget curl lsb-release

    # Tools ArduPilot's sim_vehicle.py expects to fetch/use.
    if ! command -v screen >/dev/null 2>&1; then
        sudo apt-get install -y screen
    fi
}

# ---------------------------------------------------------------------------
# ArduPilot SITL
# ---------------------------------------------------------------------------
install_ardupilot() {
    log "Installing ArduPilot SITL into: $ARDUPILOT_DIR"

    if [[ ! -d "$ARDUPILOT_DIR/.git" ]]; then
        git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
    else
        warn "ArduPilot already cloned at $ARDUPILOT_DIR; fetching latest."
        git -C "$ARDUPILOT_DIR" fetch --all --tags
    fi

    git -C "$ARDUPILOT_DIR" checkout "$ARDUPILOT_BRANCH"
    # Refresh submodules for this branch.
    git -C "$ARDUPILOT_DIR" submodule update --init --recursive

    log "Installing ArduPilot Python tooling (pymavlink, MAVProxy, dronekit-sitl)..."
    # sim_vehicle.py relies on these; install into user site to avoid PEP 668.
    pip3 install --user --upgrade \
        pymavlink MAVProxy dronekit dronekit-sitl future lxml pymavlink

    log "Building ArduCopter SITL (first run downloads the toolchain; this is slow)..."
    # Tools/environment_install/install-prereqs-ubuntu.sh provisions the
    # arm-none-eabi toolchain and other deps; safe to re-run.
    if [[ -f "$ARDUPILOT_DIR/Tools/environment_install/install-prereqs-ubuntu.sh" ]]; then
        bash "$ARDUPILOT_DIR/Tools/environment_install/install-prereqs-ubuntu.sh -y" || true
    fi

    cd "$ARDUPILOT_DIR"
    # --no-mission downloads a default parameter set; building once primes the
    # binary so later collect/validate runs are fast.
    python3 "Tools/autotest/sim_vehicle.py" -v ArduCopter -w --no-mission -j"$NJOBS"
    cd "$REPO_ROOT"

    log "ArduPilot SITL ready."
    log "  SITL script: $ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py"
}

# ---------------------------------------------------------------------------
# PX4-Autopilot + JMavSim
# ---------------------------------------------------------------------------
install_px4() {
    log "Installing PX4-Autopilot into: $PX4_DIR"

    if [[ ! -d "$PX4_DIR/.git" ]]; then
        git clone --recursive https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
    else
        warn "PX4 already cloned at $PX4_DIR; fetching latest."
        git -C "$PX4_DIR" fetch --all --tags
    fi

    git -C "$PX4_DIR" checkout "$PX4_BRANCH"
    git -C "$PX4_DIR" submodule update --init --recursive

    log "Running PX4 ubuntu.sh dependency installer (sudo required)..."
    bash "$PX4_DIR/Tools/setup/ubuntu.sh" || warn "ubuntu.sh reported issues; continuing."

    log "Building PX4 SITL with JMavSim (HEADLESS build to avoid GUI at runtime)..."
    cd "$PX4_DIR"
    # Pre-build so the first mission does not pay the compile cost. JMavSim is
    # built as part of `make px4_sitl jmavsim`. We run it headless briefly then
    # it is killed; the binaries remain.
    HEADLESS=1 make px4_sitl jmavsim || true
    # Kill any lingering jmavsim/px4 the build spawned.
    pkill -f "jmavsim_run.sh" 2>/dev/null || true
    pkill -f "px4 -i" 2>/dev/null || true
    cd "$REPO_ROOT"

    # Provision the PX4 multi-instance helper the project depends on for
    # `start_multiple_sitl`. (See README "run PX4 evaluation in multiple thread".)
    provision_px4_multi_instance_helper

    log "PX4 SITL ready."
    log "  PX4 root:     $PX4_DIR"
    log "  JMavSim:       $PX4_DIR/Tools/jmavsim_run.sh"
}

provision_px4_multi_instance_helper() {
    local target="$PX4_DIR/Tools/sitl_multiple_run_single.sh"
    log "Provisioning PX4 multi-instance helper: $target"
    cat > "$target" <<'HELPER'
#!/bin/bash
# Launch a single PX4 SITL instance by index. Created by ICSearcher setup.
sitl_num=0
[ -n "$1" ] && sitl_num="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
src_path="$SCRIPT_DIR/.."
build_path=${src_path}/build/px4_sitl_default
pkill -f "px4 -i $sitl_num" 2>/dev/null || true
sleep 1
export PX4_SIM_MODEL=iris
working_dir="$build_path/instance_$sitl_num"
[ ! -d "$working_dir" ] && mkdir -p "$working_dir"
pushd "$working_dir" &>/dev/null
echo "starting instance $sitl_num in $(pwd)"
"../bin/px4" -i "$sitl_num" -d "$build_path/etc" -s etc/init.d-posix/rcS
popd &>/dev/null
HELPER
    chmod +x "$target"
}

# ---------------------------------------------------------------------------
# Python project deps (Poetry)
# ---------------------------------------------------------------------------
install_python_deps() {
    log "Installing ICSearcher Python dependencies via Poetry..."
    if ! command -v poetry >/dev/null 2>&1; then
        warn "Poetry not found; installing it for the current user."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    cd "$REPO_ROOT"
    poetry install --no-root || warn "poetry install had issues (CUDA torch source may need a moment to index)."
    cd "$REPO_ROOT"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "ICSearcher simulator bootstrap"
    log "  ArduPilot -> $ARDUPILOT_DIR (branch $ARDUPILOT_BRANCH)"
    log "  PX4       -> $PX4_DIR (branch $PX4_BRANCH)"
    echo

    install_system_deps

    if (( INSTALL_ARDUPILOT )); then install_ardupilot; fi
    if (( INSTALL_PX4 ));          then install_px4;          fi

    install_python_deps

    echo
    log "Done. Next steps:"
    log "  1. Edit Cptool/config.yaml 'paths:' to match the directories above"
    log "     (they match the defaults: $ARDUPILOT_DIR / $PX4_DIR)."
    log "  2. Set 'mode: PX4' or 'mode: Ardupilot' in Cptool/config.yaml."
    log "  3. Verify:  poetry run python3 0.collect.py   (ArduPilot)"
    log "              poetry run python3 0.collect_px4.py   (PX4)"
}

main "$@"
