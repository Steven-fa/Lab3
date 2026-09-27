#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
scenario="${1:-normal}"
if [ "$#" -gt 0 ]; then shift; fi
run_dir="${TASK3_RUN_DIR:-$TASK3_BUNDLE/results/${scenario}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run_dir"
exec ros2 launch mecharm_task3 full_sorting.launch.py scenario:="$scenario" run_dir:="$run_dir" gui:="${SIM_GUI:-true}" "$@"
