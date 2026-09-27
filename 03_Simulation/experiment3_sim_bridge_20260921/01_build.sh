#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
mkdir -p "$TASK3_WS"
cd "$TASK3_WS"
colcon build --base-paths "$TASK3_BUNDLE/src" --packages-select mycobot_description mecharm_270_m5_moveit_config mecharm_task2 mecharm_task3 --cmake-args -DCMAKE_BUILD_TYPE=Release
echo "BUILD_OK $TASK3_WS"
