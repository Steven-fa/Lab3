#!/usr/bin/env bash
TASK3_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK3_WS="${TASK3_WS:-$HOME/task3_bridge_ws_20260921}"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
if [ -f "$TASK3_WS/install/setup.bash" ]; then source "$TASK3_WS/install/setup.bash"; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-95}" ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11395}"
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="$TASK3_BUNDLE/models:/usr/share/gazebo-11/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_PLUGIN_PATH="$TASK3_WS/install/mecharm_task3/lib:$TASK3_WS/install/mecharm_task2/lib:${GAZEBO_PLUGIN_PATH:-}"
export PYTHONUNBUFFERED=1 LIBGL_ALWAYS_SOFTWARE=1
