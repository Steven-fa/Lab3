#!/usr/bin/env bash

set -u

readonly PROCESS_PATTERN='^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch mecharm_pick_place fast_replay.launch.py( .*)?$|^gzserver /home/tonyt/mecharm_ws/install/mecharm_pick_place/share/mecharm_pick_place/worlds/pick_place.world '

is_running() {
    pgrep -f "$PROCESS_PATTERN" >/dev/null 2>&1
}

process_groups() {
    local pid
    for pid in $(pgrep -f "$PROCESS_PATTERN" 2>/dev/null); do
        ps -o pgid= -p "$pid"
    done | tr -d ' ' | sort -nu
}

signal_groups() {
    local signal="$1"
    local pgid
    while read -r pgid; do
        if [[ -n "$pgid" ]]; then
            kill "-$signal" -- "-$pgid" 2>/dev/null || true
        fi
    done < <(process_groups)
}

wait_until_stopped() {
    local attempts="$1"
    local index
    for ((index = 0; index < attempts; index += 1)); do
        if ! is_running; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

if ! is_running; then
    exit 0
fi

signal_groups INT
wait_until_stopped 20 && exit 0

signal_groups TERM
wait_until_stopped 10 && exit 0

signal_groups KILL
wait_until_stopped 4
