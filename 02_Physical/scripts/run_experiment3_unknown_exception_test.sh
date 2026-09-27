#!/usr/bin/env bash
set -eo pipefail

# ROS setup scripts legitimately probe variables before defining them, so
# enable nounset only after both environments have been loaded.
source /opt/ros/humble/setup.bash
source /home/nvidia/mecharm_experiment3_ws/install/setup.bash
set -u

ROOT=/home/nvidia/mecharm_tasks/experiment3
REGIONS="$ROOT/config/regions.yaml"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/ros2_runs/unknown_test_${STAMP}.jsonl"
CONSOLE="$ROOT/ros2_runs/unknown_test_${STAMP}.console.log"
PUBLISHER=/home/nvidia/mecharm_tasks/scripts/experiment3_unknown_exception_test.py

mkdir -p "$ROOT/ros2_runs"

if pgrep -f '[e]xperiment3_robot_worker' >/dev/null; then
  echo "[拒绝] 检测到机械臂工作节点仍在运行；请先停止正式任务。"
  exit 4
fi

controller_pid=""
cleanup() {
  if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
    kill -INT "$controller_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$controller_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$controller_pid" 2>/dev/null; then
      kill -TERM "$controller_pid" 2>/dev/null || true
    fi
    for _ in $(seq 1 10); do
      kill -0 "$controller_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$controller_pid" 2>/dev/null; then
      kill -KILL "$controller_pid" 2>/dev/null || true
    fi
    wait "$controller_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[安全模式] 只启动DISARMED状态机，不启动机械臂工作节点。"
echo "[测试] 向1号区域发布不受支持类别 unrecognized_object。"
ros2 run mecharm_pick_place experiment3_task_controller --ros-args \
  -p armed:=false \
  -p regions_path:="$REGIONS" \
  -p history_size:=7 \
  -p stable_votes:=5 \
  -p idle_complete_seconds:=30.0 \
  -p log_path:="$LOG" >"$CONSOLE" 2>&1 &
controller_pid=$!

sleep 2
python3 "$PUBLISHER" --regions "$REGIONS" --grid 1 --messages 20
sleep 2
cleanup
controller_pid=""
trap - EXIT INT TERM

python3 - "$LOG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
events = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
unknown = [
    event
    for event in events
    if event.get("event") == "region_skipped"
    and event.get("state") == "UNKNOWN"
    and event.get("action") == "SKIP_NO_MOTION"
]
requests = [event for event in events if event.get("event") == "transfer_requested"]
started = [event for event in events if event.get("event") == "controller_started"]

if not started or started[-1].get("armed") is not False:
    raise SystemExit("[FAIL] 状态机没有以DISARMED模式启动")
if not unknown:
    raise SystemExit("[FAIL] 没有取得 UNKNOWN -> SKIP_NO_MOTION 记录")
if requests:
    raise SystemExit(f"[FAIL] UNKNOWN测试期间产生了{len(requests)}条运动请求")

print("[PASS] UNKNOWN -> SKIP_NO_MOTION")
print("[PASS] transfer_requested=0；未启动机械臂工作节点，未发送运动命令")
print(f"[日志] {path}")
PY

echo "[控制台] $CONSOLE"
