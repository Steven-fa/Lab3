#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/nvidia/mecharm_experiment3_ws/install/setup.bash
set -u

ROOT=/home/nvidia/mecharm_tasks/experiment3
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/ros2_runs/physical_unknown_${STAMP}.jsonl"
CONSOLE="$ROOT/ros2_runs/physical_unknown_${STAMP}.console.log"
EVIDENCE="$ROOT/ros2_runs/physical_unknown_${STAMP}_evidence"
SCANNER=/home/nvidia/mecharm_tasks/scripts/experiment3_physical_unknown_test.py
MODEL="$ROOT/models/cube6_red_focus_v4_video_negative_best.pt"
REGIONS="$ROOT/config/regions.yaml"

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
    kill -0 "$controller_pid" 2>/dev/null && kill -TERM "$controller_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$controller_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$controller_pid" 2>/dev/null && kill -KILL "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[安全模式] DISARMED状态机；不启动robot worker，不能产生机械臂运动。"
baseline_args=()
if [[ "${1:-}" == "reuse" ]]; then
  previous_baseline="$(find "$ROOT/ros2_runs" -maxdepth 2 -type f -name empty_baseline.jpg -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$previous_baseline" || ! -f "$previous_baseline" ]]; then
    echo "[FAIL] 没有可复用的空格基线"
    exit 5
  fi
  baseline_args=(--baseline "$previous_baseline")
  echo "[实物测试] 复用上一轮空格基线，直接检查当前已放置物体。"
else
  echo "[实物测试] 先自动记录空格基线，再请你放入训练类别之外的物体。"
fi
ros2 run mecharm_pick_place experiment3_task_controller --ros-args \
  -p armed:=false \
  -p regions_path:="$REGIONS" \
  -p history_size:=7 \
  -p stable_votes:=5 \
  -p idle_complete_seconds:=60.0 \
  -p log_path:="$LOG" >"$CONSOLE" 2>&1 &
controller_pid=$!

sleep 2
python3 "$SCANNER" --regions "$REGIONS" --model "$MODEL" --grid 1 --evidence-dir "$EVIDENCE" "${baseline_args[@]}"
sleep 2
cleanup
controller_pid=""
trap - EXIT INT TERM

python3 - "$LOG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
unknown = [
    event for event in events
    if event.get("event") == "region_skipped"
    and event.get("state") == "UNKNOWN"
    and event.get("action") == "SKIP_NO_MOTION"
]
requests = [event for event in events if event.get("event") == "transfer_requested"]
started = [event for event in events if event.get("event") == "controller_started"]
if not started or started[-1].get("armed") is not False:
    raise SystemExit("[FAIL] 状态机没有以DISARMED模式启动")
if not unknown:
    raise SystemExit("[FAIL] 实物测试没有取得UNKNOWN -> SKIP_NO_MOTION记录")
if requests:
    raise SystemExit(f"[FAIL] 实物UNKNOWN测试期间产生了{len(requests)}条运动请求")
print("[PASS] physical UNKNOWN -> SKIP_NO_MOTION")
print("[PASS] transfer_requested=0；robot worker未启动；机械臂没有运动")
print(f"[日志] {path}")
PY

echo "[证据] $EVIDENCE"
echo "[控制台] $CONSOLE"
