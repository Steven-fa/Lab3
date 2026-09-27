#!/usr/bin/env python3
"""Closed-loop Experiment 3: YOLO regions -> fixed-pose robot transfers.

The operator authorises the complete run once.  After that, grid selection,
class routing and task progression are driven only by stable camera results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np
import yaml

from experiment3_gripper_profiles import load_gripper_profiles, profile_for_group
from mecharm_recorded_task import acquire_experiment3_control_lock

DESTINATION_BY_CLASS = {"cube": "right", "other": "left"}
READY_GROUP = {"READY_CUBE": "cube", "READY_OTHER": "other"}


def select_ready_target(regions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the lowest-numbered stable target, including replenished cells."""
    for region in sorted(regions, key=lambda value: int(value.get("grid_id", 999))):
        grid = int(region.get("grid_id", 0))
        group = READY_GROUP.get(str(region.get("state")))
        if group is None:
            continue
        if region.get("group") != group:
            continue
        return {
            "grid_id": grid,
            "class_name": str(region.get("class_name") or group),
            "group": group,
            "destination": DESTINATION_BY_CLASS[group],
            "confidence": float(region.get("confidence") or 0.0),
        }
    return None


def settled_without_ready_targets(regions: list[dict[str, Any]]) -> bool:
    """The continuous sorter finishes only when all six cells are stably empty."""
    return bool(regions) and all(
        str(region.get("state")) == "EMPTY" for region in regions
    )


def wait_for_source_resolution(
    vision: Any,
    grid: int,
    timeout_seconds: float,
    poll_seconds: float = 0.25,
) -> str | None:
    """Return EMPTY or READY_* after a transfer; READY_* means immediate reload.

    A fixed camera cannot distinguish an unremoved original object from a new
    object placed into the same cell.  Both must be handled by scheduling the
    visible object again, not by terminating the entire sorter.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        regions = list(vision.get_status().get("regions") or [])
        state = next(
            (
                str(region.get("state"))
                for region in regions
                if int(region.get("grid_id", 0)) == grid
            ),
            "CANDIDATE",
        )
        if state == "EMPTY" or state in READY_GROUP:
            return state
        time.sleep(poll_seconds)
    return None


class RunJournal:
    """Durable task-level log joining detections, motion logs and exceptions."""

    def __init__(self, path: Path, video_path: Path) -> None:
        self.path = path
        self.document: dict[str, Any] = {
            "format": "mecharm_experiment3_autonomous_run/v1",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "updated_at": None,
            "status": "starting",
            "video": str(video_path),
            "events": [],
            "objects": [],
            "summary": {},
        }
        self.save()

    def save(self) -> None:
        self.document["updated_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        next_path = self.path.with_suffix(self.path.suffix + ".next")
        with next_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.document, stream, allow_unicode=True, sort_keys=False)
        next_path.replace(self.path)

    def event(self, event: str, **values: Any) -> None:
        self.document["events"].append(
            {
                "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "event": event,
                **values,
            }
        )
        self.save()

    def begin_object(self, target: dict[str, Any]) -> dict[str, Any]:
        item = {
            **target,
            "detected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "executing",
            "motion_log": None,
            "error": None,
        }
        self.document["objects"].append(item)
        self.save()
        return item

    def finish(self, status: str) -> None:
        objects = self.document["objects"]
        completed = [item for item in objects if item.get("status") == "completed"]
        failed = [item for item in objects if item.get("status") == "failed"]
        skipped = [
            event
            for event in self.document["events"]
            if event.get("event") in {"unknown_skipped", "multiple_skipped"}
        ]
        self.document["status"] = status
        self.document["summary"] = {
            "detected_objects": len(objects),
            "completed_transfers": len(completed),
            "failed_transfers": len(failed),
            "skipped_regions": len(skipped),
            "correct_destination_count": sum(
                item.get("destination") == DESTINATION_BY_CLASS.get(item.get("group"))
                for item in completed
            ),
        }
        self.save()


def validate_preflight(
    poses_path: Path,
    model_path: Path,
    shape_model_path: Path,
    red_orange_model_path: Path,
    regions_path: Path,
    gripper_profiles_path: Path,
    retreat_extension: float,
) -> float:
    from experiment3_transfer_test import (
        build_high_retreat,
        build_pre_descent_transition,
        effective_retreat_extension,
        highest_recorded_tcp_z,
        load_transfer_route,
    )
    from experiment3_vision_preview import load_regions

    for path, label in (
        (poses_path, "17点姿态"),
        (model_path, "YOLO模型"),
        (shape_model_path, "形状复核模型"),
        (red_orange_model_path, "红色长方体/桥形二级分类模型"),
        (regions_path, "六区域配置"),
        (gripper_profiles_path, "分类夹爪配置"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label}不存在：{path}")
    regions = load_regions(regions_path)
    if [int(region["id"]) for region in regions] != list(range(1, 7)):
        raise RuntimeError("六区域配置必须完整包含GRID 1～6")
    load_gripper_profiles(gripper_profiles_path)

    document = yaml.safe_load(poses_path.read_text(encoding="utf-8"))
    highest_z = highest_recorded_tcp_z(document)
    for grid in range(1, 7):
        for destination in ("left", "right"):
            poses, _gripper = load_transfer_route(
                poses_path,
                grid,
                destination,
                require_active_correction=True,
            )
            approach = poses[f"{destination}_approach"]
            release = poses[f"{destination}_release"]
            extension = effective_retreat_extension(
                approach, release, retreat_extension
            )
            build_high_retreat(approach, release, extension)
            build_pre_descent_transition(
                approach,
                release,
                minimum_absolute_z_mm=highest_z,
            )
    return highest_z


def execute_target(
    target: dict[str, Any],
    *,
    poses_path: Path,
    highest_z: float,
    host: str,
    port: int,
    arm_speed: int,
    gripper_profiles_path: Path,
    stream_step: float,
    stream_period: float,
    retreat_extension: float,
) -> Path:
    from experiment3_transfer_test import (
        TransferRunner,
        effective_retreat_extension,
        load_transfer_route,
    )

    grid = int(target["grid_id"])
    destination = str(target["destination"])
    poses, _recorded_gripper = load_transfer_route(
        poses_path,
        grid,
        destination,
        require_active_correction=True,
    )
    gripper = profile_for_group(
        load_gripper_profiles(gripper_profiles_path), str(target["group"])
    )
    approach = poses[f"{destination}_approach"]
    release = poses[f"{destination}_release"]
    extension = effective_retreat_extension(
        approach, release, retreat_extension
    )
    runner = TransferRunner(
        host,
        port,
        arm_speed,
        poses_path.parent,
        stream_step,
        stream_period,
    )
    try:
        current = runner.health()
        print(
            f"[自动任务] GRID {grid} {target['class_name']} / {target['group']} "
            f"-> {destination}，置信度={target['confidence']:.3f}"
        )
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        runner.run_transfer(
            current,
            poses,
            gripper,
            grid,
            destination,
            int(gripper["speed"]),
            extension,
            highest_z,
        )
        return Path(runner.log_path)
    except Exception:
        if not runner.stopped:
            runner.stop()
        raise
    finally:
        runner.arm.close()
        print(f"[会话隔离] GRID {grid}连接已关闭并清理。")


def run_controller(args: argparse.Namespace) -> int:
    from experiment3_vision_preview import Handler, Vision, VisionServer

    highest_z = validate_preflight(
        args.poses,
        args.model,
        args.shape_model,
        args.red_orange_model,
        args.regions,
        args.gripper_profiles,
        args.retreat_extension,
    )
    print("[预检] 两类、六格、左右释放区共12条可能路线全部合法。")
    print(f"[预检] 17点全局最高TCP Z={highest_z:.1f}mm")
    if args.check_only:
        print("[只读检查] 未打开相机、未连接机械臂、未发送运动命令。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_root / f"autonomous_{stamp}"
    video_path = run_dir / "annotated_acceptance.mp4"
    journal = RunJournal(run_dir / "autonomous_run.yaml", video_path)
    vision = Vision(
        args.device,
        args.width,
        args.height,
        args.fps,
        args.model,
        args.shape_model,
        args.red_orange_model,
        args.regions,
        args.confidence,
        args.shape_confidence,
        args.red_orange_confidence,
        args.red_orange_margin,
        args.cuda_device,
        args.history_size,
        args.stable_votes,
        run_dir / "vision_requests.jsonl",
        dry_run=False,
        robot_connected=True,
        plan_status="QUEUED",
        video_path=video_path,
    )
    server: ThreadingHTTPServer = VisionServer(("0.0.0.0", args.preview_port), Handler)
    server.vision = vision  # type: ignore[attr-defined]
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
    )
    interrupted = False

    def stop_runtime(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop_runtime)
    signal.signal(signal.SIGTERM, stop_runtime)
    vision.start()
    server_thread.start()
    vision.set_runtime_status(task_state="WAITING_FOR_VISION", current_grid=None)
    journal.document["status"] = "waiting_for_vision"
    journal.save()
    print(f"[实时画面] http://0.0.0.0:{args.preview_port}/vision")
    print(f"[运行记录] {journal.path}")
    print(f"[验收录像] {video_path}")

    completed_grids: set[int] = set()
    completed_transfer_count = 0
    logged_non_targets: set[tuple[int, str]] = set()
    try:
        deadline = time.monotonic() + args.vision_start_timeout
        while time.monotonic() < deadline and not interrupted:
            status = vision.get_status()
            if status.get("error"):
                raise RuntimeError(f"视觉启动失败：{status['error']}")
            if status.get("model_ready") and status.get("regions") and vision.jpeg:
                break
            time.sleep(0.25)
        else:
            if interrupted:
                raise KeyboardInterrupt
            raise RuntimeError("视觉模型或摄像头未在规定时间内就绪")

        print(f"[视觉] 已稳定启动；等待{args.initial_scan_seconds:.1f}s完成六区投票。")
        time.sleep(args.initial_scan_seconds)
        print("[安全] 六个物体摆入对应网格；左右释放区和全部运动路径必须清空。")
        print("[说明] 只需授权一次；之后类别、格号和左右去向全部由视觉决定。")
        confirmation = input("确认现场安全，输入 RUN AUTO：").strip().upper()
        if confirmation != "RUN AUTO":
            journal.event("operator_cancelled")
            journal.finish("cancelled")
            print("[取消] 未连接机械臂、未发送任何运动命令。")
            return 2

        control_lock = acquire_experiment3_control_lock("legacy-autonomous-task")
        vision.set_runtime_status(task_state="SCANNING", current_grid=None)
        journal.document["status"] = "running"
        journal.event("operator_authorised")
        idle_since: float | None = None

        while not interrupted and (
            args.max_objects == 0 or completed_transfer_count < args.max_objects
        ):
            status = vision.get_status()
            if status.get("error"):
                raise RuntimeError(f"视觉运行失败：{status['error']}")
            regions = list(status.get("regions") or [])
            for region in regions:
                state = str(region.get("state"))
                grid = int(region.get("grid_id", 0))
                key = (grid, state)
                if state in {"UNKNOWN", "MULTIPLE"} and key not in logged_non_targets:
                    logged_non_targets.add(key)
                    event_name = "unknown_skipped" if state == "UNKNOWN" else "multiple_skipped"
                    journal.event(
                        event_name,
                        grid_id=grid,
                        state=state,
                        action="SKIP_NO_MOTION",
                    )
                    print(f"[跳过] GRID {grid} 状态={state}；不发送机械臂命令。")

            target = select_ready_target(regions)
            if target is None:
                if completed_transfer_count > 0 and settled_without_ready_targets(regions):
                    idle_since = idle_since or time.monotonic()
                    remaining = args.idle_complete_seconds - (time.monotonic() - idle_since)
                    vision.set_runtime_status(
                        task_state=f"NO_TARGETS_{max(0.0, remaining):.1f}s",
                        current_grid=None,
                    )
                    if remaining <= 0:
                        break
                else:
                    idle_since = None
                    vision.set_runtime_status(task_state="SCANNING", current_grid=None)
                time.sleep(0.25)
                continue

            idle_since = None
            item = journal.begin_object(target)
            grid = int(target["grid_id"])
            vision.set_runtime_status(task_state="EXECUTING", current_grid=grid)
            journal.event("transfer_started", **target)
            try:
                motion_log = execute_target(
                    target,
                    poses_path=args.poses,
                    highest_z=highest_z,
                    host=args.host,
                    port=args.port,
                    arm_speed=args.arm_speed,
                    gripper_profiles_path=args.gripper_profiles,
                    stream_step=args.stream_step,
                    stream_period=args.stream_period,
                    retreat_extension=args.retreat_extension,
                )
                vision.set_runtime_status(
                    task_state="VERIFYING_SOURCE_CLEAR", current_grid=grid
                )
                source_state = wait_for_source_resolution(
                    vision, grid, args.source_clear_timeout
                )
                if source_state is None:
                    raise RuntimeError(
                        f"GRID {grid}动作结束后视觉状态持续不稳定；"
                        "等待EMPTY或READY状态超时"
                    )
            except Exception as error:
                item["status"] = "failed"
                item["error"] = str(error)
                journal.event("transfer_failed", grid_id=grid, error=str(error))
                journal.finish("failed_safe_stop")
                vision.set_runtime_status(task_state="FAILED_SAFE_STOP", current_grid=grid)
                raise

            item["status"] = "completed"
            item["motion_log"] = str(motion_log)
            completed_grids.add(grid)
            completed_transfer_count += 1
            journal.event(
                "transfer_completed",
                grid_id=grid,
                class_name=target["class_name"],
                group=target["group"],
                destination=target["destination"],
                motion_log=str(motion_log),
                source_state_after=source_state,
            )
            if source_state in READY_GROUP:
                journal.event(
                    "source_ready_requeued",
                    grid_id=grid,
                    state=source_state,
                )
                print(
                    f"[连续补料] GRID {grid}已出现{source_state}；"
                    "作为下一件重新排队，不判为抓取失败。"
                )
            vision.set_runtime_status(task_state="CLEAR_RELEASE_AREA", current_grid=None)
            print(
                f"[自动换格] GRID {grid}完成；请立即移走释放区物体。"
                f"{args.inter_object_delay:.1f}s后继续扫描，无需输入NEXT。"
            )
            time.sleep(args.inter_object_delay)

        if interrupted:
            raise KeyboardInterrupt
        journal.finish("completed")
        vision.set_runtime_status(task_state="COMPLETED", current_grid=None)
        summary = journal.document["summary"]
        print(
            f"[完成] 自动闭环结束：识别{summary['detected_objects']}个，"
            f"成功搬运{summary['completed_transfers']}个，"
            f"正确路由{summary['correct_destination_count']}个。"
        )
        print(f"[综合记录] {journal.path}")
        print(f"[验收录像] {video_path}")
        return 0
    except KeyboardInterrupt:
        journal.event("operator_stop")
        journal.finish("operator_stopped")
        vision.set_runtime_status(task_state="OPERATOR_STOP", current_grid=None)
        print("[停止] 操作员中止自动任务；当前机械臂动作已停止或由搬运器安全退出。")
        return 130
    finally:
        if "control_lock" in locals():
            control_lock.close()
        vision.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3 autonomous vision-to-motion task")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shape-model", type=Path, required=True)
    parser.add_argument("--red-orange-model", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument(
        "--gripper-profiles",
        type=Path,
        default=Path(
            "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-port", type=int, default=8091)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--shape-confidence", type=float, default=0.55)
    parser.add_argument("--red-orange-confidence", type=float, default=0.80)
    parser.add_argument("--red-orange-margin", type=float, default=0.30)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--history-size", type=int, default=7)
    parser.add_argument("--stable-votes", type=int, default=5)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument("--retreat-extension", type=float, default=0.50)
    parser.add_argument("--initial-scan-seconds", type=float, default=3.0)
    parser.add_argument("--idle-complete-seconds", type=float, default=5.0)
    parser.add_argument("--inter-object-delay", type=float, default=5.0)
    parser.add_argument("--source-clear-timeout", type=float, default=5.0)
    parser.add_argument("--vision-start-timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="0表示持续处理，直到六个区域稳定为空",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.width, args.height) != (1280, 720):
        raise RuntimeError("实验三视觉分辨率必须固定为1280x720")
    if not 1 <= args.stable_votes <= args.history_size:
        raise RuntimeError("稳定票数必须位于1和历史窗口大小之间")
    if args.max_objects < 0:
        raise RuntimeError("max-objects不能小于0")
    if args.source_clear_timeout <= 0.0:
        raise RuntimeError("source-clear-timeout必须大于0")
    return run_controller(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
