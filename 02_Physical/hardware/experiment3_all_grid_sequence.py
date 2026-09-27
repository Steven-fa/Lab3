"""Run the six Experiment 3 grid transfers in one operator-gated session."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import yaml

from experiment3_transfer_test import (
    DESTINATION_BY_CLASS,
    TransferRunner,
    build_high_retreat,
    build_pre_descent_transition,
    effective_retreat_extension,
    grasp_z_tolerance,
    highest_recorded_tcp_z,
    load_transfer_route,
)
from experiment3_gripper_profiles import load_gripper_profiles, profile_for_group
from mecharm_recorded_task import acquire_experiment3_control_lock


DEFAULT_GRID_CLASSES = ("other", "cube", "cube", "other", "cube", "cube")
INTER_GRID_RECONNECT_SETTLE_S = 1.0


def build_plan(
    poses_path: Path,
    classes: tuple[str, ...],
    retreat_extension: float,
    gripper_profiles_path: Path,
) -> tuple[list[dict], float]:
    if len(classes) != 6:
        raise RuntimeError("必须为1～6号格分别提供一个类别")
    invalid = [value for value in classes if value not in DESTINATION_BY_CLASS]
    if invalid:
        raise RuntimeError(f"无效类别：{invalid}；只能使用cube或other")

    document = yaml.safe_load(poses_path.read_text(encoding="utf-8"))
    highest_z = highest_recorded_tcp_z(document)
    class_profiles = load_gripper_profiles(gripper_profiles_path)
    plan: list[dict] = []
    for grid, object_class in enumerate(classes, start=1):
        destination = DESTINATION_BY_CLASS[object_class]
        poses, _recorded_gripper = load_transfer_route(
            poses_path,
            grid,
            destination,
            require_active_correction=True,
        )
        gripper = profile_for_group(class_profiles, object_class)
        source_approach = poses[f"grid_{grid}_approach"]
        source_grasp = poses[f"grid_{grid}_grasp"]
        destination_approach = poses[f"{destination}_approach"]
        destination_release = poses[f"{destination}_release"]
        extension = effective_retreat_extension(
            destination_approach,
            destination_release,
            retreat_extension,
        )
        # Construct every generated pose during preflight so an invalid later
        # route is rejected before the operator authorises any motion.
        build_high_retreat(destination_approach, destination_release, extension)
        build_pre_descent_transition(
            destination_approach,
            destination_release,
            minimum_absolute_z_mm=highest_z,
        )
        plan.append(
            {
                "grid": grid,
                "object_class": object_class,
                "destination": destination,
                "poses": poses,
                "gripper": gripper,
                "retreat_extension": extension,
                "source_lift_mm": float(
                    source_approach["coords"][2] - source_grasp["coords"][2]
                ),
                "source_grasp_raise_mm": float(
                    source_grasp.get("requested_extra_z_mm", 0.0)
                    if source_grasp.get("generated_source_grasp") is True
                    else 0.0
                ),
                "grasp_z_tolerance_mm": grasp_z_tolerance(source_grasp),
            }
        )
    return plan, highest_z


def plan_from_grid(plan: list[dict], start_grid: int) -> list[dict]:
    """Select a safe resume suffix without altering any per-grid route."""
    selected = [item for item in plan if int(item["grid"]) >= start_grid]
    if not selected or int(selected[0]["grid"]) != start_grid:
        raise RuntimeError(f"起始格必须在1～6之间，收到{start_grid}")
    return selected


def run_plan_with_fresh_sessions(
    plan: list[dict],
    highest_z: float,
    host: str,
    port: int,
    arm_speed: int,
    log_dir: Path,
    stream_step: float,
    stream_period: float,
    input_fn=input,
    runner_factory=TransferRunner,
    reconnect_settle_s: float = INTER_GRID_RECONNECT_SETTLE_S,
    auto_next_delay_s: float | None = None,
) -> tuple[int, list[Path]]:
    """Run every grid through a newly connected, explicitly closed client."""
    logs: list[Path] = []
    for index, item in enumerate(plan):
        if index:
            print("[换格] 上一格已返回safe并关闭控制连接。请立即移走释放区物体。")
            if auto_next_delay_s is None:
                next_confirmation = input_fn(
                    f"确认释放区和路径为空，输入 NEXT 执行{item['grid']}号格："
                ).strip().upper()
                if next_confirmation != "NEXT":
                    print(f"[取消] 已在safe停止；尚未执行{item['grid']}号格及后续格。")
                    return 2, logs
            else:
                print(
                    f"[自动继续] 留出{auto_next_delay_s:.1f}s清理释放区；"
                    f"随后自动执行{item['grid']}号格，无需输入NEXT。"
                )
                if auto_next_delay_s > 0:
                    time.sleep(auto_next_delay_s)
            if reconnect_settle_s > 0:
                time.sleep(reconnect_settle_s)

        runner = runner_factory(
            host,
            port,
            arm_speed,
            log_dir,
            stream_step,
            stream_period,
        )
        try:
            current = runner.health()
            print(
                f"[会话隔离] 已为{item['grid']}号格建立全新连接；"
                f"当前角度={np.round(current, 1).tolist()}"
            )
            if index == 0:
                confirmation = input_fn(
                    "确认全部布置正确并准备急停，输入 RUN ALL："
                ).strip().upper()
                if confirmation != "RUN ALL":
                    print("[取消] 未发送运动或夹爪命令。")
                    return 2, logs

            print(
                f"\n[序列 {item['grid']}/6] 开始：{item['grid']}号格 "
                f"{item['object_class']} -> {item['destination']}"
            )
            runner.run_transfer(
                current,
                item["poses"],
                item["gripper"],
                item["grid"],
                item["destination"],
                int(item["gripper"]["speed"]),
                item["retreat_extension"],
                highest_z,
            )
            logs.append(Path(runner.log_path))
        except Exception:
            if not runner.stopped:
                runner.stop()
            raise
        finally:
            runner.arm.close()
            print(f"[会话隔离] {item['grid']}号格连接已显式关闭并清理。")

    return 0, logs


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 six-grid sequence")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs=6,
        choices=tuple(DESTINATION_BY_CLASS),
        default=DEFAULT_GRID_CLASSES,
        metavar=("G1", "G2", "G3", "G4", "G5", "G6"),
    )
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument(
        "--gripper-profiles",
        type=Path,
        default=Path(
            "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
        ),
    )
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument("--retreat-extension", type=float, default=0.50)
    parser.add_argument("--start-grid", type=int, choices=range(1, 7), default=1)
    parser.add_argument(
        "--auto-next-delay",
        type=float,
        help="每格回到safe并关闭连接后，等待指定秒数并自动继续",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.auto_next_delay is not None and args.auto_next_delay < 0:
        parser.error("--auto-next-delay不能小于0")

    classes = tuple(str(value).lower() for value in args.classes)
    full_plan, highest_z = build_plan(
        args.poses,
        classes,
        args.retreat_extension,
        args.gripper_profiles,
    )
    plan = plan_from_grid(full_plan, args.start_grid)
    selected_grids = " -> ".join(str(item["grid"]) for item in plan)
    print(f"[预检] 实验3连续搬运：{selected_grids}")
    print(f"[数据集] {args.poses.resolve()}")
    print(f"[全局高点] TCP Z={highest_z:.1f}mm")
    for item in plan:
        print(
            f"[计划] {item['grid']}号格 {item['object_class']} -> "
            f"{item['destination']}；采集抬升={item['source_lift_mm']:.1f}mm；"
            f"抓取Z容差={item['grasp_z_tolerance_mm']:.1f}mm；"
            f"夹爪OPEN={item['gripper']['open']}、CLOSE={item['gripper']['close']}"
        )
        if item["source_grasp_raise_mm"] > 0.0:
            print(
                f"[计划] {item['grid']}号格抓取点沿人工下降路径上调"
                f"{item['source_grasp_raise_mm']:.1f}mm"
            )
    if args.auto_next_delay is None:
        print("[防碰撞] 每格完成并回safe后暂停；移走释放物，输入NEXT继续下一格。")
    else:
        print(
            f"[自动换格] 每格完成并回safe、关闭连接后等待"
            f"{args.auto_next_delay:.1f}s；请在此期间立即移走释放物。"
        )
    if args.check_only:
        print("[只读检查] 六条路线全部合法；未连接机械臂，未发送命令。")
        return 0

    print("[现场] 1～6号物体必须放好；左右释放点初始为空；全路径清空。")
    print("[类别] 1 other、2 cube、3 cube、4 other、5 cube、6 cube。")
    print("[会话隔离] 每格使用独立TCP连接；完成后立即关闭并清理桥接缓冲。")
    control_lock = acquire_experiment3_control_lock("six-grid-sequence")
    try:
        result, logs = run_plan_with_fresh_sessions(
            plan,
            highest_z,
            args.host,
            args.port,
            args.arm_speed,
            args.poses.parent,
            args.stream_step,
            args.stream_period,
            auto_next_delay_s=args.auto_next_delay,
        )
    finally:
        control_lock.close()
    if result == 0:
        print(
            f"[序列结束] {args.start_grid}～6号格已全部成功搬运；"
            "任何一格未完成都会阻止后续格执行。"
        )
        print("[分格记录] " + ", ".join(str(path) for path in logs))
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止六格连续搬运。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
