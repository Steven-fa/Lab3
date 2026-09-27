"""Repeat the recorded grid-1 low-pose route with an open gripper.

This diagnostic records controller-reported joints and TCP at 1, 3 and 6
seconds after each measured endpoint command.  No endpoint correction is sent
inside a measurement window.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import yaml

from experiment3_grid_dry_run import GridDryRunner, interpolate, load_route
from pymycobot.common import ProtocolCode


MEASURED_STAGES = ("approach_in", "grasp_down", "approach_up")
TESTED_JOINTS = {"J2": 1, "J5": 4}


def build_cycle_plan(route: list[dict], repeats: int) -> list[dict[str, Any]]:
    if len(route) != 3:
        raise ValueError("路线必须包含safe、approach和grasp")
    if repeats < 1:
        raise ValueError("重复次数必须至少为1")
    safe, approach, grasp = route
    plan: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        plan.extend(
            [
                {
                    "repeat": repeat,
                    "stage": "approach_in",
                    "label": "从safe到1号格上方",
                    "pose": approach,
                    "measure": True,
                },
                {
                    "repeat": repeat,
                    "stage": "grasp_down",
                    "label": "下降到1号格抓取点",
                    "pose": grasp,
                    "measure": True,
                },
                {
                    "repeat": repeat,
                    "stage": "approach_up",
                    "label": "从抓取点原路抬回上方",
                    "pose": approach,
                    "measure": True,
                },
                {
                    "repeat": repeat,
                    "stage": "safe_return",
                    "label": "返回公共safe姿态",
                    "pose": safe,
                    "measure": False,
                },
            ]
        )
    return plan


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for stage in MEASURED_STAGES:
        stage_samples = [sample for sample in samples if sample["stage"] == stage]
        elapsed_values = sorted({float(sample["elapsed_s"]) for sample in stage_samples})
        repeats = {int(sample["repeat"]) for sample in stage_samples}
        if len(elapsed_values) < 2 or len(repeats) < 3:
            continue
        first_time, final_time = elapsed_values[0], elapsed_values[-1]
        first_samples = [
            sample for sample in stage_samples if float(sample["elapsed_s"]) == first_time
        ]
        final_samples = [
            sample for sample in stage_samples if float(sample["elapsed_s"]) == final_time
        ]
        if len(first_samples) < 3 or len(final_samples) < 3:
            continue

        joint_summary: dict[str, dict[str, float]] = {}
        for joint in TESTED_JOINTS:
            first_errors = [float(sample["joint_error_deg"][joint]) for sample in first_samples]
            final_errors = [float(sample["joint_error_deg"][joint]) for sample in final_samples]
            joint_summary[joint] = {
                "mean_error_1s_deg": round(float(np.mean(first_errors)), 3),
                "mean_error_6s_deg": round(float(np.mean(final_errors)), 3),
                "mae_1s_deg": round(float(np.mean(np.abs(first_errors))), 3),
                "mae_6s_deg": round(float(np.mean(np.abs(final_errors))), 3),
                "error_reduction_1s_to_6s_deg": round(
                    float(np.mean(np.abs(first_errors)) - np.mean(np.abs(final_errors))),
                    3,
                ),
                "repeat_spread_6s_deg": round(float(np.ptp(final_errors)), 3),
                "suggested_command_offset_deg": round(-float(np.mean(final_errors)), 3),
            }

        tcp_errors = np.asarray(
            [sample["tcp_error_mm_deg"] for sample in final_samples], dtype=float
        )
        tcp_mean = np.mean(tcp_errors, axis=0)
        tcp_spread = np.ptp(tcp_errors, axis=0)
        summary[stage] = {
            "first_sample_s": first_time,
            "final_sample_s": final_time,
            "joint": joint_summary,
            "tcp": {
                "mean_error_6s_mm_deg": [round(float(value), 3) for value in tcp_mean],
                "axis_repeat_spread_6s_mm_deg": [
                    round(float(value), 3) for value in tcp_spread
                ],
                "max_xyz_repeat_spread_6s_mm": round(
                    float(np.max(tcp_spread[:3])), 3
                ),
            },
        }
    return summary


def float_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in values]


class LowPoseRepeatRunner(GridDryRunner):
    def __init__(
        self,
        host: str,
        port: int,
        arm_speed: int,
        output_dir: Path,
        stream_step: float,
        stream_period: float,
        route: list[dict],
        gripper: dict,
        repeats: int,
        sample_times: list[float],
    ) -> None:
        super().__init__(host, port, arm_speed, output_dir, stream_step, stream_period)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = output_dir / f"grid1_low_pose_repeat_{stamp}.yaml"
        self.route = route
        self.gripper_config = gripper
        self.repeats = repeats
        self.sample_times = sample_times
        self.plan = build_cycle_plan(route, repeats)
        self.samples: list[dict[str, Any]] = []
        self.status = "created"

    def save_report(self) -> None:
        route_document = {}
        for pose in self.route:
            route_document[str(pose["name"])] = {
                "joint_angles_deg": float_list(np.asarray(pose["angles"], dtype=float)),
                "tcp_coords_mm_deg": float_list(np.asarray(pose["coords"], dtype=float)),
            }
        document = {
            "format": "mecharm_grid1_low_pose_repeat/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": self.status,
            "configuration": {
                "repeats": self.repeats,
                "sample_times_s": self.sample_times,
                "arm_speed": self.arm_speed,
                "stream_step_deg": self.stream_step,
                "stream_period_s": self.stream_period,
                "gripper_open": int(self.gripper_config["open"]),
                "route": route_document,
            },
            "samples": self.samples,
            "summary": summarize_samples(self.samples),
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(self.report_path)

    def stream_measured_stage(self, label: str, target: np.ndarray) -> float:
        if self.commanded_angles is None:
            raise RuntimeError("位置流尚未初始化")
        points = interpolate(self.commanded_angles, target, self.stream_step)
        print(f"[运动] {label}：{len(points)} 个连续小步")
        command_time = time.monotonic()
        for index, point in enumerate(points):
            if self.stopped:
                raise KeyboardInterrupt
            encoded = [self.arm._angle2int(float(angle)) for angle in point]
            self.arm._mesg(
                ProtocolCode.SEND_ANGLES,
                encoded,
                self.arm_speed,
                _async=True,
            )
            self.commanded_angles = point.copy()
            if index + 1 == len(points):
                command_time = time.monotonic()
            else:
                time.sleep(self.stream_period)
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "measured_position_stream",
                "label": label,
                "commands": len(points),
                "target_deg": float_list(target),
                "feedback_reads": 0,
            }
        )
        return command_time

    def measure_stage(self, step: dict[str, Any]) -> None:
        pose = step["pose"]
        target_angles = np.asarray(pose["angles"], dtype=float)
        target_coords = np.asarray(pose["coords"], dtype=float)
        label = f"第{step['repeat']}/{self.repeats}轮 {step['label']}"
        command_time = self.stream_measured_stage(label, target_angles)

        for sample_at in self.sample_times:
            remaining = command_time + sample_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            actual_angles = self.read_angles()
            angle_elapsed = time.monotonic() - command_time
            actual_coords = self.read_coords()
            tcp_elapsed = time.monotonic() - command_time
            joint_errors = actual_angles - target_angles
            tcp_errors = actual_coords - target_coords
            sample = {
                "stage": step["stage"],
                "label": step["label"],
                "repeat": int(step["repeat"]),
                "elapsed_s": float(sample_at),
                "actual_angle_elapsed_s": round(angle_elapsed, 3),
                "actual_tcp_elapsed_s": round(tcp_elapsed, 3),
                "target_angles_deg": float_list(target_angles),
                "actual_angles_deg": float_list(actual_angles),
                "all_joint_error_deg": float_list(joint_errors),
                "joint_error_deg": {
                    joint: round(float(joint_errors[index]), 3)
                    for joint, index in TESTED_JOINTS.items()
                },
                "target_tcp_mm_deg": float_list(target_coords),
                "actual_tcp_mm_deg": float_list(actual_coords),
                "tcp_error_mm_deg": float_list(tcp_errors),
            }
            sample.update(getattr(self, "sample_context", {}))
            self.samples.append(sample)
            print(
                f"[采样 {sample_at:g}s] J2误差={joint_errors[1]:+.2f}°，"
                f"J5误差={joint_errors[4]:+.2f}°，"
                f"TCP ΔXYZ={np.round(tcp_errors[:3], 1).tolist()}mm"
            )
            self.status = "running"
            self.save_report()

        error = self.read_error()
        if not self.error_is_clear(error):
            raise RuntimeError(f"{label}完成后控制器报错：{error}")

    def print_summary(self) -> None:
        summary = summarize_samples(self.samples)
        names = {
            "approach_in": "进入上方点",
            "grasp_down": "低位抓取点",
            "approach_up": "抬升返回上方点",
        }
        print("\n[三轮汇总] 以下TCP均为机械臂控制器回报值。")
        for stage in MEASURED_STAGES:
            item = summary[stage]
            j2 = item["joint"]["J2"]
            j5 = item["joint"]["J5"]
            tcp = item["tcp"]
            print(
                f"[{names[stage]}] 6s均值：J2={j2['mean_error_6s_deg']:+.2f}°，"
                f"J5={j5['mean_error_6s_deg']:+.2f}°，"
                f"TCP ΔXYZ={tcp['mean_error_6s_mm_deg'][:3]}mm；"
                f"三次最大XYZ极差={tcp['max_xyz_repeat_spread_6s_mm']:.1f}mm"
            )
            print(
                f"[{names[stage]}] 仅供下一步验证的候选关节补偿："
                f"J2={j2['suggested_command_offset_deg']:+.2f}°，"
                f"J5={j5['suggested_command_offset_deg']:+.2f}°"
            )
        approach_in = summary["approach_in"]["joint"]
        approach_up = summary["approach_up"]["joint"]
        print(
            "[方向对比] 同一上方点："
            f"J2方向差={abs(approach_in['J2']['mean_error_6s_deg'] - approach_up['J2']['mean_error_6s_deg']):.2f}°，"
            f"J5方向差={abs(approach_in['J5']['mean_error_6s_deg'] - approach_up['J5']['mean_error_6s_deg']):.2f}°"
        )
        print("[注意] 候选补偿尚未发送到机械臂，必须先看三轮离散再决定是否应用。")

    def run_repeat(self, current: np.ndarray) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        open_value = int(self.gripper_config["open"])
        open_speed = min(20, int(self.gripper_config.get("speed", 20)))
        self.gripper("低位诊断全程保持 OPEN", open_value, open_speed)

        safe = np.asarray(self.route[0]["angles"], dtype=float)
        self.stream_to("诊断开始回到公共safe姿态", safe)
        self.settle_at("初始safe姿态", safe, tolerance_deg=3.0, attempts=5)

        for step in self.plan:
            if self.stopped:
                raise KeyboardInterrupt
            if step["measure"]:
                self.measure_stage(step)
            else:
                target = np.asarray(step["pose"]["angles"], dtype=float)
                self.stream_to(f"第{step['repeat']}/{self.repeats}轮 {step['label']}", target)
                self.settle_at(
                    f"第{step['repeat']}/{self.repeats}轮 safe",
                    target,
                    tolerance_deg=3.0,
                    attempts=5,
                )

        self.status = "completed"
        self.save_report()
        self.print_summary()
        print(f"[完成] 1号格低位空载三轮测试结束。完整记录：{self.report_path}")


def run_self_test() -> None:
    route = [
        {"name": "safe", "angles": np.zeros(6), "coords": np.zeros(6)},
        {"name": "grid_1_approach", "angles": np.ones(6), "coords": np.ones(6)},
        {"name": "grid_1_grasp", "angles": np.full(6, 2.0), "coords": np.full(6, 2.0)},
    ]
    plan = build_cycle_plan(route, 3)
    assert len(plan) == 12
    assert [step["stage"] for step in plan[:4]] == [
        "approach_in",
        "grasp_down",
        "approach_up",
        "safe_return",
    ]
    print("SELF_TEST=PASS")


def print_preflight(route: list[dict], gripper: dict, repeats: int) -> None:
    safe, approach, grasp = route
    descent = approach["coords"][:3] - grasp["coords"][:3]
    print("[预检] 1号格低位空载重复定位测试")
    print(
        f"[路线] safe -> grid_1_approach -> grid_1_grasp -> "
        f"grid_1_approach -> safe，共{repeats}轮"
    )
    print(f"[safe] joints={np.round(safe['angles'], 1).tolist()}")
    print(f"[上方点] joints={np.round(approach['angles'], 1).tolist()}")
    print(f"[抓取点] joints={np.round(grasp['angles'], 1).tolist()}")
    print(
        f"[下降段] Z={descent[2]:.1f}mm，"
        f"XY={float(np.linalg.norm(descent[:2])):.1f}mm"
    )
    print(
        f"[测量] 每个上方/抓取/抬升端点在1、3、6秒记录J2、J5和TCP；"
        f"夹爪全程OPEN={gripper['open']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-1 low-pose repeatability test")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-times", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/diagnostics"),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.poses is None:
        raise RuntimeError("必须提供--poses")
    if args.repeats != 3:
        raise RuntimeError("低位诊断固定要求重复3轮")
    if args.sample_times != [1.0, 3.0, 6.0]:
        raise RuntimeError("低位诊断固定在1、3、6秒采样")
    if not 1 <= args.arm_speed <= 12:
        raise RuntimeError("低位诊断速度必须在1~12")
    if not 0.5 <= args.stream_step <= 2.5:
        raise RuntimeError("位置流步长必须在0.5~2.5度")
    if not 0.10 <= args.stream_period <= 0.4:
        raise RuntimeError("位置流周期必须在0.10~0.4秒")

    route, gripper = load_route(args.poses, grid=1)
    print_preflight(route, gripper, args.repeats)
    if args.check_only:
        print("[只读检查] 路线、姿态和限位均合法；未连接机械臂。")
        return 0

    runner = LowPoseRepeatRunner(
        args.host,
        args.port,
        args.arm_speed,
        args.output_dir,
        args.stream_step,
        args.stream_period,
        route,
        gripper,
        args.repeats,
        args.sample_times,
    )
    try:
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 必须移走1号格及整条路径上的所有物体，手放在急停旁。")
        print("[说明] 本次会到达已录制低位抓取点，但夹爪始终张开，不会抓取。")
        confirmation = input("确认路径为空，输入 RUN 并按 Enter 开始：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送任何运动或夹爪命令。")
            return 2
        runner.run_repeat(current)
        return 0
    except Exception:
        runner.status = "stopped"
        runner.save_report()
        if not runner.stopped:
            runner.stop()
        raise
    finally:
        runner.arm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止1号格低位重复定位测试。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
