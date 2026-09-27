"""Validate J2/J3/J4 directional compensation on the grid-1 low route.

This stage keeps the gripper open and never sends a grasp command.  Recorded
angles remain the physical targets; only outgoing controller commands receive
the validated direction-dependent corrections.
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

from experiment3_directional_compensation import (
    DirectionalJointCompensator,
    load_compensation,
)
from experiment3_grid1_low_pose_repeat import (
    LowPoseRepeatRunner,
    float_list,
    print_preflight,
)
from experiment3_grid_dry_run import GridDryRunner, load_route
from pymycobot.common import ProtocolCode


JOINT_NAMES = tuple(f"J{index}" for index in range(1, 7))
STAGE_NAMES = {
    "approach_in": "进入上方点",
    "grasp_down": "低位抓取点",
    "approach_up": "抬升返回上方点",
}


def compensated_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage, label in STAGE_NAMES.items():
        stage_samples = [sample for sample in samples if sample["stage"] == stage]
        if not stage_samples:
            continue
        final_time = max(float(sample["elapsed_s"]) for sample in stage_samples)
        final_samples = [
            sample for sample in stage_samples if float(sample["elapsed_s"]) == final_time
        ]
        if len(final_samples) < 3:
            continue
        joint_errors = np.asarray(
            [sample["all_joint_error_deg"] for sample in final_samples], dtype=float
        )
        tcp_errors = np.asarray(
            [sample["tcp_error_mm_deg"] for sample in final_samples], dtype=float
        )
        mean_joint = np.mean(joint_errors, axis=0)
        mean_tcp = np.mean(tcp_errors, axis=0)
        result[stage] = {
            "label": label,
            "final_sample_s": final_time,
            "mean_joint_error_deg": {
                name: round(float(mean_joint[index]), 3)
                for index, name in enumerate(JOINT_NAMES)
            },
            "mean_tcp_error_mm_deg": float_list(mean_tcp),
            "mean_tcp_xyz_error_norm_mm": round(
                float(np.mean(np.linalg.norm(tcp_errors[:, :3], axis=1))), 3
            ),
            "max_tcp_xyz_repeat_spread_mm": round(
                float(np.max(np.ptp(tcp_errors[:, :3], axis=0))), 3
            ),
        }
    return result


class CompensatedLowPoseRunner(LowPoseRepeatRunner):
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
        compensation_document: dict[str, Any],
    ) -> None:
        super().__init__(
            host,
            port,
            arm_speed,
            output_dir,
            stream_step,
            stream_period,
            route,
            gripper,
            repeats,
            sample_times,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = output_dir / f"grid1_compensated_low_pose_repeat_{stamp}.yaml"
        self.compensator = DirectionalJointCompensator(compensation_document)
        self.sample_context: dict[str, Any] = {}

    def save_report(self) -> None:
        super().save_report()
        document = yaml.safe_load(self.report_path.read_text(encoding="utf-8"))
        document["format"] = "mecharm_grid1_compensated_low_pose_repeat/v1"
        document["compensation"] = self.compensator.metadata()
        document["compensated_summary"] = compensated_summary(self.samples)
        temporary = self.report_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(self.report_path)

    def _set_context(
        self,
        desired: np.ndarray,
        command: np.ndarray,
        directions: dict[str, str],
        pose_name: str,
        compensation_applied: bool,
    ) -> None:
        self.sample_context = {
            "pose_name": pose_name,
            "compensation_applied": compensation_applied,
            "physical_target_deg": float_list(desired),
            "compensated_command_deg": float_list(command),
            "compensation_direction": dict(directions),
        }

    def _pose_name(self, desired: np.ndarray) -> str:
        matches = [
            str(pose["name"])
            for pose in self.route
            if np.allclose(np.asarray(pose["angles"], dtype=float), desired, atol=1e-6)
        ]
        if len(matches) != 1:
            raise RuntimeError("无法把物理目标唯一映射到记录姿态")
        return matches[0]

    def stream_to(self, label: str, target: np.ndarray) -> None:
        desired = np.asarray(target, dtype=float)
        pose_name = self._pose_name(desired)
        command, directions, applied = self.compensator.command_for_pose(
            desired, pose_name
        )
        self._set_context(desired, command, directions, pose_name, applied)
        mode = "应用" if applied else "低位旁路"
        print(
            f"[补偿{mode}] {label}："
            + ", ".join(
                f"{name} {directions[name]} {command[int(item['index']) - 1] - desired[int(item['index']) - 1]:+.3f}°"
                for name, item in self.compensator.joints.items()
            )
        )
        GridDryRunner.stream_to(self, label, command)

    def stream_measured_stage(self, label: str, target: np.ndarray) -> float:
        desired = np.asarray(target, dtype=float)
        pose_name = self._pose_name(desired)
        command, directions, applied = self.compensator.command_for_pose(
            desired, pose_name
        )
        self._set_context(desired, command, directions, pose_name, applied)
        mode = "应用" if applied else "低位旁路"
        print(
            f"[补偿{mode}] {label}："
            + ", ".join(
                f"{name}={command[int(item['index']) - 1] - desired[int(item['index']) - 1]:+.3f}°({directions[name]})"
                for name, item in self.compensator.joints.items()
            )
        )
        return LowPoseRepeatRunner.stream_measured_stage(self, label, command)

    def settle_at(
        self,
        label: str,
        target: np.ndarray,
        tolerance_deg: float = 3.0,
        attempts: int = 6,
        **_: Any,
    ) -> np.ndarray:
        """Passive endpoint check: never overwrite a compensated command."""
        desired = np.asarray(target, dtype=float)
        actual = desired.copy()
        for attempt in range(1, attempts + 1):
            time.sleep(1.0)
            actual = self.read_angles()
            maximum = float(np.max(np.abs(actual - desired)))
            print(
                f"[被动校验] {label} {attempt}/{attempts}："
                f"物理目标最大误差={maximum:.2f}°，未重发目标"
            )
            if maximum <= tolerance_deg:
                return actual
        raise RuntimeError(
            f"{label}被动等待{attempts}秒后仍未达到物理目标；"
            f"最大误差={float(np.max(np.abs(actual - desired))):.2f}°"
        )

    def establish_compensated_safe(self, safe: np.ndarray, approach: np.ndarray) -> None:
        if self.commanded_angles is None:
            raise RuntimeError("位置流尚未初始化")
        preload = self.compensator.preload_commands(
            safe, approach, self.commanded_angles
        )
        print("[预紧] 在safe附近依次建立J2/J3/J4的到达方向；每次只改变一个关节。")
        for item in preload:
            joint = str(item["joint"])
            direction = str(item["direction"])
            stage = str(item["stage"])
            command = np.asarray(item["angles"], dtype=float)
            label = f"{joint} {direction} {'预紧' if stage == 'preload' else '补偿回safe'}"
            GridDryRunner.stream_to(self, label, command)
            time.sleep(1.0)
            if stage == "preload":
                actual = self.read_angles()
                index = int(self.compensator.joints[joint]["index"]) - 1
                side = float(actual[index] - safe[index])
                correct = side <= -1.5 if direction == "increasing" else side >= 1.5
                print(f"[预紧校验] {joint}距物理safe={side:+.2f}°")
                if not correct:
                    raise RuntimeError(f"{joint}未到达建立{direction}方向所需的预紧侧")
        self._set_context(
            safe,
            np.asarray(preload[-1]["angles"], dtype=float),
            dict(self.compensator.directions),
            "safe",
            True,
        )
        self.settle_at("J2/J3/J4补偿后的safe", safe, tolerance_deg=3.0, attempts=6)

    def print_compensated_summary(self) -> None:
        summary = compensated_summary(self.samples)
        print("\n[低位补偿汇总] 误差均相对于原始17点中的物理目标，而不是补偿命令。")
        for stage in ("approach_in", "grasp_down", "approach_up"):
            item = summary[stage]
            joints = item["mean_joint_error_deg"]
            print(
                f"[{item['label']}] 6s：J2={joints['J2']:+.2f}°，"
                f"J3={joints['J3']:+.2f}°，J4={joints['J4']:+.2f}°，"
                f"J5={joints['J5']:+.2f}°，TCP均值={item['mean_tcp_error_mm_deg'][:3]}mm，"
                f"XYZ误差模={item['mean_tcp_xyz_error_norm_mm']:.1f}mm"
            )
        print("[判定] 请把本汇总发回；确认低位TCP改善后，才把补偿接入正式抓取。")

    def run_repeat(self, current: np.ndarray) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        open_value = int(self.gripper_config["open"])
        open_speed = min(20, int(self.gripper_config.get("speed", 20)))
        self.gripper("补偿低位验证全程保持 OPEN", open_value, open_speed)

        safe = np.asarray(self.route[0]["angles"], dtype=float)
        approach = np.asarray(self.route[1]["angles"], dtype=float)
        GridDryRunner.stream_to(self, "先回原始safe姿态", safe)
        GridDryRunner.settle_at(
            self, "原始safe姿态", safe, tolerance_deg=3.0, attempts=5
        )
        self.establish_compensated_safe(safe, approach)

        for step in self.plan:
            if self.stopped:
                raise KeyboardInterrupt
            if step["measure"]:
                self.measure_stage(step)
            else:
                target = np.asarray(step["pose"]["angles"], dtype=float)
                label = f"第{step['repeat']}/{self.repeats}轮 {step['label']}"
                self.stream_to(label, target)
                self.settle_at(label, target, tolerance_deg=3.0, attempts=6)

        self.status = "completed"
        self.save_report()
        self.print_compensated_summary()
        print(f"[完成] 1号格补偿低位空载三轮测试结束。完整记录：{self.report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-1 compensated low-pose test")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--compensation", type=Path, required=True)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-times", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument("--stream-step", type=float, default=1.0)
    parser.add_argument("--stream-period", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/diagnostics"),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.repeats != 3 or args.sample_times != [1.0, 3.0, 6.0]:
        raise RuntimeError("低位补偿验证固定为3轮，并在1/3/6秒采样")
    if not 1 <= args.arm_speed <= 8:
        raise RuntimeError("低位补偿验证速度必须在1~8")
    if not 0.5 <= args.stream_step <= 1.5:
        raise RuntimeError("低位补偿验证关节步长必须在0.5~1.5度")
    if not 0.15 <= args.stream_period <= 0.35:
        raise RuntimeError("低位补偿验证周期必须在0.15~0.35秒")

    route, gripper = load_route(args.poses, grid=1)
    compensation_document = load_compensation(args.compensation)
    print_preflight(route, gripper, args.repeats)
    print("[补偿] J2/J3/J4安全姿态双向验证均为PASS；J1/J5/J6不补偿。")
    print("[姿态策略] 上方进入和抬升返回应用补偿；grid_1_grasp低位点使用原始记录角度。")
    print("[模式] 只验证1号格低位空载路线，夹爪保持OPEN，不抓取物体。")
    if args.check_only:
        compensator = DirectionalJointCompensator(compensation_document)
        preload = compensator.preload_commands(
            np.asarray(route[0]["angles"], dtype=float),
            np.asarray(route[1]["angles"], dtype=float),
            np.asarray(route[0]["angles"], dtype=float),
        )
        print(f"[只读检查] 配置、路线及{len(preload)}步单关节预紧序列合法；未连接机械臂。")
        return 0

    runner = CompensatedLowPoseRunner(
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
        compensation_document,
    )
    try:
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 移走1号格和整条路径上的所有物体，清空周围30cm，手放在急停旁。")
        confirmation = input("确认空载且现场安全，输入 RUN 并按 Enter 开始：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送运动或夹爪命令。")
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
        print("[停止] 操作员已中止1号格补偿低位空载验证。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
