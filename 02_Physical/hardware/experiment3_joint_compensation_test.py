"""Validate direction-aware joint compensation at the recorded safe pose.

This is deliberately separate from the production transfer path.  Each
compensated target is sent once, then feedback is sampled without corrective
commands so the validation cannot hide overshoot or slow settling.
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

from experiment3_grid_dry_run import GridDryRunner
from experiment3_joint_repeatability_test import load_safe_pose, sample_delays, to_float_list
from mecharm_recorded_task import SOFT_LOWER, SOFT_UPPER
from pymycobot.common import ProtocolCode


JOINT_INDEX = {"J2": 1, "J3": 2, "J4": 3}
FINAL_MAE_LIMIT_DEG = 0.75
DIRECTION_DIFFERENCE_LIMIT_DEG = 0.75
REPEAT_SPREAD_LIMIT_DEG = 0.50


def load_measured_errors(
    report_paths: list[Path], requested_joints: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    measured: dict[str, dict[str, float]] = {}
    for path in report_paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("status") != "completed":
            raise RuntimeError(f"基准报告未完成：{path}")
        summary = document.get("summary")
        if not isinstance(summary, dict):
            raise RuntimeError(f"基准报告缺少summary：{path}")
        for joint in requested_joints:
            item = summary.get(joint)
            if not isinstance(item, dict):
                continue
            direction_errors = item.get("direction_mean_error_deg")
            if not isinstance(direction_errors, dict):
                raise RuntimeError(f"{path}缺少{joint}方向误差")
            increasing = direction_errors.get("increasing")
            decreasing = direction_errors.get("decreasing")
            if not isinstance(increasing, (int, float)) or not isinstance(
                decreasing, (int, float)
            ):
                raise RuntimeError(f"{path}的{joint}方向误差无效")
            measured[joint] = {
                "increasing": float(increasing),
                "decreasing": float(decreasing),
            }

    missing = [joint for joint in requested_joints if joint not in measured]
    if missing:
        raise RuntimeError(f"基准报告缺少关节：{', '.join(missing)}")
    return measured


def build_compensation_trials(
    safe_angles: np.ndarray,
    measured_errors: dict[str, dict[str, float]],
    offset_deg: float,
    repeats: int,
) -> list[dict[str, Any]]:
    desired = np.asarray(safe_angles, dtype=float)
    if desired.shape != (6,) or not np.all(np.isfinite(desired)):
        raise ValueError("safe_angles必须是六个有效关节角")
    if not 2.0 <= offset_deg <= 6.0:
        raise ValueError("预紧偏移必须在2~6度")
    if repeats < 1:
        raise ValueError("重复次数必须至少为1")

    trials: list[dict[str, Any]] = []
    for joint, direction_errors in measured_errors.items():
        if joint not in JOINT_INDEX:
            raise ValueError(f"不支持补偿测试关节：{joint}")
        joint_index = JOINT_INDEX[joint]
        for direction in ("increasing", "decreasing"):
            measured_error = float(direction_errors[direction])
            if not np.isfinite(measured_error) or abs(measured_error) > 3.0:
                raise ValueError(f"{joint} {direction}补偿值超出安全验证范围")

            command_target = desired.copy()
            command_target[joint_index] = round(
                float(desired[joint_index] - measured_error), 3
            )
            pre_target = desired.copy()
            signed_offset = -offset_deg if direction == "increasing" else offset_deg
            pre_target[joint_index] = command_target[joint_index] + signed_offset
            if np.any(command_target < SOFT_LOWER) or np.any(command_target > SOFT_UPPER):
                raise ValueError(f"{joint} {direction}补偿目标超出软件限位")
            if np.any(pre_target < SOFT_LOWER) or np.any(pre_target > SOFT_UPPER):
                raise ValueError(f"{joint} {direction}预紧姿态超出软件限位")

            for repeat in range(1, repeats + 1):
                trials.append(
                    {
                        "joint": joint,
                        "joint_index": joint_index,
                        "direction": direction,
                        "repeat": repeat,
                        "measured_baseline_error_deg": measured_error,
                        "desired_target_deg": desired.copy(),
                        "command_target_deg": command_target.copy(),
                        "pre_target_deg": pre_target.copy(),
                    }
                )
    return trials


def classify_compensation(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for joint in sorted({str(sample["joint"]) for sample in samples}):
        joint_samples = [sample for sample in samples if sample["joint"] == joint]
        final_time = max(float(sample["elapsed_s"]) for sample in joint_samples)
        final_samples = [
            sample for sample in joint_samples if float(sample["elapsed_s"]) == final_time
        ]
        direction_means: dict[str, float] = {}
        direction_spreads: dict[str, float] = {}
        all_errors: list[float] = []
        for direction in ("increasing", "decreasing"):
            values = [
                float(sample["desired_error_deg"])
                for sample in final_samples
                if sample["direction"] == direction
            ]
            if not values:
                continue
            direction_means[direction] = float(np.mean(values))
            direction_spreads[direction] = float(np.ptp(values)) if len(values) > 1 else 0.0
            all_errors.extend(values)
        if len(direction_means) != 2:
            continue

        final_mae = float(np.mean(np.abs(all_errors)))
        direction_difference = abs(
            direction_means["decreasing"] - direction_means["increasing"]
        )
        repeat_spread = max(direction_spreads.values())
        result[joint] = {
            "final_sample_s": final_time,
            "final_mae_deg": round(final_mae, 3),
            "direction_mean_error_deg": {
                key: round(value, 3) for key, value in direction_means.items()
            },
            "direction_mean_difference_deg": round(direction_difference, 3),
            "repeat_spread_max_deg": round(repeat_spread, 3),
            "passed": (
                final_mae <= FINAL_MAE_LIMIT_DEG
                and direction_difference <= DIRECTION_DIFFERENCE_LIMIT_DEG
                and repeat_spread <= REPEAT_SPREAD_LIMIT_DEG
            ),
        }
    return result


class CompensationValidationRunner(GridDryRunner):
    def __init__(
        self,
        host: str,
        port: int,
        arm_speed: int,
        output_dir: Path,
        stream_step: float,
        stream_period: float,
        safe_angles: np.ndarray,
        safe_coords: np.ndarray,
        trials: list[dict[str, Any]],
        sample_times: list[float],
        gripper: dict[str, int],
        measured_errors: dict[str, dict[str, float]],
        baseline_reports: list[Path],
        offset_deg: float,
        repeats: int,
    ) -> None:
        super().__init__(host, port, arm_speed, output_dir, stream_step, stream_period)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = output_dir / f"joint_compensation_validation_{stamp}.yaml"
        self.safe_angles = safe_angles
        self.safe_coords = safe_coords
        self.trials = trials
        self.sample_times = sample_times
        self.gripper_config = gripper
        self.measured_errors = measured_errors
        self.baseline_reports = baseline_reports
        self.offset_deg = offset_deg
        self.repeats = repeats
        self.samples: list[dict[str, Any]] = []
        self.trial_records: list[dict[str, Any]] = []
        self.status = "created"

    def save_report(self) -> None:
        document = {
            "format": "mecharm_joint_compensation_validation/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": self.status,
            "configuration": {
                "safe_angles_deg": to_float_list(self.safe_angles),
                "safe_tcp_mm_deg": to_float_list(self.safe_coords),
                "measured_direction_errors_deg": self.measured_errors,
                "baseline_reports": [str(path) for path in self.baseline_reports],
                "offset_deg": self.offset_deg,
                "repeats_per_direction": self.repeats,
                "sample_times_s": self.sample_times,
                "arm_speed": self.arm_speed,
                "stream_step_deg": self.stream_step,
                "stream_period_s": self.stream_period,
            },
            "trials": self.trial_records,
            "samples": self.samples,
            "summary": classify_compensation(self.samples) if self.samples else {},
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(self.report_path)

    def send_exact_target_once(self, target: np.ndarray) -> None:
        encoded = [self.arm._angle2int(float(angle)) for angle in target]
        self.arm._mesg(
            ProtocolCode.SEND_ANGLES,
            encoded,
            self.arm_speed,
            _async=True,
        )
        self.commanded_angles = target.copy()

    def prepare_direction(self, trial: dict[str, Any]) -> np.ndarray:
        pre_target = np.asarray(trial["pre_target_deg"], dtype=float)
        desired = np.asarray(trial["desired_target_deg"], dtype=float)
        joint_index = int(trial["joint_index"])
        label = (
            f"{trial['joint']} {trial['direction']} "
            f"第{trial['repeat']}/{self.repeats}次预紧"
        )
        self.stream_to(label, pre_target)

        actual = pre_target.copy()
        for attempt in range(1, 4):
            time.sleep(1.0)
            actual = self.read_angles()
            side = float(actual[joint_index] - desired[joint_index])
            correct_side = side <= -1.5 if trial["direction"] == "increasing" else side >= 1.5
            print(
                f"[预紧校验] {label} {attempt}/3：实测{trial['joint']}="
                f"{actual[joint_index]:.2f}°，距物理目标={side:+.2f}°"
            )
            if correct_side:
                return actual
            if attempt < 3:
                self.send_exact_target_once(pre_target)
        raise RuntimeError(f"{label}未到达物理目标的正确一侧")

    def sample_trial(self, trial: dict[str, Any], pre_actual: np.ndarray) -> None:
        desired = np.asarray(trial["desired_target_deg"], dtype=float)
        command = np.asarray(trial["command_target_deg"], dtype=float)
        joint_index = int(trial["joint_index"])
        label = (
            f"{trial['joint']} {trial['direction']} "
            f"第{trial['repeat']}/{self.repeats}次"
        )
        print(
            f"[补偿目标] {label}：物理目标={desired[joint_index]:.3f}°，"
            f"发送={command[joint_index]:.3f}°；只发送一次"
        )
        start = time.monotonic()
        self.send_exact_target_once(command)
        self.trial_records.append(
            {
                "joint": trial["joint"],
                "direction": trial["direction"],
                "repeat": trial["repeat"],
                "baseline_error_deg": trial["measured_baseline_error_deg"],
                "pre_target_deg": to_float_list(np.asarray(trial["pre_target_deg"])),
                "pre_actual_deg": to_float_list(pre_actual),
                "desired_target_deg": to_float_list(desired),
                "command_target_deg": to_float_list(command),
            }
        )

        for sample_at in self.sample_times:
            remaining = start + sample_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            actual = self.read_angles()
            desired_error = float(actual[joint_index] - desired[joint_index])
            command_error = float(actual[joint_index] - command[joint_index])
            sample = {
                "joint": trial["joint"],
                "joint_index": joint_index + 1,
                "direction": trial["direction"],
                "repeat": trial["repeat"],
                "elapsed_s": float(sample_at),
                "actual_elapsed_s": round(time.monotonic() - start, 3),
                "desired_deg": round(float(desired[joint_index]), 3),
                "commanded_deg": round(float(command[joint_index]), 3),
                "actual_deg": round(float(actual[joint_index]), 3),
                "desired_error_deg": round(desired_error, 3),
                "command_tracking_error_deg": round(command_error, 3),
                "all_actual_deg": to_float_list(actual),
            }
            self.samples.append(sample)
            print(
                f"[采样 {sample_at:g}s] {trial['joint']}实测="
                f"{actual[joint_index]:.2f}°，相对物理目标误差={desired_error:+.2f}°"
            )
            self.status = "running"
            self.save_report()

    def print_summary(self) -> None:
        summary = classify_compensation(self.samples)
        print("\n[补偿验证汇总] 只有PASS的补偿才可进入下一阶段低位验证。")
        for joint, item in summary.items():
            print(
                f"[{joint}] 6s MAE={item['final_mae_deg']:.3f}°，"
                f"方向均值差={item['direction_mean_difference_deg']:.3f}°，"
                f"三次最大极差={item['repeat_spread_max_deg']:.3f}°；"
                f"结果={'PASS' if item['passed'] else 'FAIL'}"
            )

    def run_validation(self, current: np.ndarray) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        self.gripper(
            "补偿诊断全程保持 OPEN",
            int(self.gripper_config["open"]),
            min(20, int(self.gripper_config["speed"])),
        )
        self.stream_to("回到记录的safe姿态", self.safe_angles)
        self.settle_at("safe基准姿态", self.safe_angles, tolerance_deg=3.0, attempts=5)

        for index, trial in enumerate(self.trials, start=1):
            if self.stopped:
                raise KeyboardInterrupt
            error = self.read_error()
            if not self.error_is_clear(error):
                raise RuntimeError(f"第{index}次补偿验证前控制器报错：{error}")
            print(
                f"\n[验证 {index}/{len(self.trials)}] {trial['joint']} "
                f"{trial['direction']}，重复{trial['repeat']}/{self.repeats}"
            )
            pre_actual = self.prepare_direction(trial)
            self.sample_trial(trial, pre_actual)

        self.stream_to("测试结束返回safe姿态", self.safe_angles)
        self.settle_at("最终safe姿态", self.safe_angles, tolerance_deg=3.0, attempts=5)
        self.status = "completed"
        self.save_report()
        self.print_summary()
        print(f"[完成] 补偿验证结束。完整记录：{self.report_path}")


def print_preflight(
    safe_angles: np.ndarray,
    safe_coords: np.ndarray,
    measured_errors: dict[str, dict[str, float]],
    trials: list[dict[str, Any]],
    arm_speed: int,
    offset_deg: float,
) -> None:
    tested = "/".join(measured_errors)
    print(f"[预检] {tested}方向补偿验证；仅在记录的高位safe姿态附近运动")
    print(f"[safe关节] {np.round(safe_angles, 2).tolist()}")
    print(f"[safe TCP] {np.round(safe_coords, 1).tolist()}")
    for joint in measured_errors:
        index = JOINT_INDEX[joint]
        by_direction = {
            trial["direction"]: trial
            for trial in trials
            if trial["joint"] == joint and trial["repeat"] == 1
        }
        print(
            f"[{joint}] 物理目标={safe_angles[index]:.3f}°；"
            f"increasing发送={by_direction['increasing']['command_target_deg'][index]:.3f}°；"
            f"decreasing发送={by_direction['decreasing']['command_target_deg'][index]:.3f}°"
        )
    print(
        f"[方法] 每个方向重复3次；目标只发送一次；速度={arm_speed}；"
        f"预紧幅度={offset_deg:g}°；夹爪保持张开"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate direction-aware joint compensation")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--baseline-reports", type=Path, nargs="+")
    parser.add_argument("--joints", nargs="+", choices=tuple(JOINT_INDEX), default=["J3", "J4"])
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=6)
    parser.add_argument("--offset-deg", type=float, default=4.0)
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

    if args.poses is None:
        raise RuntimeError("必须提供 --poses")
    if not args.baseline_reports:
        raise RuntimeError("必须提供 --baseline-reports")
    if not 1 <= args.arm_speed <= 8:
        raise RuntimeError("补偿验证速度必须在1~8")
    if args.repeats != 3:
        raise RuntimeError("补偿验证固定要求每个方向重复3次")
    if args.sample_times != [1.0, 3.0, 6.0]:
        raise RuntimeError("补偿验证固定在1、3、6秒采样")
    sample_delays(args.sample_times)
    if not 0.5 <= args.stream_step <= 1.5:
        raise RuntimeError("连续位置流步长必须在0.5~1.5度")
    if not 0.15 <= args.stream_period <= 0.35:
        raise RuntimeError("连续位置流周期必须在0.15~0.35秒")

    requested_joints = tuple(dict.fromkeys(args.joints))
    safe_angles, safe_coords, gripper = load_safe_pose(args.poses)
    measured_errors = load_measured_errors(args.baseline_reports, requested_joints)
    trials = build_compensation_trials(
        safe_angles, measured_errors, args.offset_deg, args.repeats
    )
    print_preflight(
        safe_angles,
        safe_coords,
        measured_errors,
        trials,
        args.arm_speed,
        args.offset_deg,
    )
    if args.check_only:
        print("[只读检查] 补偿符号、关节限位和测试序列均合法；未连接机械臂。")
        return 0

    runner = CompensationValidationRunner(
        args.host,
        args.port,
        args.arm_speed,
        args.output_dir,
        args.stream_step,
        args.stream_period,
        safe_angles,
        safe_coords,
        trials,
        args.sample_times,
        gripper,
        measured_errors,
        args.baseline_reports,
        args.offset_deg,
        args.repeats,
    )
    try:
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 取下方块，清空机械臂周围30cm，并把手放在急停旁。")
        tested = "/".join(measured_errors)
        print(f"[说明] 本次只在safe附近逐个移动{tested}，不进入网格低位。")
        confirmation = input("确认现场安全，输入 RUN 并按 Enter 开始：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送任何运动或夹爪命令。")
            return 2
        runner.run_validation(current)
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
        print("[停止] 操作员已中止关节补偿验证。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
