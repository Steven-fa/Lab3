"""Measure selected-joint repeatability at the recorded Experiment 3 safe pose.

The final target is sent exactly once per trial.  Feedback is sampled at fixed
times without corrective commands so slow settling is not hidden by the test.
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
from mecharm_recorded_task import SOFT_LOWER, SOFT_UPPER
from pymycobot.common import ProtocolCode


EXPECTED_FORMAT = "mecharm_experiment3_fixed_poses/v1"
AVAILABLE_TEST_JOINTS = {
    "J1": 0,
    "J2": 1,
    "J3": 2,
    "J4": 3,
    "J5": 4,
    "J6": 5,
}
DEFAULT_TEST_JOINTS = (("J2", 1), ("J5", 4))
BACKLASH_THRESHOLD_DEG = 1.0
SLOW_RESPONSE_THRESHOLD_DEG = 1.0
REPEAT_SPREAD_THRESHOLD_DEG = 1.0
FINAL_BIAS_THRESHOLD_DEG = 2.5
COMPENSATION_MAE_THRESHOLD_DEG = 0.75
MIN_APPROACH_SIDE_DEG = 1.0


def approach_side_is_valid(
    signed_distance_deg: float,
    direction: str,
    minimum_distance_deg: float = MIN_APPROACH_SIDE_DEG,
) -> bool:
    if minimum_distance_deg <= 0:
        raise ValueError("方向确认最小角度必须大于0")
    if direction == "increasing":
        return signed_distance_deg <= -minimum_distance_deg
    if direction == "decreasing":
        return signed_distance_deg >= minimum_distance_deg
    raise ValueError(f"未知到达方向：{direction}")


def load_safe_pose(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("姿态文件不是实验3固定姿态格式")
    if not document.get("complete"):
        raise RuntimeError("实验3的17个姿态尚未完整记录")
    poses = document.get("poses")
    safe = poses.get("safe") if isinstance(poses, dict) else None
    if not isinstance(safe, dict):
        raise RuntimeError("姿态文件缺少 safe 姿态")

    angles = np.asarray(safe.get("joint_angles_deg"), dtype=float)
    coords = np.asarray(safe.get("tcp_coords_mm_deg"), dtype=float)
    if angles.shape != (6,) or coords.shape != (6,):
        raise RuntimeError("safe 姿态缺少六轴角度或TCP坐标")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(coords)):
        raise RuntimeError("safe 姿态含无效数值")
    if np.any(angles < SOFT_LOWER) or np.any(angles > SOFT_UPPER):
        raise RuntimeError("safe 姿态超出软件关节限位")

    gripper_data = document.get("gripper")
    if not isinstance(gripper_data, dict):
        raise RuntimeError("姿态文件缺少夹爪参数")
    open_value = gripper_data.get("open")
    speed = gripper_data.get("speed", 20)
    if not isinstance(open_value, int) or not 0 <= open_value <= 100:
        raise RuntimeError("夹爪 OPEN 参数无效")
    if not isinstance(speed, int) or not 1 <= speed <= 100:
        raise RuntimeError("夹爪速度无效")
    return angles, coords, {"open": open_value, "speed": speed}


def build_trials(
    safe_angles: np.ndarray,
    offset_deg: float,
    repeats: int,
    test_joints: tuple[tuple[str, int], ...] = DEFAULT_TEST_JOINTS,
) -> list[dict[str, Any]]:
    safe = np.asarray(safe_angles, dtype=float)
    if safe.shape != (6,) or not np.all(np.isfinite(safe)):
        raise ValueError("safe_angles必须是六个有效关节角")
    if offset_deg <= 0:
        raise ValueError("测试偏移必须大于0")
    if repeats < 1:
        raise ValueError("重复次数必须至少为1")

    trials: list[dict[str, Any]] = []
    if not test_joints:
        raise ValueError("至少选择一个测试关节")

    for joint_name, joint_index in test_joints:
        for direction, signed_offset in (("increasing", -offset_deg), ("decreasing", offset_deg)):
            pre_target = safe.copy()
            pre_target[joint_index] += signed_offset
            if np.any(pre_target < SOFT_LOWER) or np.any(pre_target > SOFT_UPPER):
                raise ValueError(f"{joint_name}的预备姿态超出软件关节限位")
            for repeat in range(1, repeats + 1):
                trials.append(
                    {
                        "joint": joint_name,
                        "joint_index": joint_index,
                        "direction": direction,
                        "repeat": repeat,
                        "pre_target_deg": pre_target.copy(),
                        "target_deg": safe.copy(),
                    }
                )
    return trials


def sample_delays(sample_times: list[float]) -> list[float]:
    if not sample_times or any(value <= 0 for value in sample_times):
        raise ValueError("采样时间必须为正数")
    if sample_times != sorted(sample_times) or len(set(sample_times)) != len(sample_times):
        raise ValueError("采样时间必须严格递增")
    return [sample_times[0], *[b - a for a, b in zip(sample_times, sample_times[1:])]]


def classify_repeatability(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    joints = sorted({str(sample["joint"]) for sample in samples})
    for joint in joints:
        joint_samples = [sample for sample in samples if sample["joint"] == joint]
        elapsed_values = sorted({float(sample["elapsed_s"]) for sample in joint_samples})
        if len(elapsed_values) < 2:
            continue
        first_time, final_time = elapsed_values[0], elapsed_values[-1]
        first_errors = [
            float(sample["tested_joint_error_deg"])
            for sample in joint_samples
            if float(sample["elapsed_s"]) == first_time
        ]
        final_samples = [
            sample for sample in joint_samples if float(sample["elapsed_s"]) == final_time
        ]
        final_errors = [float(sample["tested_joint_error_deg"]) for sample in final_samples]
        direction_means: dict[str, float] = {}
        direction_spreads: dict[str, float] = {}
        complete = True
        for direction in ("increasing", "decreasing"):
            values = [
                float(sample["tested_joint_error_deg"])
                for sample in final_samples
                if sample["direction"] == direction
            ]
            if not values:
                complete = False
                break
            direction_means[direction] = float(np.mean(values))
            direction_spreads[direction] = float(np.ptp(values)) if len(values) > 1 else 0.0

        if not complete:
            continue

        first_mae = float(np.mean(np.abs(first_errors)))
        final_mae = float(np.mean(np.abs(final_errors)))
        slow_improvement = first_mae - final_mae
        direction_difference = abs(
            direction_means["decreasing"] - direction_means["increasing"]
        )
        repeat_spread = max(direction_spreads.values())
        result[joint] = {
            "first_sample_s": first_time,
            "final_sample_s": final_time,
            "first_mae_deg": round(first_mae, 3),
            "final_mae_deg": round(final_mae, 3),
            "error_reduction_deg": round(slow_improvement, 3),
            "direction_mean_error_deg": {
                key: round(value, 3) for key, value in direction_means.items()
            },
            "direction_mean_difference_deg": round(direction_difference, 3),
            "repeat_spread_deg": {
                key: round(value, 3) for key, value in direction_spreads.items()
            },
            "repeat_spread_max_deg": round(repeat_spread, 3),
            "candidate_command_correction_deg": {
                key: round(-value, 3) for key, value in direction_means.items()
            },
            "slow_response_suspected": slow_improvement > SLOW_RESPONSE_THRESHOLD_DEG,
            "backlash_suspected": direction_difference > BACKLASH_THRESHOLD_DEG,
            "poor_repeatability_suspected": repeat_spread > REPEAT_SPREAD_THRESHOLD_DEG,
            "final_tracking_bias_suspected": final_mae > FINAL_BIAS_THRESHOLD_DEG,
            "compensation_validation_recommended": (
                final_mae > COMPENSATION_MAE_THRESHOLD_DEG
                or direction_difference > BACKLASH_THRESHOLD_DEG
            ),
        }
    return result


def to_float_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in values]


class JointRepeatabilityRunner(GridDryRunner):
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
        offset_deg: float,
        repeats: int,
        test_joints: tuple[tuple[str, int], ...] = DEFAULT_TEST_JOINTS,
    ) -> None:
        super().__init__(host, port, arm_speed, output_dir, stream_step, stream_period)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = output_dir / f"joint_repeatability_{stamp}.yaml"
        self.safe_angles = safe_angles
        self.safe_coords = safe_coords
        self.trials = trials
        self.sample_times = sample_times
        self.gripper_config = gripper
        self.offset_deg = offset_deg
        self.repeats = repeats
        self.test_joints = test_joints
        self.samples: list[dict[str, Any]] = []
        self.trial_records: list[dict[str, Any]] = []
        self.status = "created"

    def save_report(self) -> None:
        summary = classify_repeatability(self.samples) if self.samples else {}
        document = {
            "format": "mecharm_joint_repeatability/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": self.status,
            "configuration": {
                "safe_angles_deg": to_float_list(self.safe_angles),
                "safe_tcp_mm_deg": to_float_list(self.safe_coords),
                "tested_joints": [name for name, _index in self.test_joints],
                "offset_deg": self.offset_deg,
                "repeats_per_direction": self.repeats,
                "sample_times_s": self.sample_times,
                "arm_speed": self.arm_speed,
                "stream_step_deg": self.stream_step,
                "stream_period_s": self.stream_period,
            },
            "trials": self.trial_records,
            "samples": self.samples,
            "summary": summary,
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

    def prepare_approach_side(self, trial: dict[str, Any]) -> np.ndarray:
        pre_target = np.asarray(trial["pre_target_deg"], dtype=float)
        target = np.asarray(trial["target_deg"], dtype=float)
        joint_index = int(trial["joint_index"])
        direction = str(trial["direction"])
        label = (
            f"{trial['joint']} {direction} 第{trial['repeat']}/{self.repeats}次预备姿态"
        )
        self.stream_to(label, pre_target)

        actual = pre_target.copy()
        for attempt in range(1, 4):
            time.sleep(1.0)
            actual = self.read_angles()
            signed_side = float(actual[joint_index] - target[joint_index])
            on_correct_side = approach_side_is_valid(signed_side, direction)
            other_errors = np.delete(np.abs(actual - pre_target), joint_index)
            print(
                f"[预备校验] {label} {attempt}/3：实测{trial['joint']}="
                f"{actual[joint_index]:.2f}°，距目标={signed_side:+.2f}°"
            )
            if on_correct_side and float(np.max(other_errors)) <= 3.5:
                return actual
            if attempt < 3:
                self.send_exact_target_once(pre_target)
        raise RuntimeError(f"{label}未稳定在目标的正确到达方向，停止测试")

    def sample_one_trial(self, trial: dict[str, Any], pre_actual: np.ndarray) -> None:
        target = np.asarray(trial["target_deg"], dtype=float)
        joint_index = int(trial["joint_index"])
        label = f"{trial['joint']} {trial['direction']} 第{trial['repeat']}/{self.repeats}次"
        print(f"[单次目标] {label}：只发送一次 safe 目标；随后仅在1/3/6秒读取")
        start = time.monotonic()
        self.send_exact_target_once(target)
        trial_record = {
            "joint": trial["joint"],
            "direction": trial["direction"],
            "repeat": trial["repeat"],
            "pre_target_deg": to_float_list(np.asarray(trial["pre_target_deg"])),
            "pre_actual_deg": to_float_list(pre_actual),
            "target_deg": to_float_list(target),
        }
        self.trial_records.append(trial_record)

        for sample_at in self.sample_times:
            remaining = start + sample_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            actual = self.read_angles()
            actual_elapsed = time.monotonic() - start
            errors = actual - target
            sample = {
                "joint": trial["joint"],
                "joint_index": joint_index + 1,
                "direction": trial["direction"],
                "repeat": trial["repeat"],
                "elapsed_s": float(sample_at),
                "actual_elapsed_s": round(actual_elapsed, 3),
                "target_deg": round(float(target[joint_index]), 3),
                "actual_deg": round(float(actual[joint_index]), 3),
                "tested_joint_error_deg": round(float(errors[joint_index]), 3),
                "all_actual_deg": to_float_list(actual),
                "all_error_deg": to_float_list(errors),
            }
            self.samples.append(sample)
            print(
                f"[采样 {sample_at:g}s] {trial['joint']}目标="
                f"{target[joint_index]:.2f}°，实测={actual[joint_index]:.2f}°，"
                f"误差={errors[joint_index]:+.2f}°"
            )
            self.status = "running"
            self.save_report()

        try:
            trial_record["tcp_after_6s_mm_deg"] = to_float_list(self.read_coords())
        except RuntimeError as error:
            trial_record["tcp_read_error"] = str(error)
        self.save_report()

    def print_summary(self) -> None:
        summary = classify_repeatability(self.samples)
        print("\n[诊断汇总] 判据只提示可能原因，不自动判定舵机损坏。")
        for joint, item in summary.items():
            flags = []
            if item["slow_response_suspected"]:
                flags.append("慢响应")
            if item["backlash_suspected"]:
                flags.append("方向齿隙/迟滞")
            if item["poor_repeatability_suspected"]:
                flags.append("重复性偏差")
            if item["final_tracking_bias_suspected"]:
                flags.append("最终跟踪偏差")
            print(
                f"[{joint}] 1s MAE={item['first_mae_deg']:.2f}°，"
                f"6s MAE={item['final_mae_deg']:.2f}°，"
                f"方向均值差={item['direction_mean_difference_deg']:.2f}°，"
                f"三次最大极差={item['repeat_spread_max_deg']:.2f}°；"
                f"提示={'、'.join(flags) if flags else '未见明显异常'}"
            )
            if item["compensation_validation_recommended"]:
                corrections = item["candidate_command_correction_deg"]
                print(
                    f"[{joint}候选补偿] increasing={corrections['increasing']:+.3f}°，"
                    f"decreasing={corrections['decreasing']:+.3f}°；"
                    "仅供下一阶段验证，尚未写入正式路线"
                )

    def run_repeatability(self, current: np.ndarray) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        self.gripper(
            "诊断前保持 OPEN",
            int(self.gripper_config["open"]),
            min(20, int(self.gripper_config["speed"])),
        )
        self.stream_to("回到最新记录的 safe 姿态", self.safe_angles)
        self.settle_at("safe 基准姿态", self.safe_angles, tolerance_deg=3.0, attempts=5)

        for index, trial in enumerate(self.trials, start=1):
            if self.stopped:
                raise KeyboardInterrupt
            error = self.read_error()
            if not self.error_is_clear(error):
                raise RuntimeError(f"第{index}次测试前控制器报错：{error}")
            print(
                f"\n[测试 {index}/{len(self.trials)}] {trial['joint']}，"
                f"{trial['direction']}方向到达，重复{trial['repeat']}/{self.repeats}"
            )
            pre_actual = self.prepare_approach_side(trial)
            self.sample_one_trial(trial, pre_actual)

        self.stream_to("测试结束返回 safe 姿态", self.safe_angles)
        self.settle_at("最终 safe 姿态", self.safe_angles, tolerance_deg=3.0, attempts=5)
        self.status = "completed"
        self.save_report()
        self.print_summary()
        tested = "/".join(name for name, _index in self.test_joints)
        print(f"[完成] {tested}双方向重复定位测试结束。完整记录：{self.report_path}")


def run_self_test() -> None:
    safe = np.array([-0.43, 30.49, -27.86, 147.21, -54.75, 130.60])
    trials = build_trials(safe, 6.0, 3)
    assert len(trials) == 12
    assert sample_delays([1.0, 3.0, 6.0]) == [1.0, 2.0, 3.0]
    print("SELF_TEST=PASS")


def print_preflight(
    safe_angles: np.ndarray,
    safe_coords: np.ndarray,
    trials: list[dict[str, Any]],
    sample_times: list[float],
    arm_speed: int,
    test_joints: tuple[tuple[str, int], ...] = DEFAULT_TEST_JOINTS,
) -> None:
    tested = "/".join(name for name, _index in test_joints)
    print(f"[预检] {tested}安全重复定位诊断；测试目标为最新记录的 safe 姿态")
    print(f"[safe关节] {np.round(safe_angles, 2).tolist()}")
    print(f"[safe TCP] {np.round(safe_coords, 1).tolist()}")
    for joint_name, joint_index in test_joints:
        values = [
            trial["pre_target_deg"][joint_index]
            for trial in trials
            if trial["joint"] == joint_name
        ]
        print(
            f"[{joint_name}] 目标={safe_angles[joint_index]:.2f}°，"
            f"两侧预备角={min(values):.2f}° / {max(values):.2f}°"
        )
    print(
        f"[方法] 每个关节从正/反两个方向各重复3次；目标只发送一次，"
        f"在{sample_times}秒读取误差"
    )
    print(f"[参数] 机械臂速度={arm_speed}；夹爪只张开，不执行抓取")


def main() -> int:
    parser = argparse.ArgumentParser(description="Selected-joint bidirectional repeatability test")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--offset-deg", type=float, default=6.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-times", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument(
        "--joints",
        nargs="+",
        choices=tuple(AVAILABLE_TEST_JOINTS),
        default=["J2", "J5"],
        help="one or more joints to test independently",
    )
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/nvidia/mecharm_tasks/experiment3/diagnostics"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.poses is None:
        raise RuntimeError("必须提供 --poses")
    if not 1 <= args.arm_speed <= 12:
        raise RuntimeError("安全诊断速度必须在1~12")
    if not 2.0 <= args.offset_deg <= 8.0:
        raise RuntimeError("安全诊断偏移必须在2~8度")
    if args.repeats != 3:
        raise RuntimeError("本诊断固定要求每个方向重复3次")
    sample_delays(args.sample_times)
    if args.sample_times != [1.0, 3.0, 6.0]:
        raise RuntimeError("本诊断固定在1、3、6秒采样")
    if not 0.5 <= args.stream_step <= 2.5:
        raise RuntimeError("连续位置流步长必须在0.5~2.5度")
    if not 0.10 <= args.stream_period <= 0.4:
        raise RuntimeError("连续位置流周期必须在0.10~0.4秒")

    selected_names = list(dict.fromkeys(args.joints))
    test_joints = tuple((name, AVAILABLE_TEST_JOINTS[name]) for name in selected_names)
    safe_angles, safe_coords, gripper = load_safe_pose(args.poses)
    trials = build_trials(safe_angles, args.offset_deg, args.repeats, test_joints)
    print_preflight(
        safe_angles,
        safe_coords,
        trials,
        args.sample_times,
        args.arm_speed,
        test_joints,
    )
    if args.check_only:
        print("[只读检查] 配置、限位、双方向测试计划均合法；未连接机械臂。")
        return 0

    runner = JointRepeatabilityRunner(
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
        args.offset_deg,
        args.repeats,
        test_joints,
    )
    try:
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 取下所有物体，清空机械臂周围30cm，并把手放在急停旁。")
        tested = "/".join(name for name, _index in test_joints)
        print(
            f"[范围] 先回safe；随后仅{tested}逐个在safe附近正负"
            f"{args.offset_deg:g}度运动。"
        )
        confirmation = input("确认扶稳且现场安全，输入 RUN 并按 Enter 开始：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送任何运动或夹爪命令。")
            return 2
        runner.run_repeatability(current)
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
        print("[停止] 操作员已中止关节重复定位诊断。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
