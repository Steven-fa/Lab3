"""Low-speed, open-gripper dry run for one Experiment 3 grid pose pair."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import signal
import time

import numpy as np
import yaml

from mecharm_recorded_task import (
    RecordedTaskRunner,
    SOFT_LOWER,
    SOFT_UPPER,
    legacy_set_gripper_value,
)
from pymycobot.common import ProtocolCode


EXPECTED_FORMAT = "mecharm_experiment3_fixed_poses/v1"
CONTACT_JOINT_INDEX = 4  # J5: old arm shows repeatable gravity/backlash error here.
CONTACT_J5_TOLERANCE_DEG = 5.5
CONTACT_OTHER_JOINT_TOLERANCE_DEG = 2.5
CONTACT_STABILITY_SPREAD_DEG = 0.75
CONTACT_TCP_XYZ_TOLERANCE_MM = 8.0
CONTACT_TCP_RPY_TOLERANCE_DEG = 8.0


def interpolate(start: np.ndarray, target: np.ndarray, max_step: float) -> list[np.ndarray]:
    count = max(1, math.ceil(float(np.max(np.abs(target - start))) / max_step))
    return [start + (target - start) * (index / count) for index in range(1, count + 1)]


def load_route(path: Path, grid: int) -> tuple[list[dict], dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("姿态文件不是实验3固定姿态格式")
    if not document.get("complete"):
        raise RuntimeError("17个实验3姿态尚未全部录制完成")
    poses = document.get("poses")
    if not isinstance(poses, dict):
        raise RuntimeError("姿态文件缺少 poses")

    names = ["safe", f"grid_{grid}_approach", f"grid_{grid}_grasp"]
    route = []
    for name in names:
        pose = poses.get(name)
        if not isinstance(pose, dict):
            raise RuntimeError(f"姿态文件缺少 {name}")
        angles = np.asarray(pose.get("joint_angles_deg"), dtype=float)
        coords = np.asarray(pose.get("tcp_coords_mm_deg"), dtype=float)
        if angles.shape != (6,) or coords.shape != (6,):
            raise RuntimeError(f"{name} 缺少六轴角度或TCP坐标")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(coords)):
            raise RuntimeError(f"{name} 含无效数值")
        if np.any(angles < SOFT_LOWER) or np.any(angles > SOFT_UPPER):
            raise RuntimeError(f"{name} 超出软件关节限位")
        route.append({"name": name, "angles": angles, "coords": coords})

    gripper = document.get("gripper", {})
    open_value = gripper.get("open")
    if not isinstance(open_value, int) or not 0 <= open_value <= 100:
        raise RuntimeError("夹爪 OPEN 参数无效")
    return route, gripper


class GridDryRunner(RecordedTaskRunner):
    def __init__(
        self,
        host: str,
        port: int,
        arm_speed: int,
        log_dir: Path,
        stream_step: float,
        stream_period: float,
    ) -> None:
        super().__init__(host, port, arm_speed, log_dir)
        self.stream_step = stream_step
        self.stream_period = stream_period
        self.commanded_angles: np.ndarray | None = None

    def settle_at(
        self,
        label: str,
        target: np.ndarray,
        tolerance_deg: float = 2.5,
        attempts: int = 3,
        allow_stable_contact: bool = False,
        target_coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """Verify an endpoint, with a narrow TCP-backed allowance for low poses."""
        if allow_stable_contact and (
            target_coords is None or np.asarray(target_coords).shape != (6,)
        ):
            raise RuntimeError("低位姿态联合校验缺少六维TCP目标")
        actual = target.copy()
        angle_samples: list[np.ndarray] = []
        for attempt in range(1, attempts + 1):
            time.sleep(1.0)
            actual = self.read_angles()
            angle_samples.append(actual.copy())
            difference = np.abs(actual - target)
            maximum = float(np.max(difference))
            print(
                f"[端点校验] {label} {attempt}/{attempts}："
                f"最大关节误差={maximum:.2f}°，实测={np.round(actual, 1).tolist()}"
            )
            self.events.append(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "endpoint_feedback",
                    "label": label,
                    "attempt": attempt,
                    "actual_deg": [round(float(value), 3) for value in actual],
                    "maximum_error_deg": round(maximum, 3),
                }
            )
            if maximum <= tolerance_deg:
                return actual

            if allow_stable_contact and len(angle_samples) >= 3:
                sample_array = np.stack(angle_samples[-3:])
                stable_spread = float(np.max(np.ptp(sample_array, axis=0)))
                other_errors = np.delete(difference, CONTACT_JOINT_INDEX)
                j5_error = float(difference[CONTACT_JOINT_INDEX])
                joints_acceptable = (
                    float(np.max(other_errors)) <= CONTACT_OTHER_JOINT_TOLERANCE_DEG
                    and j5_error <= CONTACT_J5_TOLERANCE_DEG
                    and stable_spread <= CONTACT_STABILITY_SPREAD_DEG
                )
                if joints_acceptable:
                    actual_coords = self.read_coords()
                    target_coords_array = np.asarray(target_coords, dtype=float)
                    xyz_error = float(
                        np.linalg.norm(actual_coords[:3] - target_coords_array[:3])
                    )
                    rpy_difference = (
                        actual_coords[3:] - target_coords_array[3:] + 180.0
                    ) % 360.0 - 180.0
                    rpy_error = float(np.max(np.abs(rpy_difference)))
                    if (
                        xyz_error <= CONTACT_TCP_XYZ_TOLERANCE_MM
                        and rpy_error <= CONTACT_TCP_RPY_TOLERANCE_DEG
                    ):
                        print(
                            f"[低位联合校验] {label}：J5偏差={j5_error:.2f}°，"
                            f"三次稳定度={stable_spread:.2f}°，TCP位置误差={xyz_error:.1f}mm，"
                            f"姿态误差={rpy_error:.1f}°；允许继续"
                        )
                        self.events.append(
                            {
                                "time": datetime.now().isoformat(timespec="milliseconds"),
                                "event": "endpoint_acceptance",
                                "label": label,
                                "acceptance": "stable_contact_tcp",
                                "j5_error_deg": round(j5_error, 3),
                                "sample_spread_deg": round(stable_spread, 3),
                                "tcp_xyz_error_mm": round(xyz_error, 3),
                                "tcp_rpy_error_deg": round(rpy_error, 3),
                            }
                        )
                        return actual

            if attempt < attempts:
                encoded = [self.arm._angle2int(float(angle)) for angle in target]
                self.arm._mesg(
                    ProtocolCode.SEND_ANGLES,
                    encoded,
                    self.arm_speed,
                    _async=True,
                )
        maximum = float(np.max(np.abs(actual - target)))
        raise RuntimeError(f"{label} 三次补偿后仍未到位，最大误差={maximum:.2f}°")

    def stream_to(self, label: str, target: np.ndarray) -> None:
        if self.commanded_angles is None:
            raise RuntimeError("位置流尚未初始化")
        points = interpolate(self.commanded_angles, target, self.stream_step)
        print(f"[运动] {label}：{len(points)} 个连续小步")
        for point in points:
            if self.stopped:
                raise KeyboardInterrupt
            encoded = [self.arm._angle2int(float(angle)) for angle in point]
            self.arm._mesg(
                ProtocolCode.SEND_ANGLES,
                encoded,
                self.arm_speed,
                _async=True,
            )
            self.commanded_angles = point
            time.sleep(self.stream_period)
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "position_stream",
                "label": label,
                "commands": len(points),
                "target_deg": [round(float(value), 3) for value in target],
                "feedback_reads": 0,
            }
        )

    def run_dry(self, current: np.ndarray, route: list[dict], gripper: dict) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        open_value = int(gripper["open"])
        gripper_speed = min(20, int(gripper.get("speed", 20)))
        print(f"[夹爪] 仅发送 OPEN={open_value}，本次不会发送 CLOSE")
        legacy_set_gripper_value(self.arm, open_value, gripper_speed)
        time.sleep(1.2)

        safe, approach, grasp = route
        execution = [
            ("回公共安全姿态", safe["angles"]),
            ("到网格上方", approach["angles"]),
            ("下降到空载抓取高度", grasp["angles"]),
            ("原路抬回网格上方", approach["angles"]),
            ("返回公共安全姿态", safe["angles"]),
        ]
        for label, target in execution:
            self.stream_to(label, target)
            self.settle_at(label, target)
            if label == "下降到空载抓取高度":
                print("[观察] 抓取点已通过实测关节校验；夹爪保持张开，停留3秒观察朝向")
                time.sleep(3.0)

        time.sleep(0.8)
        final_angles = self.read_angles()
        final_error = float(np.max(np.abs(final_angles - safe["angles"])))
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "final_feedback",
                "actual_deg": [round(float(value), 3) for value in final_angles],
                "safe_pose_max_error_deg": round(final_error, 3),
            }
        )
        self.save_log("completed")
        print(f"[完成] 空载往返结束；安全点最大关节误差={final_error:.1f}°")
        print(f"[记录] {self.log_path}")


def self_test() -> None:
    start = np.zeros(6)
    target = np.array([0.0, 5.0, -2.0, 0.0, 0.0, 0.0])
    points = interpolate(start, target, 2.0)
    assert len(points) == 3
    assert np.allclose(points[-1], target)
    assert max(float(np.max(np.abs(b - a))) for a, b in zip([start, *points[:-1]], points)) <= 2.0
    assert float(np.max(np.abs(np.array([1.0, 2.0]) - np.array([1.0, 4.5])))) == 2.5
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 one-grid dry run")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.14)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.arm_speed <= 20:
        raise RuntimeError("首次空载验证速度必须在 1~20")
    if not 0.5 <= args.stream_step <= 3.0:
        raise RuntimeError("连续位置流步长必须在 0.5~3.0 度")
    if not 0.08 <= args.stream_period <= 0.4:
        raise RuntimeError("连续位置流周期必须在 0.08~0.4 秒")

    route, gripper = load_route(args.poses, args.grid)
    safe, approach, grasp = route
    descent = approach["coords"][:3] - grasp["coords"][:3]
    print(f"[预检] 实验3 grid_{args.grid} 空载往返；不会抓取物体")
    print(f"[预检] 夹爪全程 OPEN={gripper['open']}，不会发送 CLOSE")
    print(
        "[路线] 当前姿态 -> safe -> grid_{}_approach -> grid_{}_grasp "
        "-> grid_{}_approach -> safe".format(args.grid, args.grid, args.grid)
    )
    print(
        f"[抓取段] Z下降={descent[2]:.1f}mm，"
        f"XY变化={float(np.linalg.norm(descent[:2])):.1f}mm"
    )
    print(
        f"[参数] 机械臂低速={args.arm_speed}，关节步长<={args.stream_step:.1f}°，"
        f"发送周期={args.stream_period:.2f}s，运动途中不读取反馈"
    )
    if args.check_only:
        print("[只读检查] 记录文件和路线合法；未连接机械臂，未发送命令。")
        return 0

    runner = GridDryRunner(
        args.host,
        args.port,
        args.arm_speed,
        args.poses.parent,
        args.stream_step,
        args.stream_period,
    )
    try:
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 清空机械臂周围至少30cm，确认网格1无物体，手放在急停旁。")
        confirmation = input("输入 RUN 并按 Enter 开始空载往返：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送任何运动或夹爪命令。")
            return 2
        runner.run_dry(current, route, gripper)
        return 0
    except Exception:
        if not runner.stopped:
            runner.stop()
        raise
    finally:
        runner.arm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止空载验证。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
