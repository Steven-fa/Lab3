"""Keyboard teaching recorder for a physical mechArm 270 Pi."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import select
import shutil
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
VENV_PACKAGES = ROOT / ".venv_mecharm_hw" / "Lib" / "site-packages"
if VENV_PACKAGES.is_dir():
    sys.path.insert(0, str(VENV_PACKAGES))

from pymycobot import MechArmSocket  # noqa: E402


if os.name == "nt":
    import msvcrt
else:
    msvcrt = None
    import termios
    import tty


JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]

SOFT_LOWER = np.array([-165.0, -90.0, -180.0, -165.0, -115.0, -175.0])
SOFT_UPPER = np.array([165.0, 90.0, 65.0, 165.0, 115.0, 175.0])


@contextmanager
def raw_keyboard():
    """Enable one-key input on both Windows and an SSH Linux terminal."""
    if os.name == "nt":
        yield
        return
    if not sys.stdin.isatty():
        raise RuntimeError("采点程序需要交互式终端，请使用 ssh -tt 启动")
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def read_key() -> str:
    if os.name == "nt":
        return msvcrt.getwch()
    return sys.stdin.read(1)


def key_available(timeout: float) -> bool:
    if os.name == "nt":
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                return True
            time.sleep(0.01)
        return False
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(ready)


def classify_suffix(suffix: str | None) -> str:
    if suffix == "a":
        return "grasp"
    if suffix == "b":
        return "release"
    return "waypoint"


def read_q_sequence(timeout: float = 0.55) -> tuple[str, str | None]:
    if key_available(timeout):
        suffix = read_key().lower()
        return classify_suffix(suffix), suffix
    return "waypoint", None


class RealRecorder:
    def __init__(
        self,
        host: str,
        port: int,
        output_root: Path,
        gripper_calibration: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.arm = MechArmSocket(host, port)
        self.frames: list[dict] = []
        self.free = False
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_directory = output_root / f"real_teach_{stamp}"
        self.output_path = self.output_directory / "keyframes.yaml"
        self.gripper_calibration = gripper_calibration

    @staticmethod
    def retry_query(label: str, query, valid, attempts: int = 6):
        last = None
        for attempt in range(attempts):
            last = query()
            if valid(last):
                return last
            if attempt + 1 < attempts:
                time.sleep(0.75)
        raise RuntimeError(f"{label}连续 {attempts} 次无有效响应：{last!r}")

    def angles(self) -> list[float]:
        return self.retry_query(
            "读取关节角",
            self.arm.get_angles,
            lambda value: isinstance(value, list) and len(value) == 6,
        )

    def coords(self) -> list[float]:
        return self.retry_query(
            "读取末端坐标",
            self.arm.get_coords,
            lambda value: isinstance(value, list) and len(value) == 6,
        )

    @staticmethod
    def error_is_clear(value) -> bool:
        return value == 0 or (
            isinstance(value, (list, tuple))
            and len(value) in (6, 7)
            and all(item == 0 for item in value)
        )

    def read_error(self):
        for method_name in ("get_error_information", "read_next_error"):
            query = getattr(self.arm, method_name, None)
            if query is None:
                continue
            try:
                return self.retry_query(
                    f"读取错误状态({method_name})",
                    query,
                    lambda value: value != -1,
                    attempts=2,
                )
            except RuntimeError:
                continue
        raise RuntimeError("机械臂的错误状态接口均未返回有效数据")

    def check_connection(self) -> None:
        angles = self.angles()
        error = self.read_error()
        if not self.error_is_clear(error):
            raise RuntimeError(f"控制器当前错误：{error}")
        print(f"[连接] {self.host}:{self.port}")
        print(f"[错误码] {error}")
        print(f"[当前角度] {np.round(angles, 1).tolist()}")

    def wait_for_servo_state(self, expected: int, timeout: float) -> bool:
        """Poll both aggregate and per-axis state; old firmware can cache either."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            aggregate = self.arm.is_all_servo_enable()
            if aggregate == expected:
                return True
            per_axis = [self.arm.is_servo_enable(axis) for axis in range(1, 7)]
            valid = [value for value in per_axis if value in (0, 1)]
            if len(valid) == 6 and all(value == expected for value in valid):
                return True
            time.sleep(0.5)
        return False

    def enable_free_mode(self, verification_timeout: float = 5.0) -> None:
        if self.free:
            print("[提示] 已经处于可手拖状态。")
            return
        print("[安全] 正在释放六轴保持力，请继续用手托住机械臂……")
        # Set this before sending the command so every later error path runs
        # the re-lock cleanup. Firmware v1.2 often returns -1 or a stale 1 even
        # though release_all_servos() has already released the physical arm.
        self.free = True
        self.arm.release_all_servos()
        if self.wait_for_servo_state(0, verification_timeout):
            print("[自由模式] 六轴释放状态已确认，可以缓慢手拖示教。")
        else:
            print("[自由模式] 旧固件未返回可靠释放状态；录制继续。")
            print("[提示] 若机械臂不能手拖，请按 l 锁定后再按 f 重试。")

    def lock_servos(
        self, force: bool = False, verification_timeout: float = 8.0
    ) -> None:
        if not self.free and not force:
            print("[锁定] 六轴当前没有被本程序释放。")
            return
        print("[锁定] 正在恢复六轴保持力，请先稳住机械臂……")
        self.arm.focus_all_servos()
        if not self.wait_for_servo_state(1, verification_timeout):
            raise RuntimeError("六轴未能重新锁定，请继续托住机械臂并重新上电")
        self.free = False
        print("[锁定] 六轴保持力已恢复。")

    def document(self) -> dict:
        return {
            "format": "mecharm_real_keyframes/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "recorded_at": self.frames[0]["captured_at"] if self.frames else None,
            "robot": "mechArm 270 Pi",
            "connection": {"host": self.host, "port": self.port},
            "position_unit": "m",
            "angle_unit": "rad",
            "source_coords_unit": "mm_deg",
            "joint_names": JOINT_NAMES,
            "gripper_installed": True,
            "sequence_complete": self.sequence_complete(),
            "keyframe_count": len(self.frames),
            "keyframes": self.frames,
        }

    def sequence_complete(self) -> bool:
        carrying = False
        for frame in self.frames:
            if frame["type"] == "grasp":
                if carrying:
                    return False
                carrying = True
            elif frame["type"] == "release":
                if not carrying:
                    return False
                carrying = False
        return not carrying

    def carrying(self) -> bool:
        state = False
        for frame in self.frames:
            if frame["type"] == "grasp":
                state = True
            elif frame["type"] == "release":
                state = False
        return state

    def save(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if self.gripper_calibration and self.gripper_calibration.is_file():
            calibration_target = self.output_directory / "gripper_calibration.yaml"
            if not calibration_target.exists():
                shutil.copy2(self.gripper_calibration, calibration_target)
        temporary = self.output_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.document(), stream, allow_unicode=True, sort_keys=False
            )
        temporary.replace(self.output_path)

    def capture(self, kind: str) -> None:
        if not self.free:
            raise RuntimeError("六轴尚未进入自由模式；请托住机械臂并先按 f")
        if kind == "grasp" and self.carrying():
            raise RuntimeError("已有未释放的抓取点；请先记录 qb 或按 u 撤销")
        if kind == "release" and not self.carrying():
            raise RuntimeError("尚未记录抓取点；必须先记录 qa，再记录 qb")
        angles_deg = self.angles()
        coords = self.coords()
        angle_array = np.asarray(angles_deg, dtype=float)
        if np.any(angle_array < SOFT_LOWER) or np.any(angle_array > SOFT_UPPER):
            invalid = [
                index + 1
                for index, (value, lower, upper) in enumerate(
                    zip(angle_array, SOFT_LOWER, SOFT_UPPER)
                )
                if value < lower or value > upper
            ]
            raise RuntimeError(f"当前姿态超出旧款 mechArm 安全范围，拒绝记录：J{invalid}")
        position_m = [value / 1000.0 for value in coords[:3]]
        quaternion = Rotation.from_euler("xyz", coords[3:], degrees=True).as_quat()
        captured_at = datetime.now().isoformat(timespec="milliseconds")
        frame = {
            "index": len(self.frames) + 1,
            "type": kind,
            "captured_at": captured_at,
            "ee_pose": {
                "position": [float(value) for value in position_m],
                "orientation_xyzw": [float(value) for value in quaternion],
            },
            "joint_positions": {
                name: math_value
                for name, math_value in zip(
                    JOINT_NAMES, [float(np.radians(value)) for value in angles_deg]
                )
            },
            "joint_angles_deg": [float(value) for value in angles_deg],
            "tcp_coords_mm_deg": [float(value) for value in coords],
            "gripper_action": (
                "close" if kind == "grasp" else "open" if kind == "release" else None
            ),
        }
        self.frames.append(frame)
        self.save()
        xyz = frame["tcp_coords_mm_deg"][:3]
        print(
            f"[已记录] #{frame['index']} {kind}: "
            f"xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm"
        )
        print(
            "[六轴姿态] "
            + ", ".join(
                f"J{index}={angle:.2f}°"
                for index, angle in enumerate(angles_deg, start=1)
            )
        )
        print(
            "[合法] 六个关节均位于当前旧款 mechArm 软件安全范围内；"
            "该点可作为安全抬升/返回姿态候选。"
        )
        print(f"[已保存] {self.output_path}")

    def undo(self) -> None:
        if not self.frames:
            print("[提示] 当前没有可撤销的关键点。")
            return
        removed = self.frames.pop()
        self.save()
        print(f"[撤销] 已删除 #{removed['index']} {removed['type']}")

    def run(self) -> None:
        self.check_connection()
        print(f"[新记录] {self.output_path}")
        print()
        print("请先用手托住机械臂，然后按 f 进入自由模式（无需 Enter）。")
        print("  f     释放六轴，进入自由模式")
        print("  l     重新锁定六轴")
        print("  q     保存普通关键点")
        print("  qa    保存夹取点标签（只记录，不执行夹爪闭合）")
        print("  qb    保存释放点标签（只记录，不执行夹爪张开）")
        print("  u     撤销最后一个关键点")
        print("  s     显示当前保存状态")
        print("  Esc   保存、锁定六轴并退出")
        print()

        try:
            with raw_keyboard():
                while True:
                    key = read_key()
                    lowered = key.lower()
                    try:
                        if lowered == "f":
                            self.enable_free_mode()
                        elif lowered == "l":
                            self.lock_servos()
                        elif lowered == "q":
                            kind, suffix = read_q_sequence()
                            if suffix not in (None, "a", "b"):
                                print(f"[提示] q{suffix} 未识别，按普通关键点记录。")
                            self.capture(kind)
                        elif lowered == "u":
                            self.undo()
                        elif lowered == "s":
                            self.save()
                            print(
                                f"[状态] 已保存 {len(self.frames)} 点，"
                                f"序列合法={self.sequence_complete()}：{self.output_path}"
                            )
                        elif key == "\x1b":
                            if self.carrying():
                                print("[拒绝退出] 已记录 qa 但还没有 qb；请补充释放点或按 u 撤销。")
                                continue
                            break
                    except RuntimeError as error:
                        print(f"[拒绝] {error}")
        except KeyboardInterrupt:
            print("\n[停止] 收到 Ctrl+C。")
        finally:
            self.save()
            # Always send focus_all_servos() on exit. This covers stale status,
            # exceptions during release verification, and interrupted startup.
            self.lock_servos(force=True)
            print(f"[完成] 已保存 {len(self.frames)} 个关键点：{self.output_path}")


def self_test() -> None:
    assert classify_suffix(None) == "waypoint"
    assert classify_suffix("a") == "grasp"
    assert classify_suffix("b") == "release"
    rotation = Rotation.from_euler("xyz", [10.0, -20.0, 30.0], degrees=True)
    recovered = rotation.as_euler("xyz", degrees=True)
    assert np.allclose(recovered, [10.0, -20.0, 30.0])

    class StaleFirmware:
        def __init__(self) -> None:
            self.released = False
            self.focused = False

        def release_all_servos(self):
            self.released = True
            return -1

        def focus_all_servos(self):
            self.focused = True
            return -1

        def is_all_servo_enable(self):
            return 1

        def is_servo_enable(self, _axis):
            return 1

    fake_recorder = object.__new__(RealRecorder)
    fake_recorder.arm = StaleFirmware()
    fake_recorder.free = False
    fake_recorder.enable_free_mode(verification_timeout=0.01)
    assert fake_recorder.free
    assert fake_recorder.arm.released
    fake_recorder.lock_servos(force=True, verification_timeout=0.01)
    assert not fake_recorder.free
    assert fake_recorder.arm.focused
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "hardware_demos"
    )
    parser.add_argument("--gripper-calibration", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    recorder = RealRecorder(
        args.host,
        args.port,
        args.output_root,
        gripper_calibration=args.gripper_calibration,
    )
    recorder.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
