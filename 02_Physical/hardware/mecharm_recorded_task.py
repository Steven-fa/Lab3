"""Replay physical teaching keyframes with calibrated adaptive-gripper values."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
VENV_PACKAGES = ROOT / ".venv_mecharm_hw" / "Lib" / "site-packages"
if VENV_PACKAGES.is_dir():
    sys.path.insert(0, str(VENV_PACKAGES))

from pymycobot import MechArmSocket  # noqa: E402
from pymycobot.common import ProtocolCode  # noqa: E402


DEFAULT_DEMO = (
    ROOT / "hardware_demos" / "real_teach_20260903_161121" / "keyframes.yaml"
)
# Official mechArm 270 joint ranges from the Elephant Robotics API reference.
SOFT_LOWER = np.array([-165.0, -90.0, -180.0, -165.0, -115.0, -175.0])
SOFT_UPPER = np.array([165.0, 90.0, 65.0, 165.0, 115.0, 175.0])
RELEASE_LIFT_MM = 60.0
RELEASE_MIN_CLEARANCE_MM = 40.0
GRIPPER_POLL_INTERVAL_S = 0.35
GRIPPER_SETTLE_SAMPLES = 3
GRIPPER_SETTLE_SPREAD = 2
GRIPPER_MIN_MOVEMENT = 3
GRIPPER_TARGET_TOLERANCE = 3
GRIPPER_TIMEOUT_S = 6.0
GRIPPER_MAX_SAMPLES = math.ceil(GRIPPER_TIMEOUT_S / GRIPPER_POLL_INTERVAL_S)
EXPERIMENT3_CONTROL_LOCK = Path("/tmp/mecharm_experiment3_control.lock")


def acquire_experiment3_control_lock(owner: str):
    """Hold one Jetson-wide advisory lock for every real Experiment 3 client."""
    import fcntl

    EXPERIMENT3_CONTROL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    stream = EXPERIMENT3_CONTROL_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        holder = stream.read().strip() or "未知进程"
        stream.close()
        raise RuntimeError(
            f"机械臂控制权已被占用：{holder}；请先关闭原程序，禁止并发连接9000端口"
        ) from None
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()} owner={owner}\n")
    stream.flush()
    return stream
GRIPPER_STALL_SAMPLES = 6
GRIPPER_CALIBRATED_OPEN_MIN = 84
GRIPPER_RETRY_PAUSE_S = 0.75
GRIPPER_CLEAR_MAX_SAMPLES = 6
# Successful physical measurements from the release pose.  These two joint
# keypoints replace the slower Cartesian IK/correction loop.
RELEASE_LIFT_KEYPOINTS_DEG = (
    np.array([121.8, 33.2, -33.1, -5.2, 71.0, 30.0]),
    np.array([121.9, 28.4, -34.9, -5.3, 72.7, 30.0]),
)
SKIPPED_RUNTIME_KEYFRAMES = {9}


class GripperActuationError(RuntimeError):
    """A gripper command produced no trustworthy physical response."""


def legacy_gripper_value(arm) -> int:
    """Read the adaptive gripper with the legacy 2.9.9 zero-argument frame."""
    return arm._mesg(ProtocolCode.GET_GRIPPER_VALUE, has_reply=True)


def legacy_set_gripper_value(arm, value: int, speed: int):
    """Send one legacy 2.9.9 gripper frame without a type byte or retry."""
    return arm._mesg(
        ProtocolCode.SET_GRIPPER_VALUE,
        value,
        speed,
        has_reply=False,
        _async=True,
    )


def legacy_set_gripper_state(arm, flag: int, speed: int):
    """Force the legacy adaptive gripper open/closed with its state frame."""
    return arm._mesg(
        ProtocolCode.SET_GRIPPER_STATE,
        flag,
        speed,
        has_reply=False,
        _async=True,
    )


def load_task(keyframes_path: Path, gripper_path: Path) -> tuple[list[dict], dict]:
    keyframes_document = yaml.safe_load(keyframes_path.read_text(encoding="utf-8"))
    gripper = yaml.safe_load(gripper_path.read_text(encoding="utf-8"))
    if keyframes_document.get("format") != "mecharm_real_keyframes/v1":
        raise RuntimeError("关键点文件不是真机录制格式")
    frames = keyframes_document.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("关键点文件为空")
    if gripper.get("format") != "mecharm_gripper_calibration/v1":
        raise RuntimeError("夹爪标定文件格式无效")
    if not gripper.get("complete"):
        raise RuntimeError("夹爪 OPEN/CLOSE 标定尚未完成")
    open_value = gripper.get("open")
    close_value = gripper.get("close")
    if not all(isinstance(value, int) and 0 <= value <= 100 for value in (open_value, close_value)):
        raise RuntimeError("夹爪 OPEN/CLOSE 必须在 0~100")
    if close_value >= open_value:
        raise RuntimeError("夹爪 CLOSE 必须小于 OPEN")

    carrying = False
    for expected_index, frame in enumerate(frames, start=1):
        if frame.get("index") != expected_index:
            raise RuntimeError("关键点编号不连续")
        kind = frame.get("type")
        if kind not in {"waypoint", "grasp", "release"}:
            raise RuntimeError(f"第 {expected_index} 点类型无效：{kind}")
        angles = np.asarray(frame.get("joint_angles_deg"), dtype=float)
        if angles.shape != (6,):
            raise RuntimeError(f"第 {expected_index} 点缺少六轴角度")
        if np.any(angles < SOFT_LOWER) or np.any(angles > SOFT_UPPER):
            raise RuntimeError(f"第 {expected_index} 点超出软件限位")
        if kind == "grasp":
            if carrying:
                raise RuntimeError("关键点包含重复夹取")
            carrying = True
        elif kind == "release":
            if not carrying:
                raise RuntimeError("释放点出现在夹取点之前")
            carrying = False
    if carrying:
        raise RuntimeError("夹取点之后缺少释放点")
    return frames, gripper


def expanded_targets(
    start: np.ndarray, target: np.ndarray, max_step: float = 15.0
) -> list[np.ndarray]:
    count = max(1, math.ceil(float(np.max(np.abs(target - start))) / max_step))
    return [start + (target - start) * (index / count) for index in range(1, count + 1)]


def gripper_completion_reason(
    samples: list[int],
    target: int,
    before: int,
    *,
    require_close_target: bool = False,
    close_target_tolerance: int = GRIPPER_TARGET_TOLERANCE,
    close_target_lower_tolerance: int | None = None,
    close_target_upper_tolerance: int | None = None,
    close_contact_samples: int = GRIPPER_SETTLE_SAMPLES,
    close_contact_min_movement: int = GRIPPER_MIN_MOVEMENT,
    close_contact_min_value: int = 0,
    close_contact_max_value: int = 100,
) -> str | None:
    """Require observable evidence for close and a calibrated range for open."""
    if not samples:
        return None
    current = samples[-1]
    lower_tolerance = (
        close_target_tolerance
        if close_target_lower_tolerance is None
        else close_target_lower_tolerance
    )
    upper_tolerance = (
        close_target_tolerance
        if close_target_upper_tolerance is None
        else close_target_upper_tolerance
    )

    # Opening is a prerequisite for a safe descent.  A small move followed by
    # a stall (for example 79 -> 82) is not open enough and must not pass.
    if target >= before:
        if len(samples) < GRIPPER_SETTLE_SAMPLES:
            return None
        recent = samples[-GRIPPER_SETTLE_SAMPLES:]
        if max(recent) - min(recent) > GRIPPER_SETTLE_SPREAD:
            return None
        if abs(current - target) <= GRIPPER_TARGET_TOLERANCE:
            return "target"
        if target >= 95 and current >= GRIPPER_CALIBRATED_OPEN_MIN:
            return "calibrated-open"
        return None

    # Closing without any measurable movement is ambiguous: it may be real
    # object contact, a missed command, or a cable/servo fault.  Never infer
    # contact automatically from stationary feedback.
    if before - current < GRIPPER_MIN_MOVEMENT:
        return None
    if len(samples) < GRIPPER_SETTLE_SAMPLES:
        return None
    recent = samples[-GRIPPER_SETTLE_SAMPLES:]
    if max(recent) - min(recent) > GRIPPER_SETTLE_SPREAD:
        return None
    if target - lower_tolerance <= current <= target + upper_tolerance:
        return "target"
    if require_close_target:
        # A loaded legacy gripper can pause briefly while continuing an
        # asynchronous close.  Do not mistake that mid-travel plateau for
        # object contact when a calibrated class target is available.
        return None
    # A loaded gripper normally cannot reach its free-air target.  Accept that
    # physical stop only after a longer stable window, enough observed travel,
    # and a plausible contact position.  This prevents a brief high mid-travel
    # pause from being mistaken for a held object.
    if len(samples) < close_contact_samples:
        return None
    contact_window = samples[-close_contact_samples:]
    if max(contact_window) - min(contact_window) > GRIPPER_SETTLE_SPREAD:
        return None
    if before - current < close_contact_min_movement:
        return None
    if close_contact_min_value <= current <= close_contact_max_value:
        return "stable/contact"
    return None


def gripper_attempt_is_stalled(
    samples: list[int],
    target: int,
    before: int,
    stall_samples: int | None = None,
    *,
    require_close_target: bool = False,
    close_target_tolerance: int = GRIPPER_TARGET_TOLERANCE,
    close_target_lower_tolerance: int | None = None,
    close_target_upper_tolerance: int | None = None,
    close_contact_samples: int = GRIPPER_SETTLE_SAMPLES,
    close_contact_min_movement: int = GRIPPER_MIN_MOVEMENT,
    close_contact_min_value: int = 0,
    close_contact_max_value: int = 100,
) -> bool:
    """Stop an unproductive attempt early while preserving slow real motion."""
    required_samples = GRIPPER_STALL_SAMPLES if stall_samples is None else stall_samples
    if required_samples < GRIPPER_SETTLE_SAMPLES:
        raise ValueError("夹爪停滞判定样本数不能小于稳定判定样本数")
    if len(samples) < required_samples:
        return False
    recent = samples[-required_samples:]
    if max(recent) - min(recent) > GRIPPER_SETTLE_SPREAD:
        return False
    return gripper_completion_reason(
        samples,
        target,
        before,
        require_close_target=require_close_target,
        close_target_tolerance=close_target_tolerance,
        close_target_lower_tolerance=close_target_lower_tolerance,
        close_target_upper_tolerance=close_target_upper_tolerance,
        close_contact_samples=close_contact_samples,
        close_contact_min_movement=close_contact_min_movement,
        close_contact_min_value=close_contact_min_value,
        close_contact_max_value=close_contact_max_value,
    ) is None


class RecordedTaskRunner:
    def __init__(self, host: str, port: int, arm_speed: int, demo_dir: Path) -> None:
        self.arm = MechArmSocket(host, port)
        self.arm_speed = arm_speed
        self.motion_poll_interval = 0.6
        self.stopped = False
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = demo_dir / "runs" / f"run_{stamp}.yaml"
        self.events: list[dict] = []

    @staticmethod
    def retry_query(label: str, query, valid, attempts: int = 6):
        last = None
        for attempt in range(attempts):
            last = query()
            if valid(last):
                return last
            if attempt + 1 < attempts:
                time.sleep(0.7)
        raise RuntimeError(f"{label}连续 {attempts} 次无有效响应：{last!r}")

    def read_angles(self) -> np.ndarray:
        values = self.retry_query(
            "读取六轴角度",
            self.arm.get_angles,
            lambda value: isinstance(value, list) and len(value) == 6,
        )
        return np.asarray(values, dtype=float)

    def read_error(self):
        queries = []
        for method_name in ("get_error_information", "read_next_error"):
            method = getattr(self.arm, method_name, None)
            if method is not None:
                queries.append((method_name, method))
        last_error = None
        for method_name, query in queries:
            try:
                return self.retry_query(
                    f"读取错误状态({method_name})",
                    query,
                    lambda value: value != -1,
                    attempts=2,
                )
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(f"所有错误状态接口均无有效响应：{last_error}")

    @staticmethod
    def error_is_clear(value) -> bool:
        return value == 0 or (
            isinstance(value, list) and bool(value) and all(item == 0 for item in value)
        )

    def read_coords(self) -> np.ndarray:
        values = self.retry_query(
            "读取末端坐标",
            self.arm.get_coords,
            lambda value: isinstance(value, list) and len(value) == 6,
        )
        return np.asarray(values, dtype=float)

    def health(self) -> np.ndarray:
        power = self.retry_query(
            "读取上电状态", self.arm.is_power_on, lambda value: value != -1
        )
        enabled = self.retry_query(
            "读取舵机状态",
            self.arm.is_all_servo_enable,
            lambda value: value != -1,
        )
        error = self.read_error()
        if power != 1:
            raise RuntimeError("机械臂未上电")
        if enabled != 1:
            raise RuntimeError("六轴尚未锁定")
        if not self.error_is_clear(error):
            raise RuntimeError(f"控制器错误：{error}")
        return self.read_angles()

    def save_log(self, status: str) -> None:
        document = {
            "format": "mecharm_real_run/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "arm_speed": self.arm_speed,
            "events": self.events,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.log_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(self.log_path)

    def stop(self, *_args) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self.arm.stop()
        finally:
            self.events.append(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "emergency_stop",
                }
            )
            self.save_log("stopped")
            print("\n[急停] 已发送机械臂 stop()；夹爪保持当前状态。")

    def handle_sigint(self, *_args) -> None:
        self.stop()
        raise KeyboardInterrupt

    def move_segment(
        self, label: str, target: np.ndarray, tolerance_deg: float = 5.0
    ) -> None:
        if self.stopped:
            raise KeyboardInterrupt
        before = self.health()
        print(f"[运动] {label}: {np.round(target, 1).tolist()}")
        command_result = self.arm.send_angles(target.tolist(), self.arm_speed)
        deadline = time.monotonic() + 22.0
        actual = before
        reached = False
        while time.monotonic() < deadline:
            time.sleep(self.motion_poll_interval)
            actual = self.read_angles()
            error = self.read_error()
            if not self.error_is_clear(error):
                self.stop()
                raise RuntimeError(f"{label} 运动中报错：{error}")
            if float(np.max(np.abs(actual - target))) <= tolerance_deg:
                reached = True
                break
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "arm_move",
                "label": label,
                "target_deg": [round(float(value), 3) for value in target],
                "actual_deg": [round(float(value), 3) for value in actual],
                "command_result": command_result,
                "reached": reached,
            }
        )
        self.save_log("running")
        if not reached:
            self.stop()
            difference = float(np.max(np.abs(actual - target)))
            raise RuntimeError(f"{label} 到位超时，最大误差 {difference:.1f}°")

    def move_to(self, label: str, target: np.ndarray) -> None:
        start = self.read_angles()
        targets = expanded_targets(start, target)
        for index, intermediate in enumerate(targets, start=1):
            self.move_segment(f"{label} ({index}/{len(targets)})", intermediate)

    def gripper(
        self,
        label: str,
        value: int,
        speed: int,
        command_attempts: int = 1,
        confirm_stationary_close: bool = False,
        command_mode: str = "value",
        stop_on_failure: bool = True,
        clear_before_retry: bool = False,
        stall_samples: int | None = None,
        require_close_target: bool = False,
        close_target_tolerance: int = GRIPPER_TARGET_TOLERANCE,
        close_target_lower_tolerance: int | None = None,
        close_target_upper_tolerance: int | None = None,
        close_contact_samples: int = GRIPPER_SETTLE_SAMPLES,
        close_contact_min_movement: int = GRIPPER_MIN_MOVEMENT,
        close_contact_min_value: int = 0,
        close_contact_max_value: int = 100,
    ) -> int:
        if self.stopped:
            raise KeyboardInterrupt
        if command_attempts < 1:
            raise ValueError("夹爪命令尝试次数必须至少为1")
        if command_mode not in {"value", "state"}:
            raise ValueError("夹爪命令模式必须为value或state")
        if close_target_tolerance < 0:
            raise ValueError("夹爪闭合目标容差不能小于0")
        if close_target_lower_tolerance is not None and close_target_lower_tolerance < 0:
            raise ValueError("夹爪闭合目标下偏差不能小于0")
        if close_target_upper_tolerance is not None and close_target_upper_tolerance < 0:
            raise ValueError("夹爪闭合目标上偏差不能小于0")
        if close_contact_samples < GRIPPER_SETTLE_SAMPLES:
            raise ValueError("夹爪接触稳定样本数不能小于普通稳定样本数")
        if close_contact_min_movement < GRIPPER_MIN_MOVEMENT:
            raise ValueError("夹爪接触最小位移不能小于普通可验证位移")
        if not 0 <= close_contact_min_value <= 100:
            raise ValueError("夹爪接触位置下限必须在0~100")
        if not close_contact_min_value <= close_contact_max_value <= 100:
            raise ValueError("夹爪接触位置范围必须满足0<=下限<=上限<=100")
        initial_before = self.retry_query(
            "读取夹爪初始位置",
            lambda: legacy_gripper_value(self.arm),
            lambda result: isinstance(result, int) and 0 <= result <= 100,
        )
        print(f"[夹爪] {label}: target={value}, before={initial_before}")
        if (
            value >= GRIPPER_CALIBRATED_OPEN_MIN
            and initial_before >= GRIPPER_CALIBRATED_OPEN_MIN
        ):
            self.events.append(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "gripper",
                    "label": label,
                    "target": value,
                    "before": initial_before,
                    "actual": initial_before,
                    "samples": [initial_before],
                    "completion": "already-open",
                    "command_result": None,
                    "attempt_count": 0,
                    "attempts": [],
                }
            )
            self.save_log("running")
            print(
                f"[夹爪] {label} 已处于张开安全区，"
                f"回读={initial_before}；自检只读通过，未发送OPEN命令"
            )
            return initial_before
        before = initial_before
        all_samples: list[int] = []
        attempt_records: list[dict] = []
        completion = None
        after = before
        command_result = None
        for attempt_index in range(1, command_attempts + 1):
            if command_mode == "state":
                state_flag = 0 if value >= initial_before else 1
                sent_mode = "state-open" if state_flag == 0 else "state-close"
                command_result = legacy_set_gripper_state(
                    self.arm, state_flag, speed
                )
            else:
                sent_mode = "value"
                command_result = legacy_set_gripper_value(self.arm, value, speed)
            samples: list[int] = []
            ended_by = "timeout"
            for _sample_index in range(GRIPPER_MAX_SAMPLES):
                time.sleep(GRIPPER_POLL_INTERVAL_S)
                after = self.retry_query(
                    "读取夹爪位置",
                    lambda: legacy_gripper_value(self.arm),
                    lambda result: isinstance(result, int) and 0 <= result <= 100,
                )
                samples.append(after)
                all_samples.append(after)
                completion = gripper_completion_reason(
                    samples,
                    value,
                    before,
                    require_close_target=require_close_target,
                    close_target_tolerance=close_target_tolerance,
                    close_target_lower_tolerance=close_target_lower_tolerance,
                    close_target_upper_tolerance=close_target_upper_tolerance,
                    close_contact_samples=close_contact_samples,
                    close_contact_min_movement=close_contact_min_movement,
                    close_contact_min_value=close_contact_min_value,
                    close_contact_max_value=close_contact_max_value,
                )
                if completion is not None:
                    ended_by = "completed"
                    break
                if gripper_attempt_is_stalled(
                    samples,
                    value,
                    before,
                    stall_samples=stall_samples,
                    require_close_target=require_close_target,
                    close_target_tolerance=close_target_tolerance,
                    close_target_lower_tolerance=close_target_lower_tolerance,
                    close_target_upper_tolerance=close_target_upper_tolerance,
                    close_contact_samples=close_contact_samples,
                    close_contact_min_movement=close_contact_min_movement,
                    close_contact_min_value=close_contact_min_value,
                    close_contact_max_value=close_contact_max_value,
                ):
                    ended_by = "stalled"
                    break
            attempt_records.append(
                {
                    "attempt": attempt_index,
                    "command_mode": sent_mode,
                    "before": before,
                    "samples": samples.copy(),
                    "command_result": command_result,
                    "completion": completion,
                    "ended_by": ended_by,
                }
            )
            if completion is not None:
                break
            if attempt_index < command_attempts:
                if clear_before_retry:
                    print(
                        f"[夹爪命令清除] {label} 第{attempt_index}次无有效动作；"
                        "先发送RELEASE(254)，确认停止后才允许重试。"
                    )
                    clear_result = legacy_set_gripper_state(self.arm, 254, speed)
                    clear_samples: list[int] = []
                    clear_confirmed = False
                    for _clear_index in range(
                        min(GRIPPER_MAX_SAMPLES, GRIPPER_CLEAR_MAX_SAMPLES)
                    ):
                        time.sleep(GRIPPER_POLL_INTERVAL_S)
                        after = self.retry_query(
                            "清除后读取夹爪位置",
                            lambda: legacy_gripper_value(self.arm),
                            lambda result: isinstance(result, int)
                            and 0 <= result <= 100,
                        )
                        clear_samples.append(after)
                        all_samples.append(after)
                        if len(clear_samples) >= GRIPPER_SETTLE_SAMPLES:
                            recent = clear_samples[-GRIPPER_SETTLE_SAMPLES:]
                            if max(recent) - min(recent) <= GRIPPER_SETTLE_SPREAD:
                                clear_confirmed = True
                                break
                    attempt_records[-1]["clear_barrier"] = {
                        "command": "state-release-254",
                        "command_result": clear_result,
                        "samples": clear_samples,
                        "confirmed_stopped": clear_confirmed,
                    }
                    if not clear_confirmed:
                        print(
                            f"[夹爪命令清除失败] {label} RELEASE后仍未稳定，"
                            "禁止继续追加命令。"
                        )
                        break
                    print(
                        f"[夹爪命令清除完成] {label} 回读稳定在{after}；"
                        "现在只重试一次。"
                    )
                else:
                    print(
                        f"[夹爪重试] {label} 第{attempt_index}次没有达到可验证状态，"
                        f"回读={after}，结束原因={ended_by}；等待后重发一次。"
                    )
                    time.sleep(GRIPPER_RETRY_PAUSE_S)
                before = after

        if completion is None and confirm_stationary_close and value < initial_before:
            if self.confirm_stationary_close(label, initial_before, after, all_samples):
                completion = "operator-confirmed-contact"

        if completion is None:
            # The last asynchronous state command must also be cancelled.  A
            # later socket reconnect must never allow this failed command to
            # execute after the arm has moved to another grid.
            if clear_before_retry and attempt_records and (
                "clear_barrier" not in attempt_records[-1]
            ):
                print(
                    f"[夹爪最终清除] {label} 最后一次命令未产生有效动作；"
                    "发送RELEASE(254)并等待反馈稳定。"
                )
                clear_result = legacy_set_gripper_state(self.arm, 254, speed)
                clear_samples: list[int] = []
                clear_confirmed = False
                for _clear_index in range(
                    min(GRIPPER_MAX_SAMPLES, GRIPPER_CLEAR_MAX_SAMPLES)
                ):
                    time.sleep(GRIPPER_POLL_INTERVAL_S)
                    after = self.retry_query(
                        "最终清除后读取夹爪位置",
                        lambda: legacy_gripper_value(self.arm),
                        lambda result: isinstance(result, int)
                        and 0 <= result <= 100,
                    )
                    clear_samples.append(after)
                    all_samples.append(after)
                    if len(clear_samples) >= GRIPPER_SETTLE_SAMPLES:
                        recent = clear_samples[-GRIPPER_SETTLE_SAMPLES:]
                        if max(recent) - min(recent) <= GRIPPER_SETTLE_SPREAD:
                            clear_confirmed = True
                            break
                attempt_records[-1]["clear_barrier"] = {
                    "command": "state-release-254",
                    "command_result": clear_result,
                    "samples": clear_samples,
                    "confirmed_stopped": clear_confirmed,
                    "final": True,
                }
                if not clear_confirmed:
                    print(
                        f"[夹爪最终清除失败] {label} RELEASE后仍未稳定；"
                        "禁止继续运动。"
                    )
            self.events.append(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "gripper_failure",
                    "label": label,
                    "target": value,
                    "before": initial_before,
                    "actual": after,
                    "samples": all_samples,
                    "attempt_count": len(attempt_records),
                    "attempts": attempt_records,
                    "reason": "no_verified_motion_or_open_range",
                }
            )
            self.save_log("running")
            if stop_on_failure:
                self.stop()
            detail = (
                "没有可验证的夹爪位移，且未获得现场HOLD确认"
                if value < initial_before and confirm_stationary_close
                else "强驱动后仍没有可验证的夹爪闭合位移"
                if value < initial_before
                else f"未稳定达到张开安全区>={GRIPPER_CALIBRATED_OPEN_MIN}"
            )
            raise GripperActuationError(
                f"{label} {detail}：target={value}, "
                f"before={initial_before}, samples={all_samples}"
            )
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "gripper",
                "label": label,
                "target": value,
                "before": initial_before,
                "actual": after,
                "samples": all_samples,
                "completion": completion,
                "command_result": command_result,
                "attempt_count": len(attempt_records),
                "attempts": attempt_records,
            }
        )
        self.save_log("running")
        print(
            f"[夹爪] {label} 完成，回读={after}，确认方式={completion}"
        )
        return after

    def confirm_stationary_close(
        self,
        label: str,
        before: int,
        after: int,
        samples: list[int],
    ) -> bool:
        print(
            f"[夹爪待确认] {label} 命令后没有可验证闭合位移："
            f"{before} -> {after}。程序无法区分物体已填满夹爪与命令未执行。"
        )
        print("[现场检查] 保持机械臂静止，确认两侧夹指都已压紧物体。")
        try:
            answer = input(
                "确认确实夹牢请输入 HOLD；直接按 Enter 或输入其他内容将安全停止："
            )
        except EOFError:
            return False
        return answer.strip().upper() == "HOLD"

    def vertical_release_lift(self) -> None:
        before = self.read_coords()
        print(
            f"[竖直抬升] 起点 xyz={np.round(before[:3], 1).tolist()} mm，"
            f"执行 {len(RELEASE_LIFT_KEYPOINTS_DEG)} 个实测固定关键点"
        )
        total = len(RELEASE_LIFT_KEYPOINTS_DEG)
        for index, target in enumerate(RELEASE_LIFT_KEYPOINTS_DEG, start=1):
            self.move_segment(
                f"释放后竖直抬升固定关键点 {index}/{total}",
                target,
                tolerance_deg=5.0,
            )
            actual_coords = self.read_coords()
            self.events.append(
                {
                    "time": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "vertical_lift_step",
                    "step": index,
                    "target_deg": [round(float(value), 3) for value in target],
                    "actual_xyz_mm": [
                        round(float(value), 3) for value in actual_coords[:3]
                    ],
                }
            )
            self.save_log("running")
            print(
                f"[竖直抬升] 固定关键点 {index}/{total} 实际 xyz="
                f"{np.round(actual_coords[:3], 1).tolist()} mm"
            )
        after = self.read_coords()
        delta = after[:3] - before[:3]
        xy_drift = float(np.linalg.norm(delta[:2]))
        z_lift = float(delta[2])
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "vertical_release_lift",
                "before_xyz_mm": [round(float(value), 3) for value in before[:3]],
                "after_xyz_mm": [round(float(value), 3) for value in after[:3]],
                "delta_xyz_mm": [round(float(value), 3) for value in delta],
                "xy_drift_mm": round(xy_drift, 3),
                "z_lift_mm": round(z_lift, 3),
            }
        )
        self.save_log("running")
        print(
            f"[竖直抬升] 实际 Δxyz={np.round(delta, 1).tolist()} mm，"
            f"XY漂移={xy_drift:.1f} mm"
        )
        if z_lift < RELEASE_MIN_CLEARANCE_MM or xy_drift > 15.0:
            self.stop()
            raise RuntimeError(
                f"释放后抬升验证失败：Z={z_lift:.1f} mm，"
                f"XY漂移={xy_drift:.1f} mm"
            )

    def run(self, frames: list[dict], gripper: dict) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        current = self.health()
        print(f"[真机] 当前角度：{np.round(current, 1).tolist()}")
        self.gripper("运行前 OPEN", int(gripper["open"]), int(gripper["speed"]))

        for frame in frames:
            index = int(frame["index"])
            kind = str(frame["type"])
            target = np.asarray(frame["joint_angles_deg"], dtype=float)
            self.move_to(f"关键点 {index} / {kind}", target)
            if kind == "grasp":
                self.gripper(
                    f"关键点 {index} CLOSE",
                    int(gripper["close"]),
                    int(gripper["speed"]),
                )
            elif kind == "release":
                self.gripper(
                    f"关键点 {index} OPEN",
                    int(gripper["open"]),
                    int(gripper["speed"]),
                )
                self.vertical_release_lift()
        self.save_log("completed")
        print(f"[完成] 高速精简抓放路线已完整执行。运行记录：{self.log_path}")


def print_preflight(frames: list[dict], gripper: dict) -> None:
    print(f"[预检] 关键点：{len(frames)}")
    print("[预检] 类型：" + " -> ".join(str(frame["type"]) for frame in frames))
    print(
        f"[预检] 夹爪：OPEN={gripper['open']}, CLOSE={gripper['close']}, "
        f"speed={gripper['speed']}"
    )
    maximum = max(
        float(
            np.max(
                np.abs(
                    np.asarray(current["joint_angles_deg"], dtype=float)
                    - np.asarray(previous["joint_angles_deg"], dtype=float)
                )
            )
        )
        for previous, current in zip(frames, frames[1:])
    )
    print(f"[预检] 原始关键点最大关节变化：{maximum:.1f}°（执行时自动细分）")
    release_angles = np.asarray(
        next(frame for frame in frames if frame["type"] == "release")[
            "joint_angles_deg"
        ],
        dtype=float,
    )
    lift_max_step = max(
        float(np.max(np.abs(current - previous)))
        for previous, current in zip(
            [release_angles, *RELEASE_LIFT_KEYPOINTS_DEG[:-1]],
            RELEASE_LIFT_KEYPOINTS_DEG,
        )
    )
    print(
        f"[预检] 释放后：OPEN={gripper['open']}，"
        f"执行 {len(RELEASE_LIFT_KEYPOINTS_DEG)} 个实测固定抬升关键点，"
        f"最大关节步长 {lift_max_step:.1f}°"
    )
    print(
        "[预检] 原关键点 9 已跳过；固定抬升完成后直接前往关键点 10。"
        f"最终安全条件：Z抬升 >= {RELEASE_MIN_CLEARANCE_MM:.0f} mm，"
        "XY漂移 <= 15 mm"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--gripper", type=Path)
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--gripper-speed", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    gripper_path = args.gripper or args.keyframes.with_name(
        "gripper_calibration.yaml"
    )
    frames, gripper = load_task(args.keyframes, gripper_path)
    frames = [
        frame
        for frame in frames
        if int(frame["index"]) not in SKIPPED_RUNTIME_KEYFRAMES
    ]
    if args.gripper_speed is not None:
        if not 1 <= args.gripper_speed <= 100:
            raise RuntimeError("夹爪速度必须在 1~100")
        gripper = dict(gripper)
        gripper["speed"] = args.gripper_speed
    print_preflight(frames, gripper)
    if not args.execute:
        print("[只读] 未发送任何真机命令。")
        return 0

    runner = RecordedTaskRunner(
        args.host, args.port, args.arm_speed, args.keyframes.parent
    )
    current = runner.health()
    print(f"[当前角度] {np.round(current, 1).tolist()}")
    print("[安全] 清空底座周围 30 cm，把工件放在示教夹取位置，并准备急停。")
    print("[急停] 运动中按 Ctrl+C；程序不会自动松开正在夹持的工件。")
    if not args.yes:
        confirmation = input("输入 RUN 并按 Enter 开始完整真机测试：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送运动命令。")
            return 2
    try:
        runner.run(frames, gripper)
    except Exception:
        if not runner.stopped:
            runner.stop()
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止完整测试。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
