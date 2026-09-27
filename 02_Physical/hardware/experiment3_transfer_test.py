"""Low-speed single-object transfer using Experiment 3 taught poses."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np
import yaml

from experiment3_directional_compensation import (
    DirectionalJointCompensator,
    load_compensation,
)
from experiment3_grid_dry_run import EXPECTED_FORMAT, GridDryRunner
from experiment3_gripper_profiles import load_gripper_profiles, profile_for_group
from mecharm_recorded_task import (
    GRIPPER_CALIBRATED_OPEN_MIN,
    GRIPPER_MAX_SAMPLES,
    GripperActuationError,
    SOFT_LOWER,
    SOFT_UPPER,
    legacy_gripper_value,
    acquire_experiment3_control_lock,
)
from pymycobot import MechArmSocket
from pymycobot.common import ProtocolCode


LEGACY_REPLAY_LOWER = SOFT_LOWER.copy()
LEGACY_REPLAY_UPPER = SOFT_UPPER.copy()
# This old mechArm reports the two hand-taught lower-row poses slightly above
# the generic SDK's J2 +90 degree limit.  Raw position streaming already
# bypasses that SDK check; keep the compatibility allowance narrow.
LEGACY_REPLAY_LOWER[1] = -92.0
LEGACY_REPLAY_UPPER[1] = 92.0
DESTINATION_BY_CLASS = {"cube": "right", "other": "left"}
MINIMUM_RELEASE_RETREAT_MM = 75.0
CARRY_LIFT_MAX_ATTEMPTS = 9
CARRY_LIFT_POLL_INTERVAL_S = 0.75
CARRY_LIFT_TAUGHT_ALLOWANCE_MM = 15.0
CARRY_LIFT_XY_TOLERANCE_MM = 12.0
RELEASE_RETREAT_MAX_ATTEMPTS = 9
RELEASE_RETREAT_POLL_INTERVAL_S = 0.75
RELEASE_RETREAT_REQUIRED_LIFT_MM = 75.0
# The controller reports TCP coordinates at 0.1 mm resolution and the old arm
# can dither by a few tenths while holding position.  Treat only this narrow
# measurement band as equivalent to the commanded 75 mm retreat; larger
# shortfalls still fail closed.
RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM = 0.5
RELEASE_RETREAT_EXTENSION_STEP = 0.25
RELEASE_RETREAT_MAX_EXTENSION = 3.0
GRASP_CONVERGENCE_MAX_ATTEMPTS = 9
GRASP_CONVERGENCE_POLL_INTERVAL_S = 0.75
GRASP_TCP_XY_TOLERANCE_MM = 4.0
GRASP_TCP_Z_TOLERANCE_MM = 3.0
GRASP_STABLE_SAMPLES = 2
GRASP_OTHER_JOINT_TOLERANCE_DEG = 3.0
GRASP_J5_SAFETY_GUARD_DEG = 8.0
MANUAL_POSE_MAX_ATTEMPTS = 8
MANUAL_POSE_POLL_INTERVAL_S = 0.75
MANUAL_POSE_XYZ_TOLERANCE_MM = 8.0
MANUAL_POSE_RPY_TOLERANCE_DEG = 8.0
MANUAL_POSE_JOINT_GUARD_DEG = 8.0
MANUAL_POSE_STABLE_JOINT_MOTION_DEG = 1.0
MANUAL_POSE_STABLE_TCP_MOTION_MM = 1.5
MANUAL_RELEASE_XYZ_TOLERANCE_MM = 15.0
MANUAL_RELEASE_JOINT_GUARD_DEG = 3.0
MANUAL_GRASP_TCP_XY_TOLERANCE_MM = 5.0
MANUAL_GRASP_TCP_Z_TOLERANCE_MM = MANUAL_POSE_XYZ_TOLERANCE_MM
PRE_DESCENT_MIN_CLEARANCE_MM = 50.0
PRE_DESCENT_TARGET_CLEARANCE_MM = 65.0
PRE_DESCENT_TARGET_HEADROOM_MM = 5.0
PRE_DESCENT_MAX_ATTEMPTS = 8
PRE_DESCENT_POLL_INTERVAL_S = 0.75
PRE_DESCENT_MEASUREMENT_TOLERANCE_MM = 1.0
SOURCE_GRASP_EXTRA_Z_MM_BY_GRID = {3: 7.0, 6: 10.0}
GRID3_APPROACH_EXTRA_Z_MM = 20.0
SOURCE_APPROACH_MAX_ATTEMPTS = 8
SOURCE_APPROACH_POLL_INTERVAL_S = 0.75
SOURCE_APPROACH_HEIGHT_TOLERANCE_MM = 1.0
SOURCE_APPROACH_XY_GUARD_MM = 15.0
GRIPPER_SESSION_ATTEMPTS = 3
GRIPPER_RECONNECT_SETTLE_S = 1.0
GRIPPER_RECONNECT_MAX_DRIFT_DEG = 3.0
GRIPPER_HELD_MAX_VALUE = 78
GRIPPER_HOLD_MAX_OPENING_DRIFT = 15
GRIPPER_HOLD_MAX_CLOSING_DRIFT = 10
GRIPPER_HOLD_MAX_SAMPLES = 5
GRIPPER_HOLD_STABLE_SAMPLES = 2
GRIPPER_HOLD_POLL_INTERVAL_S = 0.25
CLASS_CLOSE_TARGET_TOLERANCE = 5
CUBE_CLOSE_TARGET = 69
CUBE_CLOSE_LOWER_TOLERANCE = 5
CUBE_CLOSE_UPPER_TOLERANCE = 3
CUBE_CLOSE_CONTACT_VALUE = CUBE_CLOSE_TARGET + CUBE_CLOSE_UPPER_TOLERANCE + 1
# Production close accepts a loaded-object plateau only after about 2.8 s of
# stable feedback.  Values above 73 are treated as a high mid-travel pause,
# while at least five feedback units of closing proves the command acted.
CLASS_CLOSE_CONTACT_SAMPLES = 8
CLASS_CLOSE_CONTACT_MIN_MOVEMENT = 5
CLASS_CLOSE_CONTACT_MAX_VALUE = 73
DEFAULT_GRIPPER_PROFILES = Path(
    "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
)


def release_retreat_lift_is_sufficient(actual_lift_mm: float) -> bool:
    return (
        float(actual_lift_mm) + RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM
        >= RELEASE_RETREAT_REQUIRED_LIFT_MM
    )


def hold_is_stable(
    reference: int,
    current: int,
    max_opening_drift: int = GRIPPER_HOLD_MAX_OPENING_DRIFT,
    max_closing_drift: int = GRIPPER_HOLD_MAX_CLOSING_DRIFT,
    maximum_held_value: int = GRIPPER_HELD_MAX_VALUE,
) -> bool:
    """Keep feedback near the first contact; large closing means a slipped load."""
    minimum_contact_value = max(0, reference - max_closing_drift)
    maximum_contact_value = min(
        maximum_held_value,
        reference + max_opening_drift,
    )
    return minimum_contact_value <= current <= maximum_contact_value


def grasp_z_tolerance(pose: dict) -> float:
    """Use the measured manual-pose tolerance for hand-corrected grasp points."""
    if pose.get("manual_command_calibrated") is True:
        return MANUAL_GRASP_TCP_Z_TOLERANCE_MM
    return GRASP_TCP_Z_TOLERANCE_MM


def grasp_xy_tolerance(pose: dict) -> float:
    """Allow the measured XY repeatability envelope only for hand-corrected points."""
    if pose.get("manual_command_calibrated") is True:
        return MANUAL_GRASP_TCP_XY_TOLERANCE_MM
    return GRASP_TCP_XY_TOLERANCE_MM


def manual_pose_acceptance_limits(pose: dict) -> tuple[float, float]:
    """Use joint-faithful validation for the shared hand-taught release poses."""
    if str(pose.get("name", "")).endswith("_release"):
        return MANUAL_RELEASE_XYZ_TOLERANCE_MM, MANUAL_RELEASE_JOINT_GUARD_DEG
    return MANUAL_POSE_XYZ_TOLERANCE_MM, MANUAL_POSE_JOINT_GUARD_DEG


def build_raised_source_grasp(
    approach: dict,
    grasp: dict,
    extra_z_mm: float,
) -> dict:
    """Raise a grasp along its hand-taught path without requiring old-firmware IK."""
    extra_z = float(extra_z_mm)
    if extra_z <= 0.0:
        raise RuntimeError("抓取点的上调高度必须大于0mm")
    approach_angles = np.asarray(approach["angles"], dtype=float)
    grasp_angles = np.asarray(grasp["angles"], dtype=float)
    approach_coords = np.asarray(approach["coords"], dtype=float)
    grasp_coords = np.asarray(grasp["coords"], dtype=float)
    taught_lift = float(approach_coords[2] - grasp_coords[2])
    if taught_lift <= extra_z:
        raise RuntimeError(
            f"{grasp['name']}到上方点只有{taught_lift:.1f}mm，"
            f"不能在不越过上方点的情况下上调{extra_z:.1f}mm"
        )
    fraction = extra_z / taught_lift
    angles = grasp_angles + fraction * (approach_angles - grasp_angles)
    if np.any(angles < LEGACY_REPLAY_LOWER) or np.any(angles > LEGACY_REPLAY_UPPER):
        raise RuntimeError(f"{grasp['name']}上调后的抓取点超出软件关节限位")
    coords = grasp_coords + fraction * (approach_coords - grasp_coords)
    return {
        "name": f"{grasp['name'].replace('_grasp', '')}_raised_grasp",
        "angles": angles,
        "coords": coords,
        "manual_command_calibrated": True,
        "generated_source_grasp": True,
        "requested_extra_z_mm": extra_z,
        "original_grasp_z_mm": float(grasp_coords[2]),
        "generated_fraction": fraction,
    }


def build_raised_source_approach(
    approach: dict,
    grasp: dict,
    extra_z_mm: float,
) -> dict:
    """Extend a taught grasp-to-approach path to gain real source clearance."""
    extra_z = float(extra_z_mm)
    if extra_z <= 0.0:
        raise RuntimeError("抓取上方点的额外抬升必须大于0mm")
    approach_angles = np.asarray(approach["angles"], dtype=float)
    grasp_angles = np.asarray(grasp["angles"], dtype=float)
    approach_coords = np.asarray(approach["coords"], dtype=float)
    grasp_coords = np.asarray(grasp["coords"], dtype=float)
    taught_lift = float(approach_coords[2] - grasp_coords[2])
    if taught_lift < 10.0:
        raise RuntimeError(
            f"{approach['name']}仅比抓取点高{taught_lift:.1f}mm，"
            "无法沿原采集方向可靠生成加高点"
        )
    scale = (taught_lift + extra_z) / taught_lift
    angles = grasp_angles + scale * (approach_angles - grasp_angles)
    if np.any(angles < LEGACY_REPLAY_LOWER) or np.any(angles > LEGACY_REPLAY_UPPER):
        raise RuntimeError("3号格加高上方点超出软件关节限位")
    coords = grasp_coords + scale * (approach_coords - grasp_coords)
    return {
        "name": f"{approach['name'].replace('_approach', '')}_raised_approach",
        "angles": angles,
        "coords": coords,
        "manual_command_calibrated": True,
        "generated_source_approach": True,
        "requested_extra_z_mm": extra_z,
        "original_approach_z_mm": float(approach_coords[2]),
        "original_approach_xy_mm": approach_coords[:2].copy(),
        "generated_scale": scale,
    }


def build_high_retreat(
    approach: dict, release: dict, extension: float
) -> dict:
    """Extend the verified release-to-approach joint direction after release."""
    angles = approach["angles"] + extension * (
        approach["angles"] - release["angles"]
    )
    if np.any(angles < LEGACY_REPLAY_LOWER) or np.any(angles > LEGACY_REPLAY_UPPER):
        raise RuntimeError("释放后高撤离姿态超出软件关节限位")
    return {"name": "release_retreat_high", "angles": angles}


def effective_retreat_extension(
    approach: dict,
    release: dict,
    requested_extension: float,
    minimum_lift_mm: float = MINIMUM_RELEASE_RETREAT_MM,
) -> float:
    """Raise a short taught retreat enough to preserve a safe post-release lift."""
    taught_lift = float(approach["coords"][2] - release["coords"][2])
    if taught_lift < 10.0:
        raise RuntimeError(
            f"释放上方点仅比释放点高{taught_lift:.1f}mm，无法可靠生成高撤离点"
        )
    required = minimum_lift_mm / taught_lift - 1.0
    effective = max(float(requested_extension), required)
    if effective > RELEASE_RETREAT_MAX_EXTENSION:
        raise RuntimeError(
            f"需要的释放后延长比例{effective:.2f}过大，请重新采集更高的释放上方点"
        )
    return effective


def build_pre_descent_transition(
    approach: dict,
    release: dict,
    minimum_clearance_mm: float = PRE_DESCENT_MIN_CLEARANCE_MM,
    target_clearance_mm: float = PRE_DESCENT_TARGET_CLEARANCE_MM,
    target_headroom_mm: float = PRE_DESCENT_TARGET_HEADROOM_MM,
    minimum_absolute_z_mm: float | None = None,
) -> dict | None:
    """Create a high pass-through pose when a taught release approach is too low."""
    approach_coords = np.asarray(approach["coords"], dtype=float)
    release_coords = np.asarray(release["coords"], dtype=float)
    taught_clearance = float(approach_coords[2] - release_coords[2])
    required_absolute_z = float(release_coords[2] + minimum_clearance_mm)
    required_target_z = float(release_coords[2] + target_clearance_mm)
    if minimum_absolute_z_mm is not None:
        required_absolute_z = max(required_absolute_z, float(minimum_absolute_z_mm))
        required_target_z = max(required_target_z, float(minimum_absolute_z_mm))
    if float(approach_coords[2]) >= required_absolute_z:
        return None
    if taught_clearance < 10.0:
        raise RuntimeError(
            f"{approach['name']}仅比{release['name']}高{taught_clearance:.1f}mm，"
            "无法沿现有点位可靠生成高位过渡点"
        )
    if required_target_z < required_absolute_z:
        raise RuntimeError("自动高位目标不能低于最低进场高度")
    if target_headroom_mm < 0.0:
        raise RuntimeError("自动高位跟随余量不能为负数")
    # The old arm can settle 1-2 mm below a generated joint target.  Command
    # above the safety line instead of weakening the physical acceptance line.
    target_absolute_z = required_target_z + float(target_headroom_mm)
    required_clearance = required_target_z - float(release_coords[2])
    generated_clearance = target_absolute_z - float(release_coords[2])
    extension = generated_clearance / taught_clearance - 1.0
    if extension > RELEASE_RETREAT_MAX_EXTENSION:
        raise RuntimeError(
            f"{approach['name']}需要的自动高位扩展比例{extension:.2f}过大"
        )
    generated = build_high_retreat(approach, release, extension)
    scale = 1.0 + extension
    generated_coords = release_coords + scale * (approach_coords - release_coords)
    side = str(approach["name"]).split("_", 1)[0]
    generated.update(
        {
            "name": f"{side}_generated_high_approach",
            "coords": generated_coords,
            "manual_command_calibrated": True,
            "generated": True,
            "generated_extension": extension,
            "required_clearance_mm": required_clearance,
            "generated_clearance_mm": float(generated_coords[2] - release_coords[2]),
            "target_headroom_mm": float(target_headroom_mm),
        }
    )
    return generated


def validate_active_correction_document(document: dict) -> None:
    correction = document.get("correction") if isinstance(document, dict) else None
    if not isinstance(correction, dict) or correction.get("active") is not True:
        raise RuntimeError("正式运动拒绝未激活的校正数据集")
    if int(correction.get("corrected_pose_count", 0)) != 17:
        raise RuntimeError("正式运动要求17个点全部完成校正")


def highest_recorded_tcp_z(document: dict) -> float:
    poses = document.get("poses") if isinstance(document, dict) else None
    if not isinstance(poses, dict) or not poses:
        raise RuntimeError("无法读取17点的全局最高TCP Z")
    values: list[float] = []
    for pose in poses.values():
        try:
            value = float(pose["tcp_coords_mm_deg"][2])
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("17点数据中存在无效TCP Z") from None
        if not np.isfinite(value):
            raise RuntimeError("17点数据中存在无效TCP Z")
        values.append(value)
    return max(values)


def load_transfer_route(
    path: Path,
    grid: int,
    destination: str,
    require_active_correction: bool = False,
) -> tuple[dict, dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("姿态文件不是实验3固定姿态格式")
    if require_active_correction:
        validate_active_correction_document(document)
    if not document.get("complete"):
        raise RuntimeError("实验3的17个姿态尚未全部录制完成")
    poses = document.get("poses")
    if not isinstance(poses, dict):
        raise RuntimeError("姿态文件缺少 poses")

    names = [
        "safe",
        f"grid_{grid}_approach",
        f"grid_{grid}_grasp",
        f"{destination}_approach",
        f"{destination}_release",
    ]
    selected: dict[str, dict] = {}
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
        if np.any(angles < LEGACY_REPLAY_LOWER) or np.any(angles > LEGACY_REPLAY_UPPER):
            raise RuntimeError(f"{name} 超出软件关节限位")
        selected[name] = {
            "name": name,
            "angles": angles,
            "coords": coords,
            "manual_command_calibrated": pose.get("manual_command_calibrated") is True,
        }

    grasp_raise_mm = SOURCE_GRASP_EXTRA_Z_MM_BY_GRID.get(grid)
    if grasp_raise_mm is not None:
        approach_name = f"grid_{grid}_approach"
        grasp_name = f"grid_{grid}_grasp"
        original_approach = selected[approach_name]
        original_grasp = selected[grasp_name]
        selected[grasp_name] = build_raised_source_grasp(
            original_approach,
            original_grasp,
            grasp_raise_mm,
        )
    if grid == 3:
        selected[approach_name] = build_raised_source_approach(
            original_approach,
            original_grasp,
            GRID3_APPROACH_EXTRA_Z_MM,
        )

    gripper = document.get("gripper", {})
    open_value = gripper.get("open")
    close_value = gripper.get("close")
    if not all(isinstance(value, int) and 0 <= value <= 100 for value in (open_value, close_value)):
        raise RuntimeError("夹爪 OPEN/CLOSE 参数无效")
    if close_value >= open_value:
        raise RuntimeError("夹爪 CLOSE 必须小于 OPEN")
    return selected, gripper


class TransferRunner(GridDryRunner):
    def __init__(
        self,
        host: str,
        port: int,
        arm_speed: int,
        log_dir: Path,
        stream_step: float,
        stream_period: float,
        compensation_document: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(host, port, arm_speed, log_dir, stream_step, stream_period)
        self.control_host = host
        self.control_port = port
        self.compensator = (
            DirectionalJointCompensator(compensation_document)
            if compensation_document is not None
            else None
        )

    def stream_pose(self, label: str, pose: dict) -> bool:
        desired = np.asarray(pose["angles"], dtype=float)
        if pose.get("manual_command_calibrated") is True:
            print(f"[人工命令角-直接应用] {pose['name']}：跳过方向补偿")
            self.stream_to(label, desired)
            return False
        compensator = getattr(self, "compensator", None)
        if compensator is None:
            self.stream_to(label, desired)
            return False
        command, directions, applied = compensator.command_for_pose(
            desired, str(pose["name"])
        )
        mode = "应用" if applied else "原始点位"
        print(
            f"[方向补偿-{mode}] {pose['name']}："
            + ", ".join(
                f"{name}={command[int(item['index']) - 1] - desired[int(item['index']) - 1]:+.3f}°({directions[name]})"
                for name, item in compensator.joints.items()
            )
        )
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "directional_compensation",
                "label": label,
                "pose_name": str(pose["name"]),
                "applied": applied,
                "physical_target_deg": [round(float(value), 3) for value in desired],
                "command_target_deg": [round(float(value), 3) for value in command],
                "directions": directions,
            }
        )
        self.stream_to(label, command)
        return applied

    def reconnect_control_session(self, label: str) -> None:
        """Flush one unresponsive bridge session without changing arm position."""
        before = self.read_angles()
        try:
            self.arm.close()
        except Exception:
            pass
        time.sleep(GRIPPER_RECONNECT_SETTLE_S)
        self.arm = MechArmSocket(self.control_host, self.control_port)
        time.sleep(GRIPPER_RECONNECT_SETTLE_S)
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)
        after = self.health()
        drift = float(np.max(np.abs(after - before)))
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "gripper_session_reconnect",
                "label": label,
                "before_deg": [round(float(value), 3) for value in before],
                "after_deg": [round(float(value), 3) for value in after],
                "maximum_drift_deg": round(drift, 3),
            }
        )
        if drift > GRIPPER_RECONNECT_MAX_DRIFT_DEG:
            raise RuntimeError(
                f"{label}重建连接后机械臂姿态漂移{drift:.2f}°，禁止继续"
            )
        print(
            f"[夹爪会话恢复] {label} 已清空旧连接并重建；"
            f"机械臂最大姿态漂移={drift:.2f}°"
        )

    def gripper_reliable(
        self,
        label: str,
        value: int,
        speed: int,
        recovery_moves: list[tuple[str, dict]],
    ) -> int:
        """Wait out latency, flush stale commands, and never skip an incomplete grid."""
        failures: list[str] = []
        is_close = value < GRIPPER_CALIBRATED_OPEN_MIN
        strict_cube_close = is_close and value == CUBE_CLOSE_TARGET
        close_contact_min_value = (
            CUBE_CLOSE_CONTACT_VALUE if strict_cube_close else 0
        )
        for session_attempt in range(1, GRIPPER_SESSION_ATTEMPTS + 1):
            try:
                return self.gripper(
                    label,
                    value,
                    speed,
                    command_attempts=1,
                    confirm_stationary_close=False,
                    command_mode="value" if is_close else "state",
                    stop_on_failure=False,
                    clear_before_retry=True,
                    stall_samples=GRIPPER_MAX_SAMPLES,
                    # A cube may physically stop one feedback unit above the
                    # direct 64..72 target range.  Admit only the captured 73
                    # plateau through the long stable-contact path; do not
                    # widen the ordinary target range or accept lower values.
                    require_close_target=False,
                    close_target_tolerance=CLASS_CLOSE_TARGET_TOLERANCE,
                    close_target_lower_tolerance=(
                        CUBE_CLOSE_LOWER_TOLERANCE
                        if strict_cube_close
                        else CLASS_CLOSE_TARGET_TOLERANCE
                    ),
                    close_target_upper_tolerance=(
                        CUBE_CLOSE_UPPER_TOLERANCE
                        if strict_cube_close
                        else CLASS_CLOSE_TARGET_TOLERANCE
                    ),
                    close_contact_samples=CLASS_CLOSE_CONTACT_SAMPLES,
                    close_contact_min_movement=CLASS_CLOSE_CONTACT_MIN_MOVEMENT,
                    close_contact_min_value=close_contact_min_value,
                    close_contact_max_value=CLASS_CLOSE_CONTACT_MAX_VALUE,
                )
            except GripperActuationError as error:
                failures.append(str(error))
                print(
                    f"[夹爪恢复 {session_attempt}/{GRIPPER_SESSION_ATTEMPTS}] "
                    f"{error}"
                )
                if session_attempt < GRIPPER_SESSION_ATTEMPTS:
                    print(
                        "[夹爪冻结] 保持当前机械臂姿态；旧命令已RELEASE清除，"
                        "现在关闭并重建TCP控制会话。"
                    )
                    self.reconnect_control_session(label)
                    continue

        print(f"[夹爪恢复耗尽] {label} 三个独立会话均未确认真实动作。")
        for move_label, pose in recovery_moves:
            self.move_verified(move_label, pose)
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "grid_blocked_gripper_failure",
                "failed_label": label,
                "failures": failures,
                "session_attempts": GRIPPER_SESSION_ATTEMPTS,
                "recovery_moves": [item[0] for item in recovery_moves],
            }
        )
        self.save_log("blocked_gripper_failure")
        raise GripperActuationError(
            f"{label}经过{GRIPPER_SESSION_ATTEMPTS}个独立会话仍无可靠响应；"
            "当前格未完成，整条序列停止，绝不跳到下一格"
        )

    def confirm_gripper_hold(self, label: str, reference: int) -> int:
        """Confirm a hold from passive samples without sending another CLOSE."""
        samples: list[int] = []
        consecutive = 0
        minimum_contact_value = max(
            0, reference - GRIPPER_HOLD_MAX_CLOSING_DRIFT
        )
        maximum_contact_value = min(
            GRIPPER_HELD_MAX_VALUE,
            reference + GRIPPER_HOLD_MAX_OPENING_DRIFT,
        )
        for attempt in range(1, GRIPPER_HOLD_MAX_SAMPLES + 1):
            current = self.retry_query(
                label,
                lambda: legacy_gripper_value(self.arm),
                lambda value: isinstance(value, int) and 0 <= value <= 100,
            )
            samples.append(current)
            held = hold_is_stable(reference, current)
            consecutive = consecutive + 1 if held else 0
            print(
                f"[夹持被动确认] {label} {attempt}/{GRIPPER_HOLD_MAX_SAMPLES}："
                f"基准={reference}，回读={current}，"
                f"首次接触保持区={minimum_contact_value}..{maximum_contact_value}，"
                f"连续稳定={consecutive}/{GRIPPER_HOLD_STABLE_SAMPLES}"
            )
            if consecutive >= GRIPPER_HOLD_STABLE_SAMPLES:
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "passive_gripper_hold_confirmation",
                        "label": label,
                        "reference": reference,
                        "samples": samples,
                        "accepted_value": current,
                        "minimum_contact_value": minimum_contact_value,
                        "maximum_contact_value": maximum_contact_value,
                        "maximum_closing_drift": GRIPPER_HOLD_MAX_CLOSING_DRIFT,
                        "maximum_opening_drift": GRIPPER_HOLD_MAX_OPENING_DRIFT,
                        "maximum_held_value": GRIPPER_HELD_MAX_VALUE,
                        "target_resends": 0,
                    }
                )
                return current
            if attempt < GRIPPER_HOLD_MAX_SAMPLES:
                time.sleep(GRIPPER_HOLD_POLL_INTERVAL_S)
        raise RuntimeError(
            f"{label}未连续确认夹持：基准={reference}，回读={samples}；"
            f"回读已离开首次接触保持区{minimum_contact_value}.."
            f"{maximum_contact_value}（可能物体滑落或夹爪接近张开区）"
        )

    def passive_settle_physical(
        self,
        label: str,
        physical_target: np.ndarray,
        tolerance_deg: float = 3.0,
        attempts: int = 6,
    ) -> np.ndarray:
        actual = np.asarray(physical_target, dtype=float).copy()
        for attempt in range(1, attempts + 1):
            time.sleep(1.0)
            actual = self.read_angles()
            maximum = float(np.max(np.abs(actual - physical_target)))
            print(
                f"[补偿端点校验] {label} {attempt}/{attempts}："
                f"物理目标最大误差={maximum:.2f}°，未重发目标"
            )
            if maximum <= tolerance_deg:
                return actual
        raise RuntimeError(
            f"{label}被动等待{attempts}秒后仍未达到补偿后的物理目标；"
            f"最大误差={float(np.max(np.abs(actual - physical_target))):.2f}°"
        )

    def passive_settle_manual_pose(self, label: str, pose: dict) -> np.ndarray:
        """Verify a manually calibrated command by its saved physical TCP."""
        target_angles = np.asarray(pose["angles"], dtype=float)
        target_coords = np.asarray(pose["coords"], dtype=float)
        xyz_tolerance, joint_guard = manual_pose_acceptance_limits(pose)
        previous_angles: np.ndarray | None = None
        previous_coords: np.ndarray | None = None
        consecutive = 0
        actual_angles = target_angles.copy()
        xyz_error = float("inf")
        rpy_error = float("inf")
        joint_error = float("inf")
        for attempt in range(1, MANUAL_POSE_MAX_ATTEMPTS + 1):
            time.sleep(MANUAL_POSE_POLL_INTERVAL_S)
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            xyz_error = float(np.linalg.norm(actual_coords[:3] - target_coords[:3]))
            rpy_delta = (actual_coords[3:] - target_coords[3:] + 180.0) % 360.0 - 180.0
            rpy_error = float(np.max(np.abs(rpy_delta)))
            joint_error = float(np.max(np.abs(actual_angles - target_angles)))
            stable = False
            if previous_angles is not None and previous_coords is not None:
                stable = (
                    float(np.max(np.abs(actual_angles - previous_angles)))
                    <= MANUAL_POSE_STABLE_JOINT_MOTION_DEG
                    and float(np.linalg.norm(actual_coords[:3] - previous_coords[:3]))
                    <= MANUAL_POSE_STABLE_TCP_MOTION_MM
                )
            acceptable = (
                xyz_error <= xyz_tolerance
                and rpy_error <= MANUAL_POSE_RPY_TOLERANCE_DEG
                and joint_error <= joint_guard
            )
            consecutive = consecutive + 1 if acceptable and stable else (1 if acceptable else 0)
            print(
                f"[人工点TCP校验] {label} {attempt}/{MANUAL_POSE_MAX_ATTEMPTS}："
                f"XYZ误差={xyz_error:.1f}mm，姿态误差={rpy_error:.1f}°，"
                f"最大关节偏差={joint_error:.2f}°，稳定={'是' if stable else '否'}"
            )
            if consecutive >= 2:
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "manual_pose_tcp_acceptance",
                        "label": label,
                        "tcp_xyz_error_mm": round(xyz_error, 3),
                        "tcp_xyz_tolerance_mm": round(xyz_tolerance, 3),
                        "tcp_rpy_error_deg": round(rpy_error, 3),
                        "joint_guard_error_deg": round(joint_error, 3),
                        "joint_guard_tolerance_deg": round(joint_guard, 3),
                        "target_resends": 0,
                    }
                )
                return actual_angles
            previous_angles = actual_angles.copy()
            previous_coords = actual_coords.copy()
        raise RuntimeError(
            f"{label}未复现人工校正TCP：XYZ={xyz_error:.1f}mm，"
            f"姿态={rpy_error:.1f}°，最大关节偏差={joint_error:.2f}°；"
            f"本点要求XYZ<={xyz_tolerance:.1f}mm且关节偏差<={joint_guard:.1f}°"
        )

    def passive_settle_generated_source_approach(
        self, label: str, pose: dict
    ) -> np.ndarray:
        """Validate the generated grid-3 approach by measured clearance."""
        target_angles = np.asarray(pose["angles"], dtype=float)
        original_xy = np.asarray(pose["original_approach_xy_mm"], dtype=float)
        original_z = float(pose["original_approach_z_mm"])
        requested_extra = float(pose["requested_extra_z_mm"])
        required_z = original_z + requested_extra
        previous_angles: np.ndarray | None = None
        previous_coords: np.ndarray | None = None
        actual_angles = target_angles.copy()
        actual_coords = np.asarray(pose["coords"], dtype=float).copy()
        horizontal_shift = float("inf")
        joint_error = float("inf")
        stable_samples = 0
        for attempt in range(1, SOURCE_APPROACH_MAX_ATTEMPTS + 1):
            time.sleep(SOURCE_APPROACH_POLL_INTERVAL_S)
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            horizontal_shift = float(np.linalg.norm(actual_coords[:2] - original_xy))
            joint_error = float(np.max(np.abs(actual_angles - target_angles)))
            stable = False
            if previous_angles is not None and previous_coords is not None:
                stable = (
                    float(np.max(np.abs(actual_angles - previous_angles)))
                    <= MANUAL_POSE_STABLE_JOINT_MOTION_DEG
                    and float(np.linalg.norm(actual_coords[:3] - previous_coords[:3]))
                    <= MANUAL_POSE_STABLE_TCP_MOTION_MM
                )
            height_ok = (
                float(actual_coords[2]) + SOURCE_APPROACH_HEIGHT_TOLERANCE_MM
                >= required_z
            )
            acceptable = (
                height_ok
                and horizontal_shift <= SOURCE_APPROACH_XY_GUARD_MM
                and joint_error <= MANUAL_POSE_JOINT_GUARD_DEG
            )
            stable_samples = stable_samples + 1 if acceptable and stable else 0
            print(
                f"[3号上方加高验证] {attempt}/{SOURCE_APPROACH_MAX_ATTEMPTS}："
                f"实测Z={actual_coords[2]:.1f}mm，目标>={required_z:.1f}mm"
                f"（容差{SOURCE_APPROACH_HEIGHT_TOLERANCE_MM:.1f}mm）；"
                f"相对原上方点XY偏移={horizontal_shift:.1f}mm，"
                f"最大关节偏差={joint_error:.2f}°，稳定={'是' if stable else '否'}"
            )
            if stable_samples >= 1:
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(f"{label}过程中控制器报错：{controller_error}")
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "generated_source_approach_clearance",
                        "label": label,
                        "requested_extra_z_mm": round(requested_extra, 3),
                        "required_z_mm": round(required_z, 3),
                        "actual_z_mm": round(float(actual_coords[2]), 3),
                        "horizontal_shift_from_original_mm": round(horizontal_shift, 3),
                        "joint_guard_error_deg": round(joint_error, 3),
                        "target_resends": 0,
                    }
                )
                return actual_angles
            previous_angles = actual_angles.copy()
            previous_coords = actual_coords.copy()
        raise RuntimeError(
            f"{label}未达到加高要求：实测Z={actual_coords[2]:.1f}mm，"
            f"目标>={required_z:.1f}mm（容差"
            f"{SOURCE_APPROACH_HEIGHT_TOLERANCE_MM:.1f}mm），"
            f"相对原上方点XY偏移={horizontal_shift:.1f}mm，"
            f"最大关节偏差={joint_error:.2f}°"
        )

    def establish_compensated_safe(self, safe: dict, following_pose: dict) -> None:
        if safe.get("manual_command_calibrated") is True:
            self.move_verified("回人工校正公共安全姿态", safe)
            return
        compensator = getattr(self, "compensator", None)
        if compensator is None:
            self.move_verified("回公共安全姿态", safe)
            return
        desired = np.asarray(safe["angles"], dtype=float)
        following = np.asarray(following_pose["angles"], dtype=float)
        self.stream_to("先回原始公共安全姿态", desired)
        GridDryRunner.settle_at(
            self, "原始公共安全姿态", desired, tolerance_deg=3.0, attempts=5
        )
        if self.commanded_angles is None:
            raise RuntimeError("位置流尚未初始化")
        preload = compensator.preload_commands(desired, following, self.commanded_angles)
        print("[方向预紧] 在safe附近依次建立J2/J3/J4方向；每次只改变一个关节。")
        for item in preload:
            joint = str(item["joint"])
            direction = str(item["direction"])
            stage = str(item["stage"])
            command = np.asarray(item["angles"], dtype=float)
            self.stream_to(
                f"{joint} {direction} {'预紧' if stage == 'preload' else '补偿回safe'}",
                command,
            )
            time.sleep(1.0)
            if stage == "preload":
                actual = self.read_angles()
                index = int(compensator.joints[joint]["index"]) - 1
                side = float(actual[index] - desired[index])
                correct = side <= -1.5 if direction == "increasing" else side >= 1.5
                print(f"[方向预紧校验] {joint}距物理safe={side:+.2f}°")
                if not correct:
                    raise RuntimeError(f"{joint}未到达建立{direction}方向所需的预紧侧")
        self.passive_settle_physical("方向补偿后的公共safe", desired)

    def move_verified(self, label: str, pose: dict) -> None:
        compensation_applied = self.stream_pose(label, pose)
        if pose.get("generated_source_approach") is True:
            self.passive_settle_generated_source_approach(label, pose)
            return
        if pose.get("manual_command_calibrated") is True:
            self.passive_settle_manual_pose(label, pose)
            return
        if compensation_applied:
            self.passive_settle_physical(label, np.asarray(pose["angles"], dtype=float))
            return
        low_contact_pose = label.startswith("下降到") and (
            "抓取点" in label or "释放点" in label
        )
        self.settle_at(
            label,
            pose["angles"],
            allow_stable_contact=low_contact_pose,
            target_coords=pose.get("coords") if low_contact_pose else None,
        )

    def move_grasp_converged(self, label: str, pose: dict) -> tuple[np.ndarray, np.ndarray]:
        """Send the grasp route once, then wait passively for TCP convergence."""
        target_angles = np.asarray(pose["angles"], dtype=float)
        target_coords = np.asarray(pose["coords"], dtype=float)
        xy_tolerance = grasp_xy_tolerance(pose)
        z_tolerance = grasp_z_tolerance(pose)
        self.stream_pose(label, pose)

        stable_samples = 0
        actual_angles = target_angles.copy()
        actual_coords = target_coords.copy()
        xyz_error = float("inf")
        xy_error = float("inf")
        z_error = float("inf")
        for attempt in range(1, GRASP_CONVERGENCE_MAX_ATTEMPTS + 1):
            time.sleep(GRASP_CONVERGENCE_POLL_INTERVAL_S)
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            joint_error = actual_angles - target_angles
            tcp_delta = actual_coords[:3] - target_coords[:3]
            xyz_error = float(np.linalg.norm(tcp_delta))
            xy_error = float(np.linalg.norm(tcp_delta[:2]))
            z_error = abs(float(tcp_delta[2]))
            other_joint_error = float(np.max(np.abs(np.delete(joint_error, 4))))
            j5_error = abs(float(joint_error[4]))
            elapsed = attempt * GRASP_CONVERGENCE_POLL_INTERVAL_S
            print(
                f"[抓取点被动收敛] {elapsed:.2f}s：XY误差={xy_error:.3f}mm，"
                f"Z误差={z_error:.3f}mm，"
                f"J2={joint_error[1]:+.2f}°，J5={joint_error[4]:+.2f}°"
            )

            within_tcp = (
                xy_error <= xy_tolerance
                and z_error <= z_tolerance
            )
            joints_safe = (
                other_joint_error <= GRASP_OTHER_JOINT_TOLERANCE_DEG
                and j5_error <= GRASP_J5_SAFETY_GUARD_DEG
            )
            stable_samples = stable_samples + 1 if within_tcp and joints_safe else 0
            if stable_samples >= GRASP_STABLE_SAMPLES:
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(
                        f"{label}等待收敛时控制器报错：{controller_error}"
                    )
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "passive_grasp_convergence",
                        "label": label,
                        "samples": attempt,
                        "elapsed_s": round(elapsed, 3),
                        "target_xyz_mm": [
                            round(float(value), 3) for value in target_coords[:3]
                        ],
                        "actual_xyz_mm": [
                            round(float(value), 3) for value in actual_coords[:3]
                        ],
                        "tcp_xyz_error_mm": round(xyz_error, 3),
                        "tcp_xy_error_mm": round(xy_error, 3),
                        "tcp_xy_tolerance_mm": round(xy_tolerance, 3),
                        "tcp_z_error_mm": round(z_error, 3),
                        "tcp_z_tolerance_mm": round(z_tolerance, 3),
                        "j2_error_deg": round(float(joint_error[1]), 3),
                        "j5_error_deg": round(float(joint_error[4]), 3),
                        "target_resends": 0,
                    }
                )
                print(
                    f"[抓取点通过] 连续{GRASP_STABLE_SAMPLES}次XY误差<="
                    f"{xy_tolerance:.1f}mm且Z误差<="
                    f"{z_tolerance:.1f}mm；未重复发送关节目标"
                )
                return actual_angles, actual_coords

        controller_error = self.read_error()
        if not self.error_is_clear(controller_error):
            raise RuntimeError(f"{label}等待收敛时控制器报错：{controller_error}")
        raise RuntimeError(
            f"{label}被动等待{GRASP_CONVERGENCE_MAX_ATTEMPTS * GRASP_CONVERGENCE_POLL_INTERVAL_S:.1f}s"
            f"后TCP仍未稳定满足XY<={xy_tolerance:.1f}mm、"
            f"Z<={z_tolerance:.1f}mm；"
            f"最后XY={xy_error:.3f}mm、Z={z_error:.3f}mm"
        )

    def move_carry_lift(self, approach: dict, grasp: dict) -> None:
        """Lift a held object and validate measured clearance, not exact joints."""
        label = "夹持后原路抬升"
        start_coords = self.read_coords()
        self.stream_pose(label, approach)
        approach_coords = np.asarray(approach["coords"], dtype=float)
        grasp_coords = np.asarray(grasp["coords"], dtype=float)
        taught_delta = approach_coords[:3] - grasp_coords[:3]
        # Judge this segment against its captured approach/grasp pair.  A fixed
        # minimum made short but intentional taught lifts (notably grid 3's
        # 17 mm segment) impossible to pass even when reproduced accurately.
        required_rise = max(
            0.0,
            float(taught_delta[2] - CARRY_LIFT_TAUGHT_ALLOWANCE_MM),
        )
        actual_coords = start_coords.copy()
        actual_angles = np.asarray(approach["angles"], dtype=float)
        actual_rise = 0.0
        xy_path_error = float("inf")
        for attempt in range(1, CARRY_LIFT_MAX_ATTEMPTS + 1):
            time.sleep(CARRY_LIFT_POLL_INTERVAL_S)
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            actual_delta = actual_coords[:3] - start_coords[:3]
            actual_rise = float(actual_delta[2])
            xy_path_error = float(np.linalg.norm(actual_delta[:2] - taught_delta[:2]))
            elapsed = attempt * CARRY_LIFT_POLL_INTERVAL_S
            print(
                f"[携物抬升] {elapsed:.2f}s：实际抬高={actual_rise:.1f}mm，"
                f"要求>={required_rise:.1f}mm，水平路径误差={xy_path_error:.1f}mm"
            )
            if (
                actual_rise >= required_rise
                and xy_path_error <= CARRY_LIFT_XY_TOLERANCE_MM
            ):
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(f"{label}过程中控制器报错：{controller_error}")
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "carry_lift_clearance",
                        "samples": attempt,
                        "elapsed_s": round(elapsed, 3),
                        "start_xyz_mm": [
                            round(float(value), 3) for value in start_coords[:3]
                        ],
                        "actual_xyz_mm": [
                            round(float(value), 3) for value in actual_coords[:3]
                        ],
                        "actual_angles_deg": [
                            round(float(value), 3) for value in actual_angles
                        ],
                        "actual_rise_mm": round(actual_rise, 3),
                        "required_rise_mm": round(required_rise, 3),
                        "xy_path_error_mm": round(xy_path_error, 3),
                        "target_resends": 0,
                    }
                )
                print(
                    f"[携物抬升通过] 实际抬高={actual_rise:.1f}mm；"
                    "按TCP净抬升允许继续，不要求六轴完全等于录制值"
                )
                return

        controller_error = self.read_error()
        if not self.error_is_clear(controller_error):
            raise RuntimeError(f"{label}过程中控制器报错：{controller_error}")
        raise RuntimeError(
            f"夹持后被动等待{CARRY_LIFT_MAX_ATTEMPTS * CARRY_LIFT_POLL_INTERVAL_S:.1f}s，"
            f"实际抬高仍只有{actual_rise:.1f}mm或水平路径误差{xy_path_error:.1f}mm"
        )

    def move_release_retreat(
        self,
        approach: dict,
        release_pose: dict,
        release_coords: np.ndarray,
        initial_extension: float,
    ) -> np.ndarray:
        """Pass through approach, then extend until measured TCP Z rises 75 mm."""
        self.stream_pose("释放后原路抬升（连续通过）", approach)
        actual_coords = np.asarray(release_coords, dtype=float).copy()
        actual_angles = np.asarray(approach["angles"], dtype=float).copy()
        actual_lift = 0.0
        extension = float(initial_extension)
        sample_count = 0
        while extension <= RELEASE_RETREAT_MAX_EXTENSION + 1e-9:
            retreat_target = build_high_retreat(approach, release_pose, extension)
            self.stream_pose(
                f"释放后Z抬升目标（extension={extension:.2f}）",
                retreat_target,
            )
            time.sleep(RELEASE_RETREAT_POLL_INTERVAL_S)
            sample_count += 1
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            actual_lift = float(actual_coords[2] - release_coords[2])
            print(
                f"[释放后强制Z抬升] extension={extension:.2f}："
                f"实际ΔZ={actual_lift:.1f}mm，要求>="
                f"{RELEASE_RETREAT_REQUIRED_LIFT_MM:.1f}mm"
            )
            if release_retreat_lift_is_sufficient(actual_lift):
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(f"释放后高撤离过程中控制器报错：{controller_error}")
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "release_retreat_clearance",
                        "samples": sample_count,
                        "final_extension": round(extension, 3),
                        "release_xyz_mm": [
                            round(float(value), 3) for value in release_coords[:3]
                        ],
                        "actual_xyz_mm": [
                            round(float(value), 3) for value in actual_coords[:3]
                        ],
                        "actual_angles_deg": [
                            round(float(value), 3) for value in actual_angles
                        ],
                        "actual_lift_mm": round(actual_lift, 3),
                        "measurement_tolerance_mm": RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM,
                        "intermediate_joint_settle_checks": 0,
                    }
                )
                print(
                    f"[高撤离通过] 释放后实际ΔZ={actual_lift:.1f}mm；"
                    + (
                        f"与75mm目标相差不超过控制器测量容差"
                        f"{RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM:.1f}mm；"
                        if actual_lift < RELEASE_RETREAT_REQUIRED_LIFT_MM
                        else ""
                    )
                    + "中间上方点连续通过，未按关节误差停车"
                )
                return actual_coords
            extension = round(extension + RELEASE_RETREAT_EXTENSION_STEP, 6)

        for _ in range(max(0, RELEASE_RETREAT_MAX_ATTEMPTS - sample_count)):
            time.sleep(RELEASE_RETREAT_POLL_INTERVAL_S)
            sample_count += 1
            actual_angles = self.read_angles()
            actual_coords = self.read_coords()
            actual_lift = float(actual_coords[2] - release_coords[2])
            print(
                f"[释放后强制Z抬升] 最大extension被动等待："
                f"实际ΔZ={actual_lift:.1f}mm，要求>="
                f"{RELEASE_RETREAT_REQUIRED_LIFT_MM:.1f}mm"
            )
            if release_retreat_lift_is_sufficient(actual_lift):
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(f"释放后高撤离过程中控制器报错：{controller_error}")
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "release_retreat_clearance",
                        "samples": sample_count,
                        "final_extension": RELEASE_RETREAT_MAX_EXTENSION,
                        "release_xyz_mm": [
                            round(float(value), 3) for value in release_coords[:3]
                        ],
                        "actual_xyz_mm": [
                            round(float(value), 3) for value in actual_coords[:3]
                        ],
                        "actual_angles_deg": [
                            round(float(value), 3) for value in actual_angles
                        ],
                        "actual_lift_mm": round(actual_lift, 3),
                        "measurement_tolerance_mm": RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM,
                        "intermediate_joint_settle_checks": 0,
                    }
                )
                if actual_lift < RELEASE_RETREAT_REQUIRED_LIFT_MM:
                    print(
                        f"[高撤离通过] 实际ΔZ={actual_lift:.1f}mm；"
                        f"与75mm目标相差不超过控制器测量容差"
                        f"{RELEASE_RETREAT_MEASUREMENT_TOLERANCE_MM:.1f}mm"
                    )
                return actual_coords

        controller_error = self.read_error()
        if not self.error_is_clear(controller_error):
            raise RuntimeError(f"释放后高撤离过程中控制器报错：{controller_error}")
        raise RuntimeError(
            f"释放后连续高撤离等待"
            f"{RELEASE_RETREAT_MAX_ATTEMPTS * RELEASE_RETREAT_POLL_INTERVAL_S:.1f}s，"
            f"实际ΔZ仍只有{actual_lift:.1f}mm，低于"
            f"{RELEASE_RETREAT_REQUIRED_LIFT_MM:.0f}mm"
        )

    def move_pre_descent_transition(
        self, approach: dict, release: dict, highest_recorded_z_mm: float
    ) -> None:
        """Reach an automatically generated high pose before the taught descent."""
        transition = build_pre_descent_transition(
            approach,
            release,
            minimum_absolute_z_mm=highest_recorded_z_mm,
        )
        if transition is None:
            taught_clearance = float(approach["coords"][2] - release["coords"][2])
            print(
                f"[自动高位] {approach['name']}已高出低位点{taught_clearance:.1f}mm，"
                "无需添加过渡点"
            )
            return
        print(
            f"[自动高位] 生成{transition['name']}：预计高出低位点"
            f"{transition['generated_clearance_mm']:.1f}mm，"
            f"最低要求={transition['required_clearance_mm']:.1f}mm，"
            f"跟随余量={transition['target_headroom_mm']:.1f}mm，"
            f"扩展比例={transition['generated_extension']:.2f}"
        )
        self.stream_pose("到释放区自动高位过渡点", transition)
        actual_lift = float("-inf")
        actual_coords = np.asarray(release["coords"], dtype=float).copy()
        required_absolute_z = max(
            float(release["coords"][2]) + PRE_DESCENT_MIN_CLEARANCE_MM,
            float(highest_recorded_z_mm),
        )
        required_relative_lift = float(transition["required_clearance_mm"])
        for attempt in range(1, PRE_DESCENT_MAX_ATTEMPTS + 1):
            time.sleep(PRE_DESCENT_POLL_INTERVAL_S)
            actual_coords = self.read_coords()
            actual_lift = float(actual_coords[2] - release["coords"][2])
            print(
                f"[自动高位验证] {attempt}/{PRE_DESCENT_MAX_ATTEMPTS}："
                f"实测Z={actual_coords[2]:.1f}mm，目标>={required_absolute_z:.1f}mm；"
                f"相对释放点ΔZ={actual_lift:.1f}mm，目标>="
                f"{required_relative_lift:.1f}mm；容差="
                f"{PRE_DESCENT_MEASUREMENT_TOLERANCE_MM:.1f}mm"
            )
            absolute_height_ok = (
                float(actual_coords[2]) + PRE_DESCENT_MEASUREMENT_TOLERANCE_MM
                >= required_absolute_z
            )
            relative_lift_ok = (
                actual_lift + PRE_DESCENT_MEASUREMENT_TOLERANCE_MM
                >= required_relative_lift
            )
            if absolute_height_ok and relative_lift_ok:
                controller_error = self.read_error()
                if not self.error_is_clear(controller_error):
                    raise RuntimeError(f"自动高位过渡过程中控制器报错：{controller_error}")
                self.events.append(
                    {
                        "time": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "generated_pre_descent_transition",
                        "pose_name": transition["name"],
                        "generated_extension": round(
                            float(transition["generated_extension"]), 3
                        ),
                        "required_absolute_z_mm": round(required_absolute_z, 3),
                        "required_relative_lift_mm": round(required_relative_lift, 3),
                        "commanded_relative_lift_mm": round(
                            float(transition["generated_clearance_mm"]), 3
                        ),
                        "target_headroom_mm": round(
                            float(transition["target_headroom_mm"]), 3
                        ),
                        "measurement_tolerance_mm": round(
                            PRE_DESCENT_MEASUREMENT_TOLERANCE_MM, 3
                        ),
                        "actual_clearance_mm": round(actual_lift, 3),
                        "actual_xyz_mm": [
                            round(float(value), 3) for value in actual_coords[:3]
                        ],
                        "target_resends": 0,
                    }
                )
                return
        raise RuntimeError(
            f"自动高位过渡被动等待"
            f"{PRE_DESCENT_MAX_ATTEMPTS * PRE_DESCENT_POLL_INTERVAL_S:.1f}s后，"
            f"实测Z={actual_coords[2]:.1f}mm、相对抬升={actual_lift:.1f}mm，"
            f"仍未在{PRE_DESCENT_MEASUREMENT_TOLERANCE_MM:.1f}mm容差内达到"
            f"绝对高度{required_absolute_z:.1f}mm和相对抬升{required_relative_lift:.1f}mm"
        )

    def run_transfer(
        self,
        current: np.ndarray,
        poses: dict,
        gripper: dict,
        grid: int,
        destination: str,
        gripper_speed: int,
        retreat_extension: float,
        highest_recorded_z_mm: float,
    ) -> None:
        # ROS 2 executes this method in a worker thread. Python only permits
        # signal handlers in the main thread; standalone CLI runs keep the
        # original Ctrl+C emergency-stop handler.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        open_value = int(gripper["open"])
        close_value = int(gripper["close"])
        safe = poses["safe"]
        source_approach = poses[f"grid_{grid}_approach"]
        source_grasp = poses[f"grid_{grid}_grasp"]
        destination_approach = poses[f"{destination}_approach"]
        destination_release = poses[f"{destination}_release"]
        print(
            f"[夹爪流程] 在safe先严格OPEN，再前往{grid}号格上方；"
            "到位后只读复核，检测到意外闭合才补发OPEN"
        )
        self.establish_compensated_safe(safe, source_approach)
        self.gripper_reliable(
            f"{grid}号格前往上方点前 OPEN",
            open_value,
            gripper_speed,
            recovery_moves=[],
        )
        self.move_verified(f"到{grid}号格上方", source_approach)
        self.gripper_reliable(
            f"{grid}号格下降前 OPEN 自检",
            open_value,
            gripper_speed,
            recovery_moves=[("夹爪失败后返回公共安全姿态", safe)],
        )
        self.move_grasp_converged(f"下降到{grid}号格抓取点", source_grasp)
        close_feedback = self.gripper_reliable(
            f"{grid}号格 CLOSE",
            close_value,
            gripper_speed,
            recovery_moves=[
                (f"夹爪失败后抬回{grid}号格上方", source_approach),
                ("夹爪失败后返回公共安全姿态", safe),
            ],
        )

        self.move_carry_lift(source_approach, source_grasp)
        held_after_lift = self.confirm_gripper_hold(
            "读取抬升后夹爪位置", close_feedback
        )
        print(f"[夹持确认] 抬升后夹爪回读={held_after_lift}")

        self.move_verified("携物回公共安全姿态", safe)
        held_before_transfer = self.confirm_gripper_hold(
            "读取横向搬运前夹爪位置", close_feedback
        )
        print(
            f"[夹持确认] 横向搬运前回读={held_before_transfer}；"
            "保持首次CLOSE命令，不重复闭合"
        )
        self.move_pre_descent_transition(
            destination_approach,
            destination_release,
            highest_recorded_z_mm,
        )
        self.move_verified(f"到{destination}侧释放点上方", destination_approach)
        self.move_verified(f"下降到{destination}侧释放点", destination_release)
        release_coords = self.read_coords()
        self.gripper_reliable(
            f"{destination}侧 OPEN 释放",
            open_value,
            gripper_speed,
            recovery_moves=[
                (f"夹爪失败后抬回{destination}侧上方", destination_approach),
                ("夹爪失败后携物返回公共安全姿态", safe),
            ],
        )

        high_coords = self.move_release_retreat(
            destination_approach,
            destination_release,
            release_coords,
            retreat_extension,
        )
        actual_lift = float(high_coords[2] - release_coords[2])
        print(
            f"[高撤离验证] 释放后实际抬升={actual_lift:.1f}mm，"
            f"高点TCP={np.round(high_coords[:3], 1).tolist()}mm"
        )
        self.move_verified("返回公共安全姿态", safe)
        print(
            "[夹爪流程] 释放点已验证OPEN；返回safe后不重复驱动夹爪。"
            "下一格只在下降抓取前再次严格验证。"
        )
        final_angles = self.read_angles()
        final_error = float(np.max(np.abs(final_angles - safe["angles"])))
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "transfer_complete",
                "source_grid": grid,
                "destination": destination,
                "held_gripper_value_after_lift": held_after_lift,
                "held_gripper_value_before_transfer": held_before_transfer,
                "release_retreat_lift_mm": round(actual_lift, 3),
                "safe_pose_max_error_deg": round(final_error, 3),
            }
        )
        self.save_log("completed")
        print(
            f"[完成] {grid}号格 -> {destination}侧搬运完成；"
            f"最终安全点误差={final_error:.1f}°"
        )
        print(f"[记录] {self.log_path}")


def self_test() -> None:
    route_names = [
        "safe",
        "grid_1_approach",
        "grid_1_grasp",
        "right_approach",
        "right_release",
    ]
    assert len(route_names) == len(set(route_names)) == 5
    assert hold_is_stable(70, 70)
    assert hold_is_stable(72, 70)
    assert hold_is_stable(70, 75)
    assert hold_is_stable(65, 72)
    assert not hold_is_stable(65, 79)
    assert not hold_is_stable(73, 54)
    assert DESTINATION_BY_CLASS["cube"] == "right"
    assert DESTINATION_BY_CLASS["other"] == "left"
    approach = {
        "name": "right_approach",
        "angles": np.array([0.0, 10.0, 0.0, 0.0, -20.0, 0.0]),
        "coords": np.array([0.0, 0.0, 65.0, 0.0, 0.0, 0.0]),
    }
    release = {
        "name": "right_release",
        "angles": np.array([0.0, 20.0, 0.0, 0.0, -10.0, 0.0]),
        "coords": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    high = build_high_retreat(approach, release, 0.6)
    assert np.allclose(high["angles"], [0.0, 4.0, 0.0, 0.0, -26.0, 0.0])
    short_approach = dict(approach, coords=np.array([0.0, 0.0, 25.0, 0.0, 0.0, 0.0]))
    assert abs(effective_retreat_extension(short_approach, release, 0.5) - 2.0) < 1e-9
    assert effective_retreat_extension(approach, release, 0.5) == 0.5
    generated = build_pre_descent_transition(short_approach, release)
    assert generated is not None
    assert abs(generated["required_clearance_mm"] - 65.0) < 1e-9
    assert abs(generated["generated_clearance_mm"] - 70.0) < 1e-9
    assert build_pre_descent_transition(approach, release) is None
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 transfer test")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--compensation", type=Path)
    parser.add_argument("--grid", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--destination", choices=("left", "right"))
    parser.add_argument("--object-class", choices=tuple(DESTINATION_BY_CLASS))
    parser.add_argument(
        "--gripper-profiles",
        type=Path,
        default=DEFAULT_GRIPPER_PROFILES,
        help="已标定的cube/other分类夹爪参数；正式动作不再使用17点文件中的旧CLOSE",
    )
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=12)
    parser.add_argument(
        "--gripper-speed",
        type=int,
        help="可选临时覆盖夹爪配置速度（仅允许1～30）",
    )
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument("--retreat-extension", type=float, default=0.50)
    parser.add_argument(
        "--require-active-correction",
        action="store_true",
        help="拒绝原始或未激活的数据集；正式运动启动器必须启用",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.poses is None:
        raise RuntimeError("必须提供--poses")
    if not 1 <= args.arm_speed <= 20:
        raise RuntimeError("首次搬运速度必须在1~20")
    if args.gripper_speed is not None and not 1 <= args.gripper_speed <= 30:
        raise RuntimeError("夹爪速度覆盖值必须在1~30")
    if not 0.5 <= args.stream_step <= 3.0:
        raise RuntimeError("位置流步长必须在0.5~3.0度")
    if not 0.08 <= args.stream_period <= 0.4:
        raise RuntimeError("位置流周期必须在0.08~0.4秒")
    if not 0.1 <= args.retreat_extension <= 2.0:
        raise RuntimeError("释放后高撤离扩展比例必须在0.1~2.0")

    destination = args.destination
    if args.object_class is not None:
        classified_destination = DESTINATION_BY_CLASS[args.object_class]
        if destination is not None and destination != classified_destination:
            raise RuntimeError(
                f"类别{args.object_class}必须去{classified_destination}侧，"
                f"不能去{destination}侧"
            )
        destination = classified_destination
    if destination is None:
        raise RuntimeError("必须提供 --object-class 或 --destination")
    object_group = args.object_class or (
        "cube" if destination == DESTINATION_BY_CLASS["cube"] else "other"
    )

    pose_document = yaml.safe_load(args.poses.read_text(encoding="utf-8"))
    highest_recorded_z_mm = highest_recorded_tcp_z(pose_document)
    poses, _recorded_gripper = load_transfer_route(
        args.poses,
        args.grid,
        destination,
        require_active_correction=args.require_active_correction,
    )
    gripper_profiles = load_gripper_profiles(args.gripper_profiles)
    gripper = profile_for_group(gripper_profiles, object_group)
    gripper_speed = (
        int(args.gripper_speed)
        if args.gripper_speed is not None
        else int(gripper["speed"])
    )
    compensation_document = None
    if args.compensation is not None:
        compensation_document = load_compensation(args.compensation)
        if compensation_document.get("production_enabled") is not True:
            raise RuntimeError("方向补偿尚未获准用于正式路线")
    source_approach = poses[f"grid_{args.grid}_approach"]["coords"]
    source_grasp = poses[f"grid_{args.grid}_grasp"]["coords"]
    source_grasp_pose = poses[f"grid_{args.grid}_grasp"]
    source_grasp_xy_tolerance = grasp_xy_tolerance(source_grasp_pose)
    source_grasp_z_tolerance = grasp_z_tolerance(source_grasp_pose)
    destination_approach = poses[f"{destination}_approach"]["coords"]
    destination_release = poses[f"{destination}_release"]["coords"]
    retreat_extension = effective_retreat_extension(
        poses[f"{destination}_approach"],
        poses[f"{destination}_release"],
        args.retreat_extension,
    )
    high_retreat = build_high_retreat(
        poses[f"{destination}_approach"],
        poses[f"{destination}_release"],
        retreat_extension,
    )
    recorded_release_lift = destination_approach[2] - destination_release[2]
    estimated_high_lift = recorded_release_lift * (1.0 + retreat_extension)
    pre_descent_transition = build_pre_descent_transition(
        poses[f"{destination}_approach"],
        poses[f"{destination}_release"],
        minimum_absolute_z_mm=highest_recorded_z_mm,
    )
    class_label = object_group
    print(
        f"[预检] {args.grid}号格，类别={class_label} -> "
        f"{destination}侧单次真实搬运"
    )
    print(f"[数据集] {args.poses.resolve()}")
    print(f"[全局高点] 17个采集点最高TCP Z={highest_recorded_z_mm:.1f}mm")
    print(
        f"[夹爪] OPEN={gripper['open']}，CLOSE={gripper['close']}，"
        f"速度={gripper_speed}，配置={args.gripper_profiles}"
    )
    print(
        "[夹爪机制] OPEN使用旧固件张开状态命令；CLOSE使用分类标定的位置命令。"
        "cube的CLOSE=69在回读64..72时直接通过；回读73必须连续稳定8个采样"
        "才按物体接触通过。other若被物体阻挡，也必须在可信接触区连续稳定"
        "8个采样才通过。已确认接触后保持原CLOSE命令，"
        "禁止发送RELEASE或重复闭合"
    )
    print(
        f"[低位收敛] 机械臂速度={args.arm_speed}；抓取点只发送一次，"
        f"被动等待最多{GRASP_CONVERGENCE_MAX_ATTEMPTS * GRASP_CONVERGENCE_POLL_INTERVAL_S:.1f}s，"
        f"XY误差连续{GRASP_STABLE_SAMPLES}次<={source_grasp_xy_tolerance:.1f}mm且"
        f"Z误差<={source_grasp_z_tolerance:.1f}mm后闭合"
    )
    print(
        "[携物抬升] 按采集点抬升量（允许15mm跟随误差）和水平路径误差验收；"
        "不再设置20mm固定下限，也不要求J2/J5完全等于录制关节角"
    )
    if compensation_document is not None:
        print(
            "[方向补偿] 仅safe和grid_1_approach启用J2/J3/J4补偿；"
            "低位抓取和释放区使用原始采集点"
        )
    if source_grasp_pose.get("generated_source_grasp") is True:
        print(
            f"[抓取点调整] {args.grid}号格沿人工下降路径上调"
            f"{source_grasp_pose['requested_extra_z_mm']:.1f}mm；"
            f"目标TCP Z={source_grasp_pose['coords'][2]:.1f}mm"
        )
    print(
        f"[源下降] Z={source_approach[2]-source_grasp[2]:.1f}mm，"
        f"XY={float(np.linalg.norm(source_grasp[:2]-source_approach[:2])):.1f}mm"
    )
    print(
        f"[释放下降] Z={destination_approach[2]-destination_release[2]:.1f}mm，"
        f"XY={float(np.linalg.norm(destination_release[:2]-destination_approach[:2])):.1f}mm"
    )
    if pre_descent_transition is None:
        print(
            f"[进场高位] 已有approach高度满足>="
            f"{PRE_DESCENT_MIN_CLEARANCE_MM:.0f}mm，不生成额外点"
        )
    else:
        print(
            f"[进场高位] 将沿现有release→approach方向自动生成"
            f"{pre_descent_transition['generated_clearance_mm']:.1f}mm高位过渡点，"
            f"预计TCP Z={pre_descent_transition['coords'][2]:.1f}mm，"
            f"不低于全局高点{highest_recorded_z_mm:.1f}mm，"
            f"扩展比例={pre_descent_transition['generated_extension']:.2f}"
        )
    print(
        f"[释放后高撤离] 自动扩展比例={retreat_extension:.2f}，"
        f"预计总抬升约{estimated_high_lift:.1f}mm，"
        f"目标关节={np.round(high_retreat['angles'], 1).tolist()}"
    )
    print("[安全路线] 抓取后先回safe，再横向前往释放区；不会低空横穿网格")
    if args.check_only:
        print("[只读检查] 路线合法；未连接机械臂，未发送命令。")
        return 0

    control_lock = acquire_experiment3_control_lock("single-transfer")
    runner = None
    try:
        runner = TransferRunner(
            args.host,
            args.port,
            args.arm_speed,
            args.poses.parent,
            args.stream_step,
            args.stream_period,
            compensation_document,
        )
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 若夹爪仍夹着物体，先托住并取下，再把正方体放回1号格中心。")
        print("[安全] 右侧释放区必须为空；清空整条路径并准备急停。")
        confirmation = input("确认现场安全，输入 RUN 并按 Enter：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送运动或夹爪命令。")
            return 2
        runner.run_transfer(
            current,
            poses,
            gripper,
            args.grid,
            destination,
            gripper_speed,
            retreat_extension,
            highest_recorded_z_mm,
        )
        return 0
    except Exception:
        if runner is not None and not runner.stopped:
            runner.stop()
        raise
    finally:
        if runner is not None:
            runner.arm.close()
        control_lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止单次搬运测试。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
