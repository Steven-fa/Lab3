"""Direction-aware joint compensation for the old mechArm 270 Pi.

The configuration stores *command* corrections.  A positive correction means
the commanded angle is greater than the recorded physical target.  Direction
is selected from consecutive recorded (physical) targets; when a joint does
not move, its last direction is retained to avoid a command discontinuity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mecharm_recorded_task import SOFT_LOWER, SOFT_UPPER


FORMAT = "mecharm_directional_joint_compensation/v1"
VALID_DIRECTIONS = ("increasing", "decreasing")


def load_compensation(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise RuntimeError("关节补偿文件格式无效")
    if document.get("enabled") is not True:
        raise RuntimeError("关节补偿配置未启用")
    if document.get("validation_result") != "PASS":
        raise RuntimeError("关节补偿尚未通过安全姿态验证")
    joints = document.get("joints")
    if not isinstance(joints, dict) or not joints:
        raise RuntimeError("关节补偿文件没有有效关节")

    seen_indices: set[int] = set()
    for name, item in joints.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"{name}补偿配置无效")
        human_index = item.get("index")
        if not isinstance(human_index, int) or not 1 <= human_index <= 6:
            raise RuntimeError(f"{name}关节编号无效")
        index = human_index - 1
        if index in seen_indices:
            raise RuntimeError("补偿配置包含重复关节编号")
        seen_indices.add(index)
        validation = item.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            raise RuntimeError(f"{name}补偿没有通过验证")
        corrections = item.get("correction_deg")
        if not isinstance(corrections, dict):
            raise RuntimeError(f"{name}缺少方向补偿值")
        for direction in VALID_DIRECTIONS:
            value = corrections.get(direction)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError(f"{name} {direction}补偿值无效")
            if abs(float(value)) > 3.0:
                raise RuntimeError(f"{name} {direction}补偿值超过3度安全上限")
    return document


class DirectionalJointCompensator:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.joints: dict[str, dict[str, Any]] = document["joints"]
        self.last_desired: np.ndarray | None = None
        self.directions: dict[str, str] = {}

    def _check_angles(self, values: np.ndarray, label: str) -> np.ndarray:
        angles = np.asarray(values, dtype=float)
        if angles.shape != (6,) or not np.all(np.isfinite(angles)):
            raise ValueError(f"{label}必须包含六个有效关节角")
        if np.any(angles < SOFT_LOWER) or np.any(angles > SOFT_UPPER):
            raise ValueError(f"{label}超出软件关节限位")
        return angles

    def choose_directions(
        self, previous_desired: np.ndarray, next_desired: np.ndarray
    ) -> dict[str, str]:
        previous = self._check_angles(previous_desired, "上一物理目标")
        target = self._check_angles(next_desired, "下一物理目标")
        result: dict[str, str] = {}
        for name, item in self.joints.items():
            index = int(item["index"]) - 1
            delta = float(target[index] - previous[index])
            if delta > 1e-6:
                direction = "increasing"
            elif delta < -1e-6:
                direction = "decreasing"
            else:
                direction = self.directions.get(name, "increasing")
            result[name] = direction
        return result

    def compensated(
        self, desired: np.ndarray, directions: dict[str, str] | None = None
    ) -> np.ndarray:
        physical = self._check_angles(desired, "物理目标")
        selected = directions if directions is not None else self.directions
        command = physical.copy()
        for name, item in self.joints.items():
            direction = selected.get(name)
            if direction not in VALID_DIRECTIONS:
                raise RuntimeError(f"{name}尚未建立有效到达方向")
            index = int(item["index"]) - 1
            command[index] += float(item["correction_deg"][direction])
        self._check_angles(command, "补偿命令")
        return command

    def prime(
        self, desired: np.ndarray, following_desired: np.ndarray
    ) -> tuple[np.ndarray, dict[str, str]]:
        physical = self._check_angles(desired, "补偿起始目标")
        self.directions = self.choose_directions(physical, following_desired)
        self.last_desired = physical.copy()
        return self.compensated(physical), dict(self.directions)

    def command_for(self, desired: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
        physical = self._check_angles(desired, "物理目标")
        if self.last_desired is None:
            raise RuntimeError("方向补偿尚未预紧初始化")
        self.directions = self.choose_directions(self.last_desired, physical)
        command = self.compensated(physical)
        self.last_desired = physical.copy()
        return command, dict(self.directions)

    def command_for_pose(
        self, desired: np.ndarray, pose_name: str
    ) -> tuple[np.ndarray, dict[str, str], bool]:
        physical = self._check_angles(desired, "物理目标")
        compensated_command, directions = self.command_for(physical)
        policy = self.document.get("pose_policy", {})
        if not isinstance(policy, dict):
            raise RuntimeError("姿态补偿策略无效")
        enabled = policy.get("default_compensation_enabled", True)
        overrides = policy.get("overrides", {})
        if not isinstance(enabled, bool) or not isinstance(overrides, dict):
            raise RuntimeError("姿态补偿策略字段无效")
        override = overrides.get(pose_name, {})
        if override:
            if not isinstance(override, dict) or not isinstance(
                override.get("compensation_enabled"), bool
            ):
                raise RuntimeError(f"{pose_name}姿态补偿覆盖配置无效")
            enabled = bool(override["compensation_enabled"])
        return (
            compensated_command if enabled else physical.copy(),
            directions,
            enabled,
        )

    def preload_commands(
        self,
        desired: np.ndarray,
        following_desired: np.ndarray,
        starting_command: np.ndarray,
    ) -> list[dict[str, Any]]:
        physical = self._check_angles(desired, "补偿起始目标")
        current = self._check_angles(starting_command, "预紧起始命令").copy()
        final_command, directions = self.prime(physical, following_desired)
        offset = float(self.document.get("preload_offset_deg", 4.0))
        if not 2.0 <= offset <= 6.0:
            raise RuntimeError("预紧偏移必须在2~6度")

        commands: list[dict[str, Any]] = []
        for name, item in self.joints.items():
            index = int(item["index"]) - 1
            direction = directions[name]
            pre = current.copy()
            pre[index] = final_command[index] + (-offset if direction == "increasing" else offset)
            self._check_angles(pre, f"{name}预紧命令")
            commands.append(
                {"joint": name, "direction": direction, "stage": "preload", "angles": pre}
            )
            current = pre.copy()
            current[index] = final_command[index]
            self._check_angles(current, f"{name}补偿起始命令")
            commands.append(
                {"joint": name, "direction": direction, "stage": "settle", "angles": current.copy()}
            )
        if not np.allclose(current, final_command):
            raise RuntimeError("预紧序列没有形成完整补偿命令")
        return commands

    def metadata(self) -> dict[str, Any]:
        return {
            "format": self.document["format"],
            "scope": self.document.get("scope"),
            "production_enabled": bool(self.document.get("production_enabled", False)),
            "baseline_report": self.document.get("baseline_report"),
            "validation_report": self.document.get("validation_report"),
            "pose_policy": self.document.get("pose_policy", {}),
            "joints": self.joints,
        }
