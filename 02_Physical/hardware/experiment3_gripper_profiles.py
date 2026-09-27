"""Validated class-specific gripper profiles shared by Experiment 3 runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EXPECTED_FORMAT = "mecharm_experiment3_gripper_profiles/v1"
VALID_GROUPS = {"cube", "other"}
MAX_SAFE_GRIPPER_SPEED = 30


def _position_value(document: dict[str, Any], name: str) -> int:
    value = document.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise RuntimeError(f"夹爪配置{name}必须是0～100的整数")
    return value


def load_gripper_profiles(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise RuntimeError(f"分类夹爪配置不存在：{path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("分类夹爪配置格式无效")
    if document.get("complete") is not True:
        raise RuntimeError("分类夹爪配置尚未完成；请先分别保存cube和other")
    open_value = _position_value(document, "open")
    cube_close = _position_value(document, "cube_close")
    other_close = _position_value(document, "other_close")
    speed = _position_value(document, "speed")
    if not 1 <= speed <= MAX_SAFE_GRIPPER_SPEED:
        raise RuntimeError(
            f"夹爪速度必须位于1～{MAX_SAFE_GRIPPER_SPEED}；"
            "正式搬运不接受测试器的高速参数"
        )
    if not other_close < cube_close < open_value:
        raise RuntimeError(
            "分类夹爪配置方向错误：必须满足other_close < cube_close < open"
        )
    return {
        "open": open_value,
        "cube_close": cube_close,
        "other_close": other_close,
        "speed": speed,
    }


def profile_for_group(profiles: dict[str, int], group: str) -> dict[str, int]:
    if group not in VALID_GROUPS:
        raise RuntimeError(f"不支持的夹爪分类：{group}")
    return {
        "open": int(profiles["open"]),
        "close": int(profiles[f"{group}_close"]),
        "speed": int(profiles["speed"]),
    }
