"""Persistent file format and validation for manually selected keyframes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


KEYFRAME_TYPES = {"waypoint", "grasp", "release"}


def classify_q_sequence(suffix: str | None) -> str:
    """Translate raw q/qa/qb keyboard input into a keyframe type."""
    if suffix is None:
        return "waypoint"
    value = suffix.lower()
    if value == "a":
        return "grasp"
    if value == "b":
        return "release"
    return "waypoint"


def validate_keyframes(keyframes: list[dict[str, Any]]) -> None:
    """Reject malformed or logically impossible keyframe sequences."""
    if not keyframes:
        raise ValueError("关键点文件中没有关键点")

    carrying = False
    for index, keyframe in enumerate(keyframes, start=1):
        kind = str(keyframe.get("type", ""))
        if kind not in KEYFRAME_TYPES:
            raise ValueError(f"第 {index} 个关键点类型无效：{kind!r}")

        pose = keyframe.get("ee_pose", {})
        if not isinstance(pose, dict):
            raise ValueError(f"第 {index} 个关键点末端位姿格式无效")
        if len(pose.get("position", [])) != 3:
            raise ValueError(f"第 {index} 个关键点缺少末端位置")
        if len(pose.get("orientation_xyzw", [])) != 4:
            raise ValueError(f"第 {index} 个关键点缺少末端姿态")

        object_pose = keyframe.get("object_pose", {})
        if not isinstance(object_pose, dict):
            raise ValueError(f"第 {index} 个关键点工件位姿格式无效")
        if len(object_pose.get("position", [])) != 3:
            raise ValueError(f"第 {index} 个关键点缺少工件位置")
        if len(object_pose.get("orientation_xyzw", [])) != 4:
            raise ValueError(f"第 {index} 个关键点缺少工件姿态")

        if kind == "grasp":
            if carrying:
                raise ValueError(f"第 {index} 个关键点重复夹取，前一次尚未释放")
            carrying = True
        elif kind == "release":
            if not carrying:
                raise ValueError(f"第 {index} 个关键点先释放、后夹取")
            carrying = False

    if carrying:
        raise ValueError("关键点序列包含夹取，但缺少后续释放点 qb")


def keyframe_document(
    keyframes: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build the stable YAML representation used by recorder and player."""
    return {
        "format": "mecharm_keyframes/v1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
    }


def write_keyframes(
    output_directory: Path,
    keyframes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Atomically persist every selected keyframe after each key press."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "keyframes.yaml"
    temporary_path = output_directory / "keyframes.yaml.tmp"
    with temporary_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            keyframe_document(keyframes, metadata),
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    temporary_path.replace(output_path)
    return output_path


def load_keyframes(path: Path) -> dict[str, Any]:
    """Load and validate a keyframe YAML file or its containing directory."""
    source = path.expanduser()
    if source.is_dir():
        source = source / "keyframes.yaml"
    if not source.is_file():
        raise ValueError(f"找不到关键点文件：{source}")
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if document.get("format") != "mecharm_keyframes/v1":
        raise ValueError("不支持的关键点文件格式")
    keyframes = document.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError("关键点文件缺少 keyframes 列表")
    validate_keyframes(keyframes)
    document["source_path"] = str(source.resolve())
    return document
