"""Pure-Python helpers for recording and learning robot demonstrations."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

import yaml


ARM_JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
GRIPPER_JOINT = "gripper_controller"
POSE_FIELDS = [
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "object_x",
    "object_y",
    "object_z",
    "object_qx",
    "object_qy",
    "object_qz",
    "object_qw",
]


def _distance(first: Iterable[float], second: Iterable[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _feature(sample: dict[str, float], joint_names: list[str]) -> list[float]:
    # The gripper is down-weighted so one open/close event does not hide arm turns.
    return [float(sample[name]) for name in joint_names] + [
        0.35 * float(sample[GRIPPER_JOINT])
    ]


def extract_keypoints(
    samples: list[dict[str, float]],
    joint_names: list[str],
    tolerance: float = 0.035,
    pause_speed: float = 0.025,
    gripper_delta: float = 0.08,
) -> list[dict[str, object]]:
    """Select motion turns, pauses and gripper events from a dense trajectory."""
    if not samples:
        return []
    if len(samples) == 1:
        return [{"index": 0, "reasons": ["开始", "结束"]}]

    reasons: dict[int, set[str]] = {0: {"开始"}, len(samples) - 1: {"结束"}}
    kept = {0, len(samples) - 1}
    features = [_feature(sample, joint_names) for sample in samples]
    times = [float(sample["time_from_start"]) for sample in samples]

    # Ramer-Douglas-Peucker in joint/gripper space with time-aware interpolation.
    pending = [(0, len(samples) - 1)]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        duration = max(times[end] - times[start], 1e-9)
        largest_error = -1.0
        largest_index = -1
        for index in range(start + 1, end):
            phase = (times[index] - times[start]) / duration
            expected = [
                left + phase * (right - left)
                for left, right in zip(features[start], features[end])
            ]
            error = _distance(features[index], expected) / math.sqrt(len(expected))
            if error > largest_error:
                largest_error = error
                largest_index = index
        if largest_error > tolerance:
            kept.add(largest_index)
            reasons.setdefault(largest_index, set()).add("轨迹转折")
            pending.extend([(start, largest_index), (largest_index, end)])

    speeds = [0.0]
    for index in range(1, len(samples)):
        delta_time = times[index] - times[index - 1]
        arm_distance = _distance(
            [samples[index][name] for name in joint_names],
            [samples[index - 1][name] for name in joint_names],
        )
        speeds.append(arm_distance / delta_time if delta_time >= 1e-4 else 0.0)

        if abs(
            float(samples[index][GRIPPER_JOINT])
            - float(samples[index - 1][GRIPPER_JOINT])
        ) >= gripper_delta:
            kept.add(index)
            reasons.setdefault(index, set()).add("夹爪状态变化")

    moving = [speed >= pause_speed for speed in speeds]
    for index in range(2, len(samples)):
        if moving[index] != moving[index - 1]:
            kept.add(index)
            label = "开始运动" if moving[index] else "停顿"
            reasons.setdefault(index, set()).add(label)

    result = []
    for index in sorted(kept):
        result.append(
            {
                "index": index,
                "reasons": sorted(reasons.get(index, {"轨迹转折"})),
            }
        )
    return result


def trajectory_statistics(
    samples: list[dict[str, float]], joint_names: list[str]
) -> dict[str, object]:
    """Compute concise, useful statistics for one demonstration."""
    if not samples:
        return {
            "sample_count": 0,
            "duration_s": 0.0,
            "joint_path_rad": 0.0,
            "end_effector_path_m": 0.0,
            "maximum_joint_speed_rad_s": 0.0,
            "gripper_event_count": 0,
        }

    joint_path = 0.0
    ee_path = 0.0
    maximum_speed = 0.0
    gripper_events = 0
    for index in range(1, len(samples)):
        joint_delta = _distance(
            [samples[index][name] for name in joint_names],
            [samples[index - 1][name] for name in joint_names],
        )
        joint_path += joint_delta
        delta_time = (
            float(samples[index]["time_from_start"])
            - float(samples[index - 1]["time_from_start"])
        )
        if delta_time >= 1e-4:
            maximum_speed = max(maximum_speed, joint_delta / delta_time)
        ee_path += _distance(
            [samples[index][name] for name in ("ee_x", "ee_y", "ee_z")],
            [samples[index - 1][name] for name in ("ee_x", "ee_y", "ee_z")],
        )
        if abs(
            float(samples[index][GRIPPER_JOINT])
            - float(samples[index - 1][GRIPPER_JOINT])
        ) >= 0.08:
            gripper_events += 1

    return {
        "sample_count": len(samples),
        "duration_s": round(float(samples[-1]["time_from_start"]), 6),
        "joint_path_rad": round(joint_path, 6),
        "end_effector_path_m": round(ee_path, 6),
        "maximum_joint_speed_rad_s": round(maximum_speed, 6),
        "gripper_event_count": gripper_events,
        "object_start_position": [
            round(float(samples[0][name]), 6)
            for name in ("object_x", "object_y", "object_z")
        ],
        "object_end_position": [
            round(float(samples[-1][name]), 6)
            for name in ("object_x", "object_y", "object_z")
        ],
    }


def _keypoint_rows(
    samples: list[dict[str, float]],
    keypoints: list[dict[str, object]],
    joint_names: list[str],
) -> list[dict[str, object]]:
    result = []
    for keypoint in keypoints:
        sample = samples[int(keypoint["index"])]
        result.append(
            {
                "index": int(keypoint["index"]),
                "time_from_start": round(float(sample["time_from_start"]), 6),
                "reasons": list(keypoint["reasons"]),
                "joint_positions": [
                    round(float(sample[name]), 8) for name in joint_names
                ],
                "gripper_position": round(float(sample[GRIPPER_JOINT]), 8),
                "end_effector_position": [
                    round(float(sample[name]), 8)
                    for name in ("ee_x", "ee_y", "ee_z")
                ],
                "object_position": [
                    round(float(sample[name]), 8)
                    for name in ("object_x", "object_y", "object_z")
                ],
            }
        )
    return result


def write_demonstration(
    output_directory: Path,
    samples: list[dict[str, float]],
    joint_names: list[str],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write dense CSV, compact YAML summary and a one-demo learned trajectory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    field_names = ["time_from_start", *joint_names, GRIPPER_JOINT, *POSE_FIELDS]
    with (output_directory / "raw_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample[name] for name in field_names})

    keypoints = extract_keypoints(samples, joint_names)
    statistics_report = trajectory_statistics(samples, joint_names)
    summary = {
        "schema_version": 1,
        "metadata": metadata,
        "joint_names": joint_names,
        "gripper_joint": GRIPPER_JOINT,
        "statistics": statistics_report,
        "keypoints": _keypoint_rows(samples, keypoints, joint_names),
    }
    with (output_directory / "summary.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, allow_unicode=True, sort_keys=False)

    learned = build_learned_trajectory([samples], joint_names, sample_count=200)
    learned["source_demonstrations"] = [str(output_directory)]
    with (output_directory / "learned_trajectory.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(learned, stream, allow_unicode=True, sort_keys=False)
    return summary


def read_demonstration(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _interpolate(
    samples: list[dict[str, float]], field: str, target_time: float
) -> float:
    if target_time <= samples[0]["time_from_start"]:
        return float(samples[0][field])
    if target_time >= samples[-1]["time_from_start"]:
        return float(samples[-1][field])
    low = 0
    high = len(samples) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if samples[middle]["time_from_start"] <= target_time:
            low = middle
        else:
            high = middle
    left_time = float(samples[low]["time_from_start"])
    right_time = float(samples[high]["time_from_start"])
    phase = (target_time - left_time) / max(right_time - left_time, 1e-9)
    return float(samples[low][field]) + phase * (
        float(samples[high][field]) - float(samples[low][field])
    )


def build_learned_trajectory(
    demonstrations: list[list[dict[str, float]]],
    joint_names: list[str],
    sample_count: int = 200,
) -> dict[str, object]:
    """Time-normalise demonstrations and learn their point-wise median path."""
    valid = [samples for samples in demonstrations if len(samples) >= 2]
    if not valid:
        raise ValueError("至少需要一条包含两个采样点的示教轨迹")
    sample_count = max(2, int(sample_count))
    durations = [float(samples[-1]["time_from_start"]) for samples in valid]
    learned_duration = statistics.median(durations)
    fields = [*joint_names, GRIPPER_JOINT, *POSE_FIELDS]
    rows: list[dict[str, float]] = []
    for index in range(sample_count):
        phase = index / (sample_count - 1)
        row = {"time_from_start": phase * learned_duration}
        for field in fields:
            values = [
                _interpolate(samples, field, phase * duration)
                for samples, duration in zip(valid, durations)
            ]
            row[field] = statistics.median(values)
        rows.append(row)

    # Smooth arm/Cartesian positions while preserving endpoints and gripper events.
    smooth_fields = [*joint_names, "ee_x", "ee_y", "ee_z"]
    for field in smooth_fields:
        original = [row[field] for row in rows]
        for index in range(1, len(rows) - 1):
            rows[index][field] = statistics.mean(
                original[max(0, index - 2): min(len(rows), index + 3)]
            )

    keypoints = extract_keypoints(rows, joint_names)
    return {
        "schema_version": 1,
        "method": "time_normalised_pointwise_median",
        "demonstration_count": len(valid),
        "joint_names": joint_names,
        "gripper_joint": GRIPPER_JOINT,
        "duration_s": round(learned_duration, 6),
        "statistics": trajectory_statistics(rows, joint_names),
        "keypoints": _keypoint_rows(rows, keypoints, joint_names),
        "samples": [
            {
                "time_from_start": round(float(row["time_from_start"]), 6),
                "joint_positions": [round(float(row[name]), 8) for name in joint_names],
                "gripper_position": round(float(row[GRIPPER_JOINT]), 8),
            }
            for row in rows
        ],
    }
