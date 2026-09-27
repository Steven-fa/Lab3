#!/usr/bin/env python3
"""Build Experiment 3 v4 from the v3 dataset plus two user videos.

The first video contains one red cuboid on a light desk.  Its bounding box is
derived from a conservative saturated-red HSV mask, not from model pseudo
labels.  The second video is a verified target-free clutter scene and receives
empty YOLO labels.  Chronological splits prevent sampled near-neighbours from
appearing in both training and evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NAMES = [
    "blue_cube",
    "green_cube",
    "pink_cube",
    "yellow_cube",
    "red_cuboid",
    "orange_bridge",
]
RED_CUBOID_CLASS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_v3_dataset(source: Path, output: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        image_src = source / "images" / split
        label_src = source / "labels" / split
        image_dst = output / "images" / split
        label_dst = output / "labels" / split
        image_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)
        images = sorted(
            path for path in image_src.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image in images:
            label = label_src / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing v3 label: {label}")
            shutil.copy2(image, image_dst / image.name)
            shutil.copy2(label, label_dst / label.name)
        counts[split] = len(images)
    return counts


def sampled_frames(video: Path, stride: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frames: list[tuple[int, np.ndarray]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append((index, frame))
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return frames


def chronological_split(count: int) -> list[str]:
    if count < 15:
        raise ValueError(f"Need at least 15 sampled frames, got {count}")
    train_end = max(1, round(count * 0.70))
    val_end = max(train_end + 1, round(count * 0.85))
    return [
        "train" if i < train_end else "val" if i < val_end else "test"
        for i in range(count)
    ]


def red_bbox(frame: np.ndarray) -> tuple[int, int, int, int, dict[str, float]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv, np.array([0, 95, 45]), np.array([16, 255, 255]))
    upper_red = cv2.inRange(hsv, np.array([164, 95, 45]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[int, int, int, int, int]] = []
    for component in range(1, count):
        x, y, box_w, box_h, area = [int(value) for value in stats[component]]
        if area >= 0.0015 * width * height:
            candidates.append((area, x, y, box_w, box_h))
    if not candidates:
        raise ValueError("No sufficiently large saturated-red component")

    area, x, y, box_w, box_h = max(candidates)
    fill_ratio = area / float(box_w * box_h)
    box_ratio = (box_w * box_h) / float(width * height)
    if fill_ratio < 0.30:
        raise ValueError(f"Red component is too sparse: fill={fill_ratio:.3f}")
    if not 0.003 <= box_ratio <= 0.45:
        raise ValueError(f"Red bounding-box area is implausible: ratio={box_ratio:.3f}")

    pad_x = max(4, round(box_w * 0.035))
    pad_y = max(4, round(box_h * 0.035))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_w + pad_x)
    y2 = min(height, y + box_h + pad_y)
    return x1, y1, x2, y2, {
        "mask_area_ratio": area / float(width * height),
        "bbox_area_ratio": ((x2 - x1) * (y2 - y1)) / float(width * height),
        "mask_fill_ratio": fill_ratio,
    }


def yolo_line(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return (
        f"{RED_CUBOID_CLASS} {center_x:.6f} {center_y:.6f} "
        f"{box_w:.6f} {box_h:.6f}\n"
    )


def save_qa_contact_sheet(items: list[tuple[np.ndarray, str]], output: Path) -> None:
    if not items:
        return
    thumb_w, thumb_h = 320, 240
    columns = 4
    rows = (len(items) + columns - 1) // columns
    canvas = np.full((rows * thumb_h, columns * thumb_w, 3), 245, np.uint8)
    for index, (image, caption) in enumerate(items):
        resized = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, thumb_h - 25), (thumb_w, thumb_h), (0, 0, 0), -1)
        cv2.putText(
            resized,
            caption,
            (7, thumb_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        canvas[
            row * thumb_h : (row + 1) * thumb_h,
            column * thumb_w : (column + 1) * thumb_w,
        ] = resized
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--positive-video", type=Path, required=True)
    parser.add_argument("--negative-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--positive-stride", type=int, default=5)
    parser.add_argument("--negative-stride", type=int, default=3)
    parser.add_argument("--positive-repeat", type=int, default=4)
    parser.add_argument("--negative-repeat", type=int, default=6)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if min(args.positive_stride, args.negative_stride) < 1:
        raise ValueError("Video strides must be positive")

    original_counts = copy_v3_dataset(source, output)
    positive = sampled_frames(args.positive_video.resolve(), args.positive_stride)
    negative = sampled_frames(args.negative_video.resolve(), args.negative_stride)
    positive_splits = chronological_split(len(positive))
    negative_splits = chronological_split(len(negative))

    manifest_items: list[dict[str, object]] = []
    qa_positive: list[tuple[np.ndarray, str]] = []
    qa_negative: list[tuple[np.ndarray, str]] = []
    train_positive_names: list[str] = []
    train_negative_names: list[str] = []

    for sample_index, ((frame_index, frame), split) in enumerate(
        zip(positive, positive_splits)
    ):
        x1, y1, x2, y2, quality = red_bbox(frame)
        name = f"v4red_{split}_{sample_index:03d}_f{frame_index:04d}.jpg"
        image_path = output / "images" / split / name
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        (output / "labels" / split / f"{Path(name).stem}.txt").write_text(
            yolo_line(frame, (x1, y1, x2, y2)), encoding="utf-8"
        )
        if split == "train":
            train_positive_names.append(name)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 8)
        if len(qa_positive) < 24 or sample_index % 7 == 0:
            qa_positive.append((annotated, f"{split} f={frame_index}"))
        manifest_items.append(
            {
                "kind": "red_cuboid_positive",
                "split": split,
                "dataset_name": name,
                "frame_index": frame_index,
                "bbox_xyxy": [x1, y1, x2, y2],
                **quality,
            }
        )

    for sample_index, ((frame_index, frame), split) in enumerate(
        zip(negative, negative_splits)
    ):
        name = f"v4neg_{split}_{sample_index:03d}_f{frame_index:04d}.jpg"
        image_path = output / "images" / split / name
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        (output / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "", encoding="utf-8"
        )
        if split == "train":
            train_negative_names.append(name)
        if len(qa_negative) < 16:
            qa_negative.append((frame, f"{split} f={frame_index}"))
        manifest_items.append(
            {
                "kind": "verified_negative",
                "split": split,
                "dataset_name": name,
                "frame_index": frame_index,
            }
        )

    source_focus = source / "train_focus.txt"
    original_focus_names = [
        Path(line.strip().replace("\\", "/")).name
        for line in source_focus.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    remote_root = args.remote_root.rstrip("/")
    focus_lines = [
        f"{remote_root}/images/train/{name}" for name in original_focus_names
    ]
    for name in train_positive_names:
        focus_lines.extend(
            [f"{remote_root}/images/train/{name}"] * args.positive_repeat
        )
    for name in train_negative_names:
        focus_lines.extend(
            [f"{remote_root}/images/train/{name}"] * args.negative_repeat
        )
    (output / "train_focus.txt").write_text(
        "\n".join(focus_lines) + "\n", encoding="utf-8"
    )

    yaml_lines = [
        f"path: {remote_root}",
        f"train: {remote_root}/train_focus.txt",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {index}: {name}" for index, name in enumerate(NAMES)]
    (output / "dataset_v4.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )

    split_counts = {
        kind: dict(Counter(str(item["split"]) for item in manifest_items if item["kind"] == kind))
        for kind in ("red_cuboid_positive", "verified_negative")
    }
    manifest = {
        "source_v3_dataset": str(source),
        "source_v3_model_expected": "cube6_red_focus_v3_hard_negative_best.pt",
        "positive_video": {
            "path": str(args.positive_video),
            "sha256": sha256(args.positive_video),
            "stride": args.positive_stride,
        },
        "negative_video": {
            "path": str(args.negative_video),
            "sha256": sha256(args.negative_video),
            "stride": args.negative_stride,
        },
        "classes": NAMES,
        "original_counts": original_counts,
        "new_counts": split_counts,
        "positive_repeat_in_train_focus": args.positive_repeat,
        "negative_repeat_in_train_focus": args.negative_repeat,
        "effective_train_entries": len(focus_lines),
        "items": manifest_items,
    }
    (output / "v4_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_qa_contact_sheet(qa_positive[:32], output / "qa_red_cuboid.jpg")
    save_qa_contact_sheet(qa_negative[:16], output / "qa_negative.jpg")
    print(json.dumps({key: value for key, value in manifest.items() if key != "items"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
