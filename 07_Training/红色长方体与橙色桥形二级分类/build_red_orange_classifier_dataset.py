#!/usr/bin/env python3
"""Build a two-class crop dataset from verified YOLO red/orange labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {4: "red_cuboid", 5: "orange_bridge"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def crop_box(image: np.ndarray, row: str, pad_ratio: float = 0.10) -> np.ndarray:
    class_id, cx, cy, box_w, box_h = map(float, row.split())
    height, width = image.shape[:2]
    x1 = (cx - box_w / 2.0) * width
    y1 = (cy - box_h / 2.0) * height
    x2 = (cx + box_w / 2.0) * width
    y2 = (cy + box_h / 2.0) * height
    pad_x, pad_y = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    return image[top:bottom, left:right]


def red_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 90, 40]), np.array([16, 255, 255])),
        cv2.inRange(hsv, np.array([164, 90, 40]), np.array([179, 255, 255])),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [tuple(int(value) for value in stats[index]) for index in range(1, count)]
    candidates = [item for item in candidates if item[4] >= 0.0015 * frame.shape[0] * frame.shape[1]]
    if not candidates:
        raise ValueError("No red object found")
    x, y, width, height, _ = max(candidates, key=lambda item: item[4])
    pad_x, pad_y = round(width * 0.10), round(height * 0.10)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(frame.shape[1], x + width + pad_x),
        min(frame.shape[0], y + height + pad_y),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-red-dir", type=Path)
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES.values():
            (output / split / class_name).mkdir(parents=True, exist_ok=True)
        for image_path in sorted((source / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")
            target_index = 0
            for row in label_path.read_text(encoding="utf-8-sig").splitlines():
                if not row.strip():
                    continue
                class_id = int(row.split()[0])
                if class_id not in CLASS_NAMES:
                    continue
                crop = crop_box(image, row)
                if crop.size == 0:
                    raise ValueError(f"Empty crop for {image_path}")
                class_name = CLASS_NAMES[class_id]
                name = f"{image_path.stem}_{target_index:02d}.jpg"
                cv2.imwrite(str(output / split / class_name / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                counts[f"{split}/{class_name}"] += 1
                target_index += 1

    if args.extra_red_dir:
        extras = sorted(
            path for path in args.extra_red_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        split_plan = ["train"] * max(0, len(extras) - 4) + ["val"] * 2 + ["test"] * 2
        for index, (image_path, split) in enumerate(zip(extras, split_plan)):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            x1, y1, x2, y2 = red_bbox(frame)
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(
                str(output / split / "red_cuboid" / f"extra_red_{index:03d}.jpg"),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 96],
            )
            counts[f"{split}/red_cuboid"] += 1

    manifest = {"source": str(source), "extra_red_dir": str(args.extra_red_dir or ""), "counts": dict(sorted(counts.items()))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
