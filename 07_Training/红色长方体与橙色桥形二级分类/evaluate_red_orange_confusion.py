#!/usr/bin/env python3
"""Measure red_cuboid/orange_bridge confusion on a labelled YOLO split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_detection_ros2.color_refinement import refine_red_orange_name


TARGET_CLASSES = {4: "red_cuboid", 5: "orange_bridge"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    items: list[dict] = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if not values:
            continue
        class_id = int(values[0])
        if class_id not in TARGET_CLASSES:
            continue
        cx, cy, box_w, box_h = map(float, values[1:5])
        items.append(
            {
                "class_id": class_id,
                "box": [
                    (cx - box_w / 2.0) * width,
                    (cy - box_h / 2.0) * height,
                    (cx + box_w / 2.0) * width,
                    (cy + box_h / 2.0) * height,
                ],
            }
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.35)
    parser.add_argument("--refine-color", action="store_true")
    args = parser.parse_args()

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    model = YOLO(str(args.model))
    counts: Counter[str] = Counter()
    details: list[dict] = []

    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        labels = ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        if not labels:
            continue
        result = model.predict(
            str(image_path), conf=0.01, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        predictions = []
        for class_id, confidence, box in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            class_id = int(class_id)
            if class_id not in TARGET_CLASSES:
                continue
            float_box = [float(value) for value in box]
            x1, y1, x2, y2 = [int(value) for value in float_box]
            raw_name = TARGET_CLASSES[class_id]
            decision = refine_red_orange_name(raw_name, image[y1:y2, x1:x2])
            effective_name = decision.class_name if args.refine_color else raw_name
            effective_id = next(
                key for key, value in TARGET_CLASSES.items() if value == effective_name
            )
            predictions.append(
                {
                    "class_id": effective_id,
                    "raw_class_id": class_id,
                    "confidence": float(confidence),
                    "box": float_box,
                    "color_corrected": bool(args.refine_color and decision.corrected),
                }
            )
        for label in labels:
            true_name = TARGET_CLASSES[label["class_id"]]
            best = max(
                predictions,
                key=lambda item: box_iou(label["box"], item["box"]),
                default=None,
            )
            overlap = box_iou(label["box"], best["box"]) if best else 0.0
            if best is None or overlap < args.match_iou:
                outcome = "missed"
                predicted_name = None
                confidence = 0.0
            else:
                predicted_name = TARGET_CLASSES[best["class_id"]]
                confidence = best["confidence"]
                outcome = "correct" if best["class_id"] == label["class_id"] else "confused"
            counts[f"{true_name}:{outcome}"] += 1
            if predicted_name:
                counts[f"{true_name}->{predicted_name}"] += 1
            details.append(
                {
                    "image": image_path.name,
                    "true": true_name,
                    "predicted": predicted_name,
                    "confidence": round(confidence, 6),
                    "iou": round(overlap, 6),
                    "outcome": outcome,
                }
            )

    summary = {
        "model": str(args.model.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "refine_color": args.refine_color,
        "counts": dict(sorted(counts.items())),
        "red_to_orange": counts["red_cuboid->orange_bridge"],
        "orange_to_red": counts["orange_bridge->red_cuboid"],
        "samples": len(details),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
