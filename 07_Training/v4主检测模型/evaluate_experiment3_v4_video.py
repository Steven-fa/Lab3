#!/usr/bin/env python3
"""Evaluate v3/v4 on the common v4 test set and its new video subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def read_ground_truth(label: Path, width: int, height: int) -> list[float]:
    values = [float(value) for value in label.read_text().split()]
    if len(values) != 5 or int(values[0]) != 4:
        raise ValueError(f"Expected one red_cuboid label in {label}")
    _, cx, cy, box_w, box_h = values
    return [
        (cx - box_w / 2.0) * width,
        (cy - box_h / 2.0) * height,
        (cx + box_w / 2.0) * width,
        (cy + box_h / 2.0) * height,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    test_images = args.dataset_root / "images" / "test"
    test_labels = args.dataset_root / "labels" / "test"
    red_paths = sorted(test_images.glob("v4red_test_*.jpg"))
    negative_paths = sorted(test_images.glob("v4neg_test_*.jpg"))
    if not red_paths or not negative_paths:
        raise FileNotFoundError("New v4 red or negative held-out frames are missing")

    red_results = []
    red_hits_025 = 0
    red_hits_050 = 0
    for path in red_paths:
        result = model.predict(str(path), conf=0.05, imgsz=640, device=args.device, verbose=False)[0]
        gt = read_ground_truth(test_labels / f"{path.stem}.txt", result.orig_shape[1], result.orig_shape[0])
        best_confidence = 0.0
        best_iou = 0.0
        for cls, confidence, xyxy in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
        ):
            if int(cls) != 4:
                continue
            overlap = box_iou(gt, [float(value) for value in xyxy])
            if overlap > best_iou:
                best_iou = overlap
                best_confidence = float(confidence)
        if best_iou >= 0.5 and best_confidence >= 0.25:
            red_hits_025 += 1
        if best_iou >= 0.5 and best_confidence >= 0.50:
            red_hits_050 += 1
        red_results.append({"image": path.name, "best_iou": best_iou, "best_confidence": best_confidence})

    negative_thresholds = (0.10, 0.25, 0.50)
    negative_stats = {
        str(threshold): {"images_with_false_positive": 0, "false_boxes": 0}
        for threshold in negative_thresholds
    }
    negative_details = []
    for path in negative_paths:
        result = model.predict(str(path), conf=0.05, imgsz=640, device=args.device, verbose=False)[0]
        boxes = [
            {"class": model.names[int(cls)], "confidence": float(confidence)}
            for cls, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
        ]
        for threshold in negative_thresholds:
            count = sum(item["confidence"] >= threshold for item in boxes)
            if count:
                negative_stats[str(threshold)]["images_with_false_positive"] += 1
                negative_stats[str(threshold)]["false_boxes"] += count
        negative_details.append({"image": path.name, "detections": boxes})

    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=640,
        batch=8,
        device=args.device,
        plots=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    report = {
        "model": str(args.model),
        "video_red_holdout": {
            "images": len(red_paths),
            "recall_iou50_conf025": red_hits_025 / len(red_paths),
            "recall_iou50_conf050": red_hits_050 / len(red_paths),
            "mean_best_iou": sum(item["best_iou"] for item in red_results) / len(red_results),
            "mean_best_confidence": sum(item["best_confidence"] for item in red_results) / len(red_results),
            "details": red_results,
        },
        "video_negative_holdout": {
            "images": len(negative_paths),
            "thresholds": negative_stats,
            "details": negative_details,
        },
        "full_v4_test_metrics": {
            key: float(value) for key, value in metrics.results_dict.items()
        },
        "per_class_map50_95": {
            str(model.names[index]): float(value)
            for index, value in enumerate(metrics.box.maps.tolist())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
