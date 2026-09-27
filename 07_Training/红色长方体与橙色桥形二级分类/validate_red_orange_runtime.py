#!/usr/bin/env python3
"""Regression-test the deployed two-stage logic on the saved real scene."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hardware-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.hardware_dir.resolve()))
    from experiment3_vision_preview import Vision

    detector = YOLO(str(args.detector.resolve()))
    pair_model = YOLO(str(args.pair_model.resolve()))
    vision = Vision.__new__(Vision)
    vision.cuda_device = args.device
    vision.red_orange_confidence = 0.80
    vision.red_orange_margin = 0.30

    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise RuntimeError(args.video)
    raw: Counter[str] = Counter()
    refined: Counter[str] = Counter()
    frame_index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index > 340:
            break
        if frame_index % 20:
            continue
        result = detector.predict(
            frame, imgsz=640, conf=0.20, device=args.device, verbose=False
        )[0]
        expected: dict[int, str] = {}
        for index, box in enumerate(result.boxes or []):
            name = str(result.names[int(box.cls[0])])
            if name not in {"red_cuboid", "orange_bridge"}:
                continue
            _x1, y1, _x2, y2 = [float(value) for value in box.xyxy[0]]
            center_y_ratio = ((y1 + y2) / 2.0) / frame.shape[0]
            if center_y_ratio >= 0.52:
                expected[index] = "red_cuboid"
            elif center_y_ratio <= 0.48:
                expected[index] = "orange_bridge"
            if index in expected:
                raw[f"{expected[index]}->{name}"] += 1
        vision._refine_red_orange(frame, result, pair_model)
        for index, wanted in expected.items():
            got = str(result.names[int(result.boxes.cls[index])])
            refined[f"{wanted}->{got}"] += 1
    capture.release()

    correct = sum(refined[f"{name}->{name}"] for name in ("red_cuboid", "orange_bridge"))
    total = sum(refined.values())
    report = {
        "raw": dict(sorted(raw.items())),
        "refined": dict(sorted(refined.items())),
        "samples": total,
        "accuracy": correct / max(total, 1),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    expected_keys = {
        "red_cuboid->red_cuboid",
        "orange_bridge->orange_bridge",
    }
    return 0 if total >= 10 and set(refined) == expected_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
