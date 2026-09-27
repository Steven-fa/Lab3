#!/usr/bin/env python3
"""Scan a video for red/orange detections and save a compact review sheet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TARGETS = {"red_cuboid", "orange_bridge"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--pair-model", type=Path)
    parser.add_argument("--pair-confidence", type=float, default=0.80)
    parser.add_argument("--pair-margin", type=float, default=0.30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    pair_model = YOLO(str(args.pair_model)) if args.pair_model else None
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    records: list[dict] = []
    thumbnails: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    frame_index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.stride:
            continue
        result = model.predict(frame, conf=args.conf, imgsz=640, device=args.device, verbose=False)[0]
        detections = []
        annotated = frame.copy()
        for class_id, confidence, xyxy in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
        ):
            raw_name = str(model.names[int(class_id)])
            if raw_name not in TARGETS:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            box = [x1, y1, x2, y2]
            name = raw_name
            pair_name = ""
            pair_confidence = 0.0
            pair_margin = 0.0
            if pair_model is not None:
                pad_x = max(1, round((x2 - x1) * 0.08))
                pad_y = max(1, round((y2 - y1) * 0.08))
                crop = frame[
                    max(0, y1 - pad_y) : min(frame.shape[0], y2 + pad_y),
                    max(0, x1 - pad_x) : min(frame.shape[1], x2 + pad_x),
                ]
                if crop.size:
                    classification = pair_model.predict(
                        crop, imgsz=224, device=args.device, verbose=False
                    )[0]
                    probabilities = classification.probs.data.tolist()
                    order = sorted(
                        range(len(probabilities)),
                        key=probabilities.__getitem__,
                        reverse=True,
                    )
                    pair_name = str(pair_model.names[order[0]])
                    pair_confidence = float(probabilities[order[0]])
                    pair_margin = pair_confidence - float(probabilities[order[1]])
                    if (
                        pair_name in TARGETS
                        and pair_confidence >= args.pair_confidence
                        and pair_margin >= args.pair_margin
                    ):
                        name = pair_name
            counts[name] += 1
            detections.append(
                {
                    "class_name": name,
                    "raw_class_name": raw_name,
                    "pair_class_name": pair_name,
                    "pair_confidence": round(pair_confidence, 6),
                    "pair_margin": round(pair_margin, 6),
                    "confidence": round(float(confidence), 6),
                    "xyxy": box,
                    "center": [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
                    "aspect_ratio": round((x2 - x1) / max(y2 - y1, 1), 6),
                }
            )
            color = (0, 0, 255) if name == "red_cuboid" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"{name} {confidence:.2f}", (x1, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        if detections:
            records.append({"frame": frame_index, "detections": detections})
            thumb = cv2.resize(annotated, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"frame {frame_index}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            thumbnails.append(thumb)
    capture.release()

    selected = thumbnails[:40]
    if selected:
        columns = 4
        rows = (len(selected) + columns - 1) // columns
        sheet = np.zeros((rows * 270, columns * 480, 3), dtype=np.uint8)
        for index, image in enumerate(selected):
            row, column = divmod(index, columns)
            sheet[row * 270 : (row + 1) * 270, column * 480 : (column + 1) * 480] = image
        cv2.imwrite(str(args.output_dir / "red_orange_candidates.jpg"), sheet)
    report = {
        "model": str(args.model.resolve()),
        "video": str(args.video.resolve()),
        "stride": args.stride,
        "pair_model": str(args.pair_model.resolve()) if args.pair_model else "",
        "pair_confidence": args.pair_confidence,
        "pair_margin": args.pair_margin,
        "counts": dict(counts),
        "sampled_frames_with_red_orange": len(records),
        "records": records,
    }
    (args.output_dir / "scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
