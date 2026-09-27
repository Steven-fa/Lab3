#!/usr/bin/env python3
"""Build a cleaned pair-classifier dataset with deployment-view crops.

The legacy detection labels contain four test boxes tagged orange_bridge even
though their pixels show a solid red cuboid.  This builder corrects those known
bad labels and adds tightly cropped, deployment-view examples from the saved
acceptance video.  The saved video shows the bridge in the upper half and the
solid cuboid in the lower half during frames 0..340.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


PAIR_NAMES = {"red_cuboid", "orange_bridge"}
KNOWN_BAD_ORANGE_LABELS = {
    "combination_03_f000860_01.jpg",
    "combination_03_f000880_00.jpg",
    "combination_03_f000900_00.jpg",
    "combination_03_f000920_00.jpg",
}


def copy_clean_base(source: Path, output: Path, counts: Counter[str]) -> None:
    for split in ("train", "val", "test"):
        for class_name in sorted(PAIR_NAMES):
            (output / split / class_name).mkdir(parents=True, exist_ok=True)
        for class_name in sorted(PAIR_NAMES):
            for path in sorted((source / split / class_name).glob("*.jpg")):
                corrected_name = class_name
                if class_name == "orange_bridge" and path.name in KNOWN_BAD_ORANGE_LABELS:
                    corrected_name = "red_cuboid"
                shutil.copy2(path, output / split / corrected_name / path.name)
                counts[f"base/{split}/{corrected_name}"] += 1
                if corrected_name != class_name:
                    counts["corrected_bad_labels"] += 1


def split_for_frame(frame_index: int) -> str:
    if frame_index <= 220:
        return "train"
    if frame_index <= 280:
        return "val"
    return "test"


def add_deployment_crops(
    video_path: Path,
    detector_path: Path,
    output: Path,
    counts: Counter[str],
    device: str,
) -> None:
    detector = YOLO(str(detector_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index > 340:
            break
        if frame_index % 10:
            continue
        result = detector.predict(frame, conf=0.20, imgsz=640, device=device, verbose=False)[0]
        height, width = frame.shape[:2]
        per_class_index: Counter[str] = Counter()
        for box in result.boxes or []:
            detected_name = str(result.names[int(box.cls[0])])
            if detected_name not in PAIR_NAMES:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in box.xyxy[0]]
            center_y_ratio = ((y1 + y2) / 2.0) / height
            if center_y_ratio >= 0.52:
                true_name = "red_cuboid"
            elif center_y_ratio <= 0.48:
                true_name = "orange_bridge"
            else:
                counts["deployment/ambiguous_skipped"] += 1
                continue
            pad_x = max(1, round((x2 - x1) * 0.03))
            pad_y = max(1, round((y2 - y1) * 0.03))
            crop = frame[
                max(0, y1 - pad_y) : min(height, y2 + pad_y),
                max(0, x1 - pad_x) : min(width, x2 + pad_x),
            ]
            if crop.size == 0:
                continue
            split = split_for_frame(frame_index)
            index = per_class_index[true_name]
            per_class_index[true_name] += 1
            name = f"deployment_f{frame_index:06d}_{index:02d}.jpg"
            cv2.imwrite(
                str(output / split / true_name / name),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            )
            counts[f"deployment/{split}/{true_name}"] += 1
    capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    counts: Counter[str] = Counter()
    copy_clean_base(args.base.resolve(), output, counts)
    add_deployment_crops(
        args.video.resolve(),
        args.detector.resolve(),
        output,
        counts,
        args.device,
    )
    manifest = {
        "base": str(args.base.resolve()),
        "video": str(args.video.resolve()),
        "detector": str(args.detector.resolve()),
        "known_bad_labels": sorted(KNOWN_BAD_ORANGE_LABELS),
        "counts": dict(sorted(counts.items())),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
