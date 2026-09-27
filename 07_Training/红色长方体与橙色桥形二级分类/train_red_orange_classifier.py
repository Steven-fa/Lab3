#!/usr/bin/env python3
"""Train the red_cuboid/orange_bridge second-stage classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--name", default="red_orange_classifier_v1")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    YOLO(args.model).train(
        data=str(args.data),
        epochs=args.epochs,
        patience=10,
        batch=32,
        imgsz=224,
        device=args.device,
        workers=2,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        hsv_h=0.001,
        hsv_s=0.08,
        hsv_v=0.14,
        degrees=12.0,
        translate=0.08,
        scale=0.18,
        fliplr=0.5,
        erasing=0.15,
        auto_augment=None,
        deterministic=True,
        seed=20260914,
        plots=False,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
