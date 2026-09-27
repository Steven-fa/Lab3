#!/usr/bin/env python3
"""Continue Experiment 3 training from v3 using new video examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="cube6_red_focus_v4_video_negative")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=10,
        batch=16,
        imgsz=640,
        device=args.device,
        workers=2,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.000075,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        box=7.5,
        cls=1.8,
        dfl=1.5,
        hsv_h=0.003,
        hsv_s=0.15,
        hsv_v=0.22,
        degrees=8.0,
        translate=0.07,
        scale=0.22,
        shear=1.5,
        perspective=0.0004,
        fliplr=0.5,
        mosaic=0.20,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=7,
        cache="disk",
        amp=True,
        deterministic=True,
        seed=20260913,
        plots=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
