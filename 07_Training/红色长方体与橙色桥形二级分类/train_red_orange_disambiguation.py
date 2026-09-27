#!/usr/bin/env python3
"""Fine-tune v4 conservatively to separate red_cuboid from orange_bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="cube6_v5_red_orange_balanced")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=8,
        batch=16,
        imgsz=640,
        device=args.device,
        workers=2,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.00005,
        lrf=0.08,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.5,
        box=7.5,
        cls=2.5,
        dfl=1.5,
        hsv_h=0.0015,
        hsv_s=0.10,
        hsv_v=0.16,
        degrees=8.0,
        translate=0.06,
        scale=0.18,
        shear=1.0,
        perspective=0.0003,
        fliplr=0.5,
        mosaic=0.12,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=6,
        cache="disk",
        amp=True,
        deterministic=True,
        seed=20260914,
        plots=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
