#!/usr/bin/env python3
"""Evaluate the pair classifier and save a simple confusion report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    counts: Counter[str] = Counter()
    details: list[dict] = []
    for true_name in ("red_cuboid", "orange_bridge"):
        for path in sorted((args.data / "test" / true_name).glob("*.jpg")):
            result = model.predict(str(path), imgsz=224, device=args.device, verbose=False)[0]
            predicted = str(model.names[int(result.probs.top1)])
            confidence = float(result.probs.top1conf)
            counts[f"{true_name}->{predicted}"] += 1
            details.append({"image": path.name, "true": true_name, "predicted": predicted, "confidence": round(confidence, 6)})
    correct = sum(counts[f"{name}->{name}"] for name in ("red_cuboid", "orange_bridge"))
    report = {"model": str(args.model.resolve()), "counts": dict(sorted(counts.items())), "samples": len(details), "accuracy": correct / max(len(details), 1), "details": details}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
