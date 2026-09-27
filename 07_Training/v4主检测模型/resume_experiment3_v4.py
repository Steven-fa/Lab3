#!/usr/bin/env python3
"""Resume an interrupted Ultralytics training run from its last checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=Path, required=True)
    args = parser.parse_args()
    YOLO(str(args.last)).train(resume=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
