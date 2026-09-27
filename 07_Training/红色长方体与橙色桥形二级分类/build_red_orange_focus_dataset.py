#!/usr/bin/env python3
"""Create a local YOLO dataset view with balanced red/orange hard-example repeats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


NAMES = [
    "blue_cube",
    "green_cube",
    "pink_cube",
    "yellow_cube",
    "red_cuboid",
    "orange_bridge",
]
TARGET_CLASSES = {4, 5}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def label_classes(label: Path) -> list[int]:
    classes: list[int] = []
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Malformed YOLO row in {label}: {line!r}")
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
        if class_id not in range(len(NAMES)) or not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"Invalid label in {label}: {line!r}")
        classes.append(class_id)
    return classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--red-repeat", type=int, default=5)
    parser.add_argument("--orange-repeat", type=int, default=5)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, Counter[int]] = {}
    train_entries: list[str] = []
    split_images: dict[str, int] = {}

    for split in ("train", "val", "test"):
        images = sorted(
            path for path in (source / "images" / split).iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        counts: Counter[int] = Counter()
        for image in images:
            label = source / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            classes = label_classes(label)
            counts.update(classes)
            if split == "train":
                repeat = 1
                if 4 in classes:
                    repeat = max(repeat, args.red_repeat)
                if 5 in classes:
                    repeat = max(repeat, args.orange_repeat)
                train_entries.extend([image.as_posix()] * repeat)
        split_images[split] = len(images)
        class_counts[split] = counts

    train_file = output / "train_red_orange_focus.txt"
    train_file.write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    yaml_file = output / "dataset_v5.yaml"
    yaml_file.write_text(
        "\n".join(
            [
                f"path: {source.as_posix()}",
                f"train: {train_file.as_posix()}",
                "val: images/val",
                "test: images/test",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": str(source),
        "split_images": split_images,
        "class_instances": {
            split: {NAMES[index]: count for index, count in sorted(counts.items())}
            for split, counts in class_counts.items()
        },
        "red_repeat": args.red_repeat,
        "orange_repeat": args.orange_repeat,
        "effective_train_entries": len(train_entries),
        "train_file": str(train_file),
        "dataset_yaml": str(yaml_file),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
