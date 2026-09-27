"""Learn a robust trajectory from one or more recorded demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .demonstration import ARM_JOINT_NAMES, build_learned_trajectory, read_demonstration


def _find_demonstrations(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("**/raw_trajectory.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="从人工示教记录学习 mechArm 轨迹")
    parser.add_argument("--input", required=True, help="示教目录或 raw_trajectory.csv")
    parser.add_argument("--output", help="输出 YAML；默认写入输入目录")
    parser.add_argument("--samples", type=int, default=200, help="学习轨迹采样数")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    demonstration_paths = _find_demonstrations(input_path)
    if not demonstration_paths:
        raise SystemExit(f"没有在 {input_path} 找到 raw_trajectory.csv")

    demonstrations = [read_demonstration(path) for path in demonstration_paths]
    learned = build_learned_trajectory(
        demonstrations, ARM_JOINT_NAMES, sample_count=args.samples
    )
    learned["source_demonstrations"] = [str(path.parent) for path in demonstration_paths]

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (input_path if input_path.is_dir() else input_path.parent)
        / "learned_trajectory.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(learned, stream, allow_unicode=True, sort_keys=False)

    print(f"已学习 {len(demonstrations)} 条示教轨迹")
    print(f"关键点数量：{len(learned['keypoints'])}")
    print(f"学习结果：{output_path}")


if __name__ == "__main__":
    main()
