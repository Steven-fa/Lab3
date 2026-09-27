"""Guided fixed-pose teaching recorder for experiment 3."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np

from mecharm_real_recorder import RealRecorder, raw_keyboard, read_key


def build_stage_plan(grid_count: int) -> list[dict[str, str]]:
    stages = [
        {
            "name": "safe",
            "instruction": "摆到公共安全姿态：夹爪抬高、机械臂收拢且不碰桌面",
        }
    ]
    for grid_id in range(1, grid_count + 1):
        stages.extend(
            [
                {
                    "name": f"grid_{grid_id}_approach",
                    "instruction": f"摆到网格 {grid_id} 正上方约 5-8 cm，夹爪中心对准格子中心",
                },
                {
                    "name": f"grid_{grid_id}_grasp",
                    "instruction": f"从上方下降到网格 {grid_id} 的实际夹取高度，手指位于物体两侧",
                },
            ]
        )
    stages.extend(
        [
            {
                "name": "left_approach",
                "instruction": "摆到网格纸左侧外部释放点正上方约 5-8 cm",
            },
            {
                "name": "left_release",
                "instruction": "下降到左侧释放高度，物体底面应距桌面约 1-2 cm",
            },
            {
                "name": "right_approach",
                "instruction": "摆到网格纸右侧外部释放点正上方约 5-8 cm",
            },
            {
                "name": "right_release",
                "instruction": "下降到右侧释放高度，物体底面应距桌面约 1-2 cm",
            },
        ]
    )
    return stages


class Experiment3PoseRecorder(RealRecorder):
    def __init__(self, host: str, port: int, output_root: Path, grid_count: int) -> None:
        super().__init__(host, port, output_root)
        self.grid_count = grid_count
        self.stages = build_stage_plan(grid_count)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_directory = output_root / f"experiment3_poses_{stamp}"
        self.output_path = self.output_directory / "experiment3_poses.yaml"

    def document(self) -> dict:
        return {
            "format": "mecharm_experiment3_fixed_poses/v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "robot": "mechArm 270 Pi",
            "connection": {"host": self.host, "port": self.port},
            "grid_count": self.grid_count,
            "expected_pose_count": len(self.stages),
            "recorded_pose_count": len(self.frames),
            "complete": len(self.frames) == len(self.stages),
            "classification": {
                "cube": {
                    "source_classes": [
                        "blue_cube",
                        "green_cube",
                        "pink_cube",
                        "yellow_cube",
                    ],
                    "destination": "right",
                },
                "other": {
                    "source_classes": ["red_cuboid", "orange_bridge"],
                    "destination": "left",
                },
            },
            "gripper": {"open": 100, "close": 62, "speed": 30},
            "pose_order": [stage["name"] for stage in self.stages],
            "poses": {frame["name"]: frame for frame in self.frames},
        }

    def show_stage(self) -> None:
        if len(self.frames) >= len(self.stages):
            print("[全部完成] 17个姿态已经记录；按 Esc 锁定并退出。")
            return
        stage = self.stages[len(self.frames)]
        print()
        print(
            "[下一项 {}/{}] {}".format(
                len(self.frames) + 1, len(self.stages), stage["name"]
            )
        )
        print("[摆放要求] {}".format(stage["instruction"]))
        print("[操作] 扶住机械臂摆好后按 q 保存；无需按 Enter。")

    def capture_stage(self) -> None:
        if not self.free:
            raise RuntimeError("六轴尚未进入自由模式；请托住机械臂并先按 f")
        if len(self.frames) >= len(self.stages):
            print("[提示] 所有姿态均已记录，请按 Esc 保存退出。")
            return
        angles = self.angles()
        coords = self.coords()
        if not (
            isinstance(angles, list)
            and len(angles) == 6
            and isinstance(coords, list)
            and len(coords) == 6
            and np.all(np.isfinite(np.asarray(angles, dtype=float)))
            and np.all(np.isfinite(np.asarray(coords, dtype=float)))
        ):
            raise RuntimeError("六轴角度或TCP坐标读取无效")
        stage = self.stages[len(self.frames)]
        frame = {
            "index": len(self.frames) + 1,
            "name": stage["name"],
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
            "joint_angles_deg": [round(float(value), 3) for value in angles],
            "tcp_coords_mm_deg": [round(float(value), 3) for value in coords],
        }
        self.frames.append(frame)
        self.save()
        print(
            "[已记录] #{:02d} {}  angles={}  TCP={}".format(
                frame["index"],
                frame["name"],
                frame["joint_angles_deg"],
                frame["tcp_coords_mm_deg"],
            )
        )
        print("[已保存] {}".format(self.output_path))
        self.show_stage()

    def undo_stage(self) -> None:
        if not self.frames:
            print("[提示] 当前没有可撤销的姿态。")
            return
        removed = self.frames.pop()
        self.save()
        print("[撤销] 已删除 #{} {}".format(removed["index"], removed["name"]))
        self.show_stage()

    def run(self) -> None:
        self.check_connection()
        self.save()
        print("[实验3] 六网格固定姿态示教")
        print("[输出] {}".format(self.output_path))
        print("[说明] 本程序只读取并记录姿态，不会自动移动，也不会操作夹爪。")
        print()
        print("请先用双手托住机械臂，然后按 f 释放六轴。")
        print("  f    进入自由模式")
        print("  l    锁定六轴")
        print("  q    保存当前提示的姿态并进入下一项")
        print("  u    撤销上一项")
        print("  s    显示进度和文件位置")
        print("  Esc  保存、锁定并退出")
        self.show_stage()

        try:
            with raw_keyboard():
                while True:
                    key = read_key()
                    lowered = key.lower()
                    try:
                        if lowered == "f":
                            self.enable_free_mode()
                        elif lowered == "l":
                            self.lock_servos()
                        elif lowered == "q":
                            self.capture_stage()
                        elif lowered == "u":
                            self.undo_stage()
                        elif lowered == "s":
                            self.save()
                            print(
                                "[进度] {}/{}；完整={}".format(
                                    len(self.frames),
                                    len(self.stages),
                                    len(self.frames) == len(self.stages),
                                )
                            )
                            print("[文件] {}".format(self.output_path))
                            self.show_stage()
                        elif key == "\x1b":
                            break
                    except RuntimeError as error:
                        print("[拒绝] {}".format(error))
        except KeyboardInterrupt:
            print("\n[停止] 收到 Ctrl+C。")
        finally:
            self.save()
            self.lock_servos(force=True)
            print(
                "[完成] 已保存 {}/{} 个姿态：{}".format(
                    len(self.frames), len(self.stages), self.output_path
                )
            )


def self_test() -> None:
    names = [stage["name"] for stage in build_stage_plan(6)]
    assert len(names) == 17
    assert len(set(names)) == 17
    assert names[:3] == ["safe", "grid_1_approach", "grid_1_grasp"]
    assert names[-4:] == [
        "left_approach",
        "left_release",
        "right_approach",
        "right_release",
    ]
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 fixed-pose recorder")
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/recordings"),
    )
    parser.add_argument("--grid-count", type=int, default=6, choices=range(4, 7))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    recorder = Experiment3PoseRecorder(
        args.host, args.port, args.output_root, args.grid_count
    )
    recorder.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("[错误] {}".format(error))
        raise SystemExit(1)
