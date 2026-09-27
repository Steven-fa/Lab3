"""Interactive command-angle correction for Experiment 3's 17 taught poses."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import time

import numpy as np
import yaml

from experiment3_grid_dry_run import EXPECTED_FORMAT, GridDryRunner
from mecharm_recorded_task import SOFT_LOWER, SOFT_UPPER, legacy_set_gripper_value
from pymycobot.common import ProtocolCode


LEGACY_LOWER = SOFT_LOWER.copy()
LEGACY_UPPER = SOFT_UPPER.copy()
LEGACY_LOWER[1] = -92.0
LEGACY_UPPER[1] = 92.0
MAX_SINGLE_JOINT_DELTA_DEG = 5.0
MAX_ALL_JOINT_DELTA_DEG = 15.0
MIN_APPROACH_CLEARANCE_MM = 10.0


def load_pose_document(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != EXPECTED_FORMAT:
        raise RuntimeError("源文件不是实验3固定姿态格式")
    order = document.get("pose_order")
    poses = document.get("poses")
    if not isinstance(order, list) or len(order) != 17 or len(set(order)) != 17:
        raise RuntimeError("源文件必须包含17个不重复姿态名称")
    if not isinstance(poses, dict) or any(name not in poses for name in order):
        raise RuntimeError("源文件缺少17点姿态数据")
    for name in order:
        angles = np.asarray(poses[name].get("joint_angles_deg"), dtype=float)
        coords = np.asarray(poses[name].get("tcp_coords_mm_deg"), dtype=float)
        if angles.shape != (6,) or coords.shape != (6,):
            raise RuntimeError(f"{name}缺少六轴角度或TCP")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(coords)):
            raise RuntimeError(f"{name}包含无效数值")
        if np.any(angles < LEGACY_LOWER) or np.any(angles > LEGACY_UPPER):
            raise RuntimeError(f"{name}超出旧款mechArm软件限位")
    return document


def find_incomplete_correction(output_root: Path) -> Path | None:
    candidates: list[Path] = []
    for path in output_root.glob("experiment3_pose_correction_*/experiment3_poses.yaml"):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            correction = document.get("correction", {}) if isinstance(document, dict) else {}
            count = int(correction.get("corrected_pose_count", 0))
            active = correction.get("active") is True
            if 0 < count < 17 and not active:
                candidates.append(path)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item)))


def resolve_pose_selector(selector: str, order: list[str]) -> str:
    token = selector.strip()
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(order):
            return order[index - 1]
    if token in order:
        return token
    raise ValueError("点位必须是1~17的编号或完整姿态名称")


def parse_pose_scope(value: str | None, order: list[str]) -> list[str]:
    if value is None:
        return list(order)
    result: list[str] = []
    for token in value.split(","):
        name = resolve_pose_selector(token.strip(), order)
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("--only至少需要一个点位")
    return result


def context_route_names(name: str, poses: dict) -> list[str]:
    """Return the production-like incoming route used to verify one pose."""
    if name == "safe":
        return ["safe"]
    grid_match = re.fullmatch(r"grid_(\d+)_(approach|grasp)", name)
    if grid_match:
        grid = grid_match.group(1)
        if grid_match.group(2) == "approach":
            route = ["safe", name]
        else:
            route = ["safe", f"grid_{grid}_approach", name]
    elif name in {"left_approach", "right_approach"}:
        route = ["safe", name]
    elif name in {"left_release", "right_release"}:
        side = name.split("_", 1)[0]
        route = ["safe", f"{side}_approach", name]
    else:
        raise ValueError(f"无法为{name}生成进入路线")
    if any(item not in poses for item in route):
        raise ValueError(f"{name}的进入路线缺少依赖点")
    return route


def vertical_route_errors(
    poses: dict, minimum_clearance_mm: float = MIN_APPROACH_CLEARANCE_MM
) -> list[str]:
    """Return approach/lower-pose pairs whose recorded TCP Z order is unsafe."""
    pairs = [
        (f"grid_{index}_approach", f"grid_{index}_grasp")
        for index in range(1, 7)
    ]
    pairs.extend(
        [
            ("left_approach", "left_release"),
            ("right_approach", "right_release"),
        ]
    )
    errors: list[str] = []
    for approach_name, lower_name in pairs:
        if approach_name not in poses or lower_name not in poses:
            errors.append(f"缺少 {approach_name} 或 {lower_name}")
            continue
        try:
            approach_z = float(poses[approach_name]["tcp_coords_mm_deg"][2])
            lower_z = float(poses[lower_name]["tcp_coords_mm_deg"][2])
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{approach_name}/{lower_name} 缺少有效TCP Z")
            continue
        clearance = approach_z - lower_z
        if not np.isfinite(clearance) or clearance < minimum_clearance_mm:
            errors.append(
                f"{approach_name} 相对 {lower_name} 的高度差仅 "
                f"{clearance:+.1f}mm（要求至少 +{minimum_clearance_mm:.1f}mm）"
            )
    return errors


def validate_target(target: np.ndarray, current: np.ndarray, max_delta: float) -> None:
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise ValueError("目标必须包含六个有效角度")
    if np.any(target < LEGACY_LOWER) or np.any(target > LEGACY_UPPER):
        raise ValueError("目标超出旧款mechArm软件关节限位")
    largest = float(np.max(np.abs(target - current)))
    if largest > max_delta:
        raise ValueError(f"本次最大角度变化{largest:.2f}°，超过{max_delta:.1f}°限制")


class PoseCorrector(GridDryRunner):
    def __init__(
        self,
        host: str,
        port: int,
        arm_speed: int,
        source_path: Path,
        output_root: Path,
        stream_step: float,
        stream_period: float,
        resume_path: Path | None = None,
        only_names: list[str] | None = None,
    ) -> None:
        if resume_path is not None:
            self.output_path = resume_path.resolve()
            self.output_directory = self.output_path.parent
            self.document = load_pose_document(self.output_path)
            correction = self.document.get("correction")
            if not isinstance(correction, dict):
                raise RuntimeError("恢复文件缺少校正会话信息")
            recorded_source = correction.get("source_pose_file")
            self.source_path = Path(recorded_source).resolve() if recorded_source else source_path.resolve()
            corrected_names = correction.get("corrected_poses", [])
            if not isinstance(corrected_names, list):
                raise RuntimeError("恢复文件的已校正点列表无效")
        else:
            source = load_pose_document(source_path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_directory = output_root / f"experiment3_pose_correction_{stamp}"
            self.output_path = self.output_directory / "experiment3_poses.yaml"
            self.source_path = source_path.resolve()
            self.document = deepcopy(source)
            corrected_names = []
        self.order: list[str] = list(self.document["pose_order"])
        self.poses: dict = self.document["poses"]
        self.corrected: set[str] = {
            name for name in corrected_names if name in self.order
        }
        self.required_names = list(only_names) if only_names is not None else list(self.order)
        if not self.required_names or any(name not in self.order for name in self.required_names):
            raise RuntimeError("定向补采点列表无效")
        correction = self.document.get("correction", {})
        previous_scope = correction.get("targeted_scope", {}) if isinstance(correction, dict) else {}
        same_scope = previous_scope.get("pose_names") == self.required_names
        previous_saved = previous_scope.get("saved_pose_names", []) if same_scope else []
        self.scope_saved: set[str] = {
            name for name in previous_saved if name in self.required_names
        }
        if only_names is not None:
            self.selected = next(
                (name for name in self.required_names if name not in self.scope_saved),
                self.required_names[0],
            )
        else:
            self.selected = next(
                (name for name in self.order if name not in self.corrected), self.order[0]
            )
        self.active_pose: str | None = None
        self.adjustment_history: list[np.ndarray] = []
        super().__init__(
            host,
            port,
            arm_speed,
            self.output_directory,
            stream_step,
            stream_period,
        )
        if resume_path is None:
            self._prepare_document()
        self.save_document()

    def _prepare_document(self) -> None:
        self.document["correction"] = {
            "format": "mecharm_experiment3_pose_correction/v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_pose_file": str(self.source_path),
            "corrected_pose_count": 0,
            "corrected_poses": [],
            "active": False,
        }

    def save_document(self) -> None:
        correction = self.document["correction"]
        correction["updated_at"] = datetime.now().isoformat(timespec="seconds")
        correction["corrected_pose_count"] = len(self.corrected)
        correction["corrected_poses"] = [name for name in self.order if name in self.corrected]
        if self.required_names != self.order:
            correction["targeted_scope"] = {
                "pose_names": list(self.required_names),
                "saved_pose_names": [
                    name for name in self.required_names if name in self.scope_saved
                ],
            }
        self.document["updated_at"] = correction["updated_at"]
        self.output_directory.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.document, stream, allow_unicode=True, sort_keys=False)
        temporary.replace(self.output_path)

    def print_list(self) -> None:
        print("\n[17点状态]")
        for index, name in enumerate(self.order, start=1):
            if self.required_names != self.order:
                if name in self.scope_saved:
                    mark = "本轮已补采"
                elif name in self.required_names:
                    mark = "本轮待补采"
                elif name in self.corrected:
                    mark = "已有校正"
                else:
                    mark = "沿用原值"
            else:
                mark = "已校正" if name in self.corrected else "待校正"
            cursor = " <- 当前选择" if name == self.selected else ""
            print(f"  {index:02d}. {name:<20} {mark}{cursor}")

    def report(self, prefix: str = "状态") -> tuple[np.ndarray, np.ndarray]:
        actual_angles = self.read_angles()
        actual_coords = self.read_coords()
        command = (
            self.commanded_angles
            if self.commanded_angles is not None
            else actual_angles
        )
        print(f"[{prefix}] 命令角={np.round(command, 2).tolist()}")
        print(f"[{prefix}] 回读角={np.round(actual_angles, 2).tolist()}")
        print(f"[{prefix}] TCP={np.round(actual_coords, 1).tolist()}")
        return actual_angles, actual_coords

    def open_gripper(self) -> None:
        gripper = self.document.get("gripper", {})
        open_value = gripper.get("open")
        speed = min(20, int(gripper.get("speed", 20)))
        if not isinstance(open_value, int) or not 0 <= open_value <= 100:
            raise RuntimeError("数据文件中的夹爪OPEN参数无效")
        before = self.retry_query(
            "读取夹爪位置",
            self.arm.get_gripper_value,
            lambda value: isinstance(value, int) and 0 <= value <= 100,
        )
        print(f"[夹爪] 正在张开：target={open_value}，before={before}，speed={speed}")
        legacy_set_gripper_value(self.arm, open_value, speed)
        samples: list[int] = []
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            time.sleep(0.35)
            value = self.arm.get_gripper_value()
            if not isinstance(value, int) or not 0 <= value <= 100:
                continue
            samples.append(value)
            reached = abs(value - open_value) <= 3
            calibrated_open_stop = open_value >= 95 and value >= 84
            if reached or calibrated_open_stop:
                print(f"[夹爪] 已张开，回读={value}")
                return
        raise RuntimeError(f"夹爪张开超时：target={open_value}，samples={samples}")

    def move_selected_via_context(self) -> None:
        route = context_route_names(self.selected, self.poses)
        print(f"[复现路线] {' -> '.join(route)}")
        for name in route:
            target = np.asarray(self.poses[name]["joint_angles_deg"], dtype=float)
            self.stream_to(f"前往 {name}", target)
            time.sleep(0.8)
        self.active_pose = self.selected
        self.adjustment_history.clear()
        self.report("到达")
        print("[调整] 可输入 j 5 +1、set 5 0、delta 六个增量或 all 六个绝对角度")

    def apply_target(self, target: np.ndarray, max_delta: float, label: str) -> None:
        if self.active_pose != self.selected or self.commanded_angles is None:
            raise RuntimeError("请先用 go 移动到当前选择的点位")
        current_command = self.commanded_angles.copy()
        validate_target(target, current_command, max_delta)
        self.adjustment_history.append(current_command)
        self.stream_to(label, target)
        time.sleep(0.8)
        self.report("调整后")

    def adjust_joint(self, joint: int, delta: float) -> None:
        if not 1 <= joint <= 6:
            raise ValueError("关节编号必须为1~6")
        if abs(delta) > MAX_SINGLE_JOINT_DELTA_DEG:
            raise ValueError("单次关节增量必须在±5°以内")
        if self.commanded_angles is None:
            raise RuntimeError("请先用 go 移动到点位")
        target = self.commanded_angles.copy()
        target[joint - 1] += delta
        self.apply_target(target, MAX_SINGLE_JOINT_DELTA_DEG, f"J{joint}调整{delta:+.2f}°")

    def set_joint(self, joint: int, angle: float) -> None:
        if not 1 <= joint <= 6:
            raise ValueError("关节编号必须为1~6")
        if self.commanded_angles is None:
            raise RuntimeError("请先用 go 移动到点位")
        target = self.commanded_angles.copy()
        target[joint - 1] = angle
        self.apply_target(target, MAX_SINGLE_JOINT_DELTA_DEG, f"J{joint}设为{angle:+.2f}°")

    def adjust_all(self, values: list[float], absolute: bool) -> None:
        if self.commanded_angles is None:
            raise RuntimeError("请先用 go 移动到点位")
        if len(values) != 6:
            raise ValueError("必须输入六个角度")
        value_array = np.asarray(values, dtype=float)
        target = value_array if absolute else self.commanded_angles + value_array
        limit = MAX_ALL_JOINT_DELTA_DEG if absolute else MAX_SINGLE_JOINT_DELTA_DEG
        self.apply_target(
            target,
            limit,
            "整组六轴绝对调整" if absolute else "整组六轴增量调整",
        )

    def undo_adjustment(self) -> None:
        if not self.adjustment_history:
            raise RuntimeError("当前点没有可撤销的调整")
        target = self.adjustment_history.pop()
        self.stream_to("撤销上一步调整", target)
        time.sleep(0.8)
        self.report("撤销后")

    def save_selected(self) -> None:
        if self.active_pose != self.selected or self.commanded_angles is None:
            raise RuntimeError("只能保存已用go到达并调整的当前点")
        time.sleep(0.8)
        feedback_angles, coords = self.report("保存采样")
        frame = self.poses[self.selected]
        frame.setdefault("original_joint_angles_deg", deepcopy(frame["joint_angles_deg"]))
        frame.setdefault("original_tcp_coords_mm_deg", deepcopy(frame["tcp_coords_mm_deg"]))
        command = self.commanded_angles.copy()
        frame["joint_angles_deg"] = [round(float(value), 3) for value in command]
        frame["tcp_coords_mm_deg"] = [round(float(value), 3) for value in coords]
        frame["manual_command_calibrated"] = True
        frame["correction_feedback_angles_deg"] = [
            round(float(value), 3) for value in feedback_angles
        ]
        frame["corrected_at"] = datetime.now().isoformat(timespec="milliseconds")
        original = np.asarray(frame["original_joint_angles_deg"], dtype=float)
        frame["correction_command_delta_deg"] = [
            round(float(value), 3) for value in command - original
        ]
        self.corrected.add(self.selected)
        self.scope_saved.add(self.selected)
        self.save_document()
        print(f"[已保存] {self.selected} 的新命令角和真实TCP")
        print(f"[文件] {self.output_path}")
        if self.required_names != self.order:
            print(
                f"[本轮补采] {len(self.scope_saved)}/{len(self.required_names)}；"
                "建议立即输入 verify 复验进入方向"
            )
        else:
            print(f"[进度] {len(self.corrected)}/17；建议立即输入 verify 复验进入方向")

    def activate(self) -> None:
        if self.required_names != self.order:
            missing = [name for name in self.required_names if name not in self.scope_saved]
            if missing:
                raise RuntimeError(f"必须完成本轮补采才能激活；尚缺：{', '.join(missing)}")
        elif len(self.corrected) != len(self.order):
            missing = [name for name in self.order if name not in self.corrected]
            raise RuntimeError(f"必须完成17点才能激活；尚缺：{', '.join(missing)}")
        geometry_errors = vertical_route_errors(self.poses)
        if geometry_errors:
            details = "\n  - ".join(geometry_errors)
            raise RuntimeError(
                "路线高度关系不合法，禁止激活。请重新校正对应的approach点："
                f"\n  - {details}"
            )
        current = self.output_directory.parent.parent / "current"
        temporary = current.with_name("current.next")
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        os.symlink(self.output_directory, temporary)
        temporary.replace(current)
        self.document["correction"]["active"] = True
        self.document["correction"]["activation_mode"] = (
            "targeted_patch" if self.required_names != self.order else "full_correction"
        )
        self.document["correction"]["activated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        self.save_document()
        print(f"[已激活] current -> {self.output_directory}")

    def help(self) -> None:
        print(
            """
[命令]
  list                         显示17点校正进度
  select 1                     选择第1点（也可写完整名称）
  go                           按正式进入方向移动到当前点
  j 5 +1                       J5相对增加1°（单次最多±5°）
  set 5 0                      把J5设到绝对0°（单次变化最多5°）
  delta d1 d2 d3 d4 d5 d6     同时输入六轴增量（每轴最多±5°）
  all a1 a2 a3 a4 a5 a6       输入六轴绝对目标（每轴单次最多变化15°）
  show                         显示命令角、回读角和TCP
  open                         立即把夹爪张开到标定OPEN值
  undo                         撤销上一次角度调整
  save                         保存当前命令角和真实TCP
  verify                       离开后按正式进入方向重新到达当前点
  next / prev                  选择下一点/上一点（不自动运动）
  activate                     17点全部保存后切换为正式数据
  quit                         保存校正文件并退出
  Ctrl+C                       立即发送stop()
""".strip()
        )

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        current = self.health()
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        print("[模式] 实验3 17点逐点六轴校正；启动后不会自动运动")
        print(f"[源文件] {self.source_path}")
        print(f"[新文件] {self.output_path}")
        print("[重要] 保存的是校正后的命令角；回读角与真实TCP另行保留")
        print("[安全] 每次go/verify前必须清空对应路线并准备急停")
        if self.corrected:
            print(
                f"[恢复会话] 已保留{len(self.corrected)}/17个校正点；"
                f"从第{self.order.index(self.selected)+1}点继续"
            )
        if self.required_names != self.order:
            print(f"[定向补采] {' -> '.join(self.required_names)}")
            print("[说明] 其他点保留当前校正文件中的已有值或原始值")
        self.open_gripper()
        self.help()
        self.print_list()

        while True:
            try:
                position = self.required_names.index(self.selected) + 1
                total = len(self.required_names)
                line = input(f"\n校正[{position}/{total} {self.selected}]> ").strip()
                if not line:
                    continue
                parts = line.split()
                command = parts[0].lower()
                if command in {"help", "h", "?"}:
                    self.help()
                elif command in {"list", "ls"}:
                    self.print_list()
                elif command in {"select", "sel"}:
                    if len(parts) != 2:
                        raise ValueError("用法：select 1 或 select grid_1_grasp")
                    selected = resolve_pose_selector(parts[1], self.order)
                    if selected not in self.required_names:
                        raise ValueError("该点不在本轮定向补采列表中")
                    self.selected = selected
                    self.active_pose = None
                    print(f"[已选择] {self.selected}；输入go才会运动")
                elif command in {"go", "verify"}:
                    print("[确认] 清空复现路线后输入 MOVE；其他输入取消。")
                    if input("MOVE> ").strip().upper() != "MOVE":
                        print("[取消] 未运动")
                        continue
                    self.move_selected_via_context()
                elif command in {"j", "joint"}:
                    if len(parts) != 3:
                        raise ValueError("用法：j 5 +1")
                    self.adjust_joint(int(parts[1]), float(parts[2]))
                elif re.fullmatch(r"j[1-6]", command):
                    if len(parts) != 2:
                        raise ValueError("用法：j5 +1")
                    self.adjust_joint(int(command[1]), float(parts[1]))
                elif command == "set":
                    if len(parts) != 3:
                        raise ValueError("用法：set 5 0")
                    self.set_joint(int(parts[1]), float(parts[2]))
                elif command in {"delta", "d"}:
                    self.adjust_all([float(value) for value in parts[1:]], False)
                elif command in {"all", "angles"}:
                    self.adjust_all([float(value) for value in parts[1:]], True)
                elif command in {"show", "status", "s"}:
                    self.report()
                elif command in {"open", "gripper-open"}:
                    self.open_gripper()
                elif command in {"undo", "u"}:
                    self.undo_adjustment()
                elif command in {"save", "write"}:
                    self.save_selected()
                elif command in {"next", "n", "prev", "p"}:
                    step = 1 if command in {"next", "n"} else -1
                    index = (
                        self.required_names.index(self.selected) + step
                    ) % len(self.required_names)
                    self.selected = self.required_names[index]
                    self.active_pose = None
                    print(f"[已选择] {self.selected}；输入go才会运动")
                elif command == "activate":
                    print("[确认] 这会让正式程序使用本次校正结果。输入 ACTIVATE 确认。")
                    if input("ACTIVATE> ").strip().upper() == "ACTIVATE":
                        self.activate()
                    else:
                        print("[取消] 未切换正式数据")
                elif command in {"quit", "exit", "q"}:
                    self.save_document()
                    print(f"[退出] 校正结果保存在 {self.output_path}")
                    return
                else:
                    raise ValueError("未知命令；输入help查看用法")
            except (ValueError, RuntimeError) as error:
                print(f"[拒绝] {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 pose command corrector")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/corrections"),
    )
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=6)
    parser.add_argument("--stream-step", type=float, default=1.0)
    parser.add_argument("--stream-period", type=float, default=0.25)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument(
        "--only",
        help="只补采逗号分隔的点位；完成这些点后可激活，其他点原样保留",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    document = load_pose_document(args.source)
    only_names = parse_pose_scope(args.only, list(document["pose_order"]))
    resume_path = (
        find_incomplete_correction(args.output_root) if args.resume_latest else None
    )
    if args.check_only:
        print(f"[只读检查] 17点数据合法：{args.source.resolve()}")
        for index, name in enumerate(document["pose_order"], start=1):
            print(f"  {index:02d}. {name}: {' -> '.join(context_route_names(name, document['poses']))}")
        print("[只读检查] 未连接机械臂、未创建校正文件、未发送运动命令。")
        if resume_path is not None:
            resume_document = yaml.safe_load(resume_path.read_text(encoding="utf-8"))
            correction = resume_document["correction"]
            print(
                f"[续作检查] 将恢复{resume_path}；"
                f"已保存{correction['corrected_pose_count']}/17点"
            )
        if args.only is not None:
            print(f"[定向补采检查] {' -> '.join(only_names)}")
        return 0
    if not 1 <= args.arm_speed <= 8:
        raise RuntimeError("校正速度必须在1~8")
    corrector = PoseCorrector(
        args.host,
        args.port,
        args.arm_speed,
        args.source,
        args.output_root,
        args.stream_step,
        args.stream_period,
        resume_path,
        only_names if args.only is not None else None,
    )
    try:
        corrector.run()
        return 0
    except KeyboardInterrupt:
        print("[停止] 已执行急停并退出校正器")
        return 130
    finally:
        corrector.arm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
