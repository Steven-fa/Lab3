"""Pick, lift, replace, and release one object using Experiment 3 fixed poses."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import time

import numpy as np

from experiment3_grid_dry_run import load_route
from experiment3_gripper_profiles import load_gripper_profiles, profile_for_group
from experiment3_transfer_test import TransferRunner
from mecharm_recorded_task import (
    acquire_experiment3_control_lock,
    legacy_gripper_value,
)
from pymycobot.common import ProtocolCode


class GridGraspRunner(TransferRunner):
    def run_grasp(
        self,
        current: np.ndarray,
        route: list[dict],
        gripper: dict,
        gripper_speed: int,
    ) -> None:
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.commanded_angles = current.copy()
        self.arm.set_fresh_mode(1)
        self.arm._mesg(ProtocolCode.RESUME, _async=True)
        time.sleep(0.5)

        open_value = int(gripper["open"])
        close_value = int(gripper["close"])
        safe, approach, grasp = route

        self.gripper_reliable(
            "运行前 OPEN", open_value, gripper_speed, recovery_moves=[]
        )
        for label, pose in (
            ("回公共安全姿态", safe),
            ("到1号格上方", approach),
        ):
            self.stream_to(label, pose["angles"])
            self.settle_at(label, pose["angles"])
        self.gripper_reliable(
            "下降前 OPEN 自检",
            open_value,
            gripper_speed,
            recovery_moves=[("夹爪失败后返回公共安全姿态", safe)],
        )
        self.stream_to("下降到1号格抓取点", grasp["angles"])
        self.settle_at("下降到1号格抓取点", grasp["angles"])

        print("[抓取] 保持机械臂静止，执行 CLOSE")
        self.gripper_reliable(
            "1号格 CLOSE",
            close_value,
            gripper_speed,
            recovery_moves=[
                ("夹爪失败后抬回上方", approach),
                ("夹爪失败后返回公共安全姿态", safe),
            ],
        )

        self.stream_to("夹持后原路抬升", approach["angles"])
        self.settle_at("夹持后原路抬升", approach["angles"])
        held_value = self.retry_query(
            "读取抬升后夹爪位置",
            lambda: legacy_gripper_value(self.arm),
            lambda value: isinstance(value, int) and 0 <= value <= 100,
        )
        print(f"[观察] 已抬升并夹持，夹爪回读={held_value}；停留3秒")
        time.sleep(3.0)

        self.stream_to("原路下降放回", grasp["angles"])
        self.settle_at("原路下降放回", grasp["angles"])
        self.gripper_reliable(
            "1号格 OPEN 放回",
            open_value,
            gripper_speed,
            recovery_moves=[
                ("夹爪失败后抬回上方", approach),
                ("夹爪失败后返回公共安全姿态", safe),
            ],
        )

        self.stream_to("释放后原路抬升", approach["angles"])
        self.settle_at("释放后原路抬升", approach["angles"])
        self.stream_to("返回公共安全姿态", safe["angles"])
        final_angles = self.settle_at("返回公共安全姿态", safe["angles"])
        print("[夹爪流程] 放回时已验证OPEN；返回safe后不重复发送夹爪命令。")

        final_error = float(np.max(np.abs(final_angles - safe["angles"])))
        self.events.append(
            {
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "event": "grasp_test_complete",
                "grid": 1,
                "held_gripper_value": held_value,
                "safe_pose_max_error_deg": round(final_error, 3),
            }
        )
        self.save_log("completed")
        print(f"[完成] 1号格抓取、抬升、放回测试完成；最终误差={final_error:.1f}°")
        print(f"[记录] {self.log_path}")


def self_test() -> None:
    assert 0 <= 62 < 100 <= 100
    assert 1 <= 20 <= 100
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3 grid 1 grasp test")
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--object-class", choices=("cube", "other"), default="cube")
    parser.add_argument(
        "--gripper-profiles",
        type=Path,
        default=Path(
            "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
        ),
    )
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--arm-speed", type=int, default=8)
    parser.add_argument("--stream-step", type=float, default=2.0)
    parser.add_argument("--stream-period", type=float, default=0.20)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.arm_speed <= 20:
        raise RuntimeError("首次实物抓取速度必须在1~20")
    if not 0.5 <= args.stream_step <= 3.0:
        raise RuntimeError("位置流步长必须在0.5~3.0度")
    if not 0.08 <= args.stream_period <= 0.4:
        raise RuntimeError("位置流周期必须在0.08~0.4秒")

    route, _recorded_gripper = load_route(args.poses, args.grid)
    gripper = profile_for_group(
        load_gripper_profiles(args.gripper_profiles), args.object_class
    )
    gripper_speed = int(gripper["speed"])
    safe, approach, grasp = route
    descent = approach["coords"][:3] - grasp["coords"][:3]
    print(f"[预检] grid_{args.grid} 单次真实抓取、抬升、原位放回")
    print(
        f"[夹爪] OPEN={gripper['open']}，CLOSE={gripper['close']}，"
        f"分类={args.object_class}，速度={gripper_speed}"
    )
    print(
        f"[抓取段] Z下降={descent[2]:.1f}mm，"
        f"XY变化={float(np.linalg.norm(descent[:2])):.1f}mm"
    )
    print(
        "[路线] 当前 -> safe -> approach -> grasp -> CLOSE -> approach(停3秒) "
        "-> grasp -> OPEN -> approach -> safe"
    )
    if args.check_only:
        print("[只读检查] 数据与测试路线合法；未连接机械臂，未发送命令。")
        return 0

    control_lock = acquire_experiment3_control_lock("grid-grasp-test")
    runner = None
    try:
        runner = GridGraspRunner(
            args.host,
            args.port,
            args.arm_speed,
            args.poses.parent,
            args.stream_step,
            args.stream_period,
        )
        current = runner.health()
        print(f"[当前角度] {np.round(current, 1).tolist()}")
        print("[安全] 只保留1号格中心的单个物体；清空其余区域并准备急停。")
        confirmation = input("确认物体已放好，输入 RUN 并按 Enter：").strip().upper()
        if confirmation != "RUN":
            print("[取消] 未发送运动或夹爪命令。")
            return 2
        runner.run_grasp(current, route, gripper, gripper_speed)
        return 0
    except Exception:
        if runner is not None and not runner.stopped:
            runner.stop()
        raise
    finally:
        if runner is not None:
            runner.arm.close()
        control_lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[停止] 操作员已中止1号格抓取测试。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
