"""Interactive gripper position tester for Experiment 3 class profiles.

Run this program on the Jetson.  It talks to the mechArm socket bridge and
stores separate close targets for cube and other objects without changing any
of the 17 recorded arm poses.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import yaml
from pymycobot import MechArmSocket

from experiment3_gripper_profiles import MAX_SAFE_GRIPPER_SPEED
from mecharm_recorded_task import acquire_experiment3_control_lock
from pymycobot.common import ProtocolCode


POLL_SECONDS = 0.25
MAX_SAMPLES = 24
STABLE_SAMPLES = 4
STABLE_SPREAD = 1


def valid_gripper_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100


def parse_command(line: str) -> tuple[str, Any]:
    parts = line.strip().lower().split()
    if not parts:
        return "empty", None
    if len(parts) == 1:
        try:
            return "move", int(parts[0])
        except ValueError:
            return parts[0], None
    if parts[0] in {"move", "speed", "save", "test"} and len(parts) == 2:
        if parts[0] in {"move", "speed"}:
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return "invalid", None
        return parts[0], parts[1]
    return "invalid", None


def is_stable(samples: list[int]) -> bool:
    if len(samples) < STABLE_SAMPLES:
        return False
    recent = samples[-STABLE_SAMPLES:]
    return max(recent) - min(recent) <= STABLE_SPREAD


def legacy_gripper_value(arm: MechArmSocket) -> int:
    return arm._mesg(ProtocolCode.GET_GRIPPER_VALUE, has_reply=True)


def legacy_set_gripper_value(
    arm: MechArmSocket, value: int, speed: int
) -> Any:
    return arm._mesg(
        ProtocolCode.SET_GRIPPER_VALUE,
        value,
        speed,
        has_reply=False,
        _async=True,
    )


def legacy_release_gripper(arm: MechArmSocket, speed: int) -> Any:
    return arm._mesg(
        ProtocolCode.SET_GRIPPER_STATE,
        254,
        speed,
        has_reply=False,
        _async=True,
    )


class GripperProfileTester:
    def __init__(
        self,
        host: str,
        port: int,
        output: Path,
        speed: int,
    ) -> None:
        self.host = host
        self.port = port
        self.output = output
        self.speed = speed
        self.arm = MechArmSocket(host, port)
        self.last_target: int | None = None
        self.last_actual: int | None = None
        self.profiles: dict[str, int | None] = {
            "open": 100,
            "cube_close": 69,
            "other_close": None,
        }
        self.samples: list[dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.output.is_file():
            return
        document = yaml.safe_load(self.output.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return
        for name in self.profiles:
            value = document.get(name)
            if value is None or valid_gripper_value(value):
                self.profiles[name] = value
        saved_speed = document.get("speed")
        if (
            isinstance(saved_speed, int)
            and not isinstance(saved_speed, bool)
            and 1 <= saved_speed <= MAX_SAFE_GRIPPER_SPEED
        ):
            self.speed = saved_speed
        saved_samples = document.get("samples")
        if isinstance(saved_samples, list):
            self.samples = saved_samples

    def document(self) -> dict[str, Any]:
        return {
            "format": "mecharm_experiment3_gripper_profiles/v1",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "robot": "mechArm 270 Pi",
            "connection": {"host": self.host, "port": self.port},
            "value_range": [0, 100],
            "direction": "0=closed, 100=open",
            "speed": self.speed,
            **self.profiles,
            "complete": self._profiles_are_safe(),
            "samples": self.samples,
        }

    def _profiles_are_safe(self) -> bool:
        values = [
            self.profiles[name]
            for name in ("other_close", "cube_close", "open")
        ]
        return (
            all(valid_gripper_value(value) for value in values)
            and int(values[0]) < int(values[1]) < int(values[2])
            and 1 <= self.speed <= MAX_SAFE_GRIPPER_SPEED
        )

    def save_file(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.document(), stream, allow_unicode=True, sort_keys=False
            )
        temporary.replace(self.output)

    def read_value(self, attempts: int = 6) -> int:
        last: Any = None
        for attempt in range(1, attempts + 1):
            last = legacy_gripper_value(self.arm)
            if valid_gripper_value(last):
                return int(last)
            if attempt < attempts:
                time.sleep(0.35)
        raise RuntimeError(f"连续{attempts}次无法读取夹爪位置，最后回读={last}")

    def move(self, target: int) -> None:
        if not valid_gripper_value(target):
            print("[拒绝] 位置值必须是0～100的整数。")
            return
        before = self.read_value()
        print(
            f"[发送一次] target={target}, before={before}, speed={self.speed}；"
            "不会自动重发"
        )
        result = legacy_set_gripper_value(self.arm, target, self.speed)
        samples: list[int] = []
        stopped_by = "timeout"
        for index in range(1, MAX_SAMPLES + 1):
            time.sleep(POLL_SECONDS)
            try:
                current = self.read_value(attempts=3)
            except RuntimeError as error:
                print(f"[回读 {index}/{MAX_SAMPLES}] {error}")
                continue
            samples.append(current)
            print(
                f"[实时回读 {index:02d}/{MAX_SAMPLES}] {current} "
                f"(相对起点={current - before:+d}, 距目标={current - target:+d})"
            )
            if is_stable(samples):
                stopped_by = "stable"
                break
        actual = samples[-1] if samples else before
        self.last_target = target
        self.last_actual = actual
        self.samples.append(
            {
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "target": target,
                "before": before,
                "actual": actual,
                "speed": self.speed,
                "samples": samples,
                "ended_by": stopped_by,
                "command_result": result,
            }
        )
        self.save_file()
        if actual == before:
            print("[警告] 回读没有变化；未自动重发。可输入 clear 后再测试。")
        elif stopped_by == "stable":
            print(
                f"[稳定] 命令值={target}，最终回读={actual}。"
                "请以实物夹持松紧为准。"
            )
        else:
            print(
                f"[超时] 最终回读={actual}，夹爪可能仍在运动；"
                "请等待并输入 show，暂时不要继续发送。"
            )

    def clear_pending_command(self) -> None:
        before = self.read_value()
        print(
            "[清除] 发送RELEASE(254)终止可能残留的夹爪命令；"
            "测试物必须由桌面支撑。"
        )
        legacy_release_gripper(self.arm, self.speed)
        samples: list[int] = []
        for index in range(1, 13):
            time.sleep(POLL_SECONDS)
            current = self.read_value(attempts=3)
            samples.append(current)
            print(f"[清除回读 {index:02d}/12] {current}")
            if is_stable(samples):
                print(f"[清除完成] 回读稳定在{current}（清除前{before}）。")
                return
        print("[警告] RELEASE后回读仍未稳定，禁止继续连发命令。")

    def save_profile(self, name: str) -> None:
        mapping = {
            "open": "open",
            "cube": "cube_close",
            "other": "other_close",
        }
        key = mapping.get(name)
        if key is None:
            print("[用法] save open | save cube | save other")
            return
        if self.last_target is None:
            print("[拒绝] 请先输入一个0～100位置值并确认实物效果。")
            return
        self.profiles[key] = self.last_target
        self.save_file()
        print(
            f"[已保存] {key}={self.last_target}，测试回读={self.last_actual}"
        )
        print(f"[文件] {self.output}")

    def test_profile(self, name: str) -> None:
        mapping = {
            "open": "open",
            "cube": "cube_close",
            "other": "other_close",
        }
        key = mapping.get(name)
        if key is None:
            print("[用法] test open | test cube | test other")
            return
        value = self.profiles.get(key)
        if not valid_gripper_value(value):
            print(f"[拒绝] {key}尚未保存。")
            return
        self.move(int(value))

    def show(self) -> None:
        actual = self.read_value()
        print(
            f"[状态] actual={actual}, speed={self.speed}, "
            f"open={self.profiles['open']}, "
            f"cube_close={self.profiles['cube_close']}, "
            f"other_close={self.profiles['other_close']}"
        )
        print(f"[文件] {self.output}")

    def reconnect(self) -> None:
        try:
            self.arm.close()
        except Exception:
            pass
        time.sleep(0.75)
        self.arm = MechArmSocket(self.host, self.port)
        time.sleep(0.75)
        print(f"[重连] 当前夹爪回读={self.read_value()}")

    def run(self) -> None:
        print(f"[连接] Jetson -> mechArm {self.host}:{self.port}")
        print(f"[当前夹爪] {self.read_value()}")
        print("[安全] 测试物必须放在桌面上，禁止悬空夹持；手指远离夹指。")
        print("[说明] 0=闭合，100=张开；这是位置值，不是角度制的度数。")
        print()
        print("直接输入数字：  例如 72      立即测试位置72")
        print("保存参数：      save cube | save other | save open")
        print("复测参数：      test cube | test other | test open")
        print("其他命令：      show | speed 20 | clear | reconnect | help | quit")
        print()
        self.save_file()

        while True:
            try:
                line = input("gripper> ")
            except EOFError:
                print("\n[退出] 输入流已关闭。")
                break
            command, value = parse_command(line)
            if command == "empty":
                continue
            if command == "move":
                self.move(value)
            elif command == "speed":
                if (
                    isinstance(value, int)
                    and 1 <= value <= MAX_SAFE_GRIPPER_SPEED
                ):
                    self.speed = value
                    self.save_file()
                    print(f"[速度] {self.speed}")
                else:
                    print(
                        f"[拒绝] 正式搬运速度必须是1～"
                        f"{MAX_SAFE_GRIPPER_SPEED}的整数。"
                    )
            elif command == "save":
                self.save_profile(str(value))
            elif command == "test":
                self.test_profile(str(value))
            elif command in {"show", "status"}:
                self.show()
            elif command == "clear":
                self.clear_pending_command()
            elif command == "reconnect":
                self.reconnect()
            elif command in {"help", "?"}:
                print("数字0～100 | save/test open/cube/other | speed N")
                print("show | clear | reconnect | quit")
            elif command in {"quit", "exit", "q"}:
                break
            else:
                print("[无法识别] 输入 help 查看命令。")

        self.save_file()
        print("[完成] 测试结束；夹爪保持当前状态，没有自动追加命令。")
        print(f"[已保存] {self.output}")

    def close(self) -> None:
        try:
            self.arm.close()
        except Exception:
            pass


def self_test() -> None:
    assert parse_command("72") == ("move", 72)
    assert parse_command("move 44") == ("move", 44)
    assert parse_command("save cube") == ("save", "cube")
    assert parse_command("speed 20") == ("speed", 20)
    assert parse_command("not a command") == ("invalid", None)
    assert valid_gripper_value(0)
    assert valid_gripper_value(100)
    assert not valid_gripper_value(-1)
    assert not valid_gripper_value(101)
    assert is_stable([70, 70, 71, 70])
    assert not is_stable([70, 72, 70, 72])
    print("SELF_TEST=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.100.2")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.speed <= MAX_SAFE_GRIPPER_SPEED:
        raise RuntimeError(f"speed必须位于1～{MAX_SAFE_GRIPPER_SPEED}")
    lock_stream = acquire_experiment3_control_lock("gripper-profile-tester")
    try:
        tester = GripperProfileTester(args.host, args.port, args.output, args.speed)
        try:
            tester.run()
        finally:
            tester.close()
    finally:
        lock_stream.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[停止] 操作员按Ctrl+C中止；没有自动追加夹爪命令。")
        raise SystemExit(130)
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1)
