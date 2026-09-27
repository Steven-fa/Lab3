"""Raw-terminal keyboard frontend for q/qa/qb keyframe recording."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from .keyframes import classify_q_sequence


SERVICE_NAMES = {
    "waypoint": "/mecharm_demo/record_waypoint",
    "grasp": "/mecharm_demo/record_grasp",
    "release": "/mecharm_demo/record_release",
}

LABELS = {
    "waypoint": "普通关键点 q",
    "grasp": "夹取点 qa",
    "release": "释放点 qb",
}


class KeyframeKeyboard(Node):
    def __init__(self) -> None:
        super().__init__("keyframe_keyboard")
        self.record_clients = {
            kind: self.create_client(Trigger, service)
            for kind, service in SERVICE_NAMES.items()
        }
        self.start_client = self.create_client(
            Trigger, "/mecharm_demo/start_recording"
        )
        self.stop_client = self.create_client(
            Trigger, "/mecharm_demo/stop_recording"
        )

    def call(self, client, label: str, timeout: float = 8.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            print(f"[失败] 服务不可用：{label}")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        response = future.result() if future.done() else None
        if response is None:
            print(f"[失败] 服务超时：{label}")
            return False
        prefix = "成功" if response.success else "提示"
        print(f"[{prefix}] {response.message}")
        return bool(response.success)

    def ensure_recording(self) -> None:
        self.call(self.start_client, "开始关键点录制", timeout=3.0)

    def record(self, kind: str) -> None:
        self.call(self.record_clients[kind], LABELS[kind])

    def save(self) -> None:
        self.call(self.stop_client, "保存关键点")


def _read_q_sequence(timeout: float = 0.55) -> tuple[str, str | None]:
    """Read q and an optional immediate a/b suffix without requiring Enter."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return "waypoint", None
    suffix = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
    return classify_q_sequence(suffix), suffix


def main(args=None) -> None:
    if not sys.stdin.isatty():
        raise SystemExit("keyframe_keyboard 必须在交互式 WSL 终端中运行")

    rclpy.init(args=args)
    node = KeyframeKeyboard()
    original_settings = termios.tcgetattr(sys.stdin)
    try:
        node.ensure_recording()
        print("\n关键点键盘已就绪（无需按 Enter）")
        print("  q     保存普通关键点")
        print("  qa    保存夹取点并闭合夹爪")
        print("  qb    保存释放点并张开夹爪")
        print("  s     保存本次关键点")
        print("  Esc   保存并退出键盘程序\n")
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            if key.lower() == "q":
                kind, suffix = _read_q_sequence()
                if suffix not in (None, "a", "b"):
                    print(f"[提示] 未识别 q{suffix}，按普通关键点 q 处理")
                node.record(kind)
            elif key.lower() == "s":
                node.save()
            elif key == "\x1b":
                node.save()
                break
    except KeyboardInterrupt:
        node.save()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
