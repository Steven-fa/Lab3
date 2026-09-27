"""ROS 2 feedback worker that executes one verified real-arm transfer at a time."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml


GROUP_DESTINATION = {"cube": "right", "other": "left"}


class Experiment3RobotWorker(Node):
    def __init__(self) -> None:
        super().__init__("experiment3_robot_worker")
        self.declare_parameter("armed", False)
        self.declare_parameter("hardware_root", "/home/nvidia/mecharm_tasks")
        self.declare_parameter(
            "poses_path",
            "/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml",
        )
        self.declare_parameter(
            "gripper_profiles_path",
            "/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml",
        )
        self.declare_parameter("host", "192.168.100.2")
        self.declare_parameter("port", 9000)
        self.declare_parameter("arm_speed", 8)
        self.declare_parameter("stream_step", 2.0)
        self.declare_parameter("stream_period", 0.20)
        self.declare_parameter("retreat_extension", 0.50)
        self.declare_parameter("request_topic", "/experiment3/transfer_requests")
        self.declare_parameter("feedback_topic", "/experiment3/transfer_feedback")

        self.armed = bool(self.get_parameter("armed").value)
        self.hardware_root = Path(str(self.get_parameter("hardware_root").value))
        self.poses_path = Path(str(self.get_parameter("poses_path").value))
        self.gripper_profiles_path = Path(
            str(self.get_parameter("gripper_profiles_path").value)
        )
        if not self.hardware_root.is_dir():
            raise RuntimeError(f"hardware_root不存在：{self.hardware_root}")
        if not self.poses_path.is_file():
            raise RuntimeError(f"17点姿态不存在：{self.poses_path}")
        root_value = str(self.hardware_root)
        if root_value not in sys.path:
            sys.path.insert(0, root_value)
        self.transfer_module = importlib.import_module("experiment3_transfer_test")
        self.gripper_profiles_module = importlib.import_module(
            "experiment3_gripper_profiles"
        )
        self.gripper_profiles = self.gripper_profiles_module.load_gripper_profiles(
            self.gripper_profiles_path
        )
        self.control_lock = (
            self.transfer_module.acquire_experiment3_control_lock(
                "ros2-experiment3-worker"
            )
            if self.armed
            else None
        )
        self.active_runner: Any | None = None
        self.busy = False
        self.worker_thread: threading.Thread | None = None
        self.feedback_publisher = self.create_publisher(
            String, str(self.get_parameter("feedback_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("request_topic").value),
            self.on_request,
            10,
        )
        mode = "ARMED" if self.armed else "DISARMED"
        self.get_logger().warning(
            f"Robot worker {mode}; poses={self.poses_path}; "
            f"gripper OPEN={self.gripper_profiles['open']}, "
            f"cube={self.gripper_profiles['cube_close']}, "
            f"other={self.gripper_profiles['other_close']}"
        )

    def publish_feedback(self, request_id: int, status: str, **values: Any) -> None:
        message = String()
        message.data = json.dumps(
            {"request_id": request_id, "status": status, **values},
            ensure_ascii=False,
        )
        self.feedback_publisher.publish(message)

    def on_request(self, message: String) -> None:
        try:
            request = json.loads(message.data)
            request_id = int(request["request_id"])
            grid = int(request["grid_id"])
            group = str(request["group"])
            destination = str(request["destination"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Invalid request: {error}")
            return
        if not self.armed:
            self.publish_feedback(request_id, "FAILED", error="robot worker is disarmed")
            return
        if self.busy:
            self.publish_feedback(request_id, "FAILED", error="robot worker is busy")
            return
        if grid not in range(1, 7) or GROUP_DESTINATION.get(group) != destination:
            self.publish_feedback(
                request_id,
                "FAILED",
                error="invalid grid/group/destination mapping",
            )
            return
        self.busy = True
        self.worker_thread = threading.Thread(
            target=self.execute_request,
            args=(request,),
            daemon=False,
        )
        self.worker_thread.start()

    def execute_request(self, request: dict[str, Any]) -> None:
        request_id = int(request["request_id"])
        grid = int(request["grid_id"])
        group = str(request["group"])
        destination = str(request["destination"])
        runner = None
        try:
            module = self.transfer_module
            document = yaml.safe_load(self.poses_path.read_text(encoding="utf-8"))
            highest_z = module.highest_recorded_tcp_z(document)
            poses, _recorded_gripper = module.load_transfer_route(
                self.poses_path,
                grid,
                destination,
                require_active_correction=True,
            )
            gripper = self.gripper_profiles_module.profile_for_group(
                self.gripper_profiles, group
            )
            self.get_logger().info(
                f"GRID {grid} {group}: OPEN={gripper['open']}, "
                f"CLOSE={gripper['close']}, speed={gripper['speed']}"
            )
            approach = poses[f"{destination}_approach"]
            release = poses[f"{destination}_release"]
            extension = module.effective_retreat_extension(
                approach,
                release,
                float(self.get_parameter("retreat_extension").value),
            )
            runner = module.TransferRunner(
                str(self.get_parameter("host").value),
                int(self.get_parameter("port").value),
                int(self.get_parameter("arm_speed").value),
                self.poses_path.parent,
                float(self.get_parameter("stream_step").value),
                float(self.get_parameter("stream_period").value),
            )
            self.active_runner = runner
            current = runner.health()
            self.publish_feedback(request_id, "STARTED", grid_id=grid)
            runner.run_transfer(
                current,
                poses,
                gripper,
                grid,
                destination,
                int(gripper["speed"]),
                extension,
                highest_z,
            )
            self.publish_feedback(
                request_id,
                "COMPLETED",
                grid_id=grid,
                destination=destination,
                group=group,
                gripper_open=int(gripper["open"]),
                gripper_close=int(gripper["close"]),
                motion_log=str(runner.log_path),
            )
        except Exception as error:
            if runner is not None and not runner.stopped:
                try:
                    runner.stop()
                except Exception:
                    pass
            self.publish_feedback(
                request_id,
                "FAILED",
                grid_id=grid,
                error=str(error),
                motion_log=str(runner.log_path) if runner is not None else None,
            )
        finally:
            if runner is not None:
                try:
                    runner.arm.close()
                except Exception:
                    pass
            self.active_runner = None
            self.busy = False

    def destroy_node(self) -> bool:
        runner = self.active_runner
        if runner is not None and not runner.stopped:
            try:
                runner.stop()
            except Exception:
                pass
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        worker_still_running = (
            self.worker_thread is not None and self.worker_thread.is_alive()
        )
        if worker_still_running:
            self.get_logger().error(
                "Robot worker did not stop within 5s; retaining the exclusive "
                "control lock until process exit"
            )
        elif self.control_lock is not None:
            self.control_lock.close()
            self.control_lock = None
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Experiment3RobotWorker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
