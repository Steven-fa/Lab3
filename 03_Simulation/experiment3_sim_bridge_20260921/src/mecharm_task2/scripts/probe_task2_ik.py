#!/usr/bin/env python3
"""Probe downward-facing link6 poses while calibrating the task-2 simulation."""

import math
import sys

import rclpy
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node


class Probe(Node):
    def __init__(self):
        super().__init__("probe_task2_ik")
        self.client = self.create_client(GetPositionIK, "/compute_ik")

    def call(self, x, y, z, yaw):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "arm"
        ik.avoid_collisions = False
        ik.ik_link_name = "link6"
        ik.pose_stamped.header.frame_id = "base"
        ik.pose_stamped.pose.position.x = x
        ik.pose_stamped.pose.position.y = y
        ik.pose_stamped.pose.position.z = z
        ik.pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
        ik.pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
        ik.timeout.sec = 1
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        values = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        )
        names = [
            "joint1_to_base",
            "joint2_to_joint1",
            "joint3_to_joint2",
            "joint4_to_joint3",
            "joint5_to_joint4",
            "joint6_to_joint5",
        ]
        return [values[name] for name in names]


def main():
    rclpy.init()
    node = Probe()
    if not node.client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("MoveIt IK service is unavailable")
        sys.exit(1)
    found = 0
    for z in (0.08, 0.10, 0.12, 0.14, 0.16):
        for yaw_deg in (0, 45, -45, 90, -90, 135, -135, 180):
            values = node.call(0.18, 0.068, z, math.radians(yaw_deg))
            if values is not None:
                found += 1
                rounded = ", ".join(f"{value:.4f}" for value in values)
                print(f"IK_CANDIDATE z={z:.3f} yaw={yaw_deg:+d} joints=[{rounded}]")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
