#!/usr/bin/env python3
"""Verify the arm KDL solver using the current link6 pose."""

import sys

import rclpy
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.node import Node


class IkVerifier(Node):
    def __init__(self):
        super().__init__("verify_moveit_ik")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

    def run(self):
        if not self.fk_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("Forward-kinematics service is unavailable")
            return False
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("Inverse-kinematics service is unavailable")
            return False

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = "base"
        fk_request.fk_link_names = ["link6"]
        fk_request.robot_state.is_diff = True
        fk_future = self.fk_client.call_async(fk_request)
        rclpy.spin_until_future_complete(self, fk_future, timeout_sec=10.0)
        if not fk_future.done() or fk_future.result() is None:
            self.get_logger().error("Forward-kinematics request timed out")
            return False
        fk_response = fk_future.result()
        if fk_response.error_code.val != MoveItErrorCodes.SUCCESS or not fk_response.pose_stamped:
            self.get_logger().error(f"Forward kinematics failed: {fk_response.error_code.val}")
            return False

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = "arm"
        ik_request.ik_request.robot_state.is_diff = True
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.ik_link_name = "link6"
        ik_request.ik_request.pose_stamped = fk_response.pose_stamped[0]
        ik_request.ik_request.timeout.sec = 5

        ik_future = self.ik_client.call_async(ik_request)
        rclpy.spin_until_future_complete(self, ik_future, timeout_sec=10.0)
        if not ik_future.done() or ik_future.result() is None:
            self.get_logger().error("Inverse-kinematics request timed out")
            return False
        ik_response = ik_future.result()
        if ik_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Inverse kinematics failed: {ik_response.error_code.val}")
            return False

        solved = dict(
            zip(
                ik_response.solution.joint_state.name,
                ik_response.solution.joint_state.position,
            )
        )
        arm_joint_count = sum(name.startswith("joint") for name in solved)
        self.get_logger().info(f"MOVEIT_IK_OK link=link6 solved_arm_joints={arm_joint_count}")
        return arm_joint_count >= 6


def main():
    rclpy.init()
    node = IkVerifier()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
