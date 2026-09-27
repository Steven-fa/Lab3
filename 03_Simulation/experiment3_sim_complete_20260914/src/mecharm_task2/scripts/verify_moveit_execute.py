#!/usr/bin/env python3
"""Plan and execute a small motion on fake hardware, then return Home."""

import sys

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node


JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
TEST_TARGET = [0.20, -0.15, -0.20, 0.15, 0.10, 0.20]
HOME_TARGET = [0.0] * 6


class ExecuteVerifier(Node):
    def __init__(self):
        super().__init__("verify_moveit_execute")
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def execute(self, target, label):
        if not self.client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error("MoveGroup action server is unavailable")
            return False

        goal_message = MoveGroup.Goal()
        request = goal_message.request
        request.group_name = "arm"
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnectkConfigDefault"
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.10
        request.max_acceleration_scaling_factor = 0.10
        request.start_state.is_diff = True

        constraints = Constraints()
        constraints.name = label
        for name, position in zip(JOINT_NAMES, target):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = 0.01
            joint.tolerance_below = 0.01
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        request.goal_constraints.append(constraints)

        goal_message.planning_options.plan_only = False
        goal_message.planning_options.replan = True
        goal_message.planning_options.replan_attempts = 2

        send_future = self.client.send_goal_async(goal_message)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"{label}: goal request timed out")
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{label}: goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"{label}: execution timed out")
            return False
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"{label}: MoveIt error={result.error_code.val}")
            return False

        self.get_logger().info(f"MOVEIT_EXECUTE_OK target={label}")
        return True


def main():
    rclpy.init()
    node = ExecuteVerifier()
    ok = node.execute(TEST_TARGET, "verification_target")
    if ok:
        ok = node.execute(HOME_TARGET, "Home")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
