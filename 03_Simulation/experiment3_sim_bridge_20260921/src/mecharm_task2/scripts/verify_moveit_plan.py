#!/usr/bin/env python3
"""Request one collision-checked arm plan without executing it."""

import sys

import rclpy
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from rclpy.node import Node


JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
JOINT_TARGET = [0.20, -0.15, -0.20, 0.15, 0.10, 0.20]


class PlanVerifier(Node):
    def __init__(self):
        super().__init__("verify_moveit_plan")
        self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")

    def run(self):
        if not self.client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("MoveIt planning service is unavailable")
            return False

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = "arm"
        motion_request.pipeline_id = "ompl"
        motion_request.planner_id = "RRTConnectkConfigDefault"
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.start_state.is_diff = True

        goal = Constraints()
        goal.name = "task2_verification_goal"
        for name, target in zip(JOINT_NAMES, JOINT_TARGET):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = target
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            goal.joint_constraints.append(constraint)
        motion_request.goal_constraints.append(goal)

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("MoveIt planning request timed out")
            return False

        response = future.result().motion_plan_response
        point_count = len(response.trajectory.joint_trajectory.points)
        if response.error_code.val != MoveItErrorCodes.SUCCESS or point_count == 0:
            self.get_logger().error(
                f"Planning failed: error_code={response.error_code.val}, points={point_count}"
            )
            return False

        self.get_logger().info(
            f"MOVEIT_PLAN_OK group=arm points={point_count} planning_time={response.planning_time:.3f}s"
        )
        return True


def main():
    rclpy.init()
    node = PlanVerifier()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
