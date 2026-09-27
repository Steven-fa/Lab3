#!/usr/bin/env python3
"""Find collision-free A/B grasp poses for the Gazebo wrapper geometry."""

import itertools
import math
import random
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
LOWER = [-2.792527, -1.3089, -3.0543, -2.7052, -2.0071, -3.14]
UPPER = [2.792527, 2.0943, 1.1344, 2.7052, 2.0071, 3.14]
POINT_A = (0.18, 0.08, 0.02)
POINT_B = (0.18, -0.08, 0.02)
GRASP_OFFSET = (0.0, 0.012, -0.10)
SAFE_HEIGHT = 0.12


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotate(q, vector):
    x, y, z, w = q
    vx, vy, vz = vector
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


class Search(Node):
    def __init__(self):
        super().__init__("search_grasp_ik")
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def apply_target(self, add):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        collision = CollisionObject()
        collision.header.frame_id = "base"
        collision.id = "task2_target"
        collision.operation = CollisionObject.ADD if add else CollisionObject.REMOVE
        if add:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [0.04, 0.04, 0.04]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = POINT_A
            pose.orientation.w = 1.0
            collision.primitives.append(primitive)
            collision.primitive_poses.append(pose)
        request.scene.world.collision_objects.append(collision)
        future = self.scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        return bool(future.done() and future.result() and future.result().success)

    def solve(self, point, q, seed):
        world_offset = rotate(q, GRASP_OFFSET)
        position = [point[i] - world_offset[i] for i in range(3)]
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "arm"
        ik.avoid_collisions = True
        ik.ik_link_name = "link6"
        ik.robot_state.joint_state.name = list(JOINT_NAMES)
        ik.robot_state.joint_state.position = list(seed)
        ik.pose_stamped.header.frame_id = "base"
        ik.pose_stamped.pose.position.x = position[0]
        ik.pose_stamped.pose.position.y = position[1]
        ik.pose_stamped.pose.position.z = position[2]
        (
            ik.pose_stamped.pose.orientation.x,
            ik.pose_stamped.pose.orientation.y,
            ik.pose_stamped.pose.orientation.z,
            ik.pose_stamped.pose.orientation.w,
        ) = q
        ik.timeout.nanosec = 400_000_000
        future = self.ik.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done() or future.result() is None:
            return None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        values = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        return [values[name] for name in JOINT_NAMES]

    def solve_chain(self, q, seed):
        a_safe = (POINT_A[0], POINT_A[1], POINT_A[2] + SAFE_HEIGHT)
        b_safe = (POINT_B[0], POINT_B[1], POINT_B[2] + SAFE_HEIGHT)
        result = []
        current = seed
        for point in (a_safe, POINT_A, a_safe, b_safe, POINT_B, b_safe):
            current = self.solve(point, q, current)
            if current is None:
                return None
            result.append(current)
        return result


def main():
    rclpy.init()
    node = Search()
    if not node.ik.wait_for_service(timeout_sec=10.0) or not node.scene.wait_for_service(
        timeout_sec=10.0
    ):
        node.get_logger().error("MoveIt services are unavailable")
        sys.exit(1)
    if not node.apply_target(False):
        node.get_logger().error("Could not temporarily remove the target collision object")
        sys.exit(1)

    rng = random.Random(270)
    found = None
    try:
        angles = (-45, -30, 0, 30, 45)
        yaws = (-90, -45, 0, 45, 90, 135, 180)
        for roll_deg, pitch_deg, yaw_deg in itertools.product(angles, angles, yaws):
            q = quaternion_from_euler(
                math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
            )
            seeds = [[0.0] * 6]
            seeds.extend(
                [rng.uniform(low, high) for low, high in zip(LOWER, UPPER)]
                for _ in range(3)
            )
            for seed in seeds:
                chain = node.solve_chain(q, seed)
                if chain is not None:
                    found = (roll_deg, pitch_deg, yaw_deg, q, chain)
                    break
            if found:
                break
    finally:
        node.apply_target(True)

    if not found:
        node.get_logger().error("No collision-free A/B grasp chain was found")
        code = 1
    else:
        roll_deg, pitch_deg, yaw_deg, q, chain = found
        print(
            f"GRASP_CHAIN_OK rpy_deg=[{roll_deg}, {pitch_deg}, {yaw_deg}] "
            f"quaternion={[round(value, 7) for value in q]}"
        )
        labels = ("A_SAFE", "A_GRASP", "A_LIFT", "B_SAFE", "B_RELEASE", "B_LIFT")
        for label, values in zip(labels, chain):
            print(f"{label}={[round(value, 7) for value in values]}")
        code = 0
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
