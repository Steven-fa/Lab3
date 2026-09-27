#!/usr/bin/env python3
"""Find an open-jaw A pose that is collision-free with the cube still present."""

import itertools
import random
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

import search_jointspace_grasp as search


def add_target(checker):
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    obj = CollisionObject()
    obj.header.frame_id = "base"
    obj.id = "task2_target"
    obj.operation = CollisionObject.ADD
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [0.04, 0.04, 0.04]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = search.POINT_A
    pose.orientation.w = 1.0
    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    request.scene.world.collision_objects.append(obj)
    future = checker.scene.call_async(request)
    rclpy.spin_until_future_complete(checker, future, timeout_sec=4.0)
    return bool(future.done() and future.result() and future.result().success)


def main():
    rclpy.init()
    checker = search.CollisionChecker()
    if not checker.validity.wait_for_service(timeout_sec=12.0):
        sys.exit(1)
    if not checker.scene.wait_for_service(timeout_sec=12.0) or not add_target(checker):
        sys.exit(2)

    rng = random.Random(2705)
    seeds = [np.zeros(6)] + [
        np.array([rng.uniform(lo, hi) for lo, hi in zip(search.LOWER, search.UPPER)])
        for _ in range(90)
    ]
    offsets = [
        (-0.0006, 0.010, 0.105),
        (-0.005, -0.020, 0.0675),
        (0.015, -0.020, 0.060),
    ]
    offsets.extend(
        (x, y, z)
        for x, y, z in itertools.product(
            (-0.020, -0.010, 0.0, 0.010, 0.020),
            (-0.015, -0.005, 0.005, 0.015, 0.025),
            (0.0725, 0.0775, 0.0825, 0.0875),
        )
    )

    seen = set()
    for offset in offsets:
        rounded_offset = tuple(round(value, 5) for value in offset)
        if rounded_offset in seen:
            continue
        seen.add(rounded_offset)
        search.GRASP_OFFSET = np.array([*offset, 1.0])
        search.OFFSET_DIRECTION = (
            search.GRASP_OFFSET[:3] / np.linalg.norm(search.GRASP_OFFSET[:3])
        )
        candidates = search.unique_solutions(search.GRASP_POINT_A, seeds, limit=12)
        for joints in candidates:
            valid, contacts = checker.is_valid(joints)
            if valid:
                print(
                    f"OPEN_GRASP_VALID offset={list(rounded_offset)} "
                    f"joints={[round(float(value), 7) for value in joints]}"
                )
                checker.destroy_node()
                rclpy.shutdown()
                return
        print(f"OFFSET_REJECTED offset={list(rounded_offset)} candidates={len(candidates)}")

    print("NO_OPEN_GRASP_CAVITY_FOUND")
    checker.destroy_node()
    rclpy.shutdown()
    sys.exit(3)


if __name__ == "__main__":
    main()
