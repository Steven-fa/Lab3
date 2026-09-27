#!/usr/bin/env python3
"""Numerically find collision-free grasp/lift joint states for task 2.

The calculation uses the exact URDF joint transforms.  MoveIt is then asked to
validate each solution against self-collision and the table before it is printed.
"""

import math
import random
import sys

import numpy as np
from scipy.optimize import least_squares

import rclpy
from moveit_msgs.msg import CollisionObject, RobotState
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
LOWER = np.array([-2.792527, -1.3089, -3.0543, -2.7052, -2.0071, -3.14])
UPPER = np.array([2.792527, 2.0943, 1.1344, 2.7052, 2.0071, 3.14])
POINT_A = np.array([0.18, 0.08, 0.02])
POINT_B = np.array([0.18, -0.08, 0.02])
GRASP_POINT_A = POINT_A.copy()
SAFE_RISE = 0.10
GRASP_OFFSET = np.array([-0.0006, 0.010, 0.105, 1.0])
OFFSET_DIRECTION = GRASP_OFFSET[:3] / np.linalg.norm(GRASP_OFFSET[:3])


def translation(xyz):
    matrix = np.eye(4)
    matrix[:3, 3] = xyz
    return matrix


def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def origin(xyz, rpy):
    return translation(xyz) @ rotation_z(rpy[2]) @ rotation_y(rpy[1]) @ rotation_x(rpy[0])


ORIGINS = [
    origin([0, 0, 0.1], [0, 0, 0]),
    origin([0, 0, 0.038], [-math.pi / 2, 0, 0]),
    origin([0, -0.1, 0], [0, 0, 0]),
    origin([0.108, -0.005, -0.001], [0, math.pi / 2, 0]),
    origin([-0.001, 0, 0], [0, -math.pi / 2, 0]),
    origin([0.06, 0, 0], [0, math.pi / 2, 0]),
]


def link6_transform(joints):
    transform = np.eye(4)
    for fixed, angle in zip(ORIGINS, joints):
        transform = transform @ fixed @ rotation_z(angle)
    return transform


def residual(joints, target):
    transform = link6_transform(joints)
    grasp = (transform @ GRASP_OFFSET)[:3]
    direction = transform[:3, :3] @ OFFSET_DIRECTION
    # Keep the line between the two fingertip pads horizontal.  Constraining
    # only the approach direction leaves a free roll that can put one pad into
    # the tabletop even when the grasp centre itself is correct.
    jaw_axis = np.array([1.0, 0.0, 0.0])
    jaw_axis -= np.dot(jaw_axis, OFFSET_DIRECTION) * OFFSET_DIRECTION
    jaw_axis /= np.linalg.norm(jaw_axis)
    world_jaw_axis = transform[:3, :3] @ jaw_axis
    # Centimetre-scale position errors dominate; the direction term keeps the
    # real fingers pointing down towards the table.
    return np.concatenate((
        (grasp - target) * 100.0,
        (direction - [0, 0, -1]) * 1.5,
        (world_jaw_axis - [1, 0, 0]) * 1.2,
    ))


class CollisionChecker(Node):
    def __init__(self):
        super().__init__("search_jointspace_grasp")
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def remove_target(self):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        collision = CollisionObject()
        collision.header.frame_id = "base"
        collision.id = "task2_target"
        collision.operation = CollisionObject.REMOVE
        request.scene.world.collision_objects.append(collision)
        future = self.scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        # MoveIt may report success=false when the object was already removed by
        # a previous calibration pass.  In that case the desired state already
        # holds, so a completed service response is sufficient here.
        return bool(future.done() and future.result() is not None)

    def is_valid(self, joints):
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state = RobotState()
        request.robot_state.joint_state = JointState()
        request.robot_state.joint_state.name = JOINT_NAMES + ["gripper_controller"]
        request.robot_state.joint_state.position = list(joints) + [0.15]
        future = self.validity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return False, ["validity service timeout"]
        response = future.result()
        contacts = sorted({
            f"{contact.contact_body_1}<->{contact.contact_body_2}"
            for contact in response.contacts
        })
        return response.valid, contacts


def unique_solutions(target, seeds, limit=40):
    solutions = []
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            args=(target,),
            bounds=(LOWER + 1e-5, UPPER - 1e-5),
            max_nfev=700,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        transform = link6_transform(result.x)
        position_error = np.linalg.norm((transform @ GRASP_OFFSET)[:3] - target)
        direction = transform[:3, :3] @ OFFSET_DIRECTION
        downward = float(np.dot(direction, [0, 0, -1]))
        if position_error > 0.0015 or downward < 0.94:
            continue
        if any(np.linalg.norm(result.x - existing) < 0.08 for existing in solutions):
            continue
        solutions.append(result.x)
        if len(solutions) >= limit:
            break
    return solutions


def find_valid(checker, label, target, seeds):
    candidates = unique_solutions(target, seeds)
    print(f"{label}_NUMERICAL_CANDIDATES={len(candidates)}")
    rejected = []
    for index, candidate in enumerate(candidates):
        valid, contacts = checker.is_valid(candidate)
        if valid:
            grasp = (link6_transform(candidate) @ GRASP_OFFSET)[:3]
            print(
                f"{label}={[round(float(v), 7) for v in candidate]} "
                f"grasp={[round(float(v), 5) for v in grasp]} VALID"
            )
            return candidate
        if index < 12:
            print(
                f"{label}_INVALID_{index + 1}="
                f"{[round(float(v), 5) for v in candidate]} contacts={contacts}"
            )
        rejected.extend(contacts)
    print(f"{label}_REJECTED_CONTACTS={sorted(set(rejected))}")
    return None


def main():
    rclpy.init()
    checker = CollisionChecker()
    if not checker.validity.wait_for_service(timeout_sec=12.0):
        print("STATE_VALIDITY_UNAVAILABLE")
        sys.exit(1)
    if not checker.scene.wait_for_service(timeout_sec=12.0) or not checker.remove_target():
        print("PLANNING_SCENE_UNAVAILABLE")
        sys.exit(1)

    rng = random.Random(270)
    random_seeds = [np.zeros(6)] + [
        np.array([rng.uniform(lo, hi) for lo, hi in zip(LOWER, UPPER)])
        for _ in range(360)
    ]

    results = {}
    for label, target in (
        ("A_GRASP", GRASP_POINT_A),
        ("A_SAFE", GRASP_POINT_A + [0, 0, SAFE_RISE]),
        # Release 10 mm above the table; the cube then settles under gravity.
        ("B_RELEASE", POINT_B + [0, 0, 0.01]),
        ("B_SAFE", POINT_B + [0, 0, SAFE_RISE]),
    ):
        preferred = list(random_seeds)
        if results:
            preferred = list(results.values()) + preferred
        answer = find_valid(checker, label, target, preferred)
        if answer is None:
            checker.destroy_node()
            rclpy.shutdown()
            sys.exit(2)
        results[label] = answer

    print("JOINTSPACE_GRASP_CHAIN_OK")
    checker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
