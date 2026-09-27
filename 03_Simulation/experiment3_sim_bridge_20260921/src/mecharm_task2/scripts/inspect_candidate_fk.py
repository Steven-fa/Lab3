#!/usr/bin/env python3
"""Inspect MoveIt FK and contact geometry for the current A candidate."""

import numpy as np
import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetStateValidity
from rclpy.node import Node
from sensor_msgs.msg import JointState


NAMES = [
    "joint1_to_base", "joint2_to_joint1", "joint3_to_joint2",
    "joint4_to_joint3", "joint5_to_joint4", "joint6_to_joint5",
    "gripper_controller",
]
VALUES = [0.458, 1.10467, -0.95098, -0.13233, 1.41814, -1.08915, -0.65]
PAD_LOCAL = np.array([-0.041031, 0.050470, -0.008134])


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def main():
    rclpy.init()
    node = Node("inspect_candidate_fk")
    fk = node.create_client(GetPositionFK, "/compute_fk")
    validity = node.create_client(GetStateValidity, "/check_state_validity")
    fk.wait_for_service(timeout_sec=10.0)
    validity.wait_for_service(timeout_sec=10.0)
    state = RobotState()
    state.joint_state = JointState()
    state.joint_state.name = NAMES
    state.joint_state.position = VALUES
    request = GetPositionFK.Request()
    request.header.frame_id = "base"
    request.fk_link_names = ["link6", "gripper_left1", "gripper_right1"]
    request.robot_state = state
    future = fk.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    poses = {
        name: stamped.pose
        for name, stamped in zip(request.fk_link_names, future.result().pose_stamped)
    }
    pose = poses["gripper_right1"]
    origin = np.array([pose.position.x, pose.position.y, pose.position.z])
    pad = origin + rotation(pose.orientation) @ PAD_LOCAL
    print(f"RIGHT1_ORIGIN={origin.round(6).tolist()}")
    print(f"RIGHT_PAD_WORLD={pad.round(6).tolist()}")
    link6 = poses["link6"]
    r6 = rotation(link6.orientation)
    p6 = np.array([link6.position.x, link6.position.y, link6.position.z])
    for name in ("gripper_left1", "gripper_right1"):
        item = poses[name]
        relative = r6.T @ (
            np.array([item.position.x, item.position.y, item.position.z]) - p6
        )
        print(f"MOVEIT_RELATIVE_{name}={relative.round(6).tolist()}")
    desired = {
        "gripper_left1": np.array([0.025, -0.010, 0.082]),
        "gripper_right1": np.array([-0.035, -0.010, 0.082]),
    }
    for name, target in desired.items():
        item = poses[name]
        world_target = p6 + r6 @ target
        local = rotation(item.orientation).T @ (
            world_target - np.array([item.position.x, item.position.y, item.position.z])
        )
        print(f"MOVEIT_PAD_LOCAL_{name}={local.round(6).tolist()}")
    check = GetStateValidity.Request()
    check.group_name = "arm"
    check.robot_state = state
    future = validity.call_async(check)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    response = future.result()
    print(f"VALID={response.valid}")
    for contact in response.contacts:
        print(
            f"CONTACT {contact.contact_body_1}<->{contact.contact_body_2} "
            f"depth={contact.depth:.6f} position="
            f"[{contact.position.x:.6f},{contact.position.y:.6f},{contact.position.z:.6f}]"
        )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
