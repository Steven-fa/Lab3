#!/usr/bin/env python3
"""Measure calibrated fingertip-pad locations across the gripper stroke."""

import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import LinkStates
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


PAD_LOCALS = {
    "gripper_left1": np.array([0.017862, 0.048691, -0.010549]),
    "gripper_right1": np.array([-0.019062, 0.048691, -0.010549]),
}


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


class Sweep(Node):
    def __init__(self):
        super().__init__("calibrate_gripper_sweep")
        self.action = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_action_controller/follow_joint_trajectory",
        )
        self.states = None
        self.create_subscription(LinkStates, "/gazebo/link_states", self.on_states, 1)

    def on_states(self, message):
        self.states = message

    def command(self, value):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["gripper_controller"]
        point = JointTrajectoryPoint()
        point.positions = [float(value)]
        point.time_from_start.sec = 1
        goal.trajectory.points = [point]
        future = self.action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=5.0)
        time.sleep(0.15)

    def pads(self):
        self.states = None
        while self.states is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        poses = dict(zip(self.states.name, self.states.pose))
        link6 = poses["mecharm_270_m5::link6"]
        r6 = rotation(link6.orientation)
        p6 = np.array([link6.position.x, link6.position.y, link6.position.z])
        result = {}
        for link, local in PAD_LOCALS.items():
            pose = poses[f"mecharm_270_m5::{link}"]
            world = np.array([pose.position.x, pose.position.y, pose.position.z]) + rotation(pose.orientation) @ local
            result[link] = r6.T @ (world - p6)
        return result


def main():
    rclpy.init()
    node = Sweep()
    node.action.wait_for_server(timeout_sec=10.0)
    for value in (-0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.10, 0.15):
        node.command(value)
        pads = node.pads()
        left = pads["gripper_left1"]
        right = pads["gripper_right1"]
        print(
            f"q={value:+.2f} left={left.round(5).tolist()} "
            f"right={right.round(5).tolist()} separation={abs(left[0]-right[0]):.5f}"
        )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
