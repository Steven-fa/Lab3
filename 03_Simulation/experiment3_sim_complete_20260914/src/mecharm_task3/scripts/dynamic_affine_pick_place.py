#!/usr/bin/env python3
"""Full Gazebo pick/place driven by the experiment-2 planar affine mapping.

Pipeline:
user XY -> calibrated TCP/task XY -> hull check -> dense IK chains ->
Gazebo cube reset -> descend/grasp/attach -> lift/transfer/release -> Home.

This program only talks to the ROS_DOMAIN_ID selected by the caller.  It has no
myCobot/pymycobot/serial imports and therefore cannot command the real arm.
"""

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    CollisionObject,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import validate_affine_sim as affine_base


ARM_JOINTS = affine_base.ARM_JOINTS
HOME = affine_base.HOME
OPEN = affine_base.OPEN_GRIPPER
CLOSED = 0.10
OBJECT_Z_M = 0.02
APPROACH_HEIGHT_M = 0.03
RELEASE_CLEARANCE_M = 0.01
CARTESIAN_STEP_M = 0.005
CHAIN_MAX_JOINT_STEP_RAD = math.radians(20.0)
CHAIN_MAX_JOINT_SPEED_RAD_S = 0.16
GRASP_OFFSET = (-0.0006, 0.010, 0.105)
REFERENCE_GRASP_JOINTS = [
    0.3922133,
    1.0421826,
    -0.9652633,
    0.0861514,
    1.4526762,
    -1.1872484,
]
TOUCH_LINKS = [
    "link6",
    "gripper_base",
    "gripper_left1",
    "gripper_left2",
    "gripper_left3",
    "gripper_right1",
    "gripper_right2",
    "gripper_right3",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=2, type=float, default=(0.0, 31.0))
    parser.add_argument("--end", nargs=2, type=float, default=(7.0, 31.0))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("direct", "anchored"), default="direct")
    parser.add_argument("--allow-outside-hull", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the Gazebo arm and cube. Without this flag only preflight runs.",
    )
    return parser.parse_args()


def wrapped_delta(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def interpolate_xyz(start, end, max_step=CARTESIAN_STEP_M):
    span = distance(start, end)
    count = max(1, math.ceil(span / max_step))
    return [
        tuple(start[i] + (end[i] - start[i]) * index / count for i in range(3))
        for index in range(count + 1)
    ]


def rotate_vector(q, vector):
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


class DynamicAffinePickPlace(affine_base.AffineSimulationValidator):
    def __init__(self):
        super().__init__()
        self.target_name = "task3_object_1"
        self.arm_trajectory = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.gripper = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_action_controller/follow_joint_trajectory",
        )
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.get_scene = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.get_entity = self.create_client(GetEntityState, "/gazebo/get_entity_state")
        self.set_entity = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.check_grasp = self.create_client(Trigger, "/task3/check_grasp")
        self.attach = self.create_client(Trigger, "/task3/attach")
        self.detach = self.create_client(Trigger, "/task3/detach")
        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, message):
        self.latest_joint_state = message

    def ready_full(self):
        return (
            super().ready()
            and self.arm_trajectory.wait_for_server(timeout_sec=20.0)
            and self.gripper.wait_for_server(timeout_sec=20.0)
            and self.scene.wait_for_service(timeout_sec=20.0)
            and self.get_scene.wait_for_service(timeout_sec=20.0)
            and self.get_entity.wait_for_service(timeout_sec=20.0)
            and self.set_entity.wait_for_service(timeout_sec=20.0)
            and self.check_grasp.wait_for_service(timeout_sec=20.0)
            and self.attach.wait_for_service(timeout_sec=20.0)
            and self.detach.wait_for_service(timeout_sec=20.0)
        )

    @staticmethod
    def robot_state(arm_positions, gripper_position=OPEN):
        state = RobotState()
        state.is_diff = False
        state.joint_state = JointState()
        state.joint_state.name = list(ARM_JOINTS) + [affine_base.GRIPPER_JOINT]
        state.joint_state.position = list(arm_positions) + [float(gripper_position)]
        return state

    def state_is_valid(self, arm_positions, gripper_position=OPEN):
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state = self.robot_state(arm_positions, gripper_position)
        # Preserve the planning scene's carried object when checking a sample.
        request.robot_state.is_diff = True
        future = self.validity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False, ["validity timeout"]
        response = future.result()
        contacts = sorted(
            {
                f"{contact.contact_body_1}<->{contact.contact_body_2}"
                for contact in response.contacts
            }
        )
        return bool(response.valid), contacts

    def grasp_reference_pose(self):
        request = affine_base.GetPositionFK.Request()
        request.header.frame_id = "base"
        request.fk_link_names = ["link6"]
        request.robot_state = self.robot_state(REFERENCE_GRASP_JOINTS)
        future = self.fk.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            return None, "grasp FK timeout"
        response = future.result()
        if (
            response.error_code.val != affine_base.MoveItErrorCodes.SUCCESS
            or not response.pose_stamped
        ):
            return None, f"grasp FK error {response.error_code.val}"
        return response.pose_stamped[0], None

    @staticmethod
    def link6_pose_for_object_center(object_center, reference_pose):
        q = reference_pose.pose.orientation
        quaternion = (q.x, q.y, q.z, q.w)
        world_offset = rotate_vector(quaternion, GRASP_OFFSET)
        pose = copy.deepcopy(reference_pose)
        pose.header.frame_id = "base"
        pose.pose.position.x = object_center[0] - world_offset[0]
        pose.pose.position.y = object_center[1] - world_offset[1]
        pose.pose.position.z = object_center[2] - world_offset[2]
        return pose

    def solve_object_path(self, object_points, seed, reference_pose, label):
        solutions = []
        current = list(seed)
        largest_step = 0.0
        for index, point in enumerate(object_points):
            pose = self.link6_pose_for_object_center(point, reference_pose)
            solution, reason, _ = self.solve_ik(pose, current)
            if solution is None:
                return None, {
                    "label": label,
                    "ok": False,
                    "failed_index": index,
                    "failed_object_center_m": [round(value, 6) for value in point],
                    "reason": reason,
                }
            step = max(abs(a - b) for a, b in zip(solution, current))
            if solutions:
                largest_step = max(largest_step, step)
            if solutions and step > CHAIN_MAX_JOINT_STEP_RAD:
                return None, {
                    "label": label,
                    "ok": False,
                    "failed_index": index,
                    "reason": f"adjacent IK jump {math.degrees(step):.2f} deg",
                }
            solutions.append(solution)
            current = solution
        return solutions, {
            "label": label,
            "ok": True,
            "cartesian_points": len(object_points),
            "largest_adjacent_joint_step_deg": round(math.degrees(largest_step), 3),
            "start_object_center_m": [round(value, 6) for value in object_points[0]],
            "end_object_center_m": [round(value, 6) for value in object_points[-1]],
        }

    def preflight(self, start_xy, end_xy):
        reference, reason = self.grasp_reference_pose()
        if reference is None:
            return None, {"ok": False, "reason": reason}

        start_grasp = (start_xy[0], start_xy[1], OBJECT_Z_M)
        start_safe = (start_xy[0], start_xy[1], OBJECT_Z_M + APPROACH_HEIGHT_M)
        end_safe = (end_xy[0], end_xy[1], OBJECT_Z_M + APPROACH_HEIGHT_M)
        end_release = (end_xy[0], end_xy[1], OBJECT_Z_M + RELEASE_CLEARANCE_M)

        descriptions = []
        start_vertical_points = list(reversed(interpolate_xyz(start_grasp, start_safe)))
        start_vertical, item = self.solve_object_path(
            start_vertical_points,
            REFERENCE_GRASP_JOINTS,
            reference,
            "start_vertical_safe_to_grasp",
        )
        descriptions.append(item)
        if start_vertical is None:
            return None, {"ok": False, "chains": descriptions}

        transfer_points = interpolate_xyz(start_safe, end_safe)[1:]
        transfer, item = self.solve_object_path(
            transfer_points,
            start_vertical[0],
            reference,
            "safe_transfer_start_to_end",
        )
        descriptions.append(item)
        if transfer is None:
            return None, {"ok": False, "chains": descriptions}

        end_vertical_points = interpolate_xyz(end_safe, end_release)
        end_vertical, item = self.solve_object_path(
            end_vertical_points,
            transfer[-1],
            reference,
            "end_vertical_safe_to_release",
        )
        descriptions.append(item)
        if end_vertical is None:
            return None, {"ok": False, "chains": descriptions}

        all_states = start_vertical + transfer + end_vertical
        invalid = []
        for index, state in enumerate(all_states):
            valid, contacts = self.state_is_valid(state, OPEN)
            if not valid:
                invalid.append({"index": index, "contacts": contacts})
                break
        result = {
            "ok": not invalid,
            "chains": descriptions,
            "total_ik_states": len(all_states),
            "open_gripper_preflight_invalid": invalid,
            "reference_grasp_link6_pose": affine_base.pose_as_dict(reference),
        }
        if invalid:
            return None, result
        return {
            "reference_pose": reference,
            "start_vertical": start_vertical,
            "transfer": transfer,
            "end_vertical": end_vertical,
            "start_grasp": start_grasp,
            "start_safe": start_safe,
            "end_safe": end_safe,
            "end_release": end_release,
        }, result

    def current_arm_positions(self):
        deadline = time.monotonic() + 4.0
        while self.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_joint_state is None:
            return None
        values = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        if any(name not in values for name in ARM_JOINTS):
            return None
        return [values[name] for name in ARM_JOINTS]

    @staticmethod
    def duration(seconds):
        whole = int(seconds)
        return Duration(sec=whole, nanosec=int((seconds - whole) * 1.0e9))

    def execute_chain(self, chain, gripper_position, label):
        if not chain:
            return True, {"label": label, "points": 0, "collision_samples": 0}
        previous = self.current_arm_positions()
        if previous is None:
            return False, {"label": label, "reason": "joint state unavailable"}
        checked = 0
        for target in chain:
            max_delta = max(abs(a - b) for a, b in zip(target, previous))
            if max_delta > CHAIN_MAX_JOINT_STEP_RAD:
                return False, {
                    "label": label,
                    "reason": f"runtime joint jump {math.degrees(max_delta):.2f} deg",
                }
            sample_count = max(1, math.ceil(max_delta / 0.04))
            for index in range(1, sample_count + 1):
                ratio = index / sample_count
                sample = [
                    origin + (target_value - origin) * ratio
                    for origin, target_value in zip(previous, target)
                ]
                valid, contacts = self.state_is_valid(sample, gripper_position)
                checked += 1
                if not valid:
                    return False, {
                        "label": label,
                        "reason": "collision sample failed",
                        "contacts": contacts,
                    }
            previous = list(target)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_JOINTS)
        previous = self.current_arm_positions()
        elapsed = 0.10
        for target in chain:
            joint_step = max(abs(a - b) for a, b in zip(target, previous))
            elapsed += max(0.08, joint_step / CHAIN_MAX_JOINT_SPEED_RAD_S)
            point = JointTrajectoryPoint()
            point.positions = list(target)
            point.time_from_start = self.duration(elapsed)
            trajectory.points.append(point)
            previous = list(target)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        sent = self.arm_trajectory.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=8.0)
        if not sent.done() or sent.result() is None or not sent.result().accepted:
            return False, {"label": label, "reason": "controller rejected chain"}
        self.active_goal = sent.result()
        result = self.active_goal.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result,
            timeout_sec=max(30.0, elapsed + 12.0),
        )
        if (
            not result.done()
            or result.result() is None
            or result.result().result.error_code != 0
        ):
            return False, {"label": label, "reason": "controller execution failed"}
        time.sleep(0.5)
        return True, {
            "label": label,
            "points": len(chain),
            "collision_samples": checked,
            "command_duration_s": round(elapsed, 3),
        }

    def move_gripper(self, position, label):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [affine_base.GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.velocities = [0.0]
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        sent = self.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=8.0)
        if not sent.done() or sent.result() is None or not sent.result().accepted:
            return False, f"{label}: gripper goal rejected"
        self.active_goal = sent.result()
        result = self.active_goal.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=10.0)
        if not result.done() or result.result() is None:
            return False, f"{label}: gripper timeout"
        if result.result().result.error_code != 0:
            return False, f"{label}: controller error {result.result().result.error_code}"
        time.sleep(0.5)
        return True, label

    def trigger_service(self, client, label, require_success=True):
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return None, f"{label}: service timeout"
        response = future.result()
        if require_success and not response.success:
            return None, f"{label}: {response.message}"
        return response, response.message

    def entity_position(self):
        request = GetEntityState.Request()
        request.name = self.target_name
        request.reference_frame = "world"
        future = self.get_entity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None or not future.result().success:
            return None
        p = future.result().state.pose.position
        return (p.x, p.y, p.z)

    def reset_target(self, task_point):
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = self.target_name
        request.state.reference_frame = "world"
        request.state.pose.position.x = task_point[0]
        request.state.pose.position.y = task_point[1]
        request.state.pose.position.z = task_point[2] + 0.01
        request.state.pose.orientation.w = 1.0
        future = self.set_entity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None or not future.result().success:
            return False, None
        previous = None
        stable = 0
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            time.sleep(0.2)
            current = self.entity_position()
            if current is None:
                stable = 0
                continue
            in_place = distance(current, task_point) <= 0.003
            stopped = previous is not None and distance(current, previous) <= 0.0005
            stable = stable + 1 if in_place and stopped else 0
            if stable >= 3:
                return True, current
            previous = current
        return False, previous

    def apply_scene(self, world_operation=None, attached_operation=None, world_pose=None):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        if world_operation is not None:
            obj = CollisionObject()
            obj.header.frame_id = "base"
            obj.id = self.target_name
            obj.operation = world_operation
            if world_operation == CollisionObject.ADD:
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [0.04, 0.04, 0.04]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = world_pose
                pose.orientation.w = 1.0
                if hasattr(self, 'collision_pose'):
                    pose = self.collision_pose(attached=False)
                obj.primitives.append(primitive)
                obj.primitive_poses.append(pose)
            request.scene.world.collision_objects.append(obj)
        if attached_operation is not None:
            attached = AttachedCollisionObject()
            attached.link_name = "link6"
            attached.touch_links = list(TOUCH_LINKS)
            attached.object.header.frame_id = "link6"
            attached.object.id = self.target_name
            attached.object.operation = attached_operation
            if attached_operation == CollisionObject.ADD:
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [0.04, 0.04, 0.04]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = GRASP_OFFSET
                pose.orientation.w = 1.0
                if hasattr(self, 'collision_pose'):
                    pose = self.collision_pose(attached=True)
                attached.object.primitives.append(primitive)
                attached.object.primitive_poses.append(pose)
            request.scene.robot_state.is_diff = True
            request.scene.robot_state.attached_collision_objects.append(attached)
        future = self.scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False
        if future.result().success:
            return True
        return world_operation == CollisionObject.REMOVE or attached_operation == CollisionObject.REMOVE

    def allow_target_table_contact(self, allowed):
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False
        matrix = future.result().scene.allowed_collision_matrix
        for name in ("task2_table", self.target_name):
            if name not in matrix.entry_names:
                matrix.entry_names.append(name)
                for row in matrix.entry_values:
                    row.enabled.append(False)
                new_row = AllowedCollisionEntry()
                new_row.enabled = [False] * len(matrix.entry_names)
                matrix.entry_values.append(new_row)
        table = matrix.entry_names.index("task2_table")
        target = matrix.entry_names.index(self.target_name)
        matrix.entry_values[table].enabled[target] = allowed
        matrix.entry_values[target].enabled[table] = allowed
        update = ApplyPlanningScene.Request()
        update.scene.is_diff = True
        update.scene.allowed_collision_matrix = matrix
        applied = self.scene.call_async(update)
        rclpy.spin_until_future_complete(self, applied, timeout_sec=5.0)
        return bool(applied.done() and applied.result() and applied.result().success)

    def execute_pick_place(self, prepared, start_xy, end_xy):
        steps = []
        self.step_records = steps

        def record(label, ok, detail=None):
            row = {"label": label, "ok": bool(ok)}
            if detail is not None:
                row["detail"] = detail
            steps.append(row)
            if hasattr(self, "report_step"):
                self.report_step(row)
            if not ok:
                raise RuntimeError(f"{label}: {detail}")

        response, detail = self.trigger_service(self.detach, "detach reset", False)
        record("detach_reset", response is not None, detail)
        record("remove_attached_scene", self.apply_scene(attached_operation=CollisionObject.REMOVE))
        record("strict_table_contact", self.allow_target_table_contact(False))
        ok, detail = self.move_gripper(OPEN, "open gripper")
        record("open_gripper", ok, detail)
        ok, detail = self.execute_target(HOME, "Home_before_dynamic_task")
        record("home_before_task", ok, detail)
        record("remove_old_world_target", self.apply_scene(world_operation=CollisionObject.REMOVE))

        start_point = (start_xy[0], start_xy[1], OBJECT_Z_M)
        ok, settled = self.reset_target(start_point)
        record("verify_cube_at_detected_grid", ok, None if settled is None else [round(v, 6) for v in settled])
        record(
            "add_start_world_target",
            self.apply_scene(world_operation=CollisionObject.ADD, world_pose=start_point),
        )

        ok, detail = self.execute_target(prepared["start_vertical"][0], "affine_start_safe")
        record("move_to_start_safe", ok, detail)
        record("remove_target_for_grasp", self.apply_scene(world_operation=CollisionObject.REMOVE))
        ok, detail = self.execute_chain(
            prepared["start_vertical"][1:],
            OPEN,
            "descend_to_affine_start",
        )
        record("descend_to_start", ok, detail)

        before_close = self.entity_position()
        record(
            "cube_still_at_start",
            before_close is not None and distance(before_close, start_point) <= 0.006,
            None if before_close is None else [round(v, 6) for v in before_close],
        )
        ok, detail = self.move_gripper(CLOSED, "close gripper")
        record("close_gripper", ok, detail)
        after_close = self.entity_position()
        record(
            "close_did_not_push_cube",
            after_close is not None and distance(before_close, after_close) <= 0.008,
            None if after_close is None else [round(v, 6) for v in after_close],
        )
        check, detail = self.trigger_service(self.check_grasp, "grasp alignment")
        record("grasp_alignment", check is not None, detail)
        match = re.search(r"center_error=([0-9.]+)", detail or "")
        center_error = None if match is None else float(match.group(1))
        record("grasp_center_within_10mm", center_error is not None and center_error <= 0.010, center_error)

        before_attach = self.entity_position()
        attached, detail = self.trigger_service(self.attach, "physical attach")
        record("physical_attach", attached is not None, detail)
        time.sleep(0.5)
        after_attach = self.entity_position()
        snap = None if before_attach is None or after_attach is None else distance(before_attach, after_attach)
        record("attach_snap_within_3mm", snap is not None and snap <= 0.003, snap)
        record("add_attached_collision", self.apply_scene(attached_operation=CollisionObject.ADD))
        record("allow_initial_table_contact", self.allow_target_table_contact(True))

        lift_chain = list(reversed(prepared["start_vertical"][:-1]))
        ok, detail = self.execute_chain(lift_chain, CLOSED, "lift_from_affine_start")
        record("lift_from_start", ok, detail)
        record("restore_strict_table_contact", self.allow_target_table_contact(False))
        lifted = self.entity_position()
        record(
            "cube_lifted",
            lifted is not None and lifted[2] - OBJECT_Z_M >= APPROACH_HEIGHT_M - 0.01,
            None if lifted is None else [round(v, 6) for v in lifted],
        )

        ok, detail = self.execute_chain(prepared["transfer"], CLOSED, "front_arc_bin_transfer")
        record("transfer_to_end", ok, detail)
        hold, detail = self.trigger_service(self.check_grasp, "transfer hold")
        record("transfer_hold", hold is not None, detail)

        ok, detail = self.execute_chain(
            prepared["end_vertical"][1:],
            CLOSED,
            "descend_to_affine_end_release",
        )
        record("descend_to_release", ok, detail)
        expected_release = (end_xy[0], end_xy[1], OBJECT_Z_M + RELEASE_CLEARANCE_M)
        release_pose = self.entity_position()
        record(
            "release_pose_aligned",
            release_pose is not None and distance(release_pose, expected_release) <= 0.018,
            None if release_pose is None else [round(v, 6) for v in release_pose],
        )

        record("remove_attached_collision", self.apply_scene(attached_operation=CollisionObject.REMOVE))
        ok, detail = self.move_gripper(OPEN, "open for release")
        record("open_for_release", ok, detail)
        detached, detail = self.trigger_service(self.detach, "physical detach")
        record("physical_detach", detached is not None, detail)
        time.sleep(2.0)
        final_pose = self.entity_position()
        xy_error = None if final_pose is None else math.hypot(final_pose[0] - end_xy[0], final_pose[1] - end_xy[1])
        z_error = None if final_pose is None else abs(final_pose[2] - OBJECT_Z_M)
        record(
            "final_cube_tolerance",
            xy_error is not None and xy_error <= 0.015 and z_error <= 0.008,
            {
                "pose_m": None if final_pose is None else [round(v, 6) for v in final_pose],
                "xy_error_mm": None if xy_error is None else round(xy_error * 1000.0, 3),
                "z_error_mm": None if z_error is None else round(z_error * 1000.0, 3),
            },
        )
        record(
            "restore_world_collision",
            self.apply_scene(world_operation=CollisionObject.ADD, world_pose=final_pose),
        )

        return_chain = list(reversed(prepared["end_vertical"][:-1]))
        ok, detail = self.execute_chain(return_chain, OPEN, "leave_affine_end")
        record("leave_end", ok, detail)
        ok, detail = self.execute_target(HOME, "Home_after_dynamic_task")
        record("home_after_task", ok, detail)
        return {
            "ok": True,
            "steps": steps,
            "attach_snap_mm": round(snap * 1000.0, 3),
            "final_pose_m": [round(value, 6) for value in final_pose],
            "final_xy_error_mm": round(xy_error * 1000.0, 3),
            "final_z_error_mm": round(z_error * 1000.0, 3),
        }


def main():
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    affine = summary["new_affine"]
    start_user = tuple(args.start)
    end_user = tuple(args.end)
    points = []
    for label, user_point in (("start", start_user), ("end", end_user)):
        classification = affine_base.point_classification(user_point)
        raw_mm, gazebo_xy = affine_base.registered_gazebo_xy_m(user_point, affine, args.mode)
        points.append(
            {
                "label": label,
                "user_xy": list(user_point),
                "classification": classification,
                "distance_outside_hull_user_units": round(
                    affine_base.distance_to_calibration_hull(user_point), 6
                ),
                "distance_outside_hull_physical_mm": round(
                    affine_base.distance_to_calibration_hull(user_point)
                    * affine_base.COORDINATE_DEFINITION["unit_mm"],
                    3,
                ),
                "affine_tcp_xy_mm": [round(value, 6) for value in raw_mm],
                "gazebo_task_xy_m": [round(value, 6) for value in gazebo_xy],
            }
        )

    report = {
        "status": "FAIL",
        "mode": args.mode,
        "executed": bool(args.execute),
        "points": points,
        "affine_quality": {
            "rmse_mm": affine["rmse_mm"],
            "max_error_mm": affine["max_error_mm"],
        },
        "coordinate_definition": affine_base.COORDINATE_DEFINITION,
        "simulation_xy_assumption": (
            "Gazebo object-center XY is set equal to calibrated grasp TCP XY because "
            "the new samples were measured while the gripper held a manually centered object; "
            "the configured 3D grasp offset then converts object center to link6 pose."
        ),
    }
    outside = [row["label"] for row in points if row["classification"] == "outside_calibration_hull"]
    if outside and not args.allow_outside_hull:
        report["failure"] = f"outside calibration hull: {','.join(outside)}"
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print("DYNAMIC_AFFINE_PICK_PLACE_REJECT " + report["failure"])
        print(rendered)
        return 2

    rclpy.init()
    node = DynamicAffinePickPlace()
    try:
        if not node.ready_full():
            raise RuntimeError("Gazebo/MoveIt pick-place services are unavailable")
        start_xy = tuple(points[0]["gazebo_task_xy_m"])
        end_xy = tuple(points[1]["gazebo_task_xy_m"])
        prepared, preflight = node.preflight(start_xy, end_xy)
        report["preflight"] = preflight
        if prepared is None:
            raise RuntimeError("dynamic IK preflight failed")
        if args.execute:
            report["execution"] = node.execute_pick_place(prepared, start_xy, end_xy)
        report["status"] = "PASS"
        print(
            "DYNAMIC_AFFINE_PICK_PLACE_OK "
            f"start={start_user} end={end_user} execute={args.execute}"
        )
    except RuntimeError as exc:
        report["failure"] = str(exc)
        print(f"DYNAMIC_AFFINE_PICK_PLACE_FAIL reason={exc}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
