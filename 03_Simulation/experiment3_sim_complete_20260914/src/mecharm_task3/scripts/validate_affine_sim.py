#!/usr/bin/env python3
"""Validate the experiment-2 planar affine mapping in the Gazebo MoveIt stack.

The script deliberately validates only a collision-free hover transfer.  It does
not touch the real robot and it does not yet move the Gazebo cube.  This keeps
coordinate conversion, reachability and planning separate from grasp physics.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    RobotState,
)
from moveit_msgs.srv import (
    GetMotionPlan,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
GRIPPER_JOINT = "gripper_controller"
OPEN_GRIPPER = 0.15
HOME = [0.0] * 6
IK_BRANCH_LIMIT_RAD = math.radians(120.0)

# This 3 cm hover state is the top of the collision-checked A vertical chain
# that has already completed the Gazebo A->B task.  Its FK pose is preferable
# to naively translating the grasp pose upward, because that exact translated
# pose can fall outside the IK manifold of this model.
REFERENCE_HOVER_JOINTS = [
    0.3909269,
    1.1909292,
    -1.5237074,
    0.0907308,
    1.8540972,
    -1.1523635,
]
REFERENCE_GAZEBO_POINT_M = (0.18, 0.08)
REFERENCE_USER_POINT = (0.0, 31.0)

# Convex hull of the 12-point official data set, clockwise.
CALIBRATION_HULL = [
    (-12.0, 25.0),
    (-6.0, 31.0),
    (-4.0, 32.0),
    (3.0, 32.0),
    (17.0, 31.0),
]
OFFICIAL_SAMPLES = {
    (0.0, 31.0),
    (3.0, 32.0),
    (7.0, 31.0),
    (11.0, 31.0),
    (17.0, 31.0),
    (-1.0, 32.0),
    (-4.0, 32.0),
    (-6.0, 31.0),
    (-2.0, 29.0),
    (-5.0, 28.0),
    (-7.0, 27.0),
    (-12.0, 25.0),
}
COORDINATE_DEFINITION = {
    "coordinate_system_id": "mecharm_user_plane_v1",
    "unit_mm": 5.0,
    "unit_description": "one input coordinate unit equals 0.5 cm",
    "origin": "center of the robot-base edge facing the arm/end-effector side",
    "positive_x": "rightward along that base edge when facing the arm side",
    "positive_y": "from the base toward the arm/end-effector side",
    "input_point_semantics": "object center",
    "gripper_relation": "physical setup aims to align the object center with the gripper center",
    "calibration_measurement_state": (
        "all 12 new physical samples were recorded while the gripper was holding "
        "an object resting at the ground/work plane"
    ),
    "affine_output_semantics": "gripper TCP XY at a successful ground-grasp state",
    "tcp_to_object_center_xy_offset_mm": [0.0, 0.0],
    "tcp_object_offset_status": "assumed zero from manual center alignment; calibrate later if bias is observed",
    "fit_dataset": "new_official_12_points_only",
    "old_dataset_in_fit": False,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=2, type=float, default=(0.0, 31.0))
    parser.add_argument("--end", nargs=2, type=float, default=(7.0, 31.0))
    parser.add_argument(
        "--mode",
        choices=("direct", "anchored"),
        default="anchored",
        help=(
            "direct: affine millimetres become Gazebo base metres; "
            "anchored: preserve affine deltas but register user (0,31) to Gazebo A"
        ),
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-outside-hull", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def inside_convex_hull(point, tolerance=1.0e-9):
    values = [
        cross(CALIBRATION_HULL[i], CALIBRATION_HULL[(i + 1) % len(CALIBRATION_HULL)], point)
        for i in range(len(CALIBRATION_HULL))
    ]
    return all(value <= tolerance for value in values) or all(
        value >= -tolerance for value in values
    )


def point_classification(point):
    if tuple(point) in OFFICIAL_SAMPLES:
        return "official_sample"
    if inside_convex_hull(point):
        return "inside_calibration_hull"
    return "outside_calibration_hull"


def distance_to_calibration_hull(point):
    """Return zero inside/on the hull, otherwise Euclidean user-coordinate distance."""
    if inside_convex_hull(point):
        return 0.0
    best = float("inf")
    for index, start in enumerate(CALIBRATION_HULL):
        end = CALIBRATION_HULL[(index + 1) % len(CALIBRATION_HULL)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        ratio = 0.0 if length_sq == 0.0 else (
            (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
        ) / length_sq
        ratio = min(1.0, max(0.0, ratio))
        nearest = (start[0] + ratio * dx, start[1] + ratio * dy)
        best = min(best, math.hypot(point[0] - nearest[0], point[1] - nearest[1]))
    return best


def affine_xy_mm(point, affine):
    x, y = point
    x_eq = affine["x_equation"]
    y_eq = affine["y_equation"]
    return (
        x_eq["user_x"] * x + x_eq["user_y"] * y + x_eq["intercept"],
        y_eq["user_x"] * x + y_eq["user_y"] * y + y_eq["intercept"],
    )


def registered_gazebo_xy_m(point, affine, mode):
    raw_mm = affine_xy_mm(point, affine)
    raw_m = (raw_mm[0] / 1000.0, raw_mm[1] / 1000.0)
    if mode == "direct":
        return raw_mm, raw_m
    reference_mm = affine_xy_mm(REFERENCE_USER_POINT, affine)
    translation = (
        REFERENCE_GAZEBO_POINT_M[0] - reference_mm[0] / 1000.0,
        REFERENCE_GAZEBO_POINT_M[1] - reference_mm[1] / 1000.0,
    )
    return raw_mm, (raw_m[0] + translation[0], raw_m[1] + translation[1])


def degrees(values):
    return [round(math.degrees(value), 3) for value in values]


class AffineSimulationValidator(Node):
    def __init__(self):
        super().__init__("validate_affine_sim")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        self.plan = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.move = ActionClient(self, MoveGroup, "/move_action")

    def ready(self):
        checks = (
            self.fk.wait_for_service(timeout_sec=20.0),
            self.ik.wait_for_service(timeout_sec=20.0),
            self.validity.wait_for_service(timeout_sec=20.0),
            self.plan.wait_for_service(timeout_sec=20.0),
        )
        return all(checks)

    @staticmethod
    def robot_state(arm_positions):
        state = RobotState()
        state.is_diff = False
        state.joint_state = JointState()
        state.joint_state.name = list(ARM_JOINTS) + [GRIPPER_JOINT]
        state.joint_state.position = list(arm_positions) + [OPEN_GRIPPER]
        return state

    def reference_pose(self):
        request = GetPositionFK.Request()
        request.header.frame_id = "base"
        request.fk_link_names = ["link6"]
        request.robot_state = self.robot_state(REFERENCE_HOVER_JOINTS)
        future = self.fk.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            return None, "FK timeout"
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS or not response.pose_stamped:
            return None, f"FK error {response.error_code.val}"
        return response.pose_stamped[0], None

    def solve_ik(self, pose, seed):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "arm"
        ik.avoid_collisions = True
        ik.ik_link_name = "link6"
        ik.robot_state = self.robot_state(seed)
        ik.pose_stamped = pose
        ik.timeout.sec = 4
        ik.constraints.name = "stay_on_nearby_ik_branch"
        for name, position in zip(ARM_JOINTS, seed):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(position)
            joint.tolerance_above = IK_BRANCH_LIMIT_RAD
            joint.tolerance_below = IK_BRANCH_LIMIT_RAD
            joint.weight = 1.0
            ik.constraints.joint_constraints.append(joint)
        future = self.ik.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if not future.done() or future.result() is None:
            return None, "IK timeout", None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"IK error {response.error_code.val}", None
        solved = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        if any(name not in solved for name in ARM_JOINTS):
            return None, "IK response omitted one or more arm joints", None
        solution = [solved[name] for name in ARM_JOINTS]
        max_delta = max(
            abs(math.atan2(math.sin(value - origin), math.cos(value - origin)))
            for value, origin in zip(solution, seed)
        )
        if max_delta > IK_BRANCH_LIMIT_RAD + 1.0e-6:
            return (
                None,
                f"IK branch jump {math.degrees(max_delta):.1f} deg exceeds 120 deg",
                math.degrees(max_delta),
            )
        return solution, None, math.degrees(max_delta)

    def state_is_valid(self, arm_positions):
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state = self.robot_state(arm_positions)
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

    @staticmethod
    def goal_constraints(target, label):
        constraints = Constraints()
        constraints.name = label
        for name, position in zip(ARM_JOINTS, target):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(position)
            joint.tolerance_above = 0.008
            joint.tolerance_below = 0.008
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def plan_segment(self, start, target, label):
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "arm"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnectkConfigDefault"
        motion.num_planning_attempts = 8
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = 0.10
        motion.max_acceleration_scaling_factor = 0.10
        motion.start_state = self.robot_state(start)
        motion.goal_constraints = [self.goal_constraints(target, label)]
        future = self.plan.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            return {"label": label, "ok": False, "reason": "planning timeout"}
        response = future.result().motion_plan_response
        count = len(response.trajectory.joint_trajectory.points)
        ok = response.error_code.val == MoveItErrorCodes.SUCCESS and count > 0
        return {
            "label": label,
            "ok": ok,
            "error_code": int(response.error_code.val),
            "trajectory_points": count,
            "planning_time_s": round(float(response.planning_time), 4),
        }

    def execute_target(self, target, label):
        if not self.move.wait_for_server(timeout_sec=20.0):
            return False, "MoveGroup action unavailable"
        goal = MoveGroup.Goal()
        motion = goal.request
        motion.group_name = "arm"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnectkConfigDefault"
        motion.num_planning_attempts = 8
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = 0.16
        motion.max_acceleration_scaling_factor = 0.16
        motion.start_state.is_diff = True
        motion.goal_constraints = [self.goal_constraints(target, label)]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        sent = self.move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=12.0)
        if not sent.done() or sent.result() is None or not sent.result().accepted:
            return False, "goal rejected or timed out"
        self.active_goal = sent.result()
        result = self.active_goal.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=60.0)
        if not result.done() or result.result() is None:
            return False, "execution timeout"
        code = int(result.result().result.error_code.val)
        if code == MoveItErrorCodes.SUCCESS:
            return True, f"MoveIt success ({code})"
        return False, f"MoveIt error {code}"


def pose_as_dict(pose):
    p = pose.pose.position
    q = pose.pose.orientation
    return {
        "frame": pose.header.frame_id,
        "position_m": [round(p.x, 6), round(p.y, 6), round(p.z, 6)],
        "quaternion_xyzw": [round(q.x, 7), round(q.y, 7), round(q.z, 7), round(q.w, 7)],
    }


def main():
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    affine = summary["new_affine"]
    start = tuple(args.start)
    end = tuple(args.end)
    point_rows = []
    for label, point in (("start", start), ("end", end)):
        classification = point_classification(point)
        raw_mm, gazebo_m = registered_gazebo_xy_m(point, affine, args.mode)
        point_rows.append(
            {
                "label": label,
                "user_xy": [point[0], point[1]],
                "classification": classification,
                "distance_outside_hull_user_units": round(
                    distance_to_calibration_hull(point), 6
                ),
                "affine_tcp_xy_mm": [round(raw_mm[0], 6), round(raw_mm[1], 6)],
                "gazebo_task_xy_m": [round(gazebo_m[0], 6), round(gazebo_m[1], 6)],
            }
        )

    outside = [row["label"] for row in point_rows if row["classification"] == "outside_calibration_hull"]
    if outside and not args.allow_outside_hull:
        print(f"AFFINE_SIM_REJECT outside_hull={','.join(outside)}")
        return 2

    rclpy.init()
    node = AffineSimulationValidator()
    report = {
        "status": "FAIL",
        "mode": args.mode,
        "executed": bool(args.execute),
        "affine_quality": {
            "rmse_mm": affine["rmse_mm"],
            "max_error_mm": affine["max_error_mm"],
        },
        "coordinate_definition": COORDINATE_DEFINITION,
        "registration": (
            "none"
            if args.mode == "direct"
            else "translation_only: user (0,31) registered to Gazebo A (0.18,0.08)m"
        ),
        "points": point_rows,
    }
    try:
        if not node.ready():
            report["failure"] = "MoveIt services unavailable"
            raise RuntimeError(report["failure"])

        reference, failure = node.reference_pose()
        if reference is None:
            report["failure"] = failure
            raise RuntimeError(failure)
        report["reference_link6_pose"] = pose_as_dict(reference)

        targets = []
        seed = list(REFERENCE_HOVER_JOINTS)
        for row in point_rows:
            pose = copy.deepcopy(reference)
            pose.header.frame_id = "base"
            pose.pose.position.x += row["gazebo_task_xy_m"][0] - REFERENCE_GAZEBO_POINT_M[0]
            pose.pose.position.y += row["gazebo_task_xy_m"][1] - REFERENCE_GAZEBO_POINT_M[1]
            solution, failure, branch_delta_deg = node.solve_ik(pose, seed)
            if solution is None:
                row["ik_ok"] = False
                row["ik_failure"] = failure
                report["failure"] = f"{row['label']}: {failure}"
                raise RuntimeError(report["failure"])
            valid, contacts = node.state_is_valid(solution)
            row["ik_ok"] = True
            row["max_wrapped_joint_delta_from_seed_deg"] = round(branch_delta_deg, 3)
            row["state_valid"] = valid
            row["contacts"] = contacts
            row["hover_link6_pose"] = pose_as_dict(pose)
            row["joint_angles_deg"] = degrees(solution)
            if not valid:
                report["failure"] = f"{row['label']}: IK state is in collision"
                raise RuntimeError(report["failure"])
            targets.append(solution)
            seed = solution

        segments = [
            node.plan_segment(HOME, targets[0], "home_to_start_hover"),
            node.plan_segment(targets[0], targets[1], "start_to_end_hover"),
            node.plan_segment(targets[1], HOME, "end_hover_to_home"),
        ]
        report["plans"] = segments
        if not all(segment["ok"] for segment in segments):
            report["failure"] = "one or more MoveIt plans failed"
            raise RuntimeError(report["failure"])

        if args.execute:
            executed = []
            for target, label in (
                (targets[0], "affine_start_hover"),
                (targets[1], "affine_end_hover"),
                (HOME, "Home"),
            ):
                ok, detail = node.execute_target(target, label)
                executed.append({"label": label, "ok": ok, "detail": detail})
                if not ok:
                    report["executions"] = executed
                    report["failure"] = f"execution failed at {label}: {detail}"
                    raise RuntimeError(report["failure"])
            report["executions"] = executed

        report["status"] = "PASS"
        print(
            "AFFINE_SIM_OK "
            f"mode={args.mode} start={start} end={end} "
            f"execute={args.execute}"
        )
    except RuntimeError as exc:
        print(f"AFFINE_SIM_FAIL mode={args.mode} reason={exc}")
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
