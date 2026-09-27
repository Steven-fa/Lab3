"""Execute one complete fixed-point pick-and-place cycle in simulation."""

from __future__ import annotations

import math
import time
from typing import Iterable

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class TaskExecutionError(RuntimeError):
    """Raised when a required planning, controller, or Gazebo operation fails."""


def quaternion_from_rpy(rpy: Iterable[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_conjugate(
    quaternion: Iterable[float],
) -> tuple[float, float, float, float]:
    x, y, z, w = [float(value) for value in quaternion]
    return (-x, -y, -z, w)


def quaternion_multiply(
    first: Iterable[float], second: Iterable[float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = [float(value) for value in first]
    bx, by, bz, bw = [float(value) for value in second]
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(
    quaternion: Iterable[float], vector: Iterable[float]
) -> tuple[float, float, float]:
    """Rotate a vector by a unit quaternion in xyzw order."""
    qx, qy, qz, qw = [float(value) for value in quaternion]
    vx, vy, vz = [float(value) for value in vector]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def end_effector_position_for_grasp(
    grasp_position: Iterable[float],
    orientation: Iterable[float],
    grasp_center_offset: Iterable[float],
) -> list[float]:
    """Return the end-effector origin that puts its grasp center on a world point."""
    grasp = [float(value) for value in grasp_position]
    world_offset = rotate_vector(orientation, grasp_center_offset)
    return [grasp[index] - world_offset[index] for index in range(3)]


def set_allowed_collision(
    matrix: AllowedCollisionMatrix,
    first_name: str,
    second_name: str,
    allowed: bool,
) -> None:
    """Set one symmetric ACM pair while preserving every existing entry."""
    for name in (first_name, second_name):
        if name in matrix.entry_names:
            continue
        matrix.entry_names.append(name)
        for row in matrix.entry_values:
            row.enabled.append(False)
        matrix.entry_values.append(
            AllowedCollisionEntry(enabled=[False] * len(matrix.entry_names))
        )

    first_index = matrix.entry_names.index(first_name)
    second_index = matrix.entry_names.index(second_name)
    matrix.entry_values[first_index].enabled[second_index] = allowed
    matrix.entry_values[second_index].enabled[first_index] = allowed


class PickPlaceTask(Node):
    def __init__(self, node_name: str = "pick_place_task") -> None:
        super().__init__(node_name)

        self.declare_parameter("frame_id", "base")
        self.declare_parameter("table_id", "workbench")
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("end_effector_link", "gripper_base")
        self.declare_parameter("move_action_name", "/move_action")
        self.declare_parameter("cartesian_path_service", "/compute_cartesian_path")
        self.declare_parameter("execute_trajectory_action", "/execute_trajectory")
        self.declare_parameter("cartesian_max_step", 0.005)
        self.declare_parameter(
            "arm_trajectory_action_name", "/arm_controller/follow_joint_trajectory"
        )
        self.declare_parameter(
            "gripper_action_name", "/adaptive_gripper_controller/gripper_cmd"
        )
        self.declare_parameter("apply_scene_service", "/apply_planning_scene")
        self.declare_parameter("get_entity_service", "/gazebo/get_entity_state")
        self.declare_parameter("set_entity_service", "/gazebo/set_entity_state")
        self.declare_parameter("robot_model_name", "mecharm_270_m5")
        self.declare_parameter("gazebo_reference_link", "link6")
        self.declare_parameter(
            "gazebo_reference_to_end_effector_position", [0.0, 0.0, 0.038]
        )
        self.declare_parameter(
            "gazebo_reference_to_end_effector_rpy", [1.579, 0.0, 0.0]
        )
        self.declare_parameter("object_id", "workpiece")
        self.declare_parameter("object_position", [0.18, -0.10, 0.020])
        self.declare_parameter("object_size", [0.035, 0.035, 0.050])
        self.declare_parameter("planning_object_size", [0.033, 0.033, 0.047])
        self.declare_parameter("pick_position", [0.18, -0.10, 0.020])
        self.declare_parameter("pick_orientation_rpy", [-math.pi / 2.0, 0.0, 0.0])
        self.declare_parameter("place_position", [0.18, 0.10, 0.020])
        self.declare_parameter(
            "place_orientation_rpy", [-math.pi / 2.0, 0.0, math.pi]
        )
        self.declare_parameter("grasp_center_offset", [0.0, 0.030, -0.055])
        self.declare_parameter("max_grasp_alignment_error", 0.020)
        self.declare_parameter("initial_lift_distance", 0.040)
        self.declare_parameter("safe_height", 0.09)
        self.declare_parameter("gripper_open", 0.10)
        self.declare_parameter("gripper_closed", -0.70)
        self.declare_parameter("gripper_max_effort", 12.0)
        self.declare_parameter("velocity_scaling", 0.10)
        self.declare_parameter("acceleration_scaling", 0.10)
        self.declare_parameter("planning_time", 8.0)
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("position_tolerance", 0.008)
        self.declare_parameter("orientation_tolerance", 0.10)
        self.declare_parameter("startup_delay", 3.0)
        self.declare_parameter("step_pause", 0.5)
        self.declare_parameter(
            "home_joint_names",
            [
                "joint1_to_base",
                "joint2_to_joint1",
                "joint3_to_joint2",
                "joint4_to_joint3",
                "joint5_to_joint4",
                "joint6_to_joint5",
            ],
        )
        self.declare_parameter(
            "home_joint_positions", [0.0, 0.35, -0.70, 0.35, 0.0, 0.0]
        )
        self.declare_parameter(
            "touch_links",
            [
                "gripper_base",
                "gripper_left1",
                "gripper_left2",
                "gripper_left3",
                "gripper_right1",
                "gripper_right2",
                "gripper_right3",
            ],
        )
        self.declare_parameter(
            "support_touch_links",
            [
                "gripper_left1",
                "gripper_left2",
                "gripper_right1",
                "gripper_right2",
            ],
        )

        self.move_client = ActionClient(
            self,
            MoveGroup,
            str(self.get_parameter("move_action_name").value),
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            str(self.get_parameter("execute_trajectory_action").value),
        )
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            str(self.get_parameter("gripper_action_name").value),
        )
        self.arm_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("arm_trajectory_action_name").value),
        )
        self.scene_client = self.create_client(
            ApplyPlanningScene,
            str(self.get_parameter("apply_scene_service").value),
        )
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            str(self.get_parameter("cartesian_path_service").value),
        )
        self.get_scene_client = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.get_entity_client = self.create_client(
            GetEntityState,
            str(self.get_parameter("get_entity_service").value),
        )
        self.set_entity_client = self.create_client(
            SetEntityState,
            str(self.get_parameter("set_entity_service").value),
        )

        self._gazebo_attached = False
        self._relative_object_pose = Pose()
        self._relative_object_pose.orientation.w = 1.0
        self._gazebo_relative_object_pose = Pose()
        self._gazebo_relative_object_pose.orientation.w = 1.0
        self._set_state_future = None
        self._last_move_state = ""
        self.create_timer(0.04, self._follow_gripper)

    def values(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def _wait_for_interfaces(self) -> None:
        self.get_logger().info("等待 MoveIt、机械臂/夹爪控制器和 Gazebo 服务……")
        if not self.move_client.wait_for_server(timeout_sec=45.0):
            raise TaskExecutionError("MoveIt action /move_action 不可用")
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=45.0):
            raise TaskExecutionError("MoveIt action /execute_trajectory 不可用")
        if not self.arm_trajectory_client.wait_for_server(timeout_sec=45.0):
            raise TaskExecutionError("机械臂轨迹控制 Action 不可用")
        if not self.gripper_client.wait_for_server(timeout_sec=45.0):
            raise TaskExecutionError("夹爪控制 Action 不可用")
        services = [
            (self.scene_client, "MoveIt 规划场景服务"),
            (self.cartesian_path_client, "MoveIt 笛卡尔路径服务"),
            (self.get_scene_client, "MoveIt 规划场景读取服务"),
            (self.get_entity_client, "Gazebo 获取模型状态服务"),
            (self.set_entity_client, "Gazebo 设置模型状态服务"),
        ]
        for client, label in services:
            if not client.wait_for_service(timeout_sec=45.0):
                raise TaskExecutionError(f"{label}不可用")

    def _wait_for_workpiece(self) -> None:
        object_id = str(self.get_parameter("object_id").value)
        deadline = time.monotonic() + 30.0
        while rclpy.ok() and time.monotonic() < deadline:
            request = GetEntityState.Request()
            request.name = object_id
            request.reference_frame = "world"
            future = self.get_entity_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.done() and future.result() is not None and future.result().success:
                return
            self._wait(0.5)
        raise TaskExecutionError(f"Gazebo 中未找到工件模型 {object_id}")

    def _pose_constraints(
        self, position: list[float], orientation: tuple[float, float, float, float]
    ) -> Constraints:
        frame_id = str(self.get_parameter("frame_id").value)
        link_name = str(self.get_parameter("end_effector_link").value)

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = frame_id
        position_constraint.link_name = link_name
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [float(self.get_parameter("position_tolerance").value)]
        target_pose = Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = position
        target_pose.orientation.w = 1.0
        position_constraint.constraint_region.primitives = [region]
        position_constraint.constraint_region.primitive_poses = [target_pose]
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = frame_id
        orientation_constraint.link_name = link_name
        (
            orientation_constraint.orientation.x,
            orientation_constraint.orientation.y,
            orientation_constraint.orientation.z,
            orientation_constraint.orientation.w,
        ) = orientation
        tolerance = float(self.get_parameter("orientation_tolerance").value)
        orientation_constraint.absolute_x_axis_tolerance = tolerance
        orientation_constraint.absolute_y_axis_tolerance = tolerance
        orientation_constraint.absolute_z_axis_tolerance = tolerance
        orientation_constraint.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints = [position_constraint]
        constraints.orientation_constraints = [orientation_constraint]
        return constraints

    def _joint_constraints(self) -> Constraints:
        names = [str(value) for value in self.get_parameter("home_joint_names").value]
        positions = self.values("home_joint_positions")
        constraints = Constraints()
        for name, position in zip(names, positions):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = 0.01
            joint.tolerance_below = 0.01
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def _move_goal(self, constraints: Constraints) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request.group_name = str(self.get_parameter("planning_group").value)
        goal.request.num_planning_attempts = int(
            self.get_parameter("planning_attempts").value
        )
        goal.request.allowed_planning_time = float(
            self.get_parameter("planning_time").value
        )
        goal.request.max_velocity_scaling_factor = float(
            self.get_parameter("velocity_scaling").value
        )
        goal.request.max_acceleration_scaling_factor = float(
            self.get_parameter("acceleration_scaling").value
        )
        goal.request.pipeline_id = "ompl"
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay = 0.5
        return goal

    def _move_feedback(self, message) -> None:
        state = str(message.feedback.state)
        if state and state != self._last_move_state:
            self._last_move_state = state
            self.get_logger().info(f"MoveIt 状态：{state}")

    def _execute_move(self, label: str, constraints: Constraints) -> None:
        self.get_logger().info(f"执行动作：{label}")
        self._last_move_state = ""
        send_future = self.move_client.send_goal_async(
            self._move_goal(constraints), feedback_callback=self._move_feedback
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        goal_handle = send_future.result() if send_future.done() else None
        if goal_handle is None or not goal_handle.accepted:
            raise TaskExecutionError(f"MoveIt 拒绝动作：{label}")
        result_future = goal_handle.get_result_async()
        timeout = float(self.get_parameter("planning_time").value) + 60.0
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        wrapped_result = result_future.result() if result_future.done() else None
        if wrapped_result is None:
            goal_handle.cancel_goal_async()
            raise TaskExecutionError(f"动作超时：{label}")
        error_code = wrapped_result.result.error_code.val
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or error_code != MoveItErrorCodes.SUCCESS
        ):
            raise TaskExecutionError(
                f"动作失败：{label}，MoveIt 错误码 {error_code}，"
                f"Action 状态 {wrapped_result.status}"
            )
        self.get_logger().info(f"动作完成：{label}")

    def _move_pose(
        self,
        label: str,
        position: list[float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self._execute_move(label, self._pose_constraints(position, orientation))

    def _move_cartesian_pose(
        self,
        label: str,
        position: list[float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        """Compute and execute one collision-checked straight Cartesian segment."""
        self.get_logger().info(f"执行笛卡尔直线动作：{label}")
        waypoint = Pose()
        waypoint.position.x, waypoint.position.y, waypoint.position.z = position
        (
            waypoint.orientation.x,
            waypoint.orientation.y,
            waypoint.orientation.z,
            waypoint.orientation.w,
        ) = orientation

        request = GetCartesianPath.Request()
        request.header.frame_id = str(self.get_parameter("frame_id").value)
        request.start_state.is_diff = True
        request.group_name = str(self.get_parameter("planning_group").value)
        request.link_name = str(self.get_parameter("end_effector_link").value)
        request.waypoints = [waypoint]
        request.max_step = float(self.get_parameter("cartesian_max_step").value)
        request.jump_threshold = 0.0
        request.prismatic_jump_threshold = 0.0
        request.revolute_jump_threshold = 0.0
        request.avoid_collisions = True

        plan_future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, plan_future, timeout_sec=15.0)
        response = plan_future.result() if plan_future.done() else None
        if response is None:
            raise TaskExecutionError(f"笛卡尔路径规划超时：{label}")
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or response.fraction < 0.999
        ):
            raise TaskExecutionError(
                f"笛卡尔路径不完整：{label}，完成比例 {response.fraction:.3f}，"
                f"MoveIt 错误码 {response.error_code.val}"
            )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        send_future = self.execute_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        goal_handle = send_future.result() if send_future.done() else None
        if goal_handle is None or not goal_handle.accepted:
            raise TaskExecutionError(f"MoveIt 拒绝笛卡尔轨迹：{label}")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        wrapped_result = result_future.result() if result_future.done() else None
        if (
            wrapped_result is None
            or wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped_result.result.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            raise TaskExecutionError(f"笛卡尔轨迹执行失败：{label}")
        self.get_logger().info(f"笛卡尔直线动作完成：{label}")

    def _move_grasp_pose(
        self,
        label: str,
        grasp_position: list[float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        end_effector_position = end_effector_position_for_grasp(
            grasp_position,
            orientation,
            self.values("grasp_center_offset"),
        )
        self._move_pose(label, end_effector_position, orientation)

    def _command_gripper(self, label: str, position: float) -> None:
        self.get_logger().info(f"执行动作：{label}")
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(self.get_parameter("gripper_max_effort").value)
        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result() if send_future.done() else None
        if goal_handle is None or not goal_handle.accepted:
            raise TaskExecutionError(f"夹爪控制器拒绝动作：{label}")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
        wrapped_result = result_future.result() if result_future.done() else None
        if wrapped_result is None or wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise TaskExecutionError(f"夹爪动作失败或超时：{label}")
        self.get_logger().info(f"动作完成：{label}")
        self._wait(float(self.get_parameter("step_pause").value))

    def _apply_scene(
        self, scene: PlanningScene, label: str, allow_noop: bool = False
    ) -> None:
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result() if future.done() else None
        if response is None:
            raise TaskExecutionError(f"MoveIt 规划场景更新失败：{label}")
        if not response.success:
            if allow_noop:
                self.get_logger().warning(
                    f"MoveIt 规划场景未发生变化，继续执行：{label}"
                )
                return
            raise TaskExecutionError(f"MoveIt 规划场景更新失败：{label}")

    def _remove_world_object(self) -> None:
        collision = CollisionObject()
        collision.header.frame_id = str(self.get_parameter("frame_id").value)
        collision.id = str(self.get_parameter("object_id").value)
        collision.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [collision]
        self._apply_scene(scene, "允许夹爪接近工件", allow_noop=True)

    def _set_support_contact_allowed(self, allowed: bool) -> None:
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result() if future.done() else None
        if response is None:
            raise TaskExecutionError("无法读取 MoveIt 允许碰撞矩阵")

        matrix = response.scene.allowed_collision_matrix
        table_id = str(self.get_parameter("table_id").value)
        for link in self.get_parameter("support_touch_links").value:
            set_allowed_collision(matrix, str(link), table_id, allowed)

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        state = "允许" if allowed else "恢复禁止"
        self._apply_scene(scene, f"{state}指尖接触工作台")

    def _set_object_support_contact_allowed(self, allowed: bool) -> None:
        """Temporarily permit only the held object to leave its support."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result() if future.done() else None
        if response is None:
            raise TaskExecutionError("无法读取 MoveIt 允许碰撞矩阵")

        matrix = response.scene.allowed_collision_matrix
        object_id = str(self.get_parameter("object_id").value)
        table_id = str(self.get_parameter("table_id").value)
        set_allowed_collision(matrix, object_id, table_id, allowed)

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        state = "允许初始离台" if allowed else "恢复工件与工作台碰撞检测"
        self._apply_scene(scene, state)

    def _calculate_relative_object_pose(self) -> Pose:
        pick_q = quaternion_from_rpy(self.values("pick_orientation_rpy"))
        relative = Pose()
        (
            relative.position.x,
            relative.position.y,
            relative.position.z,
        ) = self.values("grasp_center_offset")
        relative.orientation = Quaternion(
            x=-pick_q[0], y=-pick_q[1], z=-pick_q[2], w=pick_q[3]
        )
        return relative

    def _calculate_gazebo_relative_object_pose(self) -> Pose:
        """Convert the gripper-relative pose to Gazebo's uncollapsed link6 frame."""
        reference_to_gripper = self.values(
            "gazebo_reference_to_end_effector_position"
        )
        reference_to_gripper_q = quaternion_from_rpy(
            self.values("gazebo_reference_to_end_effector_rpy")
        )
        gripper_to_object = [
            self._relative_object_pose.position.x,
            self._relative_object_pose.position.y,
            self._relative_object_pose.position.z,
        ]
        rotated_offset = rotate_vector(reference_to_gripper_q, gripper_to_object)
        gripper_to_object_q = (
            self._relative_object_pose.orientation.x,
            self._relative_object_pose.orientation.y,
            self._relative_object_pose.orientation.z,
            self._relative_object_pose.orientation.w,
        )
        reference_to_object_q = quaternion_multiply(
            reference_to_gripper_q, gripper_to_object_q
        )

        relative = Pose()
        relative.position.x = reference_to_gripper[0] + rotated_offset[0]
        relative.position.y = reference_to_gripper[1] + rotated_offset[1]
        relative.position.z = reference_to_gripper[2] + rotated_offset[2]
        relative.orientation = Quaternion(
            x=reference_to_object_q[0],
            y=reference_to_object_q[1],
            z=reference_to_object_q[2],
            w=reference_to_object_q[3],
        )
        return relative

    def _gazebo_reference_name(self) -> str:
        return (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('gazebo_reference_link').value}"
        )

    def _capture_actual_gazebo_grasp_pose(self) -> None:
        """Reject distant suction and preserve the actual no-jump grasp transform."""
        expected = self._calculate_gazebo_relative_object_pose()
        request = GetEntityState.Request()
        request.name = str(self.get_parameter("object_id").value)
        request.reference_frame = self._gazebo_reference_name()
        future = self.get_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise TaskExecutionError("无法读取闭爪瞬间的工件相对位姿")

        actual = response.state.pose
        error = math.sqrt(
            (actual.position.x - expected.position.x) ** 2
            + (actual.position.y - expected.position.y) ** 2
            + (actual.position.z - expected.position.z) ** 2
        )
        max_error = float(self.get_parameter("max_grasp_alignment_error").value)
        self.get_logger().info(f"闭爪对准误差：{error * 1000.0:.1f} mm")
        if error > max_error:
            raise TaskExecutionError(
                f"夹爪没有对准工件（误差 {error * 1000.0:.1f} mm，"
                f"上限 {max_error * 1000.0:.1f} mm），拒绝浮空吸附"
            )
        self._gazebo_relative_object_pose = actual

    def _attach_planning_object(self) -> None:
        object_id = str(self.get_parameter("object_id").value)
        end_effector = str(self.get_parameter("end_effector_link").value)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = self.values("planning_object_size")

        attached = AttachedCollisionObject()
        attached.link_name = end_effector
        attached.object.header.frame_id = end_effector
        attached.object.id = object_id
        attached.object.primitives = [primitive]
        attached.object.primitive_poses = [self._relative_object_pose]
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = [
            str(value) for value in self.get_parameter("touch_links").value
        ]

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        self._apply_scene(scene, "附着工件")

    def _remove_attached_planning_object(self) -> None:
        object_id = str(self.get_parameter("object_id").value)
        end_effector = str(self.get_parameter("end_effector_link").value)
        attached = AttachedCollisionObject()
        attached.link_name = end_effector
        attached.object.id = object_id
        attached.object.operation = CollisionObject.REMOVE

        # Detaching may place the collision object back into the world at the
        # current gripper pose. Remove that copy in the same scene diff so it
        # cannot collide with the still-closed gripper during final descent.
        world_object = CollisionObject()
        world_object.header.frame_id = str(self.get_parameter("frame_id").value)
        world_object.id = object_id
        world_object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        scene.world.collision_objects = [world_object]
        self._apply_scene(scene, "准备将工件放回台面")

    def _add_world_object(self, release_position: list[float]) -> None:
        object_id = str(self.get_parameter("object_id").value)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = self.values("planning_object_size")
        world_object = CollisionObject()
        world_object.header.frame_id = str(self.get_parameter("frame_id").value)
        world_object.id = object_id
        world_object.primitives = [primitive]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = release_position
        pose.orientation.w = 1.0
        world_object.primitive_poses = [pose]
        world_object.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [world_object]
        self._apply_scene(scene, "释放工件")

    def _entity_request(self, pose: Pose, reference_frame: str) -> SetEntityState.Request:
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = str(self.get_parameter("object_id").value)
        request.state.reference_frame = reference_frame
        request.state.pose = pose
        return request

    def _follow_gripper(self) -> None:
        if not self._gazebo_attached:
            return
        if self._set_state_future is not None and not self._set_state_future.done():
            return
        reference = self._gazebo_reference_name()
        request = self._entity_request(self._gazebo_relative_object_pose, reference)
        self._set_state_future = self.set_entity_client.call_async(request)

    def _set_gazebo_release_pose(self, release_position: list[float]) -> None:
        self._gazebo_attached = False
        if self._set_state_future is not None and not self._set_state_future.done():
            rclpy.spin_until_future_complete(self, self._set_state_future, timeout_sec=2.0)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = release_position
        pose.orientation.w = 1.0
        future = self.set_entity_client.call_async(self._entity_request(pose, "world"))
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise TaskExecutionError("Gazebo 工件释放位置更新失败")

    def run(self) -> None:
        self._wait_for_interfaces()
        self._wait_for_workpiece()
        delay = float(self.get_parameter("startup_delay").value)
        if delay > 0.0:
            self.get_logger().info(f"环境已就绪，{delay:.1f} 秒后开始自动抓取")
            self._wait(delay)

        pick = self.values("pick_position")
        place = self.values("place_position")
        safe_height = float(self.get_parameter("safe_height").value)
        initial_lift_distance = float(
            self.get_parameter("initial_lift_distance").value
        )
        pick_q = quaternion_from_rpy(self.values("pick_orientation_rpy"))
        place_q = quaternion_from_rpy(self.values("place_orientation_rpy"))
        pre_pick = [pick[0], pick[1], safe_height]
        pre_place = [place[0], place[1], safe_height]
        initial_lift = [pick[0], pick[1], pick[2] + initial_lift_distance]

        self.get_logger().info("========== 开始一轮定点抓取 ==========")
        self._command_gripper("张开夹爪", self.get_parameter("gripper_open").value)
        self._move_grasp_pose("移动到 A 点上方", pre_pick, pick_q)
        self._remove_world_object()
        self._set_support_contact_allowed(True)
        self._move_grasp_pose("将夹持中心下降到 A 点", pick, pick_q)
        self._command_gripper("闭合夹爪", self.get_parameter("gripper_closed").value)

        self._relative_object_pose = self._calculate_relative_object_pose()
        self._capture_actual_gazebo_grasp_pose()
        self._gazebo_attached = True
        self._follow_gripper()
        self._wait(float(self.get_parameter("step_pause").value))

        # Lift clear of the table before adding the carried object to MoveIt.
        self._move_grasp_pose("从台面初始抬升", initial_lift, pick_q)
        self._set_support_contact_allowed(False)
        self._attach_planning_object()
        self._move_grasp_pose("抬升到 A 点安全高度", pre_pick, pick_q)
        self._move_grasp_pose("移动到 B 点上方", pre_place, place_q)

        # Remove the attached collision proxy before lowering it onto the table.
        self._remove_attached_planning_object()
        self._set_support_contact_allowed(True)
        self._move_grasp_pose("将夹持中心下降到 B 点", place, place_q)
        self._command_gripper("张开夹爪并释放", self.get_parameter("gripper_open").value)

        self._set_gazebo_release_pose(place)
        self._wait(float(self.get_parameter("step_pause").value))
        self._move_grasp_pose("从 B 点抬起", pre_place, place_q)
        self._set_support_contact_allowed(False)
        self._add_world_object(place)
        self._execute_move("机械臂回到 Home", self._joint_constraints())
        self.get_logger().info("========== 定点抓取任务完成 ==========")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceTask()
    try:
        node.run()
    except (TaskExecutionError, KeyboardInterrupt) as error:
        if isinstance(error, TaskExecutionError):
            node.get_logger().error(f"自动抓取已停止：{error}")
    except Exception as error:  # Keep launch alive and report unexpected failures clearly.
        node.get_logger().error(f"自动抓取发生未预期错误：{error!r}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
