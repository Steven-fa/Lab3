"""Plan and execute MoveIt motions between manually selected keyframes."""

from __future__ import annotations

from pathlib import Path

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Quaternion

from .keyframes import load_keyframes
from .task_executor import (
    PickPlaceTask,
    TaskExecutionError,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
)


def relative_object_pose(keyframe: dict) -> Pose:
    """Convert recorded world poses into an object pose in gripper_base."""
    ee = keyframe["ee_pose"]
    obj = keyframe["object_pose"]
    ee_position = [float(value) for value in ee["position"]]
    object_position = [float(value) for value in obj["position"]]
    ee_orientation = [float(value) for value in ee["orientation_xyzw"]]
    object_orientation = [
        float(value) for value in obj["orientation_xyzw"]
    ]
    inverse = quaternion_conjugate(ee_orientation)
    delta = [
        object_position[index] - ee_position[index] for index in range(3)
    ]
    relative_position = rotate_vector(inverse, delta)
    relative_orientation = quaternion_multiply(inverse, object_orientation)
    result = Pose()
    (
        result.position.x,
        result.position.y,
        result.position.z,
    ) = relative_position
    result.orientation = Quaternion(
        x=relative_orientation[0],
        y=relative_orientation[1],
        z=relative_orientation[2],
        w=relative_orientation[3],
    )
    return result


class KeyframePlayer(PickPlaceTask):
    def __init__(self) -> None:
        super().__init__(node_name="keyframe_player")
        self.declare_parameter("input_path", "~/mecharm_demos")
        self.declare_parameter("reset_object_before_run", False)

    def _reset_workpiece_for_replay(self, keyframe: dict) -> None:
        """Put the workpiece back at the recorded pickup pose for another run."""
        self._gazebo_attached = False
        if self._set_state_future is not None and not self._set_state_future.done():
            rclpy.spin_until_future_complete(
                self, self._set_state_future, timeout_sec=2.0
            )

        # A previous run may have left either an attached object or a world
        # collision object in MoveIt. Clear both representations before adding
        # one fresh object at the pickup position.
        self._remove_attached_planning_object()
        self._set_support_contact_allowed(False)
        self._set_object_support_contact_allowed(False)

        values = keyframe["object_pose"]
        position = [float(value) for value in values["position"]]
        orientation = [float(value) for value in values["orientation_xyzw"]]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation = Quaternion(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        )
        future = self.set_entity_client.call_async(
            self._entity_request(pose, "world")
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise TaskExecutionError("重复执行前无法复位 Gazebo 工件")

        self._remove_world_object()
        self._add_world_object(position)
        self._wait(0.25)
        self.get_logger().info("工件已复位到记录的抓取位置")

    def _capture_actual_gazebo_relative_pose(self) -> None:
        request = GetEntityState.Request()
        request.name = str(self.get_parameter("object_id").value)
        request.reference_frame = self._gazebo_reference_name()
        future = self.get_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise TaskExecutionError("无法读取夹取瞬间的 Gazebo 工件相对位姿")
        self._gazebo_relative_object_pose = response.state.pose

    def _release_at_recorded_pose(self, keyframe: dict) -> list[float]:
        self._gazebo_attached = False
        values = keyframe["object_pose"]
        position = [float(value) for value in values["position"]]
        orientation = [float(value) for value in values["orientation_xyzw"]]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation = Quaternion(
            x=orientation[0],
            y=orientation[1],
            z=orientation[2],
            w=orientation[3],
        )
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = str(self.get_parameter("object_id").value)
        request.state.reference_frame = "world"
        request.state.pose = pose
        future = self.set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise TaskExecutionError("Gazebo 工件释放位置更新失败")
        return position

    @staticmethod
    def _keyframe_pose(keyframe: dict) -> tuple[list[float], tuple[float, ...]]:
        values = keyframe["ee_pose"]
        position = [float(value) for value in values["position"]]
        orientation = tuple(
            float(value) for value in values["orientation_xyzw"]
        )
        return position, orientation

    def run_keyframes(self) -> None:
        document = load_keyframes(
            Path(str(self.get_parameter("input_path").value))
        )
        keyframes = document["keyframes"]
        gripper_open = float(document.get("gripper_open", 0.10))
        gripper_closed = float(document.get("gripper_closed", -0.70))

        self._wait_for_interfaces()
        self._wait_for_workpiece()
        if bool(self.get_parameter("reset_object_before_run").value):
            grasp_keyframe = next(
                (frame for frame in keyframes if frame["type"] == "grasp"),
                None,
            )
            if grasp_keyframe is None:
                raise TaskExecutionError("关键点中没有可用于复位工件的夹取点")
            self._reset_workpiece_for_replay(grasp_keyframe)
        self.get_logger().info(
            f"读取 {len(keyframes)} 个关键点：{document['source_path']}"
        )
        self._command_gripper("回放前张开夹爪", gripper_open)

        planning_attached = False
        support_contact_allowed = False
        object_support_contact_allowed = False
        released_waiting_for_retreat = False
        pending_world_object_position: list[float] | None = None
        for keyframe in keyframes:
            index = int(keyframe.get("index", 0))
            kind = str(keyframe["type"])
            position, orientation = self._keyframe_pose(keyframe)

            if kind == "grasp":
                self._remove_world_object()
                self._set_support_contact_allowed(True)
                support_contact_allowed = True

            label = f"关键点 {index} / {kind}"
            if str(keyframe.get("motion", "")) in {"vertical", "cartesian"}:
                self._move_cartesian_pose(label, position, orientation)
            else:
                self._move_pose(label, position, orientation)

            # Contact allowances are deliberately limited to the first exact
            # vertical lift away from the table.  All transfer motions run with
            # normal table collision checking enabled.
            if support_contact_allowed and kind != "grasp":
                self._set_support_contact_allowed(False)
                support_contact_allowed = False

            if object_support_contact_allowed and kind != "grasp":
                self._set_object_support_contact_allowed(False)
                object_support_contact_allowed = False

            if released_waiting_for_retreat:
                if support_contact_allowed:
                    self._set_support_contact_allowed(False)
                    support_contact_allowed = False
                if pending_world_object_position is not None:
                    self._add_world_object(pending_world_object_position)
                    pending_world_object_position = None
                released_waiting_for_retreat = False

            if kind == "grasp":
                self._command_gripper(f"关键点 {index} 闭合夹爪", gripper_closed)
                self._relative_object_pose = relative_object_pose(keyframe)
                self._capture_actual_gazebo_relative_pose()
                self._gazebo_attached = True
                self._follow_gripper()
                # Add the carried workpiece to MoveIt before the first lift.
                # Deferring this until after the lift lets the real workpiece
                # sweep through the table while the wrist changes posture.
                self._attach_planning_object()
                planning_attached = True
                self._set_object_support_contact_allowed(True)
                object_support_contact_allowed = True
            elif kind == "release":
                # Keep the workpiece attached and collision-checked throughout
                # the descent. Detach only after reaching the release target.
                if planning_attached:
                    self._remove_attached_planning_object()
                    planning_attached = False
                self._command_gripper(f"关键点 {index} 张开夹爪", gripper_open)
                release_position = self._release_at_recorded_pose(keyframe)
                # Keep the just-released workpiece out of the planning scene until
                # the next motion has moved the fingers clear. At the release pose,
                # normal finger/object contact would otherwise make that retreat's
                # start state invalid.
                pending_world_object_position = release_position
                released_waiting_for_retreat = True

        if support_contact_allowed:
            self._set_support_contact_allowed(False)
        if object_support_contact_allowed:
            self._set_object_support_contact_allowed(False)
        if pending_world_object_position is not None:
            self._add_world_object(pending_world_object_position)
        if bool(document.get("return_home", False)):
            self._execute_move("机械臂回到 Home", self._joint_constraints())
        self.get_logger().info("全部关键点已由 MoveIt 自动规划并执行完成")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyframePlayer()
    try:
        node.run_keyframes()
    except (TaskExecutionError, ValueError, KeyboardInterrupt) as error:
        if not isinstance(error, KeyboardInterrupt):
            node.get_logger().error(f"关键点回放停止：{error}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
