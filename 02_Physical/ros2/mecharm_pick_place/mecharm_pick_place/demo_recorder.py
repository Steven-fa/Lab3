"""ROS 2 node that records a complete human demonstration at a fixed rate."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time

import rclpy
from control_msgs.action import GripperCommand
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Quaternion
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .demonstration import ARM_JOINT_NAMES, GRIPPER_JOINT, write_demonstration
from .keyframes import write_keyframes


def _quaternion_from_rpy(values: list[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = values
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quaternion_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float], vector: list[float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _is_gripper_closed(
    position: float, open_position: float, closed_position: float
) -> bool:
    """Classify the gripper without assuming which command has the larger value."""
    return abs(position - closed_position) <= abs(position - open_position)


class DemonstrationRecorder(Node):
    """Record joints, gripper, end-effector and workpiece pose in one timeline."""

    def __init__(self) -> None:
        super().__init__("demonstration_recorder")
        self.declare_parameter("output_root", "~/mecharm_demos")
        self.declare_parameter("demo_name", "")
        self.declare_parameter("sample_rate", 20.0)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("recording_mode", "dense")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("end_effector_link", "gripper_base")
        self.declare_parameter("object_id", "workpiece")
        self.declare_parameter("joint_names", ARM_JOINT_NAMES)
        self.declare_parameter("gripper_joint", GRIPPER_JOINT)
        self.declare_parameter("assist_grasp", False)
        self.declare_parameter("robot_model_name", "mecharm_270_m5")
        self.declare_parameter("gazebo_reference_link", "link6")
        self.declare_parameter(
            "gazebo_reference_to_end_effector_position", [0.0, 0.0, 0.038]
        )
        self.declare_parameter(
            "gazebo_reference_to_end_effector_rpy", [1.579, 0.0, 0.0]
        )
        self.declare_parameter("pick_orientation_rpy", [-1.57079633, 0.0, 0.0])
        self.declare_parameter("grasp_center_offset", [0.0, 0.030, -0.055])
        self.declare_parameter("max_grasp_alignment_error", 0.020)
        self.declare_parameter("gripper_open", 0.10)
        self.declare_parameter("gripper_closed", -0.70)

        self._joint_positions: dict[str, float] = {}
        self._object_pose = None
        self._samples: list[dict[str, float]] = []
        self._keyframes: list[dict] = []
        self._recording = False
        self._first_sample_time_ns: int | None = None
        self._output_directory: Path | None = None
        self._last_saved_directory: Path | None = None
        self._assist_attached = False
        self._last_gripper_closed: bool | None = None
        self._attach_future = None
        self._set_state_future = None
        self._gazebo_relative_object_pose = Pose()
        self._gazebo_relative_object_pose.orientation.w = 1.0
        self._gripper_goal_future = None
        self._gripper_result_future = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_models, 20
        )
        self.create_service(
            Trigger, "/mecharm_demo/start_recording", self._start_recording
        )
        self.create_service(
            Trigger, "/mecharm_demo/stop_recording", self._stop_recording
        )
        self.create_service(Trigger, "/mecharm_demo/status", self._status)
        self.create_service(
            Trigger, "/mecharm_demo/record_waypoint", self._record_waypoint
        )
        self.create_service(Trigger, "/mecharm_demo/record_grasp", self._record_grasp)
        self.create_service(
            Trigger, "/mecharm_demo/record_release", self._record_release
        )
        self._get_entity_client = self.create_client(
            GetEntityState, "/gazebo/get_entity_state"
        )
        self._set_entity_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, "/adaptive_gripper_controller/gripper_cmd"
        )

        rate = max(float(self.get_parameter("sample_rate").value), 1.0)
        # Do not schedule recording from Gazebo's /clock. Gazebo Classic may
        # advance that clock in coarse batches, which makes two nominal 20 Hz
        # samples arrive almost simultaneously and creates false pause points.
        self._sampling_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0 / rate, self._sample, clock=self._sampling_clock)
        if bool(self.get_parameter("auto_start").value):
            self._begin_recording()
        mode = str(self.get_parameter("recording_mode").value)
        if mode == "keyframes":
            self.get_logger().info("关键点记录器已就绪：q / qa / qb 选择关键位置")
        else:
            self.get_logger().info(
                "示教记录器已就绪：调用 /mecharm_demo/start_recording 开始"
            )

    @property
    def joint_names(self) -> list[str]:
        return [str(name) for name in self.get_parameter("joint_names").value]

    def _on_joints(self, message: JointState) -> None:
        self._joint_positions.update(
            {name: float(position) for name, position in zip(message.name, message.position)}
        )

    def _on_models(self, message: ModelStates) -> None:
        object_id = str(self.get_parameter("object_id").value)
        try:
            self._object_pose = message.pose[message.name.index(object_id)]
        except ValueError:
            self._object_pose = None

    def values(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    def _gazebo_reference_name(self) -> str:
        return (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('gazebo_reference_link').value}"
        )

    def _expected_gazebo_grasp_pose(self) -> Pose:
        """Expected workpiece pose in Gazebo's uncollapsed link6 frame."""
        gripper_to_object = self.values("grasp_center_offset")
        pick_q = _quaternion_from_rpy(self.values("pick_orientation_rpy"))
        gripper_to_object_q = (-pick_q[0], -pick_q[1], -pick_q[2], pick_q[3])
        reference_to_gripper = self.values(
            "gazebo_reference_to_end_effector_position"
        )
        reference_to_gripper_q = _quaternion_from_rpy(
            self.values("gazebo_reference_to_end_effector_rpy")
        )
        rotated_offset = _rotate_vector(reference_to_gripper_q, gripper_to_object)
        reference_to_object_q = _quaternion_multiply(
            reference_to_gripper_q, gripper_to_object_q
        )
        pose = Pose()
        pose.position.x = reference_to_gripper[0] + rotated_offset[0]
        pose.position.y = reference_to_gripper[1] + rotated_offset[1]
        pose.position.z = reference_to_gripper[2] + rotated_offset[2]
        pose.orientation = Quaternion(
            x=reference_to_object_q[0],
            y=reference_to_object_q[1],
            z=reference_to_object_q[2],
            w=reference_to_object_q[3],
        )
        return pose

    def _request_manual_attach(self) -> None:
        if self._attach_future is not None or self._assist_attached:
            return
        if not self._get_entity_client.service_is_ready():
            self.get_logger().warning("Gazebo 位姿服务尚未就绪，本次闭爪不执行辅助抓取")
            return
        request = GetEntityState.Request()
        request.name = str(self.get_parameter("object_id").value)
        request.reference_frame = self._gazebo_reference_name()
        self._attach_future = self._get_entity_client.call_async(request)
        self._attach_future.add_done_callback(self._finish_manual_attach)

    def _finish_manual_attach(self, future) -> None:
        self._attach_future = None
        gripper_name = str(self.get_parameter("gripper_joint").value)
        current = self._joint_positions.get(gripper_name)
        open_position = float(self.get_parameter("gripper_open").value)
        closed_position = float(self.get_parameter("gripper_closed").value)
        if current is None or not _is_gripper_closed(
            current, open_position, closed_position
        ):
            return
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"读取闭爪相对位姿失败：{error}")
            return
        if response is None or not response.success:
            self.get_logger().warning("读取闭爪相对位姿失败，本次不吸附工件")
            return
        actual = response.state.pose
        expected = self._expected_gazebo_grasp_pose()
        error = math.sqrt(
            (actual.position.x - expected.position.x) ** 2
            + (actual.position.y - expected.position.y) ** 2
            + (actual.position.z - expected.position.z) ** 2
        )
        limit = float(self.get_parameter("max_grasp_alignment_error").value)
        self.get_logger().info(f"人工闭爪对准误差：{error * 1000.0:.1f} mm")
        if error > limit:
            self.get_logger().warning(
                f"夹爪未对准工件（上限 {limit * 1000.0:.1f} mm），"
                "拒绝浮空吸附；请张开、校正后再闭合"
            )
            return
        # Preserve the measured transform, so attachment never snaps the object
        # into the gripper palm or makes it jump sideways.
        self._gazebo_relative_object_pose = actual
        self._assist_attached = True
        self.get_logger().info("人工示教辅助抓取已锁定实际相对位姿")
        self._follow_gripper()

    def _follow_gripper(self) -> None:
        if not self._assist_attached:
            return
        if self._set_state_future is not None and not self._set_state_future.done():
            return
        if not self._set_entity_client.service_is_ready():
            return
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = str(self.get_parameter("object_id").value)
        request.state.reference_frame = self._gazebo_reference_name()
        request.state.pose = self._gazebo_relative_object_pose
        self._set_state_future = self._set_entity_client.call_async(request)

    def _update_grasp_assist(self) -> None:
        if not bool(self.get_parameter("assist_grasp").value):
            return
        gripper_name = str(self.get_parameter("gripper_joint").value)
        position = self._joint_positions.get(gripper_name)
        if position is None:
            return
        closed = _is_gripper_closed(
            position,
            float(self.get_parameter("gripper_open").value),
            float(self.get_parameter("gripper_closed").value),
        )
        if self._last_gripper_closed is None:
            self._last_gripper_closed = closed
        elif closed and not self._last_gripper_closed:
            self._request_manual_attach()
        elif not closed and self._last_gripper_closed:
            if self._assist_attached:
                self.get_logger().info("人工示教辅助抓取已释放工件")
            self._assist_attached = False
        self._last_gripper_closed = closed
        self._follow_gripper()

    def _new_output_directory(self) -> Path:
        root = Path(str(self.get_parameter("output_root").value)).expanduser()
        configured_name = str(self.get_parameter("demo_name").value).strip()
        name = configured_name or datetime.now().strftime("demo_%Y%m%d_%H%M%S")
        candidate = root / name
        suffix = 2
        while candidate.exists():
            candidate = root / f"{name}_{suffix}"
            suffix += 1
        return candidate.resolve()

    def _begin_recording(self) -> None:
        self._samples = []
        self._keyframes = []
        self._first_sample_time_ns = None
        self._output_directory = self._new_output_directory()
        self._recording = True
        self.get_logger().info(f"开始示教录制：{self._output_directory}")

    def _keyframe_metadata(self) -> dict:
        return {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "base_frame": str(self.get_parameter("base_frame").value),
            "end_effector_link": str(
                self.get_parameter("end_effector_link").value
            ),
            "object_id": str(self.get_parameter("object_id").value),
            "joint_names": self.joint_names,
            "gripper_joint": str(self.get_parameter("gripper_joint").value),
            "gripper_open": float(self.get_parameter("gripper_open").value),
            "gripper_closed": float(self.get_parameter("gripper_closed").value),
        }

    @staticmethod
    def _pose_values(pose: Pose) -> dict[str, list[float]]:
        return {
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    def _capture_keyframe(self, kind: str) -> tuple[bool, str]:
        if not self._recording:
            return False, "当前没有进行关键点录制"
        if str(self.get_parameter("recording_mode").value) != "keyframes":
            return False, "当前记录器不是关键点模式"

        gripper_joint = str(self.get_parameter("gripper_joint").value)
        required = [*self.joint_names, gripper_joint]
        missing = [name for name in required if name not in self._joint_positions]
        if missing:
            return False, f"尚未收到关节状态：{', '.join(missing)}"
        if self._object_pose is None:
            return False, "尚未收到 Gazebo 工件位姿"

        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.get_parameter("base_frame").value),
                str(self.get_parameter("end_effector_link").value),
                Time(),
            )
        except TransformException as error:
            return False, f"尚未取得机械臂末端位姿：{error}"

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        ee_pose = Pose()
        ee_pose.position.x = translation.x
        ee_pose.position.y = translation.y
        ee_pose.position.z = translation.z
        ee_pose.orientation = Quaternion(
            x=rotation.x,
            y=rotation.y,
            z=rotation.z,
            w=rotation.w,
        )
        keyframe = {
            "index": len(self._keyframes) + 1,
            "type": kind,
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
            "ee_pose": self._pose_values(ee_pose),
            "joint_positions": {
                name: float(self._joint_positions[name]) for name in self.joint_names
            },
            "gripper_position": float(self._joint_positions[gripper_joint]),
            "object_pose": self._pose_values(self._object_pose),
        }
        self._keyframes.append(keyframe)
        output_path = write_keyframes(
            self._output_directory,
            self._keyframes,
            self._keyframe_metadata(),
        )
        position = keyframe["ee_pose"]["position"]
        message = (
            f"已记录第 {keyframe['index']} 个 {kind}："
            f"xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})；"
            f"文件 {output_path}"
        )
        self.get_logger().info(message)
        return True, message

    def _gripper_result(self, future, label: str) -> None:
        try:
            wrapped = future.result()
        except Exception as error:
            self.get_logger().error(f"{label}失败：{error}")
            return
        if wrapped is None:
            self.get_logger().error(f"{label}失败：没有返回结果")
            return
        result = wrapped.result
        self.get_logger().info(
            f"{label}完成：position={result.position:.3f}, "
            f"reached_goal={result.reached_goal}, stalled={result.stalled}"
        )

    def _gripper_goal_response(self, future, label: str) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f"{label}发送失败：{error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"夹爪控制器拒绝动作：{label}")
            return
        self._gripper_result_future = goal_handle.get_result_async()
        self._gripper_result_future.add_done_callback(
            lambda result_future: self._gripper_result(result_future, label)
        )

    def _command_gripper(self, position: float, label: str) -> bool:
        if not self._gripper_client.server_is_ready():
            self.get_logger().warning(f"夹爪 action 尚未就绪，未执行：{label}")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 12.0
        self._gripper_goal_future = self._gripper_client.send_goal_async(goal)
        self._gripper_goal_future.add_done_callback(
            lambda future: self._gripper_goal_response(future, label)
        )
        return True

    def _record_keyframe_response(
        self, kind: str, response: Trigger.Response
    ) -> Trigger.Response:
        response.success, response.message = self._capture_keyframe(kind)
        return response

    def _record_waypoint(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        return self._record_keyframe_response("waypoint", response)

    def _record_grasp(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        result = self._record_keyframe_response("grasp", response)
        if result.success:
            self._command_gripper(
                float(self.get_parameter("gripper_closed").value),
                "qa 闭合夹爪",
            )
        return result

    def _record_release(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        result = self._record_keyframe_response("release", response)
        if result.success:
            self._command_gripper(
                float(self.get_parameter("gripper_open").value),
                "qb 张开夹爪",
            )
        return result

    def _start_recording(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._recording:
            response.success = False
            response.message = "示教录制已经在进行中"
            return response
        self._begin_recording()
        response.success = True
        response.message = f"开始录制：{self._output_directory}"
        return response

    def _sample(self) -> None:
        self._update_grasp_assist()
        if not self._recording:
            return
        if str(self.get_parameter("recording_mode").value) == "keyframes":
            return
        required = [*self.joint_names, str(self.get_parameter("gripper_joint").value)]
        if any(name not in self._joint_positions for name in required):
            return
        if self._object_pose is None:
            return

        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.get_parameter("base_frame").value),
                str(self.get_parameter("end_effector_link").value),
                Time(),
            )
        except TransformException:
            return

        # Gazebo Classic publishes /clock in coarse batches. A monotonic wall
        # timeline preserves the requested sample rate during human operation.
        now_ns = time.monotonic_ns()
        if self._first_sample_time_ns is None:
            self._first_sample_time_ns = now_ns
        relative_time = (now_ns - self._first_sample_time_ns) / 1_000_000_000.0
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        object_pose = self._object_pose
        sample = {"time_from_start": relative_time}
        sample.update({name: self._joint_positions[name] for name in self.joint_names})
        sample[GRIPPER_JOINT] = self._joint_positions[
            str(self.get_parameter("gripper_joint").value)
        ]
        sample.update(
            {
                "ee_x": translation.x,
                "ee_y": translation.y,
                "ee_z": translation.z,
                "ee_qx": rotation.x,
                "ee_qy": rotation.y,
                "ee_qz": rotation.z,
                "ee_qw": rotation.w,
                "object_x": object_pose.position.x,
                "object_y": object_pose.position.y,
                "object_z": object_pose.position.z,
                "object_qx": object_pose.orientation.x,
                "object_qy": object_pose.orientation.y,
                "object_qz": object_pose.orientation.z,
                "object_qw": object_pose.orientation.w,
            }
        )
        self._samples.append(sample)

    def _save(self) -> tuple[bool, str]:
        self._recording = False
        if str(self.get_parameter("recording_mode").value) == "keyframes":
            if not self._keyframes or self._output_directory is None:
                return False, "没有记录关键点；请先按 q、qa 或 qb"
            output_path = write_keyframes(
                self._output_directory,
                self._keyframes,
                self._keyframe_metadata(),
            )
            self._last_saved_directory = self._output_directory
            message = f"已保存 {len(self._keyframes)} 个关键点：{output_path}"
            self.get_logger().info(message)
            return True, message
        if not self._samples or self._output_directory is None:
            return False, "没有有效采样；请确认机械臂、TF 和工件已经加载"

        metadata = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "sample_rate_hz": float(self.get_parameter("sample_rate").value),
            "base_frame": str(self.get_parameter("base_frame").value),
            "end_effector_link": str(
                self.get_parameter("end_effector_link").value
            ),
            "object_id": str(self.get_parameter("object_id").value),
        }
        summary = write_demonstration(
            self._output_directory, self._samples, self.joint_names, metadata
        )
        self._last_saved_directory = self._output_directory
        keypoint_count = len(summary["keypoints"])
        sample_count = len(self._samples)
        message = (
            f"已保存 {sample_count} 个采样、{keypoint_count} 个关键点："
            f"{self._output_directory}"
        )
        self.get_logger().info(message)
        return True, message

    def _stop_recording(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if not self._recording:
            response.success = False
            response.message = "当前没有进行示教录制"
            return response
        response.success, response.message = self._save()
        return response

    def _status(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        response.success = True
        if self._recording:
            state = "已锁定工件" if self._assist_attached else "等待正确闭爪"
            if str(self.get_parameter("recording_mode").value) == "keyframes":
                response.message = (
                    f"关键点录制中，已保存 {len(self._keyframes)} 点，"
                    f"抓取辅助：{state}"
                )
            else:
                response.message = (
                    f"录制中，已采样 {len(self._samples)} 帧，抓取辅助：{state}"
                )
        elif self._last_saved_directory:
            response.message = f"空闲，上次结果：{self._last_saved_directory}"
        else:
            response.message = "空闲，尚未保存示教"
        return response

    def save_before_shutdown(self) -> None:
        has_data = self._samples or self._keyframes
        if self._recording and has_data:
            self.get_logger().warning("节点退出前自动保存当前示教")
            self._save()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemonstrationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_before_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
