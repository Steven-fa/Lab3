#!/usr/bin/env python3
"""Collision-checked, physically verified fixed-point pick/place for task 2."""

import math
import re
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
]
HOME = [0.0] * 6
A_GRASP = [0.3922133, 1.0421826, -0.9652633, 0.0861514, 1.4526762, -1.1872484]
# Dense, collision-checked grasp-centre chain at 0, 0.5, ... 3 cm above A.
A_VERTICAL_CHAIN = [
    A_GRASP,
    [0.3922131, 1.0414966, -1.0102831, 0.0857760, 1.4982133, -1.1833068],
    [0.3922130, 1.0458521, -1.0644135, 0.0855723, 1.5478058, -1.1790432],
    [0.3922130, 1.0571383, -1.1312561, 0.0855945, 1.6031590, -1.1742949],
    [0.3922132, 1.0795527, -1.2188614, 0.0859573, 1.6681103, -1.1686990],
    [0.3922137, 1.1285267, -1.3569101, 0.0870557, 1.7568524, -1.1609272],
    [0.3909269, 1.1909292, -1.5237074, 0.0907308, 1.8540972, -1.1523635],
]
# Five-millimetre horizontal steps keep the carried cube at 3 cm clearance.
HIGH_TRANSFER_CHAIN = [
    [0.3682187, 1.1071066, -1.3651500, 0.0886195, 1.7884408, -1.1819513],
    [0.3437704, 1.0326058, -1.2278771, 0.0885599, 1.7280216, -1.2117557],
    [0.3188868, 0.9848820, -1.1418509, 0.0890242, 1.6920421, -1.2398258],
    [0.2935878, 0.9481213, -1.0765947, 0.0896381, 1.6658943, -1.2674714],
    [0.2678959, 0.9181419, -1.0240264, 0.0903038, 1.6456894, -1.2950129],
    [0.2418365, 0.8930940, -0.9805547, 0.0909769, 1.6296890, -1.3225734],
    [0.2154378, 0.8719745, -0.9442201, 0.0916324, 1.6169368, -1.3502037],
    [0.1887306, 0.8541751, -0.9138258, 0.0922538, 1.6068418, -1.3779201],
    [0.1617482, 0.8393011, -0.8885874, 0.0928298, 1.5990117, -1.4057193],
    [0.1345261, 0.8270849, -0.8679687, 0.0933519, 1.5931736, -1.4335860],
    [0.1071020, 0.8173410, -0.8515940, 0.0938138, 1.5891329, -1.4614962],
    [0.0795152, 0.8099402, -0.8391992, 0.0942110, 1.5867494, -1.4894202],
    [0.0518066, 0.8047941, -0.8306017, 0.0945402, 1.5859238, -1.5173236],
    [0.0240182, 0.8018456, -0.8256832, 0.0947995, 1.5865887, -1.5451686],
    [-0.0038074, 0.8010637, -0.8243788, 0.0949880, 1.5887045, -1.5729148],
    [-0.0316271, 0.8024403, -0.8266715, 0.0951059, 1.5922561, -1.6005195],
    [-0.0593979, 0.8059897, -0.8325912, 0.0951548, 1.5972529, -1.6279386],
    [-0.0870774, 0.8117497, -0.8422168, 0.0951373, 1.6037297, -1.6551263],
    [-0.1146238, 0.8197852, -0.8556829, 0.0950573, 1.6117501, -1.6820359],
    [-0.1419966, 0.8301948, -0.8731924, 0.0949201, 1.6214123, -1.7086186],
    [-0.1691569, 0.8431213, -0.8950365, 0.0947330, 1.6328587, -1.7348236],
    [-0.1960676, 0.8587696, -0.9216289, 0.0945052, 1.6462923, -1.7605968],
    [-0.2226938, 0.8774365, -0.9535633, 0.0942492, 1.6620038, -1.7858780],
    [-0.2490028, 0.8995646, -0.9917159, 0.0939822, 1.6804212, -1.8105965],
    [-0.2749644, 0.9258461, -1.0374462, 0.0937292, 1.7022049, -1.8346618],
    [-0.3005513, 0.9574519, -1.0930372, 0.0935303, 1.7284589, -1.8579426],
    [-0.3257387, 0.9966370, -1.1628603, 0.0934598, 1.7612876, -1.8802114],
    [-0.3505047, 1.0489748, -1.2576767, 0.0936990, 1.8058537, -1.9009354],
    [-0.3748295, 1.1460709, -1.4383583, 0.0952886, 1.8912881, -1.9171327],
    [-0.3970753, 1.1906902, -1.5236842, 0.0936315, 1.9238020, -1.9363045],
    [-0.4184354, 1.1907118, -1.5237167, 0.0899719, 1.9139032, -1.9590123],
    [-0.4391086, 1.1908527, -1.5237525, 0.0859894, 1.9032459, -1.9812200],
]
# From 2.5 cm down to the 1 cm release clearance at B.
B_DESCENT_CHAIN = [
    [-0.4410610, 1.2164360, -1.5237338, 0.0879734, 1.8892663, -1.9843948],
    [-0.4428757, 1.2418990, -1.5237170, 0.0897670, 1.8746006, -1.9876373],
    [-0.4445578, 1.2672382, -1.5237017, 0.0913821, 1.8592547, -1.9909275],
    [-0.4450162, 1.1882422, -1.3260424, 0.0894652, 1.7437640, -2.0022261],
]
OPEN = 0.15
CLOSED = 0.10
POINT_A = (0.18, 0.08, 0.02)
POINT_B = (0.18, -0.08, 0.02)
GRASP_OFFSET = (-0.0006, 0.010, 0.105)
# Three centimetres is the highest continuous A-to-B branch without the old
# base flip or link3/link6 self-collision.  The cube bottom remains 3 cm clear.
APPROACH_HEIGHT = 0.03
RELEASE_CLEARANCE = 0.01
CHAIN_MAX_JOINT_SPEED = 0.16
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


class PickPlace(Node):
    def __init__(self):
        super().__init__("task2_pick_place")
        self.arm = ActionClient(self, MoveGroup, "/move_action")
        self.arm_trajectory = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.gripper = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_action_controller/follow_joint_trajectory",
        )
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.get_scene = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        self.get_entity = self.create_client(GetEntityState, "/gazebo/get_entity_state")
        self.set_entity = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.check_grasp = self.create_client(Trigger, "/task2/check_grasp")
        self.attach = self.create_client(Trigger, "/task2/attach")
        self.detach = self.create_client(Trigger, "/task2/detach")
        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)

    def on_joint_state(self, message):
        self.latest_joint_state = message

    def ready(self):
        return (
            self.arm.wait_for_server(timeout_sec=20.0)
            and self.arm_trajectory.wait_for_server(timeout_sec=20.0)
            and self.gripper.wait_for_server(timeout_sec=20.0)
            and self.scene.wait_for_service(timeout_sec=20.0)
            and self.get_scene.wait_for_service(timeout_sec=20.0)
            and self.validity.wait_for_service(timeout_sec=20.0)
            and self.get_entity.wait_for_service(timeout_sec=20.0)
            and self.set_entity.wait_for_service(timeout_sec=20.0)
            and self.check_grasp.wait_for_service(timeout_sec=20.0)
            and self.attach.wait_for_service(timeout_sec=20.0)
            and self.detach.wait_for_service(timeout_sec=20.0)
        )

    def move_arm(self, target, label):
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = "arm"
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnectkConfigDefault"
        request.num_planning_attempts = 8
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = 0.08
        request.max_acceleration_scaling_factor = 0.08
        request.start_state.is_diff = True
        constraints = Constraints()
        constraints.name = label
        for name, position in zip(ARM_JOINTS, target):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = 0.008
            joint.tolerance_below = 0.008
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        future = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=12.0)
        if not future.done() or future.result() is None or not future.result().accepted:
            self.get_logger().error(f"{label}: 规划目标未被接受")
            return False
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=55.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"{label}: 执行超时")
            return False
        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"{label}: MoveIt 错误 {code}")
            return False
        self.get_logger().info(f"轨迹完成：{label}")
        time.sleep(1.2)
        return True

    @staticmethod
    def robot_state(arm_positions, gripper_position=OPEN):
        state = RobotState()
        state.is_diff = True
        state.joint_state = JointState()
        state.joint_state.name = ARM_JOINTS + ["gripper_controller"]
        state.joint_state.position = list(arm_positions) + [float(gripper_position)]
        return state

    @staticmethod
    def duration(seconds):
        whole = int(seconds)
        return Duration(sec=whole, nanosec=int((seconds - whole) * 1e9))

    def current_arm_positions(self):
        deadline = time.monotonic() + 3.0
        while self.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_joint_state is None:
            return None
        values = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position,
        ))
        if any(name not in values for name in ARM_JOINTS):
            return None
        return [values[name] for name in ARM_JOINTS]

    def state_is_valid(self, arm_positions, gripper_position):
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state = self.robot_state(arm_positions, gripper_position)
        future = self.validity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if not future.done() or future.result() is None:
            return False, ["validity service timeout"]
        response = future.result()
        contacts = sorted({
            f"{contact.contact_body_1}<->{contact.contact_body_2}"
            for contact in response.contacts
        })
        return response.valid, contacts

    def execute_grasp_center_chain(self, chain, gripper_position, label):
        """Validate a dense grasp-centre chain, then execute it without flips."""
        previous = self.current_arm_positions()
        if previous is None:
            self.get_logger().error(f"{label}: 无法读取当前关节状态")
            return False

        # Interpolate every joint segment at <= 0.04 rad and ask MoveIt to
        # validate each sample against the live scene and attached cube.
        checked_samples = 0
        for target in chain:
            max_delta = max(abs(a - b) for a, b in zip(target, previous))
            if max_delta > math.radians(20.0):
                self.get_logger().error(
                    f"{label}: 相邻姿态跳变 {math.degrees(max_delta):.1f} deg"
                )
                return False
            sample_count = max(1, math.ceil(max_delta / 0.04))
            for index in range(1, sample_count + 1):
                ratio = index / sample_count
                sample = [
                    start + (end - start) * ratio
                    for start, end in zip(previous, target)
                ]
                valid, contacts = self.state_is_valid(sample, gripper_position)
                checked_samples += 1
                if not valid:
                    self.get_logger().error(
                        f"{label}: 碰撞抽样失败；contacts={contacts}"
                    )
                    return False
            previous = list(target)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_JOINTS)
        previous = self.current_arm_positions()
        elapsed = 0.10
        for target in chain:
            joint_step = max(abs(a - b) for a, b in zip(target, previous))
            elapsed += max(0.08, joint_step / CHAIN_MAX_JOINT_SPEED)
            point = JointTrajectoryPoint()
            point.positions = list(target)
            point.time_from_start = self.duration(elapsed)
            trajectory.points.append(point)
            previous = list(target)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send = self.arm_trajectory.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=8.0)
        if not send.done() or send.result() is None or not send.result().accepted:
            self.get_logger().error(f"{label}: 控制器拒绝连续姿态链")
            return False
        result = send.result().get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=max(30.0, elapsed + 10.0))
        if (
            not result.done()
            or result.result() is None
            or result.result().result.error_code != 0
        ):
            code = None if not result.done() or result.result() is None else result.result().result.error_code
            self.get_logger().error(f"{label}: 连续姿态链执行失败，error={code}")
            return False
        self.get_logger().info(
            f"抓取中心路径完成：{label}；points={len(trajectory.points)}；"
            f"collision_samples={checked_samples}"
        )
        time.sleep(0.8)
        return True

    def move_gripper(self, position, label):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["gripper_controller"]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.velocities = [0.0]
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        future = self.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if not future.done() or future.result() is None or not future.result().accepted:
            self.get_logger().error(f"{label}: 夹爪命令未被接受")
            return False
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"{label}: 夹爪执行超时")
            return False
        self.get_logger().info(f"夹爪完成：{label}")
        time.sleep(1.0)
        return True

    def trigger(self, client, label, require_success=True):
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"{label}: 服务无响应")
            return None
        response = future.result()
        self.get_logger().info(f"{label}: success={response.success}; {response.message}")
        if require_success and not response.success:
            return None
        return response

    def entity_position(self):
        request = GetEntityState.Request()
        request.name = "task2_target"
        request.reference_frame = "world"
        future = self.get_entity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None or not future.result().success:
            return None
        p = future.result().state.pose.position
        return (p.x, p.y, p.z)

    def reset_target_to_a(self):
        """Place the cube above A and wait until it is stably resting on the table."""
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = "task2_target"
        request.state.reference_frame = "world"
        request.state.pose.position.x = POINT_A[0]
        request.state.pose.position.y = POINT_A[1]
        # Start slightly above the table so Gazebo resolves contact naturally.
        request.state.pose.position.z = POINT_A[2] + 0.01
        request.state.pose.orientation.w = 1.0
        future = self.set_entity.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None or not future.result().success:
            self.get_logger().error("无法把目标物复位到 A 点")
            return False

        deadline = time.monotonic() + 5.0
        previous = None
        stable_samples = 0
        while time.monotonic() < deadline:
            time.sleep(0.20)
            current = self.entity_position()
            if current is None:
                stable_samples = 0
                continue
            at_a = self.distance(current, POINT_A) <= 0.003
            stopped = previous is not None and self.distance(current, previous) <= 0.0005
            stable_samples = stable_samples + 1 if at_a and stopped else 0
            if stable_samples >= 3:
                self.get_logger().info(
                    "目标物已复位并稳定在 A 点："
                    f"({current[0]:.4f}, {current[1]:.4f}, {current[2]:.4f})"
                )
                return True
            previous = current

        self.get_logger().error(f"目标物复位后未能稳定：{previous}")
        return False

    def apply_scene(self, world_operation=None, attached_operation=None, world_pose=None):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        if world_operation is not None:
            obj = CollisionObject()
            obj.header.frame_id = "base"
            obj.id = "task2_target"
            obj.operation = world_operation
            if world_operation == CollisionObject.ADD:
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [0.04, 0.04, 0.04]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = world_pose
                pose.orientation.w = 1.0
                obj.primitives.append(primitive)
                obj.primitive_poses.append(pose)
            request.scene.world.collision_objects.append(obj)
        if attached_operation is not None:
            attached = AttachedCollisionObject()
            attached.link_name = "link6"
            attached.touch_links = TOUCH_LINKS
            attached.object.header.frame_id = "link6"
            attached.object.id = "task2_target"
            attached.object.operation = attached_operation
            if attached_operation == CollisionObject.ADD:
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [0.04, 0.04, 0.04]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = GRASP_OFFSET
                pose.orientation.w = 1.0
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
        # Removing an object that a preceding calibration pass already removed
        # is idempotent and leaves the requested scene state in force.
        return world_operation == CollisionObject.REMOVE or attached_operation == CollisionObject.REMOVE

    def allow_target_table_contact(self, allowed):
        """Temporarily permit only the grasped cube/table pair to touch."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return False

        matrix = future.result().scene.allowed_collision_matrix
        for name in ("task2_table", "task2_target"):
            if name not in matrix.entry_names:
                matrix.entry_names.append(name)
                for row in matrix.entry_values:
                    row.enabled.append(False)
                from moveit_msgs.msg import AllowedCollisionEntry
                new_row = AllowedCollisionEntry()
                new_row.enabled = [False] * len(matrix.entry_names)
                matrix.entry_values.append(new_row)

        table_index = matrix.entry_names.index("task2_table")
        target_index = matrix.entry_names.index("task2_target")
        matrix.entry_values[table_index].enabled[target_index] = allowed
        matrix.entry_values[target_index].enabled[table_index] = allowed

        update = ApplyPlanningScene.Request()
        update.scene.is_diff = True
        update.scene.allowed_collision_matrix = matrix
        applied = self.scene.call_async(update)
        rclpy.spin_until_future_complete(self, applied, timeout_sec=5.0)
        return (
            applied.done()
            and applied.result() is not None
            and applied.result().success
        )

    @staticmethod
    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def run(self):
        if not self.ready():
            self.get_logger().error("实验二所需控制器或服务未就绪")
            return False

        # Defensive cleanup from a previously interrupted run.
        self.trigger(self.detach, "复位附着状态", require_success=False)
        self.apply_scene(attached_operation=CollisionObject.REMOVE)
        self.allow_target_table_contact(False)
        if not self.move_gripper(OPEN, "张开夹爪"):
            return False
        if not self.move_arm(HOME, "回到 Home"):
            return False
        # A completed previous run leaves the physical cube and its MoveIt
        # collision object at B. Reset both only after the arm is safely Home.
        if not self.apply_scene(world_operation=CollisionObject.REMOVE):
            self.get_logger().error("无法清理上一轮目标物碰撞状态")
            return False
        if not self.reset_target_to_a():
            return False
        if not self.apply_scene(
            world_operation=CollisionObject.ADD,
            world_pose=POINT_A,
        ):
            self.get_logger().error("无法在规划场景中恢复 A 点目标物")
            return False
        if not self.move_arm(A_VERTICAL_CHAIN[-1], "到达 A 点上方 3 cm"):
            return False

        # The physical Gazebo cube remains; only its duplicate MoveIt obstacle is
        # removed so the open fingers may surround it without a false goal clash.
        if not self.apply_scene(world_operation=CollisionObject.REMOVE):
            self.get_logger().error("无法从规划场景暂时移除目标物")
            return False
        if not self.execute_grasp_center_chain(
            list(reversed(A_VERTICAL_CHAIN[:-1])),
            OPEN,
            "A 点抓取中心垂直下降 3 cm",
        ):
            return False
        pre_close = self.entity_position()
        if pre_close is None or self.distance(pre_close, POINT_A) > 0.006:
            self.get_logger().error(f"目标物在夹爪闭合前发生异常移动：{pre_close}")
            return False
        if not self.move_gripper(CLOSED, "闭合夹爪"):
            return False
        after_close = self.entity_position()
        if after_close is None or self.distance(pre_close, after_close) > 0.008:
            self.get_logger().error(
                f"夹爪闭合时将目标物推移：before={pre_close}, after={after_close}"
            )
            return False
        grasp_check = self.trigger(self.check_grasp, "真实对位检查")
        if grasp_check is None:
            return False
        match = re.search(r"center_error=([0-9.]+)", grasp_check.message)
        if match is None or float(match.group(1)) > 0.010:
            self.get_logger().error("夹爪中心误差超过 10 mm，拒绝附着")
            return False
        before_attach = self.entity_position()
        if self.trigger(self.attach, "建立物理抓取") is None:
            return False
        time.sleep(0.5)
        after_attach = self.entity_position()
        snap = self.distance(before_attach, after_attach)
        if snap > 0.003:
            self.get_logger().error(f"附着导致物体跳变 {snap * 1000:.2f} mm")
            return False
        if not self.apply_scene(attached_operation=CollisionObject.ADD):
            self.get_logger().error("无法在 MoveIt 中登记随夹爪运动的目标物")
            return False

        # A grasp starts with the cube resting on the table. MoveIt/FCL expands
        # that contact by a small numerical tolerance, so permit exactly this
        # object/table pair only for the vertical escape motion.
        if not self.allow_target_table_contact(True):
            self.get_logger().error("无法启用目标物离桌接触容差")
            return False
        if not self.execute_grasp_center_chain(
            A_VERTICAL_CHAIN[1:],
            CLOSED,
            "抓取后抓取中心垂直抬升 3 cm",
        ):
            return False
        if not self.allow_target_table_contact(False):
            self.get_logger().error("无法恢复严格桌面碰撞检查")
            return False
        lifted = self.entity_position()
        if lifted is None or lifted[2] - POINT_A[2] < APPROACH_HEIGHT - 0.01:
            self.get_logger().error(f"物体没有随夹爪抬升：{lifted}")
            return False
        if self.trigger(self.check_grasp, "抬升后保持检查") is None:
            return False
        if not self.execute_grasp_center_chain(
            HIGH_TRANSFER_CHAIN,
            CLOSED,
            "经桌面中央等高平移到 B 点上方",
        ):
            return False
        if not self.execute_grasp_center_chain(
            B_DESCENT_CHAIN,
            CLOSED,
            "B 点抓取中心垂直下降到释放高度",
        ):
            return False

        release_pose = self.entity_position()
        expected_release = (
            POINT_B[0], POINT_B[1], POINT_B[2] + RELEASE_CLEARANCE,
        )
        if release_pose is None or self.distance(release_pose, expected_release) > 0.018:
            self.get_logger().error(f"释放前目标物未对准 B 点：{release_pose}")
            return False
        if not self.apply_scene(attached_operation=CollisionObject.REMOVE):
            return False
        if not self.move_gripper(OPEN, "松开夹爪"):
            return False
        if self.trigger(self.detach, "解除物理抓取") is None:
            return False
        time.sleep(2.0)
        final_pose = self.entity_position()
        if final_pose is None:
            return False
        xy_error = math.hypot(final_pose[0] - POINT_B[0], final_pose[1] - POINT_B[1])
        z_error = abs(final_pose[2] - POINT_B[2])
        if xy_error > 0.015 or z_error > 0.008:
            self.get_logger().error(
                f"最终落点超差：xy={xy_error * 1000:.1f} mm, z={z_error * 1000:.1f} mm"
            )
            return False
        if not self.apply_scene(
            world_operation=CollisionObject.ADD,
            world_pose=final_pose,
        ):
            return False
        b_return_chain = (
            list(reversed(B_DESCENT_CHAIN[:-1]))
            + [HIGH_TRANSFER_CHAIN[-1]]
        )
        if not self.execute_grasp_center_chain(
            b_return_chain,
            OPEN,
            "释放后从 B 点垂直离开",
        ):
            return False
        # Stop at y=0 on the same high chain before the final Home motion.
        central_return_chain = list(reversed(HIGH_TRANSFER_CHAIN[15:-1]))
        if not self.execute_grasp_center_chain(
            central_return_chain,
            OPEN,
            "返回桌面中央高位",
        ):
            return False
        if not self.move_arm(HOME, "返回 Home"):
            return False
        self.get_logger().info(
            "TASK2_PICK_PLACE_OK "
            f"attach_snap={snap * 1000:.2f}mm "
            f"final_xy_error={xy_error * 1000:.2f}mm "
            f"final_z_error={z_error * 1000:.2f}mm"
        )
        return True


def main():
    rclpy.init()
    node = PickPlace()
    ok = False
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
