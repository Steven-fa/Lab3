#!/usr/bin/env python3

from math import cos, sin
from typing import Iterable

import rclpy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import (
    CollisionObject,
    ObjectColor,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from .task_executor import set_allowed_collision


class SceneSetup(Node):
    """Add the lab geometry to MoveIt and publish the task-point markers."""

    def __init__(self) -> None:
        super().__init__("scene_setup")
        self.declare_parameter("frame_id", "base")
        self.declare_parameter("table_id", "workbench")
        self.declare_parameter("table_position", [0.12, 0.0, -0.03])
        self.declare_parameter("table_size", [0.52, 0.52, 0.05])
        self.declare_parameter("object_id", "workpiece")
        self.declare_parameter("object_position", [0.18, -0.10, 0.020])
        self.declare_parameter("object_size", [0.035, 0.035, 0.05])
        self.declare_parameter("planning_object_size", [0.033, 0.033, 0.047])
        self.declare_parameter("place_zone_position", [0.18, 0.10, -0.0045])
        self.declare_parameter("place_zone_radius", 0.055)
        self.declare_parameter("pick_position", [0.18, -0.10, 0.020])
        self.declare_parameter("pick_orientation_rpy", [-1.57079633, 0.0, 0.0])
        self.declare_parameter("place_position", [0.18, 0.10, 0.020])
        self.declare_parameter(
            "place_orientation_rpy", [-1.57079633, 0.0, 3.14159265]
        )
        self.declare_parameter("safe_height", 0.09)
        self.declare_parameter("gripper_open", 0.10)
        self.declare_parameter("gripper_closed", -0.70)
        self.declare_parameter("velocity_scaling", 0.10)
        self.declare_parameter("acceleration_scaling", 0.10)
        self.declare_parameter("allow_support_contact", False)
        self.declare_parameter(
            "support_touch_links",
            [
                "gripper_left1",
                "gripper_left2",
                "gripper_right1",
                "gripper_right2",
            ],
        )

        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/mecharm_task/markers", marker_qos
        )
        self.scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.get_scene_client = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

    def values(self, name: str) -> list[float]:
        return [float(value) for value in self.get_parameter(name).value]

    @staticmethod
    def pose(position: Iterable[float]) -> Pose:
        result = Pose()
        result.position.x, result.position.y, result.position.z = position
        result.orientation.w = 1.0
        return result

    def box(self, object_id: str, position: list[float], size: list[float]):
        collision = CollisionObject()
        collision.header.frame_id = self.get_parameter("frame_id").value
        collision.id = object_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        collision.primitives.append(primitive)
        collision.primitive_poses.append(self.pose(position))
        collision.operation = CollisionObject.ADD
        return collision

    @staticmethod
    def object_color(object_id: str, rgba: list[float]) -> ObjectColor:
        result = ObjectColor()
        result.id = object_id
        result.color.r, result.color.g, result.color.b, result.color.a = rgba
        return result

    def apply_scene(self) -> bool:
        if not self.scene_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(
                "MoveIt service /apply_planning_scene was not available after 30 s."
            )
            return False

        scene = PlanningScene()
        scene.is_diff = True
        table_id = self.get_parameter("table_id").value
        object_id = self.get_parameter("object_id").value
        scene.world.collision_objects = [
            self.box(
                table_id,
                self.values("table_position"),
                self.values("table_size"),
            ),
            self.box(
                object_id,
                self.values("object_position"),
                self.values("planning_object_size"),
            ),
        ]
        scene.object_colors = [
            self.object_color(table_id, [0.56, 0.58, 0.62, 1.0]),
            self.object_color(object_id, [1.0, 0.28, 0.10, 1.0]),
        ]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Timed out while applying the MoveIt planning scene.")
            return False
        if not future.result().success:
            self.get_logger().error("MoveIt rejected the task planning scene.")
            return False
        return True

    def allow_manual_support_contact(self) -> bool:
        """Allow only distal fingers to touch the table during manual teaching."""
        if not bool(self.get_parameter("allow_support_contact").value):
            return True
        if not self.get_scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("MoveIt 场景读取服务不可用，无法启用人工侧夹")
            return False
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result() if future.done() else None
        if response is None:
            self.get_logger().error("无法读取 MoveIt 允许碰撞矩阵")
            return False

        matrix = response.scene.allowed_collision_matrix
        table_id = str(self.get_parameter("table_id").value)
        for link in self.get_parameter("support_touch_links").value:
            set_allowed_collision(matrix, str(link), table_id, True)
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        apply_request = ApplyPlanningScene.Request()
        apply_request.scene = scene
        apply_future = self.scene_client.call_async(apply_request)
        rclpy.spin_until_future_complete(self, apply_future, timeout_sec=10.0)
        apply_response = apply_future.result() if apply_future.done() else None
        if apply_response is None or not apply_response.success:
            self.get_logger().error("无法启用指尖与工作台接触")
            return False
        self.get_logger().info("人工示教模式：已允许四个远端指尖接触工作台")
        return True

    def marker(
        self,
        marker_id: int,
        marker_type: int,
        position: list[float],
        scale: list[float],
        color: list[float],
        text: str = "",
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.get_parameter("frame_id").value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mecharm_fixed_points"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose = self.pose(position)
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.text = text
        return marker

    def publish_markers(self) -> None:
        table_position = self.values("table_position")
        table_size = self.values("table_size")
        surface_z = table_position[2] + table_size[2] / 2.0 + 0.002
        pick = self.values("pick_position")
        place = self.values("place_position")
        safe_z = float(self.get_parameter("safe_height").value)
        zone_radius = float(self.get_parameter("place_zone_radius").value)

        markers = MarkerArray()
        markers.markers.extend(
            [
                self.marker(
                    0,
                    Marker.CYLINDER,
                    [pick[0], pick[1], surface_z],
                    [0.065, 0.065, 0.003],
                    [1.0, 0.48, 0.05, 0.60],
                ),
                self.marker(
                    1,
                    Marker.CYLINDER,
                    [place[0], place[1], surface_z],
                    [2.0 * zone_radius, 2.0 * zone_radius, 0.003],
                    [0.10, 0.42, 1.0, 0.55],
                ),
                self.marker(
                    2,
                    Marker.SPHERE,
                    [pick[0], pick[1], safe_z],
                    [0.022, 0.022, 0.022],
                    [1.0, 0.72, 0.05, 0.95],
                ),
                self.marker(
                    3,
                    Marker.SPHERE,
                    [place[0], place[1], safe_z],
                    [0.022, 0.022, 0.022],
                    [0.10, 0.72, 1.0, 0.95],
                ),
                self.marker(
                    4,
                    Marker.TEXT_VIEW_FACING,
                    [pick[0], pick[1], surface_z + 0.065],
                    [0.0, 0.0, 0.025],
                    [1.0, 0.70, 0.05, 1.0],
                    "A / PICK",
                ),
                self.marker(
                    5,
                    Marker.TEXT_VIEW_FACING,
                    [place[0], place[1], surface_z + 0.025],
                    [0.0, 0.0, 0.025],
                    [0.10, 0.70, 1.0, 1.0],
                    "B / PLACE",
                ),
            ]
        )

        vertical_lines = self.marker(
            6,
            Marker.LINE_LIST,
            [0.0, 0.0, 0.0],
            [0.004, 0.0, 0.0],
            [0.95, 0.95, 0.95, 0.70],
        )
        vertical_lines.points = [
            Point(x=pick[0], y=pick[1], z=pick[2]),
            Point(x=pick[0], y=pick[1], z=safe_z),
            Point(x=place[0], y=place[1], z=place[2]),
            Point(x=place[0], y=place[1], z=safe_z),
        ]
        markers.markers.append(vertical_lines)
        self.marker_publisher.publish(markers)

        pick_rpy = self.values("pick_orientation_rpy")
        place_rpy = self.values("place_orientation_rpy")
        # Quaternions are logged so the later task node can consume either form.
        pick_q = self.quaternion_from_rpy(pick_rpy)
        place_q = self.quaternion_from_rpy(place_rpy)
        self.get_logger().info(
            f"Task scene ready: A=({pick[0]:.3f}, {pick[1]:.3f}, {pick[2]:.3f}), "
            f"B=({place[0]:.3f}, {place[1]:.3f}, {place[2]:.3f}), "
            f"safe_z={safe_z:.3f}"
        )
        self.get_logger().info(f"Pick quaternion xyzw={pick_q}")
        self.get_logger().info(f"Place quaternion xyzw={place_q}")

    @staticmethod
    def quaternion_from_rpy(rpy: list[float]) -> tuple[float, float, float, float]:
        roll, pitch, yaw = rpy
        cr, sr = cos(roll / 2.0), sin(roll / 2.0)
        cp, sp = cos(pitch / 2.0), sin(pitch / 2.0)
        cy, sy = cos(yaw / 2.0), sin(yaw / 2.0)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneSetup()
    try:
        if node.apply_scene() and node.allow_manual_support_contact():
            node.publish_markers()
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
