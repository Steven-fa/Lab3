#!/usr/bin/env python3
"""Publish the table and task-2 target object into the MoveIt planning scene."""

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class Task2Scene(Node):
    def __init__(self):
        super().__init__("task2_scene")
        self.declare_parameter("table_size", [0.80, 0.80, 0.05])
        self.declare_parameter("table_position", [0.15, 0.00, -0.025])
        self.declare_parameter("target_size", [0.04, 0.04, 0.04])
        self.declare_parameter("point_a", [0.18, 0.08, 0.02])
        self.declare_parameter("point_b", [0.18, -0.08, 0.02])
        self.declare_parameter("safe_height", 0.12)

        self.publisher = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.timer = self.create_timer(2.0, self.publish_scene)
        self.publish_count = 0

    @staticmethod
    def make_box(object_id, size, position):
        collision_object = CollisionObject()
        collision_object.header.frame_id = "base"
        collision_object.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(value) for value in size]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0

        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD
        return collision_object

    def publish_scene(self):
        table_size = self.get_parameter("table_size").value
        table_position = self.get_parameter("table_position").value
        target_size = self.get_parameter("target_size").value
        point_a = self.get_parameter("point_a").value

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            self.make_box("task2_table", table_size, table_position),
            self.make_box("task2_target", target_size, point_a),
        ]
        self.publisher.publish(scene)

        self.publish_count += 1
        if self.publish_count == 1:
            point_b = self.get_parameter("point_b").value
            safe_height = self.get_parameter("safe_height").value
            self.get_logger().info(
                "已载入实验二碰撞场景：桌面、A点目标物；"
                f"B点={list(point_b)}，安全高度={safe_height:.3f} m"
            )
        if self.publish_count >= 3:
            self.timer.cancel()


def main():
    rclpy.init()
    node = Task2Scene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
