#!/usr/bin/env python3
"""Publish a stable unsupported-class detection for Experiment 3 exception QA.

This utility only publishes ``vision_msgs/Detection2DArray`` messages.  It has
no mechanical-arm or gripper imports and cannot send motion commands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def region_center(regions_path: Path, grid_id: int) -> tuple[float, float]:
    document = yaml.safe_load(regions_path.read_text(encoding="utf-8"))
    for region in document.get("regions", []):
        if int(region.get("id", -1)) != grid_id:
            continue
        polygon = region.get("polygon", [])
        if len(polygon) < 3:
            raise RuntimeError(f"区域{grid_id}的多边形点不足")
        return (
            sum(float(point[0]) for point in polygon) / len(polygon),
            sum(float(point[1]) for point in polygon) / len(polygon),
        )
    raise RuntimeError(f"regions.yaml中没有区域{grid_id}")


class UnknownDetectionPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("experiment3_unknown_exception_publisher")
        self.publisher = self.create_publisher(Detection2DArray, topic, 10)

    def publish_unknown(self, center: tuple[float, float]) -> None:
        message = Detection2DArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "experiment3_exception_test"

        detection = Detection2D()
        detection.header = message.header
        detection.id = "unsupported_class_exception_test"
        detection.bbox.center.position.x = center[0]
        detection.bbox.center.position.y = center[1]
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = 30.0
        detection.bbox.size_y = 30.0

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = "unrecognized_object"
        hypothesis.hypothesis.score = 0.99
        detection.results.append(hypothesis)
        message.detections.append(detection)
        self.publisher.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--topic", default="/yolo/detections")
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--period", type=float, default=0.10)
    args = parser.parse_args()

    center = region_center(args.regions, args.grid)
    rclpy.init()
    node = UnknownDetectionPublisher(args.topic)
    try:
        deadline = time.monotonic() + 5.0
        while node.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("等待任务控制节点订阅超时")
            rclpy.spin_once(node, timeout_sec=0.10)
        for _ in range(args.messages):
            node.publish_unknown(center)
            rclpy.spin_once(node, timeout_sec=args.period)
        print(
            f"[发布完成] grid={args.grid}, class=unrecognized_object, "
            f"center=({center[0]:.1f}, {center[1]:.1f}), messages={args.messages}"
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
