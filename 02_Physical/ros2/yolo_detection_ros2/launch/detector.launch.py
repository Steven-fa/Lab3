from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="yolo_detection_ros2",
                executable="detector_node",
                name="yolo_detection_publisher",
                output="screen",
                parameters=[
                    {
                        "model": "/home/nvidia/jetson_yolo/best.pt",
                        "camera": 0,
                        "image_size": 768,
                        "confidence": 0.75,
                        "topic": "/yolo/detections",
                        "image_topic": "/camera/image_raw",
                        "show": True,
                        "save_path": "/home/nvidia/jetson_yolo/results/ros2_detected.avi",
                    }
                ],
            )
        ]
    )
