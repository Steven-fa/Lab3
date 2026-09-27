"""One-command ROS 2 launch for the complete real Experiment 3 pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("armed", default_value="false"),
        DeclareLaunchArgument(
            "model",
            default_value="/home/nvidia/mecharm_tasks/experiment3/models/cube6_red_focus_v4_video_negative_best.pt",
        ),
        DeclareLaunchArgument(
            "shape_model",
            default_value="/home/nvidia/mecharm_tasks/experiment3/models/red_pink_shape_classifier_v2.pt",
        ),
        DeclareLaunchArgument(
            "red_orange_model",
            default_value="/home/nvidia/mecharm_tasks/experiment3/models/red_orange_pair_classifier_v2.pt",
        ),
        DeclareLaunchArgument(
            "regions",
            default_value="/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml",
        ),
        DeclareLaunchArgument(
            "poses",
            default_value="/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml",
        ),
        DeclareLaunchArgument(
            "gripper_profiles",
            default_value="/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml",
        ),
        DeclareLaunchArgument(
            "hardware_root", default_value="/home/nvidia/mecharm_tasks"
        ),
        DeclareLaunchArgument("robot_host", default_value="192.168.100.2"),
        DeclareLaunchArgument("robot_port", default_value="9000"),
        DeclareLaunchArgument(
            "video",
            default_value="/home/nvidia/mecharm_tasks/experiment3/ros2_runs/latest_acceptance.avi",
        ),
        DeclareLaunchArgument(
            "task_log",
            default_value="/home/nvidia/mecharm_tasks/experiment3/ros2_runs/task_events.jsonl",
        ),
        DeclareLaunchArgument("preview_port", default_value="8091"),
    ]
    detector = Node(
        package="yolo_detection_ros2",
        executable="detector_node",
        name="experiment3_yolo_detector",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "model": LaunchConfiguration("model"),
                "shape_model": LaunchConfiguration("shape_model"),
                "shape_confidence": 0.55,
                "red_orange_model": LaunchConfiguration("red_orange_model"),
                "red_orange_confidence": 0.80,
                "red_orange_margin": 0.30,
                "camera": 0,
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fps": 30,
                "image_size": 640,
                "confidence": 0.25,
                "iou_threshold": 0.45,
                "device": "0",
                "topic": "/yolo/detections",
                "image_topic": "/camera/image_raw",
                "show": False,
                "save_path": LaunchConfiguration("video"),
                "regions_path": LaunchConfiguration("regions"),
                "preview_port": ParameterValue(
                    LaunchConfiguration("preview_port"), value_type=int
                ),
            }
        ],
    )
    controller = Node(
        package="mecharm_pick_place",
        executable="experiment3_task_controller",
        name="experiment3_task_controller",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "regions_path": LaunchConfiguration("regions"),
                "armed": ParameterValue(LaunchConfiguration("armed"), value_type=bool),
                "history_size": 7,
                "stable_votes": 5,
                "idle_complete_seconds": 5.0,
                "inter_object_delay": 5.0,
                "transfer_timeout_seconds": 180.0,
                "log_path": LaunchConfiguration("task_log"),
            }
        ],
    )
    worker = Node(
        package="mecharm_pick_place",
        executable="experiment3_robot_worker",
        name="experiment3_robot_worker",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "armed": ParameterValue(LaunchConfiguration("armed"), value_type=bool),
                "hardware_root": LaunchConfiguration("hardware_root"),
                "poses_path": LaunchConfiguration("poses"),
                "gripper_profiles_path": LaunchConfiguration("gripper_profiles"),
                "host": LaunchConfiguration("robot_host"),
                "port": ParameterValue(LaunchConfiguration("robot_port"), value_type=int),
                "arm_speed": 8,
                "stream_step": 2.0,
                "stream_period": 0.20,
                "retreat_extension": 0.50,
            }
        ],
    )
    stop_after_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=controller,
            on_exit=[EmitEvent(event=Shutdown(reason="Experiment 3 controller finished"))],
        )
    )
    stop_after_detector = RegisterEventHandler(
        OnProcessExit(
            target_action=detector,
            on_exit=[EmitEvent(event=Shutdown(reason="Experiment 3 detector exited"))],
        )
    )
    stop_after_worker = RegisterEventHandler(
        OnProcessExit(
            target_action=worker,
            on_exit=[EmitEvent(event=Shutdown(reason="Experiment 3 robot worker exited"))],
        )
    )
    return LaunchDescription(
        [
            *arguments,
            detector,
            controller,
            worker,
            stop_after_controller,
            stop_after_detector,
            stop_after_worker,
        ]
    )
