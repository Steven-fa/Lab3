"""Start the mechArm MoveIt demo with the task 2 collision scene."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "firefighter", package_name="mecharm_270_m5_moveit_config"
    ).to_moveit_configs()
    demo_description = generate_demo_launch(moveit_config)

    task_share = get_package_share_directory("mecharm_task2")
    launch_description = LaunchDescription(
        [
            SetEnvironmentVariable("LANG", "zh_CN.UTF-8"),
            SetEnvironmentVariable("LANGUAGE", "zh_CN:zh"),
            *demo_description.entities,
        ]
    )
    launch_description.add_action(
        Node(
            package="mecharm_task2",
            executable="task2_scene.py",
            name="task2_scene",
            output="screen",
            parameters=[os.path.join(task_share, "config", "task2_points.yaml")],
        )
    )
    return launch_description
