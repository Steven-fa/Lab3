"""Launch the lab in manual-teaching mode with keyframe recording available."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("mecharm_pick_place")
    experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "experiment.launch.py")
        ),
        launch_arguments={
            "gui": "true",
            "use_rviz": "true",
            "run_task": "false",
            "allow_support_contact": "true",
            "task_config": LaunchConfiguration("task_config"),
        }.items(),
    )
    recorder = Node(
        package="mecharm_pick_place",
        executable="demo_recorder",
        name="demonstration_recorder",
        output="screen",
        parameters=[
            LaunchConfiguration("task_config"),
            {
                "use_sim_time": True,
                "output_root": LaunchConfiguration("output_root"),
                "demo_name": LaunchConfiguration("demo_name"),
                "auto_start": LaunchConfiguration("auto_start"),
                "recording_mode": "keyframes",
                "assist_grasp": True,
            },
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task_config",
                default_value=os.path.join(package_share, "config", "task_points.yaml"),
            ),
            DeclareLaunchArgument(
                "output_root",
                default_value=PathJoinSubstitution(
                    [EnvironmentVariable("HOME"), "mecharm_demos"]
                ),
            ),
            DeclareLaunchArgument("demo_name", default_value=""),
            DeclareLaunchArgument("auto_start", default_value="true"),
            experiment,
            TimerAction(period=8.0, actions=[recorder]),
        ]
    )
