"""Launch the optimized demonstration in Gazebo without the RViz planning UI."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("mecharm_pick_place")
    experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "experiment.launch.py")
        ),
        launch_arguments={
            "gui": "true",
            "use_rviz": "false",
            "run_task": "false",
            "allow_support_contact": "false",
            "task_config": LaunchConfiguration("task_config"),
        }.items(),
    )
    replay = Node(
        package="mecharm_pick_place",
        executable="replay_keyframes",
        name="fast_keyframe_player",
        output="screen",
        condition=IfCondition(LaunchConfiguration("auto_start")),
        parameters=[
            LaunchConfiguration("task_config"),
            {
                "use_sim_time": True,
                "input_path": LaunchConfiguration("input_path"),
                "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                "acceleration_scaling": LaunchConfiguration(
                    "acceleration_scaling"
                ),
                "planning_time": 5.0,
                "planning_attempts": 8,
                "step_pause": 0.15,
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
                "input_path",
                default_value=os.path.join(package_share, "config", "side_pick_fast.yaml"),
            ),
            DeclareLaunchArgument("velocity_scaling", default_value="0.40"),
            DeclareLaunchArgument("acceleration_scaling", default_value="0.30"),
            DeclareLaunchArgument("auto_start", default_value="true"),
            experiment,
            # The player also waits for all interfaces and the workpiece. This delay
            # keeps startup logs readable while Gazebo and MoveIt finish loading.
            # The vendor Gazebo launch sends a three-second Home trajectory only
            # after all controllers are active. Start after that action finishes
            # so it cannot preempt the first replay trajectory.
            TimerAction(period=22.0, actions=[replay]),
        ]
    )
