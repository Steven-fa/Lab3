"""Display the mechArm 270 M5 and adaptive gripper without real hardware."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    task_share = get_package_share_directory("mecharm_task2")
    description_share = get_package_share_directory("mycobot_description")

    default_model = os.path.join(
        description_share,
        "urdf",
        "mecharm_270_m5",
        "mecharm_270_m5_adaptive_gripper.urdf",
    )
    default_rviz_config = os.path.join(
        task_share,
        "config",
        "mecharm_display.rviz",
    )

    model = LaunchConfiguration("model")
    rviz_config = LaunchConfiguration("rvizconfig")
    robot_description = ParameterValue(Command(["xacro ", model]), value_type=str)

    return LaunchDescription(
        [
            SetEnvironmentVariable("LANG", "zh_CN.UTF-8"),
            SetEnvironmentVariable("LANGUAGE", "zh_CN:zh"),
            SetEnvironmentVariable("LC_MESSAGES", "zh_CN.UTF-8"),
            DeclareLaunchArgument(
                "model",
                default_value=default_model,
                description="URDF or Xacro model to display",
            ),
            DeclareLaunchArgument(
                "rvizconfig",
                default_value=default_rviz_config,
                description="RViz configuration file",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
