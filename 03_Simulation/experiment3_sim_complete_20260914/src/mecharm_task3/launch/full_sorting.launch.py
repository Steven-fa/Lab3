"""Launch Gazebo, ros2_control, MoveIt and the task-2 scene."""

import os
import sys
from launch.actions import OpaqueFunction

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context):
    extra_share = get_package_share_directory("mecharm_task3")
    sys.path.insert(0, os.path.join(extra_share, "scripts"))
    from generate_world import generate
    config_file = os.path.join(extra_share,"config","sorting.yaml")
    run_dir = LaunchConfiguration("run_dir").perform(context)
    scenario = LaunchConfiguration("scenario").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    task_share = get_package_share_directory("mecharm_task2")
    description_file = os.path.join(
        extra_share, "urdf", "task3.urdf.xacro"
    )
    world_file = generate(config_file, scenario, seed, run_dir)
    points_file = os.path.join(task_share, "config", "task2_points.yaml")
    default_rviz = os.path.join(
        get_package_share_directory("moveit_setup_app_plugins"),
        "templates",
        "config",
        "moveit.rviz",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "firefighter", package_name="mecharm_270_m5_moveit_config"
        )
        .robot_description(file_path=description_file)
        .to_moveit_configs()
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
            )
        ),
        launch_arguments={
            "world": world_file,
            "gui": LaunchConfiguration("gui"),
            "pause": "false",
            "verbose": "false",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description, {"use_sim_time": True}],
    )
    world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base"],
        parameters=[{"use_sim_time": True}],
    )
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-topic",
            "robot_description",
            "-entity",
            "mecharm_270_m5",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "0",
        ],
    )

    controller_nodes = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_action_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
    ]

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )
    common = ["--config", config_file, "--run-dir", run_dir, "--scenario", scenario]
    extras = [Node(package="mecharm_task3", executable=name+".py", output="screen", arguments=common,
                   condition=IfCondition(LaunchConfiguration("autostart")) if name=="task_controller" else None)
              for name in ("grasp_bridge", "task_controller", "acceptance_monitor")]
    extras.append(Node(package="mecharm_task3",executable="detector.py",output="screen",
                  arguments=["--config",config_file,"--run-dir",run_dir,"--backend",LaunchConfiguration("detector"),"--weights",LaunchConfiguration("weights")]))
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", default_rviz],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            {"use_sim_time": True},
        ],
    )

    return [


            SetEnvironmentVariable("LANG", "zh_CN.UTF-8"),
            SetEnvironmentVariable("LANGUAGE", "zh_CN:zh"),
            gazebo,
            robot_state_publisher,
            world_to_base,
            TimerAction(period=3.0, actions=[spawn_robot]),
            TimerAction(period=7.0, actions=controller_nodes),
            TimerAction(period=10.0, actions=[move_group, *extras, rviz]),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("scenario",default_value="normal",choices=["normal","exceptions"]),
        DeclareLaunchArgument("seed",default_value="42"),
        DeclareLaunchArgument("run_dir",default_value=os.path.expanduser("~/task3_results/latest")),
        DeclareLaunchArgument("detector",default_value="hsv",choices=["hsv","yolo"]),
        DeclareLaunchArgument("weights",default_value=""),
        OpaqueFunction(function=launch_setup),
    ])
