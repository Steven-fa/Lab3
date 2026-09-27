import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _position(parameters, key):
    return [str(value) for value in parameters[key]]


def _launch_setup(context):
    package_share = get_package_share_directory("mecharm_pick_place")
    moveit_share = get_package_share_directory("mecharm_270_moveit_config")
    task_config = LaunchConfiguration("task_config").perform(context)
    with open(task_config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
        parameters = config.get("/**", config.get("scene_setup"))["ros__parameters"]

    table_xyz = _position(parameters, "table_position")
    object_xyz = _position(parameters, "object_position")
    zone_xyz = _position(parameters, "place_zone_position")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": os.path.join(package_share, "worlds", "pick_place.world"),
            "gui": LaunchConfiguration("gui"),
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items(),
    )
    spawn_table = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_workbench",
        output="screen",
        arguments=[
            "-entity",
            parameters["table_id"],
            "-file",
            os.path.join(package_share, "models", "workbench", "model.sdf"),
            "-x",
            table_xyz[0],
            "-y",
            table_xyz[1],
            "-z",
            table_xyz[2],
        ],
    )
    spawn_object = ExecuteProcess(
        output="screen",
        cmd=[
            "gz",
            "model",
            "--spawn-file",
            os.path.join(package_share, "models", "workpiece", "model.sdf"),
            "--model-name",
            parameters["object_id"],
            "-x",
            object_xyz[0],
            "-y",
            object_xyz[1],
            "-z",
            object_xyz[2],
        ],
    )
    spawn_zone = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_placement_zone",
        output="screen",
        arguments=[
            "-entity",
            "placement_zone",
            "-file",
            os.path.join(package_share, "models", "placement_zone", "model.sdf"),
            "-x",
            zone_xyz[0],
            "-y",
            zone_xyz[1],
            "-z",
            zone_xyz[2],
        ],
    )
    delayed_table = TimerAction(period=3.0, actions=[spawn_table])
    spawn_zone_after_table = RegisterEventHandler(
        OnProcessExit(target_action=spawn_table, on_exit=[spawn_zone])
    )
    spawn_object_after_zone = RegisterEventHandler(
        OnProcessExit(target_action=spawn_zone, on_exit=[spawn_object])
    )
    scene_setup = Node(
        package="mecharm_pick_place",
        executable="scene_setup",
        output="screen",
        parameters=[
            task_config,
            {
                "use_sim_time": True,
                "allow_support_contact": LaunchConfiguration(
                    "allow_support_contact"
                ),
            },
        ],
    )
    task_executor = Node(
        package="mecharm_pick_place",
        executable="pick_place_task",
        name="pick_place_task",
        output="screen",
        parameters=[task_config, {"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("run_task")),
    )
    start_task_after_object = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_object,
            on_exit=[TimerAction(period=1.0, actions=[task_executor])],
        )
    )
    # Stagger the three lab-model requests. Gazebo Classic can drop a service
    # response when several spawn requests arrive while the GUI is still loading.
    return [
        simulation,
        delayed_table,
        spawn_zone_after_table,
        spawn_object_after_zone,
        start_task_after_object,
        scene_setup,
    ]


def generate_launch_description():
    package_share = get_package_share_directory("mecharm_pick_place")
    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("run_task", default_value="true"),
            DeclareLaunchArgument("allow_support_contact", default_value="false"),
            DeclareLaunchArgument(
                "task_config",
                default_value=os.path.join(package_share, "config", "task_points.yaml"),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
