#!/usr/bin/env bash
set -eo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-colcon-common-extensions python3-opencv python3-yaml python3-packaging ros-humble-desktop ros-humble-moveit ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-xacro ros-humble-joint-state-publisher-gui ros-humble-moveit-setup-assistant
