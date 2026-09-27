from glob import glob
import os

from setuptools import find_packages, setup


package_name = "mecharm_pick_place"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", *glob("*.md")]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (
            os.path.join("share", package_name, "models", "workbench"),
            glob("models/workbench/*"),
        ),
        (
            os.path.join("share", package_name, "models", "workpiece"),
            glob("models/workpiece/*"),
        ),
        (
            os.path.join("share", package_name, "models", "placement_zone"),
            glob("models/placement_zone/*"),
        ),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="mechArm lab",
    maintainer_email="student@example.com",
    description="Fixed-point pick-and-place scene and parameters for mechArm 270.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "scene_setup = mecharm_pick_place.scene_setup:main",
            "pick_place_task = mecharm_pick_place.task_executor:main",
            "demo_recorder = mecharm_pick_place.demo_recorder:main",
            "keyframe_keyboard = mecharm_pick_place.keyframe_keyboard:main",
            "replay_keyframes = mecharm_pick_place.keyframe_player:main",
            "learn_demonstrations = mecharm_pick_place.demo_learner:main",
            "experiment3_task_controller = mecharm_pick_place.experiment3_task_controller:main",
            "experiment3_robot_worker = mecharm_pick_place.experiment3_robot_worker:main",
        ],
    },
)
