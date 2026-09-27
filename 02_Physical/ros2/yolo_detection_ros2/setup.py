from glob import glob
from setuptools import find_packages, setup


package_name = "yolo_detection_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Student",
    maintainer_email="student@example.com",
    description="Publish camera images and generic YOLO detections as ROS 2 messages.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = yolo_detection_ros2.detector_node:main",
        ],
    },
)
