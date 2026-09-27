# mecharm_task2

Pure offline RViz entry point for the mechArm 270 M5 with adaptive gripper.
It deliberately starts no serial-port or real-hardware communication node.

```bash
source /opt/ros/humble/setup.bash
source ~/task2_ws/install/setup.bash
ros2 launch mecharm_task2 display.launch.py
```
