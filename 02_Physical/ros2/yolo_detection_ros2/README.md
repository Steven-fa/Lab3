# YOLO ROS 2 Detection Publisher

This package opens a camera, runs the trained mouse and cup detector, displays
the annotated stream, saves an optional result video and publishes one
`vision_msgs/msg/Detection2DArray` message per frame.

Build it in a ROS 2 Humble workspace:

```bash
mkdir -p ~/yolo_ros2_ws/src
cp -r ros2/yolo_detection_ros2 ~/yolo_ros2_ws/src/
cd ~/yolo_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run the default Jetson configuration:

```bash
ros2 launch yolo_detection_ros2 detector.launch.py
```

Verify the published results in a second terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/yolo_ros2_ws/install/setup.bash
ros2 topic info /yolo/detections
ros2 topic echo /yolo/detections --once
ros2 topic hz /yolo/detections
```

Parameters such as the camera and model path can be overridden directly:

```bash
ros2 run yolo_detection_ros2 detector_node --ros-args \
  -p camera:=1 \
  -p model:=/home/nvidia/jetson_yolo/best.pt \
  -p mouse_confidence:=0.75 \
  -p cup_confidence:=0.75
```

For a repeatable integration check without a live camera, pass a recorded file
through the optional `source` parameter. When `source` is set, it takes
precedence over `camera` and the node exits after the final frame:

```bash
ros2 run yolo_detection_ros2 detector_node --ros-args \
  -p source:=/home/nvidia/jetson_yolo/results/test.avi \
  -p model:=/home/nvidia/jetson_yolo/best.pt \
  -p show:=false
```

To capture a submission-ready runtime record (topic metadata, one non-empty
detection message, topic frequency, node log and annotated video), run the
repository helper while a mouse or cup is visible:

```bash
cd /home/nvidia/jetson_yolo
bash scripts/capture_jetson_ros2_evidence.sh 0 \
  /home/nvidia/jetson_yolo/best.pt \
  /home/nvidia/yolo_ros2_ws
```
