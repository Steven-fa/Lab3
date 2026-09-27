# 实验三：桌面物体自动分类整理

## 系统结构

```text
/dev/video0
  -> yolo_detection_ros2/detector_node
  -> /camera/image_raw + /yolo/detections
  -> experiment3_task_controller（状态机）
  -> /experiment3/transfer_requests
  -> experiment3_robot_worker
  -> 固定点抓取、左/右分类放置
```

`experiment3_real.launch.py`可一次启动相机、YOLO、状态机和真机工作节点。

视觉端采用两级判别：v4主模型先检测六类物体和边界框；仅当候选属于
`red_cuboid / orange_bridge`时，再把目标裁剪图送入二级形状分类模型复核。
二级结果置信度低于0.80或与次高类别差值小于0.30时保留主模型结果，
避免为了修复红色长方体而误改其他四类。

## 分类规则

- `blue_cube / green_cube / pink_cube / yellow_cube` -> `cube` -> 右侧，夹爪CLOSE=69。
- `red_cuboid / orange_bridge` -> `other` -> 左侧，夹爪CLOSE=40。
- 释放统一OPEN=100。

## 首次部署

1. 连接Jetson，运行`同步原Jetson当前配置.cmd`备份原设备的区域和17姿态。
2. 运行`部署实验3到Jetson.cmd`上传源码、模型、配置并执行`colcon build`。
3. 若换了相机或桌面布局，运行`start_experiment3_region_calibrator.cmd`重新生成区域。
4. 若换了机械臂或夹爪，运行`start_experiment3_pose_recorder.cmd`和`start_experiment3_gripper_angle_test.cmd`重新采集。

## 正式启动

```powershell
& ".\start_experiment3_ros2_auto.cmd"
```

输入一次`RUN AUTO`后，类别、格号和左右去向全部由视觉与状态机确定。`start_experiment3_auto.cmd`是非ROS备用入口；`start_experiment3_motion.cmd`和`start_experiment3_all_grids.cmd`是固定点人工验证入口。

## 关键配置

- `config/regions.yaml`：六个图像区域，由区域标定器生成。
- `current/experiment3_poses.yaml`：17个真机姿态，由采点程序生成。
- `config/gripper_profiles.yaml`：分类夹爪参数。
- `models/*.pt`：当前v4主检测模型、红/粉辅助分类模型，以及红色长方体/桥形二级分类模型。

缺少区域或姿态文件时不要复制其他设备的数据硬跑，应重新标定。
