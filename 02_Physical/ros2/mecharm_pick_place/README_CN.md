# mechArm 270 定点抓取实验场景

本包按照《机械臂定点抓取实验要求》搭建工作台、固定取物点 A、标准抓取物、固定放置区 B 和安全高度，并把工作台与抓取物同步加入 MoveIt 规划场景。默认在环境就绪后自动执行一轮完整抓取。

## 一键启动

```bash
source /opt/ros/humble/setup.bash
source ~/mecharm_ws/install/setup.bash
ros2 launch mecharm_pick_place experiment.launch.py
```

自动动作顺序：张开夹爪 → A 点上方 → A 点下降 → 闭合并附着工件 → 抬起 → B 点上方 → B 点下降 → 释放工件 → 抬起 → Home。

如果只想加载模型和场景、暂不执行动作：

```bash
ros2 launch mecharm_pick_place experiment.launch.py run_task:=false
```

无界面验证：

```bash
ros2 launch mecharm_pick_place experiment.launch.py gui:=false use_rviz:=false
```

## 只需要修改的实验参数

编辑 `config/task_points.yaml`：

- `object_position`：Gazebo 中物体 A 的中心坐标。
- `place_zone_position`：放置区 B 的中心坐标。
- `pick_position` / `pick_orientation_rpy`：A 点工件中心及抓取朝向。
- `place_position` / `place_orientation_rpy`：B 点工件中心及放置朝向。
- `grasp_center_offset`：工件中心在 `gripper_base` 坐标系中的实测偏移；
  默认 `[0, 0.030, -0.055]`，对应自适应夹爪真实指尖位置，通常无需修改。
- `safe_height`：取放上方的统一安全高度。
- `gripper_open` / `gripper_closed`：夹爪开合位置。
- `velocity_scaling` / `acceleration_scaling`：任务速度与加速度比例。

任务节点会根据 `grasp_center_offset` 把工件中心目标换算为
`gripper_base` 目标，再通过 MoveIt 和夹爪 Action 执行动作。闭爪后程序读取
Gazebo 中的实际相对位姿：误差超过 20 mm 就停止，避免工件远距离“吸附”后
浮空移动；误差合格时保留闭爪瞬间的真实相对位姿，搬运过程不会跳变。

## 人工关键点示教与自动规划

示教模式不会自动运行抓取任务：

```bash
ros2 launch mecharm_pick_place teaching.launch.py demo_name:=side_pick_01
```

示教模式会启用对准检查和抓取辅助：只有闭爪误差不超过 20 mm 时工件才随
夹爪移动，并保留实际闭爪相对位姿，避免吸到掌部或突然跳动。

在第二个 WSL 终端运行键盘记录器：

```bash
ros2 run mecharm_pick_place keyframe_keyboard
```

`q` 保存普通点，`qa` 保存夹取点并闭爪，`qb` 保存释放点并开爪，`s` 保存。
程序只记录这些关键位置。使用 `replay_keyframes` 后，MoveIt 会自动规划相邻
关键点之间的完整轨迹。

完整人工操作步骤见 `MANUAL_TEACHING_CN.md`。
