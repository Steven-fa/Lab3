# mechArm 270 关键点示教与自动规划

这套流程只保存操作者指定的关键位置，不再以 20 Hz 记录整段轨迹。每个关键点
同时保存六个关节角、夹爪位置、末端 XYZ/四元数和工件位姿。回放时 MoveIt 按
关键点顺序逐段规划，中间轨迹由规划器自动生成。

## 一、启动 WSL 和示教环境

先关闭仍在运行的旧 Gazebo：在它所在的 WSL 终端按 `Ctrl+C`。然后在
PowerShell 输入：

```powershell
wsl -d Ubuntu-22.04
```

进入 Ubuntu 后执行：

```bash
source /opt/ros/humble/setup.bash
cd ~/mecharm_ws
colcon build --packages-select mecharm_pick_place --symlink-install
source install/setup.bash
ros2 launch mecharm_pick_place teaching.launch.py demo_name:=side_pick_01
```

该命令会启动 Gazebo、RViz、MoveIt 和示教记录器，但不会自动执行抓取程序。
默认记录目录是 `~/mecharm_demos`。

## 二、打开第二个 WSL 终端

在另一个 PowerShell 窗口执行：

```powershell
wsl -d Ubuntu-22.04
```

然后加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
source ~/mecharm_ws/install/setup.bash
```

启动无须按 Enter 的关键点键盘：

```bash
ros2 run mecharm_pick_place keyframe_keyboard
```

教学环境默认自动开始一份关键点记录。键盘程序显示“关键点键盘已就绪”后即可
操作。需要检查状态时可执行：

```bash
ros2 service call /mecharm_demo/status std_srvs/srv/Trigger "{}"
```

看到“关键点录制中”即可。

## 三、在 RViz 中手动操作机械臂

1. 在 RViz 左侧找到 `MotionPlanning` 面板，规划组选择 `arm`。
2. `Start State` 使用 `Current State`。
3. 在三维画面中找到夹爪末端的六自由度交互标记：
   - 拖动直线箭头改变 X/Y/Z；
   - 拖动圆环改变滚转、俯仰和偏航；
   - 每次只移动一小段，先点击 `Plan` 检查，再点击 `Execute`；也可使用
     `Plan & Execute`。
4. 如果目标机械臂显示为红色，说明存在碰撞或不可达，不要执行；把目标稍微
   抬高或移近机械臂后重新规划。
5. 到达重要位置后使用第二个终端记录：`q` 为普通点，快速连续按 `qa` 为
   夹取点，快速连续按 `qb` 为释放点。按键不需要 Enter。

建议按以下顺序完成一次正确示范：

1. Home，夹爪张开；需要把 Home 加入回放时按 `q`。
2. 移动到 A 点侧向预抓位置，按 `q`。
3. 沿水平方向慢慢接近物体，让物体位于两个指尖之间，不进入夹爪掌部。
4. 到达正确夹取位置后快速按 `qa`；程序会保存夹取位姿并自动闭合夹爪。
   - 示教辅助器会计算工件与正确指尖中心的误差；不超过 20 mm 才锁定工件；
   - 锁定时保留闭爪瞬间的真实相对位姿，所以工件不会跳进掌部或横向漂移；
   - 若终端提示“拒绝浮空吸附”，先张开夹爪，校正位置后重新闭合。
5. 先垂直抬起 4～7 cm，确认物体离开桌面，按 `q`。
6. 移动到 B 点上方，按 `q`。
7. 垂直下降到放置高度。
8. 到达释放位置后快速按 `qb`；程序会保存释放位姿并自动张开夹爪。
9. 抬起夹爪到撤离点，按 `q`；需要回 Home 时再回 Home 并按 `q`。

当前侧夹几何的参考值如下，单位为米；它们是 `gripper_base` 的目标，不是
工件中心：

| 位置 | X | Y | Z | RPY |
|---|---:|---:|---:|---|
| A 预抓 | 0.18 | -0.045 | 0.12 | `[-1.5708, 0, 0]` |
| A 夹取 | 0.18 | -0.045 | 0.05 | `[-1.5708, 0, 0]` |
| B 上方 | 0.18 | 0.045 | 0.12 | `[-1.5708, 0, 3.1416]` |
| B 放置 | 0.18 | 0.045 | 0.05 | `[-1.5708, 0, 3.1416]` |

这里的 5.5 cm 水平偏移来自厂家夹爪真实指尖长度。A 点指尖朝负 Y 方向伸出，
B 点为镜像姿态。

## 四、q / qa / qb 的含义

- `q`：保存当前末端位置和姿态，不改变夹爪；程序等待约 0.55 秒确认没有
  后续 `a/b`，所以普通点会略有延迟。
- `qa`：保存为 `grasp`，随后自动发送 `gripper_closed` 闭爪命令。
- `qb`：保存为 `release`，随后自动发送 `gripper_open` 张开命令。
- `s`：保存并结束本次记录。
- `Esc`：保存并退出键盘程序。

每次按键后都会立即覆盖写入 `keyframes.yaml`，即使窗口意外关闭，已经按过的
关键点也不会丢失。

## 五、保存并查看结果

动作全部完成后，在关键点键盘中按 `s` 或 `Esc`。也可以使用服务：

```bash
ros2 service call /mecharm_demo/stop_recording std_srvs/srv/Trigger "{}"
```

结果文件是 `~/mecharm_demos/<本次名称>/keyframes.yaml`。其中只包含人工选中
的关键点，不包含冗余的连续轨迹。

## 六、按关键点自动规划并回放

保存后先退出关键点键盘，再运行：

```bash
ros2 run mecharm_pick_place replay_keyframes --ros-args \
  -p input_path:=$HOME/mecharm_demos/side_pick_01/keyframes.yaml
```

回放器会逐点调用 MoveIt：普通点只移动机械臂；`grasp` 点到达后闭爪并带走
工件；`release` 点到达后开爪并恢复工件到规划场景。相邻关键点之间不使用
人工轨迹，而是重新进行碰撞检测和路径规划。
