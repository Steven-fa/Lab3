# 实验三原场景仿真（2026-09-14历史版本）

本目录记录原场景仿真：Gazebo俯视图像、六个固定网格、红蓝两类方块、图像检测、网格关联、分类区选择、抓放任务、自动循环、异常处理、独立验收与录像。

最新四方块两桥型仿真见 [桥型版本](../experiment3_sim_bridge_20260921/README_BRIDGE.md)。本目录的正常与异常结果仅对应2026年9月14日版本。

查看 [实验要求核查与验收报告](docs/实验要求核查与验收报告.md) 和 [红色归位轨迹修正说明](docs/红色归位轨迹修正.md)，其中包含本次测量、异常测试以及计划路径与实测关节角的对照。

本目录程序不连接真实机械臂。已完成的实机验收见 [实机材料](../../04_Evidence/) 和 [结果索引](../../05_Records/)。[历史真机验证计划](docs/真机待验证项.md)不代表当前实机完成状态。

## 最快启动

运行环境为 Ubuntu 22.04、ROS 2 Humble、Gazebo Classic 11。Windows 使用 WSL2 Ubuntu-22.04 与 WSLg。将整个目录复制到 Ubuntu 的用户目录后打开终端；Windows 挂载目录也可用。

首次构建：

```bash
bash 01_build.sh
```

一条命令启动全部节点并自动执行六物体分类：

```bash
bash 02_run_demo.sh normal
```

相机、检测、机械臂控制、状态机、日志和录像都由同一个 Launch 启动。等待加载完成即可，任务开始后不需要指定类别、网格或分类区，也不需要再次触发抓取。物体初始类别由场景随机种子打乱，控制器只根据检测消息决策。

直接 ROS 启动方式（先 source 工作区）：

```bash
source ~/task3_ws_20260914/install/setup.bash
ros2 launch mecharm_task3 full_sorting.launch.py run_dir:=$HOME/task3_demo scenario:=normal
```

直接 ROS 启动前还须设置本包 `scripts/env.sh` 中的模型路径、插件路径和隔离域；最方便的方式是在包目录先运行 `source scripts/env.sh`。`02_run_demo.sh` 已自动完成这些设置。

异常场景：先按 Ctrl+C 停止当前 Launch，再执行：

```bash
bash 02_run_demo.sh exceptions
```

此场景包含真实空网格、真实绿色未知物体，以及明确标记的不可达逆解和夹爪保持张开故障注入。不可达与抓取失败均要求返回安全位或停止，不能把错误当成功。

如需自动完成正常与异常两轮验收，确认没有同域同端口实例后执行 `python3 scripts/run_acceptance_suite.py`。它等待真实验收结果并关闭自身启动的实例，不会把启动成功当作验收通过。

默认打开 Gazebo，按 Ctrl+C 停止整套系统。没有图形桌面时，可尝试 `SIM_GUI=false bash 02_run_demo.sh normal`，但 Gazebo 相机仍需可用的渲染环境。可加 `rviz:=true` 显示 MoveIt 视图。

## 查看结果

每次运行产生独立 `results/normal_时间` 或 `results/exceptions_时间` 目录。任务结束后窗口保留，但录像自动封口保存。

录像按 8 FPS 编码。本机软件渲染与检测的实际出帧率较低，因此随包录像的播放时长短于实际运行时间；任务实际耗时以日志及 `acceptance.json` 为准。视频首尾帧与编码信息见 `docs/video_validation.json`。

- `acceptance.json`：独立验收结果，最终应为 `status: PASS`。
- `task_result.json`：任务状态、逐物体结果与分类计数。
- `simulation_demo.mp4`：实际 Gazebo 相机图像录像，叠加检测框和状态。
- `camera_initial.png`、`camera_final.png`：起始与结束截图。
- `detections.jsonl`：图像时间戳、类别、框与置信度。
- `task_states.jsonl`：状态转移、网格判断、请求和反馈。
- `motion_events.jsonl`、`jobs/`：抓取阶段、控制器结果、超时或错误恢复。
- `safety.jsonl`：关节采样与物理接触监测。
- `ground_truth.json`：仅用于场景生成和独立验收。检测器、任务控制器不读取它。

正常验收要求六个物体至少五个进入与真实类别对应的分类区，同时至少包含两类。验收监测通过 Gazebo 最终位置独立检查，不直接相信控制器报告的成功数。

## 配置与接口

固定网格、分类区落点、相机参数、HSV 检测模型、置信度门限、任务时限和状态机状态均在 `src/mecharm_task3/config/sorting.yaml`。修改配置后重新构建。正常场景的摆放顺序可用 `seed:=7` 等参数改变。

| 接口 | 类型 | 用途 |
|---|---|---|
| `/task3/camera/overhead/image_raw` | `sensor_msgs/Image` | Gazebo 俯视 RGB 图像 |
| `/task3/detections` | `vision_msgs/Detection2DArray` | 类别、置信度和检测框 |
| `/task3/annotated_image` | `sensor_msgs/Image` | 带检测框的图像 |
| `/task3/grasp/request` | `std_msgs/String` JSON | 带唯一 job_id 的长任务请求 |
| `/task3/grasp/status` | `std_msgs/String` JSON | 就绪、忙碌、阶段反馈与心跳 |
| `/task3/grasp/result` | `std_msgs/String` JSON | 关联 job_id 的结果或拒绝原因 |
| `/task3/grasp/cancel` | `std_msgs/String` JSON | 按 job_id 取消 |
| `/task3/state` | `std_msgs/String` JSON | 自动任务状态 |
| `/task3/emergency_stop` | `std_msgs/String` JSON | 监测异常时停止新任务并取消当前动作 |

这里使用实验要求允许的“等价的可反馈长时间任务接口”，并非自定义 ROS Action 类型。它实现忙碌拒绝、请求关联、重复请求过滤、阶段反馈、180 秒任务超时及取消。底层使用 ROS FollowJointTrajectory Action，MoveIt 提供逆解与碰撞检查。详细消息约定见 [接口说明](docs/接口和状态机.md)。

检测框字段按本机 Humble 的官方消息定义实现，中心为 `bbox.center.position.x/y`。[ROS vision_msgs 文档](https://docs.ros.org/en/ros2_packages/humble/api/vision_msgs/msg/BoundingBox2D.html)

## 检测模型和已有训练成果

默认仿真使用 `sorting.yaml` 中的 HSV 颜色模型，分类为 `red_cube` 与 `blue_cube`；绿色等有色目标发布 `unknown`。模型只读相机像素，不读物体名称、类别真值或物体坐标。置信度是颜色轮廓在旋转外接矩形中的覆盖率，避免正常旋转方块被水平框中的空白误降分；它不是经过概率校准的神经网络分数。发布的检测框仍为消息要求的水平矩形。面积和长宽比用于排除噪点和长条分类区底色。

`real_vision_reference/` 保留来自队友 `cube_detection.zip` 的实际训练权重和说明，包括强化检测模型、红粉分类器及相应原始脚本。该目录属于已有训练成果；原始说明中的指标不是本次重新测得。

识别节点提供可选 Ultralytics 后端，额外安装 `ultralytics` 后可使用：

```bash
bash 02_run_demo.sh normal detector:=yolo weights:="$PWD/real_vision_reference/models/cube6_red_focus_v2_best.pt"
```

该后端将已有模型的 `red_cuboid` 映射到红色分类区、`blue_cube` 映射到蓝色分类区，其他类别作为未知。模型在真实物体上训练，不保证直接适应 Gazebo 合成图像；本次仿真验收使用 HSV 后端。红粉二次分类器保留为真机识别参考，未作为本次仿真算法的一部分。

## 依赖和工作区

新电脑须先有 ROS 2 Humble 的 Ubuntu 软件源，再按需运行 `bash install_dependencies.sh`。不要把 Linux `.so` 当作 Windows 程序运行。

构建输出默认放在 `~/task3_ws_20260914`。可用 `TASK3_WS=/其他Linux目录` 覆盖，但构建与启动必须保持一致。源包需保留，重新构建会读取其中的源码。

脚本默认本机通信 `ROS_LOCALHOST_ONLY=1`、ROS 域 94、Gazebo 端口 11394。多个实例须使用不同域和端口；不要重复启动同域同端口的实例。脚本关闭外部 Gazebo 模型数据库，并携带机械臂、夹爪、地面和太阳资源。

## 仿真结果的适用范围

本包为固定工位取物：每个网格中方块的朝向与对应夹爪姿态对齐，类别仍由真实相机图像自动判断。当前夹爪开口不足以稳定处理 40 mm 方块的任意对角朝向，因此没有宣称实现任意方向抓取。

抓取插件先检查目标对位和闭爪状态，再建立物理约束搬运，不将物体瞬移到夹爪。它用于任务链和规划验收，不是摩擦抓取力学精度证明。

碰撞监测记录机器人与非允许物体之间超过 2 mm 的接触穿透；固定基座与桌面、夹爪与当前抓取物接触属于正常接触。MoveIt 同时负责场景、自碰撞和轨迹检查。关节监测允许 0.015 rad 的数值容差。结果只反映当前建模精度下的仿真，不代表实机安全标定或定位精度。
