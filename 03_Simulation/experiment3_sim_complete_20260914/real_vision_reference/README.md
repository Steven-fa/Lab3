# 六类桌面物体目标检测

本项目将录制视频抽帧并生成 YOLO 数据集，训练六类目标检测模型，供实验三的 ROS 2/Jetson 识别节点使用。类别编号固定如下：

0. `blue_cube` 蓝色立方体
1. `green_cube` 绿色立方体
2. `pink_cube` 粉色立方体
3. `yellow_cube` 黄色立方体
4. `red_cuboid` 红色长方体
5. `orange_bridge` 橙色桥状物体

## 已完成的数据流程

原始视频位于 `raw/videos`。运行下列命令会按 2 FPS 抽取单类视频、按 1.5 FPS 抽取组合视频，自动生成检测框，并按时间块拆分训练集、验证集和测试集：

```powershell
D:\Anaconda\envs\object_detection\python.exe scripts\build_dataset.py
```

数据集说明：

- `dataset/images/{train,val,test}`：抽取的图像。
- `dataset/labels/{train,val,test}`：YOLO 格式检测标注。
- `dataset/all`：便于浏览或用 LabelImg/其他工具复核的完整副本。
- `dataset/manifest.csv`：每张图像的来源视频、帧号、时间、划分和类别。
- `dataset/summary.json`：数据量与类别统计。
- `runs/dataset_qa`：随机抽取的带框质检图。

自动标注以物体颜色分割为基础；单类视频使用文件名类别先验，组合视频中的粉色立方体和红色长方体再按旋转轮廓长宽比区分。训练前如需提高实验报告的可信度，可重点人工复核 `runs/dataset_qa`、验证集和测试集。

当前数据库包含 625 张图像和 1078 个目标；训练、验证、测试图像数分别为 442、86、97。暖色物体相互接触时使用距离变换和分水岭分成独立实例，再区分粉色立方体与红色长方体。

## 训练与导出

```powershell
D:\Anaconda\envs\object_detection\python.exe scripts\train_yolo.py
```

默认使用 `yolo11n.pt`、640 像素输入、100 个 epoch、RTX 4080（CUDA 设备 0）。结果包括：

- `models/cube6_yolo11n_best.pt`：PyTorch/Ultralytics 权重。
- `models/cube6_yolo11n_best.onnx`：便于 Jetson 或 ROS 2 部署的 ONNX 模型。
- `models/metrics.json`：验证集和测试集指标。
- `runs/cube6_yolo11n`：训练曲线、混淆矩阵和权重。

本次最佳权重在独立测试集（97 张图像、151 个目标）上的 Precision 为 94.96%、Recall 为 90.49%、mAP50 为 93.18%、mAP50-95 为 89.49%。桌面 RTX 4080 测试的模型推理约为 3 ms/张；Jetson 是否达到实验要求的 5 FPS 必须在目标设备上实测。

针对实物摄像头中的角度、轻微遮挡及红粉混淆问题，实时和视频程序默认使用第二轮强化检测器 `models/cube6_red_focus_v2_best.pt`，并加载 `models/red_pink_shape_classifier_v2.pt` 做二次判别。强化检测器的测试集整体召回率为 92.09%，其中粉色立方体召回率 88.46%、红色长方体召回率 83.64%；专项分类器在独立测试裁剪集上的 Top-1 准确率为 94.12%。只有原类别为粉色、专项模型判断为红色且置信度达到 0.55 时才改成红色，已识别为红色的框不会被反向降级成粉色。

实时程序的默认检测阈值由 0.35 调整为 0.25，以保留遮挡或极端角度下置信度偏低的候选框。原高定位精度模型 `models/cube6_yolo11n_best.pt` 仍然保留，可通过 `--model` 手动指定。

## 视频推理

```powershell
D:\Anaconda\envs\object_detection\python.exe scripts\predict_video.py raw\videos\combination_04.mp4 `
  --output runs\combination_04_result.mp4 `
  --csv runs\combination_04_result.csv
```

输出视频实时叠加类别、检测框和置信度；CSV 保存每帧检测结果，可进一步转换为 ROS 2 的 `vision_msgs/Detection2DArray`。

## 实时摄像头测试

```powershell
D:\Anaconda\envs\object_detection\python.exe scripts\live_camera.py --camera 1
```

程序默认同时使用六类检测模型和红/粉专项分类器。按 `Q` 或 `Esc` 关闭，按 `S` 保存带检测结果的截图。

## 实验三对接要点

识别节点应订阅 `sensor_msgs/Image`，使用本项目模型推理，并发布 `vision_msgs/Detection2DArray`。任务节点根据检测框中心映射到固定取物网格，再按类别选择对应料盒。为了满足验收，应记录类别、置信度、网格编号、抓取结果、放置结果及异常状态，并由单个 Launch 文件启动相机、识别、机械臂和任务控制节点。
