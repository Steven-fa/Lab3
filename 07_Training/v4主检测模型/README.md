# 实验 3 YOLO v4：红色长方体与背景负样本强化

## 结论

本模型从 `cube6_red_focus_v3_hard_negative_best.pt` 继续训练，保留原六类数据，新增：

- 第一段视频：红色长方体正样本 88 张（train/val/test = 62/13/13）。
- 第二段视频：确认不含六类目标的负样本 29 张（train/val/test = 20/5/4）。
- 最佳权重来自第 33/35 轮，验证集 mAP50-95 = 0.9642。

## v3 与 v4 对照

| 测试项目 | v3 | v4 |
|---|---:|---:|
| 新红色长方体测试帧召回（IoU≥0.5、conf≥0.5） | 0/13 | 13/13 |
| 新红色长方体平均置信度 | 0.000 | 0.881 |
| 第二视频负样本误报图像数（conf≥0.25） | 4/4 | 0/4 |
| 含新增样本的完整测试集 mAP50-95 | 0.8336 | 0.8755 |
| 原 v3 测试集 mAP50-95 | 0.8699 | 0.8800 |

原 v3 测试集上的六类回归结果均稳定：蓝、绿、黄、红和橙色类提升；粉色类变化约 -0.0009，属于测量波动。

## 最佳模型

- Windows：`F:\PycharmProjects\robomaster\artifacts\experiment3_v4_red_negative_20260913\cube6_red_focus_v4_video_negative_best.pt`
- Jetson：`/home/nvidia/mecharm_tasks/experiment3/models/cube6_red_focus_v4_video_negative_best.pt`
- SHA-256：`5b775fd632042bda64e8034bf76402b5f9923a9c02a2bb06f6655c0a78d31e08`

当前自动分类程序仍保持之前指定的 v2 权重，本次训练没有擅自切换在线模型。

## 可复现文件

- `build_experiment3_v4_video_dataset.py`：复制 v3 数据并提取/标注两段视频。
- `train_experiment3_v4_video_focus.py`：从 v3 低学习率续训。
- `resume_experiment3_v4.py`：SSH 中断后从 `last.pt` 恢复。
- `evaluate_experiment3_v4_video.py`：新红色正样本、负样本和完整测试集评测。
- `v4_manifest.json`：视频哈希、抽帧索引、分组和自动框质量信息。
- `evaluation_v3_baseline.json`、`evaluation_v4.json`、`evaluation_v4_on_original_v3_test.json`：评测明细。
- `qa_red_cuboid.jpg`、`qa_negative.jpg`：样本与标注可视化检查。
- `results.csv`、`results.png`、`confusion_matrix_normalized.png`：训练曲线和混淆矩阵。

## 数据完整性

所有图像均有对应标签，类别编号限制在 0–5，YOLO 坐标均位于 0–1。红色长方体框由保守 HSV 红色掩膜生成并进行面积、填充率和边界检查；负样本使用空标签。测试帧按时间段与训练帧分离。
