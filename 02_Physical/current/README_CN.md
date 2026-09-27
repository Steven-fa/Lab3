# 当前17姿态文件位置

正式运行需要本目录下存在`experiment3_poses.yaml`。

- 使用原Jetson：运行上一级的`同步原Jetson当前配置.cmd`下载当前实测文件。
- 更换机械臂或夹爪：运行`start_experiment3_pose_recorder.cmd`重新采集，再将生成文件激活为`/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml`。

禁止把一台机械臂的低位抓取姿态直接用于另一台机械臂。

