# 设备专用配置

`regions.example.yaml`只是安全的空模板，不能用于正式运行。

连接原Jetson后运行上一级目录中的`同步原Jetson当前配置.cmd`，会下载：

- `config/regions.yaml`
- `config/gripper_profiles.yaml`
- `current/experiment3_poses.yaml`

如果更换相机、机械臂底座位置或夹爪安装方式，必须使用标定与采点程序重新生成，不要照搬原设备点位。

