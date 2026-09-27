@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_TASKS=/home/nvidia/mecharm_tasks"
set "REMOTE_WS=/home/nvidia/mecharm_experiment3_ws"

where ssh >nul 2>&1 || goto missing_ssh
where scp >nul 2>&1 || goto missing_ssh
where tar >nul 2>&1 || goto missing_tar

echo [DEPLOY] 上传实验三ROS 2包...
tar --exclude=__pycache__ --exclude=*.pyc -C "ros2" -cf - yolo_detection_ros2 mecharm_pick_place | ssh -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "mkdir -p '%REMOTE_WS%/src' && tar -m -C '%REMOTE_WS%/src' -xf -"
if errorlevel 1 goto failed

echo [DEPLOY] 上传真机运动、视觉标定和异常处理核心模块...
ssh -o BatchMode=yes "%JETSON_USER%@%JETSON_HOST%" "mkdir -p '%REMOTE_TASKS%' '%REMOTE_TASKS%/experiment3/models' '%REMOTE_TASKS%/experiment3/config' '%REMOTE_TASKS%/experiment3/current'"
if errorlevel 1 goto failed
for %%F in (hardware\*.py) do (
  scp -q -o BatchMode=yes "%%F" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/"
  if errorlevel 1 goto failed
)
for %%F in (scripts\*.py scripts\*.sh) do (
  scp -q -o BatchMode=yes "%%F" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/"
  if errorlevel 1 goto failed
)
for %%F in (models\*.pt) do (
  scp -q -o BatchMode=yes "%%F" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/experiment3/models/"
  if errorlevel 1 goto failed
)
scp -q -o BatchMode=yes "config\gripper_profiles.yaml" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/experiment3/config/gripper_profiles.yaml"
if errorlevel 1 goto failed

if exist "config\regions.yaml" scp -q -o BatchMode=yes "config\regions.yaml" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/experiment3/config/regions.yaml"
if exist "current\experiment3_poses.yaml" scp -q -o BatchMode=yes "current\experiment3_poses.yaml" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASKS%/experiment3/current/experiment3_poses.yaml"

echo [BUILD] 在Jetson构建ROS 2包...
ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "bash -lc 'source /opt/ros/humble/setup.bash && cd /home/nvidia/mecharm_experiment3_ws && colcon build --symlink-install --packages-select yolo_detection_ros2 mecharm_pick_place'"
if errorlevel 1 goto failed

echo [DONE] 实验三核心代码部署完成。
pause
exit /b 0

:missing_ssh
echo [ERROR] 未安装Windows OpenSSH客户端。
pause
exit /b 10

:missing_tar
echo [ERROR] Windows找不到tar命令。
pause
exit /b 11

:failed
echo [ERROR] 部署失败，请检查Jetson连接、SSH密钥和终端输出。
pause
exit /b 3
