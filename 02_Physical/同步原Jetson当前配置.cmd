@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"

where scp >nul 2>&1 || goto missing_ssh
if not exist "config" mkdir "config"
if not exist "current" mkdir "current"

echo [SYNC] 从原Jetson下载当前六区域、夹爪参数和17姿态...
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%:/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml" "config\regions.yaml"
if errorlevel 1 goto failed
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%:/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml" "config\gripper_profiles.yaml"
if errorlevel 1 goto failed
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%:/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml" "current\experiment3_poses.yaml"
if errorlevel 1 goto failed

echo [DONE] 当前设备专用配置已同步到本代码包。
pause
exit /b 0

:missing_ssh
echo [ERROR] 未安装Windows OpenSSH客户端。
pause
exit /b 10

:failed
echo [ERROR] 同步失败；Jetson必须开机并连接到192.168.55.1，且本机SSH密钥可用。
pause
exit /b 3

