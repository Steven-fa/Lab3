@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 Pose Recorder
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_pose_recorder.py"
set "REMOTE_OUTPUT=/home/nvidia/mecharm_tasks/experiment3/recordings"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [MODE] Experiment 3 guided 17-pose teaching recorder
echo [SAFETY] Hold the robot with both hands before pressing f.
echo [NOTICE] Recording keys do not require Enter.
echo.

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --host 192.168.100.2 --port 9000 --output-root '%REMOTE_OUTPUT%' --grid-count 6"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Experiment 3 pose recording closed normally.
) else (
  echo [STOPPED] Experiment 3 recorder exited with code %RC%.
)
pause
exit /b %RC%
