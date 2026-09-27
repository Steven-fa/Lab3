@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - Grid 1 Low Pose Repeatability
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_grid1_low_pose_repeat.py"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "OUTPUT_DIR=/home/nvidia/mecharm_tasks/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] safe -^> grid1 approach -^> grid1 grasp -^> approach -^> safe, 3 cycles.
echo [SAMPLE] J2, J5 and controller TCP are recorded at 1 s, 3 s and 6 s.
echo [GRIPPER] Stays OPEN for the entire test. No grasp command is sent.
echo [SAFETY] Remove every object from grid 1 and the whole path; keep a hand near E-stop.
echo [NOTICE] No motion occurs until you type RUN and press Enter.
echo.

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --poses '%POSES%' --host 192.168.100.2 --port 9000 --arm-speed 8 --repeats 3 --sample-times 1 3 6 --stream-step 2.0 --stream-period 0.20 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Grid 1 low-pose repeatability test completed.
) else (
  echo [STOPPED] Grid 1 low-pose repeatability test exited with code %RC%.
)
pause
exit /b %RC%
