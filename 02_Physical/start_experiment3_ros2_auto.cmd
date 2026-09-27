@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Experiment 3 ROS 2 Autonomous Acceptance
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_PREVIEW_PORT=18091"
set "REMOTE_PREVIEW_PORT=8091"

echo [TASK] One ROS 2 Launch starts camera, YOLO, state machine and real-arm worker.
echo [AUTO] The camera selects grid, class and left/right destination automatically.
echo [SAFETY] Place objects in calibrated grids, clear both release paths, and keep E-stop ready.
echo [CONTINUOUS] Remove each released object during the 5-second interval; source grids may be refilled immediately.
echo [FINISH] The task ends only after at least one transfer and all six grids stay empty for 5 seconds.
echo [VIEW] Live detections will open at http://127.0.0.1:%LOCAL_PREVIEW_PORT%/vision
echo.

ssh -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "pgrep -f '[e]xperiment3_(robot_worker|all_grid_sequence|transfer_test|gripper_profile_tester)' > /dev/null"
if "%ERRORLEVEL%"=="0" goto controller_running

set /p "CONFIRM=Type RUN AUTO to arm the complete experiment: "
if /I not "%CONFIRM%"=="RUN AUTO" (
  echo [CANCELLED] ROS nodes were not started and no robot command was sent.
  pause
  exit /b 2
)

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:%LOCAL_PREVIEW_PORT%/vision'; for($i=0; $i -lt 60; $i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%LOCAL_PREVIEW_PORT%); $c.Close(); Start-Process $u; break } catch { Start-Sleep -Seconds 1 } }"
ssh -tt -L 127.0.0.1:%LOCAL_PREVIEW_PORT%:127.0.0.1:%REMOTE_PREVIEW_PORT% -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "bash -lc 'source /opt/ros/humble/setup.bash && source /home/nvidia/mecharm_experiment3_ws/install/setup.bash && mkdir -p /home/nvidia/mecharm_tasks/experiment3/ros2_runs && STAMP=$(date +%%Y%%m%%d_%%H%%M%%S) && ros2 launch mecharm_pick_place experiment3_real.launch.py armed:=true preview_port:=%REMOTE_PREVIEW_PORT% video:=/home/nvidia/mecharm_tasks/experiment3/ros2_runs/acceptance_$STAMP.avi task_log:=/home/nvidia/mecharm_tasks/experiment3/ros2_runs/task_$STAMP.jsonl'"
set "RC=%ERRORLEVEL%"
echo.
echo [STOPPED] ROS 2 Experiment 3 launch exited with code %RC%.
pause
exit /b %RC%

:controller_running
echo [ERROR] A robot controller or gripper tester is still running on Jetson.
echo [ACTION] Type quit in the gripper tester, or stop the previous controller with Ctrl+C.
pause
exit /b 4
