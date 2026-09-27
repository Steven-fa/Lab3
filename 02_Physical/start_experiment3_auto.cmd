@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 - Autonomous Vision Sorting
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_ROOT=/home/nvidia/mecharm_tasks"
set "REMOTE_PYTHON=/home/nvidia/venvs/yolo/bin/python"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "MODEL=/home/nvidia/mecharm_tasks/experiment3/models/cube6_red_focus_v4_video_negative_best.pt"
set "SHAPE_MODEL=/home/nvidia/mecharm_tasks/experiment3/models/red_pink_shape_classifier_v2.pt"
set "RED_ORANGE_MODEL=/home/nvidia/mecharm_tasks/experiment3/models/red_orange_pair_classifier_v2.pt"
set "REGIONS=/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml"
set "GRIPPER_PROFILES=/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
set "RUN_ROOT=/home/nvidia/mecharm_tasks/experiment3/autonomous_runs"
set "LOCAL_PORT=18091"
set "REMOTE_PORT=8091"

echo [TASK] Fully automatic Experiment 3: camera - YOLO - grid - class - robot.
echo [ROUTING] cube to right, other to left. No class or grid input is accepted.
echo [SAFETY] One RUN AUTO confirmation is required before any robot command.
echo [VIEW] http://127.0.0.1:%LOCAL_PORT%/vision
echo.

ssh -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "pgrep -f '[e]xperiment3_(autonomous_task|vision_preview|region_calibrator).py' > /dev/null"
if "%ERRORLEVEL%"=="0" goto already_running

for %%F in (experiment3_autonomous_task.py experiment3_vision_preview.py experiment3_region_calibrator.py experiment3_transfer_test.py experiment3_directional_compensation.py experiment3_grid_dry_run.py experiment3_gripper_profiles.py mecharm_recorded_task.py) do (
  scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\%%F" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/%%F"
  if errorlevel 1 goto upload_error
)

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:%LOCAL_PORT%/vision'; for($i=0; $i -lt 60; $i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%LOCAL_PORT%); $c.Close(); Start-Process $u; break } catch { Start-Sleep -Seconds 1 } }"
ssh -tt -L 127.0.0.1:%LOCAL_PORT%:127.0.0.1:%REMOTE_PORT% -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "'%REMOTE_PYTHON%' -u '%REMOTE_ROOT%/experiment3_autonomous_task.py' --poses '%POSES%' --model '%MODEL%' --shape-model '%SHAPE_MODEL%' --red-orange-model '%RED_ORANGE_MODEL%' --regions '%REGIONS%' --gripper-profiles '%GRIPPER_PROFILES%' --run-root '%RUN_ROOT%' --device /dev/video0 --width 1280 --height 720 --fps 30 --preview-port %REMOTE_PORT% --host 192.168.100.2 --port 9000 --arm-speed 8 --stream-step 2.0 --stream-period 0.20 --retreat-extension 0.50 --inter-object-delay 5"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Autonomous visual sorting finished. Review the YAML log and MP4 video shown above.
) else (
  echo [STOPPED] Autonomous task exited with code %RC%. No later target is executed after a failed transfer.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the latest autonomous task files to Jetson.
pause
exit /b 3

:already_running
echo [ERROR] A camera or autonomous Experiment 3 process is already running on Jetson.
echo [ACTION] Close its terminal with Ctrl+C, then run this launcher again.
pause
exit /b 4
