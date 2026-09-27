@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 Vision Only
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_PYTHON=/home/nvidia/venvs/yolo/bin/python"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_vision_preview.py"
set "REMOTE_MODEL=/home/nvidia/mecharm_tasks/experiment3/models/cube6_red_focus_v4_video_negative_best.pt"
set "REMOTE_SHAPE=/home/nvidia/mecharm_tasks/experiment3/models/red_pink_shape_classifier_v2.pt"
set "REMOTE_RED_ORANGE=/home/nvidia/mecharm_tasks/experiment3/models/red_orange_pair_classifier_v2.pt"
set "REMOTE_REGIONS=/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml"
set "LOCAL_PORT=18090"
set "REMOTE_PORT=8090"

echo [MODE] YOLO + six regions, VISION ONLY. The robot will not move.
echo [CAMERA] /dev/video0, 1280x720
echo [OPEN] http://127.0.0.1:%LOCAL_PORT%/vision
echo [NOTICE] Stop the calibration window first. Keep this terminal open while testing.
echo.

start "" "http://127.0.0.1:%LOCAL_PORT%/vision"
ssh -tt -L 127.0.0.1:%LOCAL_PORT%:127.0.0.1:%REMOTE_PORT% -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "'%REMOTE_PYTHON%' -u '%REMOTE_SCRIPT%' --device /dev/video0 --width 1280 --height 720 --fps 30 --port %REMOTE_PORT% --model '%REMOTE_MODEL%' --shape-model '%REMOTE_SHAPE%' --red-orange-model '%REMOTE_RED_ORANGE%' --regions '%REMOTE_REGIONS%' --confidence 0.25 --shape-confidence 0.55 --red-orange-confidence 0.80 --red-orange-margin 0.30 --cuda-device 0 --history-size 7 --stable-votes 5"
set "RC=%ERRORLEVEL%"
echo.
echo [STOPPED] Vision-only preview exited with code %RC%.
pause
exit /b %RC%
