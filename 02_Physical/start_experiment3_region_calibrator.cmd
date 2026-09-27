@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 Region Calibration
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_region_calibrator.py"
set "REMOTE_CONFIG=/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml"
set "LOCAL_PORT=18090"
set "REMOTE_PORT=8090"

echo [MODE] Six-region camera calibration only; the robot will not move.
echo [CAMERA] /dev/video0, 1280x720
echo [OUTPUT] %REMOTE_CONFIG%
echo [OPEN] http://127.0.0.1:%LOCAL_PORT%/calibrate
echo [NOTICE] Keep this terminal open while calibrating. Ctrl+C stops the camera server.
echo.

start "" "http://127.0.0.1:%LOCAL_PORT%/calibrate"
ssh -tt -L 127.0.0.1:%LOCAL_PORT%:127.0.0.1:%REMOTE_PORT% -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --device /dev/video0 --width 1280 --height 720 --fps 30 --port %REMOTE_PORT% --output '%REMOTE_CONFIG%'"
set "RC=%ERRORLEVEL%"
echo.
echo [STOPPED] Region calibration server exited with code %RC%.
pause
exit /b %RC%
