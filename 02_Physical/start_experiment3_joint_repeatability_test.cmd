@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - J2 J5 Repeatability Test
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_joint_repeatability_test.py"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "OUTPUT_DIR=/home/nvidia/mecharm_tasks/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] J2 and J5 approach the same safe pose from both directions, 3 repeats each.
echo [SAMPLE] Read-only angle samples at 1 s, 3 s and 6 s after each target command.
echo [SAFETY] Remove all objects, clear 30 cm around the arm and keep a hand near E-stop.
echo [NOTICE] No motion occurs until you type RUN and press Enter.
echo.

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --poses '%POSES%' --host 192.168.100.2 --port 9000 --arm-speed 8 --offset-deg 6 --repeats 3 --sample-times 1 3 6 --stream-step 2.0 --stream-period 0.20 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] J2 J5 repeatability test completed.
) else (
  echo [STOPPED] J2 J5 repeatability test exited with code %RC%.
)
pause
exit /b %RC%
