@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - All Six Joint Repeatability Test
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_SCRIPT=%~dp0hardware\experiment3_joint_repeatability_test.py"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_all_joint_repeatability_test.py"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "OUTPUT_DIR=/home/nvidia/mecharm_tasks/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] J1 through J6 independently, safe pose +/-4 degrees.
echo [METHOD] Each joint approaches from both directions, 3 repeats per direction.
echo [SAMPLE] Read-only angle samples at 1 s, 3 s and 6 s after one target command.
echo [GRIPPER] Stays OPEN. Remove the cube and all objects from the path.
echo [SAFETY] Speed 6, one joint at a time; clear 30 cm and keep a hand near E-stop.
echo [TIME] The full six-joint test takes approximately 7 to 10 minutes.
echo [NOTICE] No motion occurs until you type RUN and press Enter.
echo.

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_SCRIPT%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT%"
if errorlevel 1 (
  echo [ERROR] Could not upload the all-joint diagnostic to Jetson.
  pause
  exit /b 3
)

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --poses '%POSES%' --host 192.168.100.2 --port 9000 --joints J1 J2 J3 J4 J5 J6 --arm-speed 6 --offset-deg 4 --repeats 3 --sample-times 1 3 6 --stream-step 1.0 --stream-period 0.25 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] All six-joint repeatability test completed.
) else (
  echo [STOPPED] All-joint test exited with code %RC%.
)
pause
exit /b %RC%
