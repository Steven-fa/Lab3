@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - J3 J4 J5 Isolation Test
cd /d "%~dp0"

set "TEST_JOINT=%~1"
if not defined TEST_JOINT set "TEST_JOINT=J5"
if /I "%TEST_JOINT%"=="J3" set "TEST_JOINT=J3"& goto joint_ok
if /I "%TEST_JOINT%"=="J4" set "TEST_JOINT=J4"& goto joint_ok
if /I "%TEST_JOINT%"=="J5" set "TEST_JOINT=J5"& goto joint_ok
echo [ERROR] Joint must be J3, J4 or J5.
echo [USAGE] %~nx0 J5
pause
exit /b 2

:joint_ok
set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_SCRIPT=%~dp0hardware\experiment3_joint_repeatability_test.py"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_joint_isolation_test.py"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "OUTPUT_DIR=/home/nvidia/mecharm_tasks/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] %TEST_JOINT% only, safe pose +/-4 degrees, 3 repeats from each direction.
echo [SAMPLE] Read-only angle samples at 1 s, 3 s and 6 s after one target command.
echo [GRIPPER] Stays OPEN. Remove the cube and every other object from the path.
echo [SAFETY] Clear 30 cm around the arm and keep a hand near E-stop.
echo [NOTICE] Uploading the diagnostic does not move the robot.
echo.

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_SCRIPT%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT%"
if errorlevel 1 (
  echo [ERROR] Could not upload the diagnostic to Jetson.
  pause
  exit /b 3
)

echo [READY] Diagnostic uploaded. No motion occurs until you type RUN and press Enter.
echo.
ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --poses '%POSES%' --host 192.168.100.2 --port 9000 --joints %TEST_JOINT% --arm-speed 6 --offset-deg 4 --repeats 3 --sample-times 1 3 6 --stream-step 1.0 --stream-period 0.25 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] %TEST_JOINT% isolation test completed.
) else (
  echo [STOPPED] %TEST_JOINT% isolation test exited with code %RC%.
)
pause
exit /b %RC%
