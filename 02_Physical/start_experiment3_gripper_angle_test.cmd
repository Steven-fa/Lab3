@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 - Gripper Class Profile Tester
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_SCRIPT=%~dp0hardware\experiment3_gripper_profile_tester.py"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_gripper_profile_tester.py"
set "REMOTE_CONFIG=/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [MODE] Interactive cube/other gripper position test.
echo [SAFETY] Put every test object on the table. Never suspend an object during calibration.
echo.

ssh -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "pgrep -f '[e]xperiment3_(robot_worker|all_grid_sequence|transfer_test|gripper_profile_tester)' > /dev/null"
if "%ERRORLEVEL%"=="0" goto controller_running

ssh -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "mkdir -p /home/nvidia/mecharm_tasks/experiment3/config"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_SCRIPT%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT%"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --host 192.168.100.2 --port 9000 --speed 20 --output '%REMOTE_CONFIG%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo [DONE] Gripper profile test closed normally.
) else (
    echo [STOPPED] Gripper profile tester exited with code %RC%.
)
pause
exit /b %RC%

:controller_running
echo [ERROR] A robot controller or another gripper tester is already running on Jetson.
echo [ACTION] Stop its terminal with Ctrl+C before starting a second control session.
pause
exit /b 4

:upload_error
echo [ERROR] Could not upload the gripper tester to Jetson.
pause
exit /b 3
