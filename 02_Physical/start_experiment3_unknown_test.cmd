@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Experiment 3 UNKNOWN Exception Test
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT_DIR=/home/nvidia/mecharm_tasks/scripts"

echo [TEST] UNKNOWN -^> SKIP_NO_MOTION
echo [SAFETY] DISARMED controller only. Robot worker is not started.
echo [NOTICE] This validates the unsupported-class ROS 2 interface path; no robot motion is possible.
echo.

ssh -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "mkdir -p %REMOTE_SCRIPT_DIR%"
if errorlevel 1 goto failed

scp -q "%~dp0scripts\experiment3_unknown_exception_test.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT_DIR%/"
if errorlevel 1 goto failed
scp -q "%~dp0scripts\run_experiment3_unknown_exception_test.sh" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT_DIR%/"
if errorlevel 1 goto failed

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "bash %REMOTE_SCRIPT_DIR%/run_experiment3_unknown_exception_test.sh"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] UNKNOWN exception test passed.
) else (
  echo [STOPPED] UNKNOWN exception test failed with code %RC%.
)
pause
exit /b %RC%

:failed
echo [ERROR] Could not deploy the safe UNKNOWN exception test to Jetson.
pause
exit /b 3
