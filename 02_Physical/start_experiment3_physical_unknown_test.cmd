@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Experiment 3 Physical UNKNOWN Test
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_SCRIPT_DIR=/home/nvidia/mecharm_tasks/scripts"
set "REMOTE_MODE="
if /I "%~1"=="reuse" set "REMOTE_MODE=reuse"

echo [TEST] Physical untrained object -^> UNKNOWN -^> SKIP_NO_MOTION
echo [SAFETY] DISARMED controller only. Robot worker is not started.
echo [SCENE] Grid 1 must be empty when this program starts.
echo.

ssh -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "mkdir -p %REMOTE_SCRIPT_DIR%"
if errorlevel 1 goto failed
scp -q "%~dp0scripts\experiment3_physical_unknown_test.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT_DIR%/"
if errorlevel 1 goto failed
scp -q "%~dp0scripts\run_experiment3_physical_unknown_test.sh" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT_DIR%/"
if errorlevel 1 goto failed

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "bash %REMOTE_SCRIPT_DIR%/run_experiment3_physical_unknown_test.sh %REMOTE_MODE%"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Physical UNKNOWN acceptance test passed.
) else (
  echo [STOPPED] Physical UNKNOWN acceptance test failed with code %RC%.
)
pause
exit /b %RC%

:failed
echo [ERROR] Could not deploy physical UNKNOWN test to Jetson.
pause
exit /b 3
