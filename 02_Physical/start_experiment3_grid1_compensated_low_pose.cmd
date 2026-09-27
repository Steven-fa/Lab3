@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - Grid 1 Compensated Low Pose Validation
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_MAIN=%~dp0hardware\experiment3_grid1_compensated_low_pose_repeat.py"
set "LOCAL_COMP=%~dp0hardware\experiment3_directional_compensation.py"
set "LOCAL_REPEAT=%~dp0hardware\experiment3_grid1_low_pose_repeat.py"
set "LOCAL_BASE=%~dp0hardware\experiment3_grid_dry_run.py"
set "LOCAL_CONFIG=%~dp0hardware\config\experiment3_joint_compensation.yaml"
set "REMOTE_ROOT=/home/nvidia/mecharm_tasks"
set "REMOTE_MAIN=%REMOTE_ROOT%/experiment3_grid1_compensated_low_pose_repeat.py"
set "REMOTE_COMP=%REMOTE_ROOT%/experiment3_directional_compensation.py"
set "REMOTE_REPEAT=%REMOTE_ROOT%/experiment3_grid1_low_pose_repeat.py"
set "REMOTE_BASE=%REMOTE_ROOT%/experiment3_grid_dry_run.py"
set "REMOTE_CONFIG=%REMOTE_ROOT%/experiment3/config/joint_compensation.yaml"
set "POSES=%REMOTE_ROOT%/experiment3/current/experiment3_poses.yaml"
set "OUTPUT_DIR=%REMOTE_ROOT%/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] Validate PASSed J2/J3/J4 direction compensation on grid 1 low route.
echo [PHYSICAL TARGET] The original 17 recorded poses are unchanged.
echo [GRIPPER] Stays OPEN for all three cycles; no grasp command is sent.
echo [SAFETY] Remove all objects, clear 30 cm, keep a hand near E-stop.
echo [NOTICE] No motion occurs until you type RUN and press Enter.
echo.

ssh -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "mkdir -p '%REMOTE_ROOT%/experiment3/config' '%OUTPUT_DIR%'"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_MAIN%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_MAIN%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_COMP%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_COMP%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_REPEAT%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_REPEAT%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_BASE%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_BASE%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_CONFIG%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_CONFIG%"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_MAIN%' --poses '%POSES%' --compensation '%REMOTE_CONFIG%' --host 192.168.100.2 --port 9000 --arm-speed 6 --repeats 3 --sample-times 1 3 6 --stream-step 1.0 --stream-period 0.25 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Compensated grid-1 low-pose validation completed.
) else (
  echo [STOPPED] Compensated low-pose validation exited with code %RC%.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the compensated low-pose validator to Jetson.
pause
exit /b 3
