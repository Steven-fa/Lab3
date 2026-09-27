@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 - New Pose Motion Controller
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_SCRIPT=%~dp0hardware\experiment3_transfer_test.py"
set "LOCAL_COMP=%~dp0hardware\experiment3_directional_compensation.py"
set "LOCAL_BASE=%~dp0hardware\experiment3_grid_dry_run.py"
set "LOCAL_TASK=%~dp0hardware\mecharm_recorded_task.py"
set "LOCAL_GRIPPER_PROFILES=%~dp0hardware\experiment3_gripper_profiles.py"
set "REMOTE_SCRIPT=/home/nvidia/mecharm_tasks/experiment3_transfer_test.py"
set "REMOTE_COMP=/home/nvidia/mecharm_tasks/experiment3_directional_compensation.py"
set "REMOTE_BASE=/home/nvidia/mecharm_tasks/experiment3_grid_dry_run.py"
set "REMOTE_TASK=/home/nvidia/mecharm_tasks/mecharm_recorded_task.py"
set "REMOTE_GRIPPER_PROFILES=/home/nvidia/mecharm_tasks/experiment3_gripper_profiles.py"
set "GRIPPER_PROFILES=/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"

echo [DATA] Active 17-pose set: /home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml
echo [CLASS] cube -^> right, other -^> left
echo.
set /p "GRID=Source grid number (1-6): "
if not "%GRID%"=="1" if not "%GRID%"=="2" if not "%GRID%"=="3" if not "%GRID%"=="4" if not "%GRID%"=="5" if not "%GRID%"=="6" (
  echo [ERROR] Grid must be 1, 2, 3, 4, 5, or 6.
  pause
  exit /b 2
)

set /p "OBJECT_CLASS=Object class (cube/other): "
if /I "%OBJECT_CLASS%"=="cube" goto class_ok
if /I "%OBJECT_CLASS%"=="other" goto class_ok
echo [ERROR] Object class must be cube or other.
pause
exit /b 2

:class_ok
echo.
echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TASK] Grid %GRID%, class %OBJECT_CLASS%
echo [SAFETY] Keep only the selected object in place; clear both release areas and the transfer path.
echo.

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_SCRIPT%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_SCRIPT%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_COMP%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_COMP%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_BASE%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_BASE%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_TASK%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_TASK%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_GRIPPER_PROFILES%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_GRIPPER_PROFILES%"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_SCRIPT%' --poses '%POSES%' --require-active-correction --grid %GRID% --object-class %OBJECT_CLASS% --gripper-profiles '%GRIPPER_PROFILES%' --host 192.168.100.2 --port 9000 --arm-speed 8 --stream-step 2.0 --stream-period 0.20 --retreat-extension 0.50"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Experiment 3 movement completed.
) else (
  echo [STOPPED] Movement exited with code %RC%.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the latest Experiment 3 movement program to Jetson.
pause
exit /b 3
