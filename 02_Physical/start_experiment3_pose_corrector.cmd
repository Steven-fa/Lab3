@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 Pose Corrector
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_ROOT=/home/nvidia/mecharm_tasks"
set "SOURCE=%REMOTE_ROOT%/experiment3/current/experiment3_poses.yaml"
set "OUTPUT=%REMOTE_ROOT%/experiment3/corrections"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [MODE] Interactive correction of all six commanded joint angles for 17 poses.
echo [SOURCE] %SOURCE%
echo [SAFETY] No automatic motion. A go or verify command also requires MOVE confirmation.
echo.

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_pose_corrector.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_pose_corrector.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_grid_dry_run.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_grid_dry_run.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\mecharm_recorded_task.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/mecharm_recorded_task.py"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_ROOT%/experiment3_pose_corrector.py' --source '%SOURCE%' --output-root '%OUTPUT%' --resume-latest --host 192.168.100.2 --port 9000 --arm-speed 6 --stream-step 1.0 --stream-period 0.25"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Pose correction session closed normally.
) else (
  echo [STOPPED] Pose corrector exited with code %RC%.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the pose corrector to Jetson.
pause
exit /b 3
