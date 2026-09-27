@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm Experiment 3 - Grid 1 to 6 Sequence
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "REMOTE_ROOT=/home/nvidia/mecharm_tasks"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "GRIPPER_PROFILES=/home/nvidia/mecharm_tasks/experiment3/config/gripper_profiles.yaml"
set "START_GRID=%~1"
if "%START_GRID%"=="" set "START_GRID=1"

echo [TASK] Run grids %START_GRID% through 6 in one session.
echo [CLASS] 1=other, 2=cube, 3=cube, 4=other, 5=cube, 6=cube.
echo [AUTO] After each grid returns safe, clear the release point within 5 seconds.
echo.

ssh -q -o BatchMode=yes -o ConnectTimeout=10 "%JETSON_USER%@%JETSON_HOST%" "pgrep -f '[e]xperiment3_all_grid_sequence.py' > /dev/null"
if "%ERRORLEVEL%"=="0" goto already_running

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_all_grid_sequence.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_all_grid_sequence.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_transfer_test.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_transfer_test.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_directional_compensation.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_directional_compensation.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_grid_dry_run.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_grid_dry_run.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\mecharm_recorded_task.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/mecharm_recorded_task.py"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%~dp0hardware\experiment3_gripper_profiles.py" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_ROOT%/experiment3_gripper_profiles.py"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_ROOT%/experiment3_all_grid_sequence.py' --poses '%POSES%' --gripper-profiles '%GRIPPER_PROFILES%' --classes other cube cube other cube cube --host 192.168.100.2 --port 9000 --arm-speed 8 --stream-step 2.0 --stream-period 0.20 --retreat-extension 0.50 --start-grid %START_GRID% --auto-next-delay 5"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] Every requested grid completed successfully. No grid was skipped.
) else (
  echo [STOPPED] Current grid did not complete; later grids were not executed. Exit code %RC%.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the latest six-grid sequence to Jetson.
pause
exit /b 3

:already_running
echo [ERROR] Another six-grid sequence is already running on Jetson.
echo [ACTION] Return to its terminal, or press Ctrl+C there before starting again.
pause
exit /b 4
