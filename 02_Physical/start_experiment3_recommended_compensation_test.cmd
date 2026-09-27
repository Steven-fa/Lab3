@echo off
setlocal EnableExtensions
chcp 65001 >nul
title mechArm - J2 J3 J4 Compensation Validation
cd /d "%~dp0"

set "JETSON_USER=nvidia"
set "JETSON_HOST=192.168.55.1"
set "LOCAL_MAIN=%~dp0hardware\experiment3_joint_compensation_test.py"
set "LOCAL_BASE=%~dp0hardware\experiment3_joint_repeatability_test.py"
set "REMOTE_MAIN=/home/nvidia/mecharm_tasks/experiment3_recommended_compensation_test.py"
set "REMOTE_BASE=/home/nvidia/mecharm_tasks/experiment3_joint_repeatability_test.py"
set "POSES=/home/nvidia/mecharm_tasks/experiment3/current/experiment3_poses.yaml"
set "BASELINE_REPORT=/home/nvidia/mecharm_tasks/experiment3/diagnostics/joint_repeatability_20260907_202425.yaml"
set "OUTPUT_DIR=/home/nvidia/mecharm_tasks/experiment3/diagnostics"

echo [PATH] Windows -^> Jetson %JETSON_HOST% -^> mechArm 192.168.100.2:9000
echo [TEST] Validate latest direction-aware compensation for J2, J3 and J4.
echo [SOURCE] Uses the completed six-joint report; no hand-entered correction values.
echo [GRIPPER] Stays OPEN. Remove the cube and all objects from the path.
echo [SAFETY] Safe pose only, speed 6, one joint at a time, +/-4 degree preload.
echo [TIME] The validation takes approximately 4 to 6 minutes.
echo [NOTICE] No motion occurs until you type RUN and press Enter.
echo.

scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_MAIN%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_MAIN%"
if errorlevel 1 goto upload_error
scp -q -o BatchMode=yes -o ConnectTimeout=10 "%LOCAL_BASE%" "%JETSON_USER%@%JETSON_HOST%:%REMOTE_BASE%"
if errorlevel 1 goto upload_error

ssh -tt -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "%JETSON_USER%@%JETSON_HOST%" "python3 -u '%REMOTE_MAIN%' --poses '%POSES%' --baseline-reports '%BASELINE_REPORT%' --joints J2 J3 J4 --host 192.168.100.2 --port 9000 --arm-speed 6 --offset-deg 4 --repeats 3 --sample-times 1 3 6 --stream-step 1.0 --stream-period 0.25 --output-dir '%OUTPUT_DIR%'"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [DONE] J2 J3 J4 compensation validation completed.
) else (
  echo [STOPPED] Compensation validation exited with code %RC%.
)
pause
exit /b %RC%

:upload_error
echo [ERROR] Could not upload the compensation validator to Jetson.
pause
exit /b 3
