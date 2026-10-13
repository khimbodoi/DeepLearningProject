@echo off
title TensorBoard Server
echo ==============================================
echo   Dang khoi dong TensorBoard...
echo   Dang mo trinh duyet tai http://localhost:6006
echo ==============================================

:: Tu dong mo trinh duyet
start "" "http://localhost:6006"

:: Chay TensorBoard bang Python 3.13
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.13.exe" (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.13.exe" -m tensorboard.main --logdir="./tensorboard_log" --port=6006
) else (
    python -m tensorboard.main --logdir="./tensorboard_log" --port=6006
)

pause
