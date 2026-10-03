@echo off
cd /d "%~dp0"
echo Starting Hand Gesture Recognition...
echo Model: TFLite (fast CPU inference)
echo.
.venv\Scripts\python.exe webcam.py
echo.
echo Webcam closed. Press any key to exit.
pause
