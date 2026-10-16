@echo off
setlocal
cd /d "%~dp0"
set ESP32_PORT=COM5
set ESP32_BAUD=115200
"%~dp0.venv\Scripts\python.exe" "%~dp0auto_start.py"
endlocal
