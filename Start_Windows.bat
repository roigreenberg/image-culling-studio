@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ImageCulling Studio is not installed yet.
    echo Please run Install_Windows.bat first!
    pause
    exit /b 1
)

echo Starting ImageCulling Studio at http://127.0.0.1:8765 ...
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765'"
.\.venv\Scripts\python.exe run_webapp.py --host 127.0.0.1 --port 8765
pause
