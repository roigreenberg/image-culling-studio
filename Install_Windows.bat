@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================================================
echo   ImageCulling Studio - Windows Installer (No Git or Python required)
echo ========================================================================
echo.

REM 1) Locate or bootstrap Astral uv (standalone binary, installs Python 3.10 without admin)
set "UV_EXE="
where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    set "UV_EXE=uv"
) else if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
) else if exist "%LOCALAPPDATA%\uv\uv.exe" (
    set "UV_EXE=%LOCALAPPDATA%\uv\uv.exe"
)

if "%UV_EXE%"=="" (
    echo [1/5] Installing uv package manager ^(auto-manages Python 3.10^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    ) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
    ) else (
        where uv >nul 2>nul && set "UV_EXE=uv"
    )
)

if "%UV_EXE%"=="" (
    echo ERROR: Could not install uv automatically. Please install Python 3.10 from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
)

echo [2/5] Ensuring Python 3.10 and creating virtual environment ^(.venv^)...
"%UV_EXE%" python install 3.10
if errorlevel 1 (
    echo ERROR: Failed to install Python 3.10 via uv.
    pause
    exit /b 1
)

"%UV_EXE%" venv .venv --python 3.10 --seed
if errorlevel 1 (
    echo ERROR: Failed to create .venv.
    pause
    exit /b 1
)

REM 2) Detect NVIDIA GPU for CUDA acceleration
set "HAS_NVIDIA=0"
nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 set "HAS_NVIDIA=1"

if "%HAS_NVIDIA%"=="1" (
    echo [3/5] NVIDIA GPU detected! Installing CUDA PyTorch + onnxruntime-gpu...
    "%UV_EXE%" pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo ERROR: Failed to install CUDA PyTorch.
        pause
        exit /b 1
    )
    "%UV_EXE%" pip install --python .venv\Scripts\python.exe onnxruntime-gpu
) else (
    echo [3/5] No NVIDIA GPU detected. Installing CPU PyTorch + onnxruntime...
    "%UV_EXE%" pip install --python .venv\Scripts\python.exe torch torchvision onnxruntime
)
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch / ONNX Runtime.
    pause
    exit /b 1
)

echo [4/5] Installing ImageCulling Studio dependencies...
"%UV_EXE%" pip install --python .venv\Scripts\python.exe -r requirements-base.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b 1
)

echo [5/5] Downloading required Project Kestrel AI model weights ^(~480 MB^)...
.\.venv\Scripts\python.exe setup_models.py
if errorlevel 1 (
    echo WARNING: Some models could not be downloaded automatically. You can retry with:
    echo   .\.venv\Scripts\python.exe setup_models.py
)

echo.
echo ========================================================================
echo   Installation complete!
echo   Double-click Start_Windows.bat to launch ImageCulling Studio.
echo ========================================================================
pause
