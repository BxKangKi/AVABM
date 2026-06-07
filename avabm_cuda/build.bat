@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

title AVABM CUDA Auto Build and Install Script

cd /d "%~dp0"
echo [Current Directory] %CD%

set "CONFIG_FILE=%~dp0..\config.txt"
if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do (
        if not "%%~A"=="" call set "%%~A=%%~B"
    )
) else (
    echo [Warning] ..\config.txt not found. Built-in build defaults will be used.
)

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if not defined TORCH_NVCC_FLAGS set "TORCH_NVCC_FLAGS=-allow-unsupported-compiler"
if not defined MAX_JOBS set "MAX_JOBS=16"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"

if not exist "%CONDA_ACTIVATE_PATH%" (
    echo [Error] Miniforge not found: %CONDA_ACTIVATE_PATH%
    pause
    exit /b 1
)

call "%CONDA_ACTIVATE_PATH%" base

if not exist "%USERPROFILE%\miniforge3\envs\%CONDA_ENV%" (
    echo [Info] Creating '%CONDA_ENV%' environment...
    call conda create -n "%CONDA_ENV%" python=3.12 ipykernel -y
) else (
    echo [Info] '%CONDA_ENV%' already exists.
)

echo [Info] Activating '%CONDA_ENV%'
call "%CONDA_ACTIVATE_PATH%" "%CONDA_ENV%"

echo [Info] Setting CUDA build environment from config.txt...
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

echo [Info] CUDA_HOME=%CUDA_HOME%
echo [Info] MAX_JOBS=%MAX_JOBS%
echo [Info] TORCH_NVCC_FLAGS=%TORCH_NVCC_FLAGS%

echo [Info] Installing package...
%PYTHON_COMMAND% -m pip install -e . --no-build-isolation

echo [Info] Moving generated .pyd files to parent directory...
for /r %%f in (*.pyd) do (
    echo Moving %%~nxf
    copy /Y "%%f" "..\"
)

echo =======================================================
echo [Info] Task completed.
echo =======================================================

pause
endlocal
