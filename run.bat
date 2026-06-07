@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "CONFIG_FILE=%~dp0config.txt"
if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do (
        if not "%%~A"=="" call set "%%~A=%%~B"
    )
) else (
    echo [Warning] config.txt not found. Built-in defaults will be used.
)

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"

if exist "%CONDA_ACTIVATE_PATH%" (
    call "%CONDA_ACTIVATE_PATH%"
    call conda activate "%CONDA_ENV%"
) else (
    echo [Warning] Conda activate script not found: %CONDA_ACTIVATE_PATH%
    echo [Warning] Running with the current shell Python.
)

%PYTHON_COMMAND% main.py
pause
endlocal
