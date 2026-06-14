@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "CONFIG_FILE=%~dp0config.txt"
if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    call :load_config "%CONFIG_FILE%"
    if errorlevel 1 (
        echo [Error] Failed to read config.txt safely.
        pause
        exit /b 1
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

set "BUILD_HELPER=%~dp0avabm_cuda\build_helper.py"
if not exist "%BUILD_HELPER%" (
    echo [Error] Missing CUDA build helper: %BUILD_HELPER%
    echo [Error] Apply the latest code patch, then run avabm_cuda\build.bat.
    pause
    exit /b 1
)

rem Syntax: check errorlevel from high to low after verify-run.
rem Logic: Stop before main.py if an old or missing CUDA binary would be imported.
%PYTHON_COMMAND% "%BUILD_HELPER%" verify-run
if errorlevel 3 (
    echo [Error] CUDA build fingerprint check failed.
    pause
    exit /b 1
)
if errorlevel 2 (
    echo [Error] Compiled CUDA module is missing or does not match the current source/build config.
    echo [Error] Run avabm_cuda\build.bat once. Repeated runs will skip if nothing changed.
    pause
    exit /b 1
)

%PYTHON_COMMAND% main.py
pause
endlocal
exit /b 0

:load_config
rem Syntax: findstr selects only ASCII KEY=VALUE lines and ignores comments.
rem Logic: Avoid executing UTF-8 comments or malformed config text as commands.
for /f "tokens=1,* delims==" %%A in ('findstr /R /C:"^[A-Za-z_][A-Za-z0-9_]*=" "%~1" 2^>nul') do (
    call set "%%~A=%%~B"
)
exit /b 0
