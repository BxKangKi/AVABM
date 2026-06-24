@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "CONFIG_FILE=%~dp0..\config.txt"
if exist "%CONFIG_FILE%" (
    for /f "tokens=1,* delims==" %%A in ('findstr /R /C:"^[A-Za-z_][A-Za-z0-9_]*=" "%CONFIG_FILE%" 2^>nul') do (
        call set "%%~A=%%~B"
    )
)

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"

if exist "%CONDA_ACTIVATE_PATH%" (
    call "%CONDA_ACTIVATE_PATH%" "%CONDA_ENV%"
)

%PYTHON_COMMAND% "%~dp0build_helper.py" hardclean
if errorlevel 1 (
    echo [Error] Hard clean failed.
    pause
    exit /b 1
)
echo [Info] Hard clean completed. Run avabm_cuda\build.bat to rebuild.
pause
endlocal
exit /b 0
