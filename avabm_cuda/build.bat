@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

title AVABM CUDA Incremental Build Script
cd /d "%~dp0"
echo [Current Directory] %CD%

set "CONFIG_FILE=%~dp0..\config.txt"
if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    call :load_config "%CONFIG_FILE%"
    if errorlevel 1 (
        echo [Error] Failed to read config.txt safely.
        pause
        exit /b 1
    )
) else (
    echo [Warning] ..\config.txt not found. Built-in build defaults will be used.
)

if /I "%~1"=="clean" set "CUDA_FORCE_REBUILD=1"
if /I "%~1"=="rebuild" set "CUDA_FORCE_REBUILD=1"

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if not defined TORCH_NVCC_FLAGS set "TORCH_NVCC_FLAGS=-allow-unsupported-compiler"
if not defined TORCH_DONT_CHECK_COMPILER_ABI set "TORCH_DONT_CHECK_COMPILER_ABI=1"
if not defined MAX_JOBS set "MAX_JOBS=4"
if not defined CUDA_BUILD_MAX_JOBS set "CUDA_BUILD_MAX_JOBS=4"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.6"
if not defined CUDA_SUPPRESS_HEADER_WARNINGS set "CUDA_SUPPRESS_HEADER_WARNINGS=1"
if not defined CUDA_SHOW_WARNINGS set "CUDA_SHOW_WARNINGS=0"
if not defined CUDA_VERBOSE_BUILD set "CUDA_VERBOSE_BUILD=0"
if not defined CUDA_BUILD_LOG_TAIL set "CUDA_BUILD_LOG_TAIL=120"
if not defined CUDA_FORCE_REBUILD set "CUDA_FORCE_REBUILD=0"
if not defined CUDA_SKIP_IF_UP_TO_DATE set "CUDA_SKIP_IF_UP_TO_DATE=1"
if not defined CUDA_INCREMENTAL_BUILD set "CUDA_INCREMENTAL_BUILD=1"
if not defined CUDA_BUILD_MODE set "CUDA_BUILD_MODE=release"
if not defined CUDA_CXX_STANDARD set "CUDA_CXX_STANDARD=17"
if not defined CUDA_DISABLE_MSVC_GL set "CUDA_DISABLE_MSVC_GL=1"
if not defined CUDA_USE_FULL_TORCH_EXTENSION_HEADER set "CUDA_USE_FULL_TORCH_EXTENSION_HEADER=0"
if not defined CUDA_OPT_LEVEL set "CUDA_OPT_LEVEL=3"
if not defined CUDA_FAST_MATH set "CUDA_FAST_MATH=1"
if not defined CUDA_FAST_EQUIV_MATH set "CUDA_FAST_EQUIV_MATH=1"
if not defined CUDA_USE_ASYNC_MEMSET_CLEAR set "CUDA_USE_ASYNC_MEMSET_CLEAR=1"
if not defined CUDA_SPAWN_GRID_INSERT_FASTPATH set "CUDA_SPAWN_GRID_INSERT_FASTPATH=1"
if not defined SPEED_MIN_CRUISE_ENABLED set "SPEED_MIN_CRUISE_ENABLED=1"
if not defined SPEED_MIN_CRUISE_KMH set "SPEED_MIN_CRUISE_KMH=40.0"
if not defined CUDA_NVCC_THREADS set "CUDA_NVCC_THREADS=4"
if not defined CUDA_SPLIT_COMPILE set "CUDA_SPLIT_COMPILE=0"
if not defined MSVC_VCVARS64_BAT if defined VCVARS64_BAT set "MSVC_VCVARS64_BAT=%VCVARS64_BAT%"
if not defined MSVC_VCVARS64_BAT set "MSVC_VCVARS64_BAT="

call :cap_max_jobs

if not exist "%CONDA_ACTIVATE_PATH%" (
    echo [Error] Miniforge not found: %CONDA_ACTIVATE_PATH%
    pause
    exit /b 1
)

call "%CONDA_ACTIVATE_PATH%" base

if not exist "%USERPROFILE%\miniforge3\envs\%CONDA_ENV%" (
    echo [Info] Creating '%CONDA_ENV%' environment...
    call conda create -n "%CONDA_ENV%" python=3.12 ipykernel -y
    if errorlevel 1 (
        echo [Error] Failed to create Conda environment.
        pause
        exit /b 1
    )
) else (
    echo [Info] '%CONDA_ENV%' already exists.
)

echo [Info] Activating '%CONDA_ENV%'
call "%CONDA_ACTIVATE_PATH%" "%CONDA_ENV%"
if errorlevel 1 (
    echo [Error] Failed to activate Conda environment.
    pause
    exit /b 1
)

call :ensure_msvc_env
if errorlevel 1 (
    pause
    exit /b 1
)

echo [Info] Setting CUDA build environment from config.txt...
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
set "USE_NINJA=1"
set "CMAKE_BUILD_PARALLEL_LEVEL=%MAX_JOBS%"
set "MAX_JOBS=%MAX_JOBS%"

echo [Info] CUDA_HOME=%CUDA_HOME%
echo [Info] MAX_JOBS=%MAX_JOBS%  [cap=%CUDA_BUILD_MAX_JOBS%]
echo [Info] TORCH_NVCC_FLAGS=%TORCH_NVCC_FLAGS%
echo [Info] TORCH_DONT_CHECK_COMPILER_ABI=%TORCH_DONT_CHECK_COMPILER_ABI%
echo [Info] TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo [Info] CUDA_BUILD_MODE=%CUDA_BUILD_MODE%
echo [Info] CUDA_CXX_STANDARD=%CUDA_CXX_STANDARD%
echo [Info] CUDA_DISABLE_MSVC_GL=%CUDA_DISABLE_MSVC_GL%
echo [Info] CUDA_USE_FULL_TORCH_EXTENSION_HEADER=%CUDA_USE_FULL_TORCH_EXTENSION_HEADER%
echo [Info] CUDA_OPT_LEVEL=%CUDA_OPT_LEVEL%
echo [Info] CUDA_FAST_EQUIV_MATH=%CUDA_FAST_EQUIV_MATH%
echo [Info] CUDA_USE_ASYNC_MEMSET_CLEAR=%CUDA_USE_ASYNC_MEMSET_CLEAR%
echo [Info] CUDA_SPAWN_GRID_INSERT_FASTPATH=%CUDA_SPAWN_GRID_INSERT_FASTPATH%
echo [Info] SPEED_MIN_CRUISE_ENABLED=%SPEED_MIN_CRUISE_ENABLED%
echo [Info] SPEED_MIN_CRUISE_KMH=%SPEED_MIN_CRUISE_KMH%
echo [Info] CUDA_NVCC_THREADS=%CUDA_NVCC_THREADS%
echo [Info] CUDA_SPLIT_COMPILE=%CUDA_SPLIT_COMPILE%
echo [Info] CUDA_SHOW_WARNINGS=%CUDA_SHOW_WARNINGS%
echo [Info] CUDA_INCREMENTAL_BUILD=%CUDA_INCREMENTAL_BUILD%
echo [Info] CUDA_FORCE_REBUILD=%CUDA_FORCE_REBUILD%

where cl.exe >nul 2>nul
if errorlevel 1 (
    echo =======================================================
    echo [Error] cl.exe is still not available after MSVC setup.
    echo [Error] Install Visual Studio 2022 Build Tools with Desktop C++ workload.
    echo [Error] Or set MSVC_VCVARS64_BAT in config.txt.
    echo =======================================================
    pause
    exit /b 1
)
for /f "tokens=*" %%C in ('where cl.exe 2^>nul') do (
    echo [Info] cl.exe=%%C
    goto :after_cl_print
)
:after_cl_print

set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "CC=cl"
set "CXX=cl"
set "TORCH_DONT_CHECK_COMPILER_ABI=%TORCH_DONT_CHECK_COMPILER_ABI%"
if /I "%CUDA_SHOW_WARNINGS%"=="0" (
    rem Syntax: CL is an environment variable automatically prepended to cl.exe options.
    rem Logic: Do not use /w, because it conflicts with distutils /W3 and creates D9025 noise.
    set "CL=/wd4996 /wd4819 /wd4251 /wd4275 /wd4244 /wd4267 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 %CL%"
) else (
    set "CL=/wd4996 /wd4819 %CL%"
)

where nvcc.exe >nul 2>nul
if errorlevel 1 (
    echo =======================================================
    echo [Error] nvcc.exe was not found. Check CUDA_HOME in config.txt.
    echo =======================================================
    pause
    exit /b 1
)

where ninja.exe >nul 2>nul
if errorlevel 1 (
    echo [Info] ninja.exe not found. Installing ninja into the Conda environment...
    call conda install -n "%CONDA_ENV%" ninja -y
    if errorlevel 1 (
        echo [Error] Failed to install ninja.
        pause
        exit /b 1
    )
)

set "BUILD_HELPER=%~dp0build_helper.py"
if not exist "%BUILD_HELPER%" (
    echo [Error] Missing build helper: %BUILD_HELPER%
    pause
    exit /b 1
)

rem Syntax: Branch labels keep clean/check/incremental paths explicit.
rem Logic: If the fingerprint matches, skip setup.py and avoid a long nvcc rebuild.
if /I "%CUDA_FORCE_REBUILD%"=="1" goto :full_clean_rebuild
if /I "%CUDA_SKIP_IF_UP_TO_DATE%"=="1" goto :maybe_skip_build
goto :prepare_incremental_build

:full_clean_rebuild
echo [Info] Full clean rebuild requested.
%PYTHON_COMMAND% "%BUILD_HELPER%" clean
if errorlevel 1 (
    pause
    exit /b 1
)
goto :after_prepare_build

:maybe_skip_build
%PYTHON_COMMAND% "%BUILD_HELPER%" check
if errorlevel 3 (
    echo [Error] Build status check failed.
    pause
    exit /b 1
)
if errorlevel 2 goto :prepare_incremental_build
goto :build_skipped

:prepare_incremental_build
if /I "%CUDA_INCREMENTAL_BUILD%"=="1" (
    echo [Info] Preparing incremental build: remove stale .pyd only, keep build cache.
    %PYTHON_COMMAND% "%BUILD_HELPER%" prepare
) else (
    echo [Info] Incremental build disabled, cleaning full build cache.
    %PYTHON_COMMAND% "%BUILD_HELPER%" clean
)
if errorlevel 1 (
    pause
    exit /b 1
)

:after_prepare_build
set "BUILD_LOG_DIR=%~dp0build_logs"
if not exist "%BUILD_LOG_DIR%" mkdir "%BUILD_LOG_DIR%"
set "BUILD_LOG=%BUILD_LOG_DIR%\build_last.log"

echo [Info] Building CUDA extension in place...
echo [Info] Full build log: %BUILD_LOG%
echo [Info] Normal repeated builds will be skipped after this binary is stamped.
if /I "%CUDA_VERBOSE_BUILD%"=="1" (
    %PYTHON_COMMAND% setup.py build_ext --inplace
) else (
    %PYTHON_COMMAND% setup.py build_ext --inplace > "%BUILD_LOG%" 2>&1
)
if errorlevel 1 (
    echo =======================================================
    echo [Error] CUDA build failed. No old .pyd will be used.
    echo [Error] The last %CUDA_BUILD_LOG_TAIL% log lines are printed below.
    echo [Error] Full log: %BUILD_LOG%
    echo =======================================================
    if exist "%BUILD_LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail %CUDA_BUILD_LOG_TAIL%"
    pause
    exit /b 1
)
if /I not "%CUDA_VERBOSE_BUILD%"=="1" (
    echo [Info] Build succeeded. Short log tail:
    if exist "%BUILD_LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail 40"
)

%PYTHON_COMMAND% "%BUILD_HELPER%" mark
if errorlevel 1 (
    pause
    exit /b 1
)

echo =======================================================
echo [Info] CUDA build completed successfully.
echo =======================================================
pause
endlocal
exit /b 0

:build_skipped
echo =======================================================
echo [Info] CUDA build skipped: compiled module already matches the source fingerprint.
echo [Info] Use "avabm_cuda\build.bat clean" only when you need a full rebuild.
echo =======================================================
pause
endlocal
exit /b 0

:load_config
rem Syntax: findstr selects only ASCII KEY=VALUE lines and ignores all comments.
rem Logic: Avoid executing UTF-8 comments or malformed lines inside CMD parenthesis blocks.
for /f "tokens=1,* delims==" %%A in ('findstr /R /C:"^[A-Za-z_][A-Za-z0-9_]*=" "%~1" 2^>nul') do (
    call set "%%~A=%%~B"
)
exit /b 0

:cap_max_jobs
set "_MJ=%MAX_JOBS%"
set "_CAP=%CUDA_BUILD_MAX_JOBS%"
set /a _MJ_NUM=%_MJ% >nul 2>nul
if errorlevel 1 set "_MJ_NUM=4"
set /a _CAP_NUM=%_CAP% >nul 2>nul
if errorlevel 1 set "_CAP_NUM=4"
if %_CAP_NUM% LEQ 0 set "_CAP_NUM=4"
if %_MJ_NUM% LEQ 0 set "_MJ_NUM=4"
if %_MJ_NUM% GTR %_CAP_NUM% set "_MJ_NUM=%_CAP_NUM%"
set "MAX_JOBS=%_MJ_NUM%"
exit /b 0

:ensure_msvc_env
where cl.exe >nul 2>nul
if not errorlevel 1 (
    echo [Info] MSVC cl.exe is already on PATH.
    exit /b 0
)

if defined MSVC_VCVARS64_BAT (
    if exist "%MSVC_VCVARS64_BAT%" (
        echo [Info] Loading MSVC environment: %MSVC_VCVARS64_BAT%
        call "%MSVC_VCVARS64_BAT%" >nul
        where cl.exe >nul 2>nul
        if not errorlevel 1 exit /b 0
    ) else (
        echo [Warning] MSVC_VCVARS64_BAT was set but not found: %MSVC_VCVARS64_BAT%
    )
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL=%%I"
    )
)

if defined VS_INSTALL (
    if exist "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
        echo [Info] Loading MSVC environment from Visual Studio installation.
        call "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
        where cl.exe >nul 2>nul
        if not errorlevel 1 exit /b 0
    )
)

if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo [Info] Loading MSVC Community 2022 environment.
    call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    where cl.exe >nul 2>nul
    if not errorlevel 1 exit /b 0
)

if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo [Info] Loading MSVC BuildTools 2022 environment.
    call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    where cl.exe >nul 2>nul
    if not errorlevel 1 exit /b 0
)

if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    echo [Info] Loading MSVC Professional 2022 environment.
    call "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    where cl.exe >nul 2>nul
    if not errorlevel 1 exit /b 0
)

if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    echo [Info] Loading MSVC Enterprise 2022 environment.
    call "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
    where cl.exe >nul 2>nul
    if not errorlevel 1 exit /b 0
)

echo =======================================================
echo [Error] MSVC cl.exe was not found.
echo [Error] Install Visual Studio 2022 Build Tools with Desktop development with C++.
echo [Error] If it is already installed, set this in config.txt:
echo [Error] MSVC_VCVARS64_BAT=C:\path\to\VC\Auxiliary\Build\vcvars64.bat
echo =======================================================
exit /b 1
