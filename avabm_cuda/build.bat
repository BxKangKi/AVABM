@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

title AVABM CUDA Native-GPU Build Script
cd /d "%~dp0"
echo [Current Directory] %CD%

set "CONFIG_FILE=%~dp0..\config.txt"
if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    rem Read config inline instead of CALLing a label before the script has been normalized by CMD.
    rem This avoids "The system cannot find the batch label specified - load_config" on some Windows shells.
    for /f "tokens=1,* delims==" %%A in ('findstr /R /C:"^[A-Za-z_][A-Za-z0-9_]*=" "%CONFIG_FILE%" 2^>nul') do (
        call set "%%~A=%%~B"
    )
) else (
    echo [Warning] ..\config.txt not found. Built-in build defaults will be used.
)

if /I "%~1"=="clean" set "CUDA_FORCE_REBUILD=1"
if /I "%~1"=="rebuild" set "CUDA_FORCE_REBUILD=1"
if /I "%~1"=="hardclean" set "CUDA_FORCE_REBUILD=1"
if /I "%~1"=="cleanall" set "CUDA_FORCE_REBUILD=1"
if /I "%~1"=="clean-all" set "CUDA_FORCE_REBUILD=1"

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined CUDA_HOME set "CUDA_HOME=auto"
if not defined TORCH_NVCC_FLAGS set "TORCH_NVCC_FLAGS=-allow-unsupported-compiler"
if not defined TORCH_DONT_CHECK_COMPILER_ABI set "TORCH_DONT_CHECK_COMPILER_ABI=1"
if not defined MAX_JOBS set "MAX_JOBS=1"
if not defined CUDA_BUILD_MAX_JOBS set "CUDA_BUILD_MAX_JOBS=1"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=auto"
if not defined CUDA_SUPPRESS_HEADER_WARNINGS set "CUDA_SUPPRESS_HEADER_WARNINGS=1"
if not defined CUDA_SHOW_WARNINGS set "CUDA_SHOW_WARNINGS=0"
if not defined CUDA_VERBOSE_BUILD set "CUDA_VERBOSE_BUILD=0"
if not defined CUDA_BUILD_LOG_TAIL set "CUDA_BUILD_LOG_TAIL=200"
if not defined CUDA_FORCE_REBUILD set "CUDA_FORCE_REBUILD=0"
if not defined CUDA_SKIP_IF_UP_TO_DATE set "CUDA_SKIP_IF_UP_TO_DATE=1"
if not defined CUDA_INCREMENTAL_BUILD set "CUDA_INCREMENTAL_BUILD=1"
if not defined CUDA_BUILD_MODE set "CUDA_BUILD_MODE=release"
if not defined CUDA_CXX_STANDARD set "CUDA_CXX_STANDARD=17"
if not defined CUDA_DISABLE_MSVC_GL set "CUDA_DISABLE_MSVC_GL=1"
if not defined CUDA_USE_FULL_TORCH_EXTENSION_HEADER set "CUDA_USE_FULL_TORCH_EXTENSION_HEADER=0"
if not defined CUDA_BINDING_OPT_LEVEL set "CUDA_BINDING_OPT_LEVEL=1"
if not defined CUDA_OPT_LEVEL set "CUDA_OPT_LEVEL=2"
if not defined CUDA_PTXAS_OPT_LEVEL set "CUDA_PTXAS_OPT_LEVEL=1"
if not defined CUDA_PTXAS_ALLOW_EXPENSIVE_OPT set "CUDA_PTXAS_ALLOW_EXPENSIVE_OPT=0"
if not defined CUDA_FAST_MATH set "CUDA_FAST_MATH=0"
if not defined CUDA_FAST_EQUIV_MATH set "CUDA_FAST_EQUIV_MATH=1"
if not defined CUDA_USE_ASYNC_MEMSET_CLEAR set "CUDA_USE_ASYNC_MEMSET_CLEAR=1"
if not defined CUDA_SPAWN_GRID_INSERT_FASTPATH set "CUDA_SPAWN_GRID_INSERT_FASTPATH=1"
if not defined SPEED_MIN_CRUISE_ENABLED set "SPEED_MIN_CRUISE_ENABLED=1"
if not defined SPEED_MIN_CRUISE_KMH set "SPEED_MIN_CRUISE_KMH=40.0"
if not defined CUDA_NVCC_THREADS set "CUDA_NVCC_THREADS=0"
if not defined CUDA_SPLIT_COMPILE set "CUDA_SPLIT_COMPILE=0"
if not defined CUDA_NVCC_TIME_LOG set "CUDA_NVCC_TIME_LOG=1"
if not defined CUDA_NATIVE_GPU_ONLY set "CUDA_NATIVE_GPU_ONLY=1"
if not defined CUDA_ARCH_FALLBACK set "CUDA_ARCH_FALLBACK=8.6"
if not defined CUDA_ARCH_KEEP_PTX set "CUDA_ARCH_KEEP_PTX=0"
if not defined CUDA_DEVICE_FORCEINLINE set "CUDA_DEVICE_FORCEINLINE=0"
if not defined CUDA_DEVICE_NOINLINE set "CUDA_DEVICE_NOINLINE=0"
if not defined CUDA_VALIDATE_NVCC_ARCH set "CUDA_VALIDATE_NVCC_ARCH=1"
if not defined CUDA_PTXAS_SAFE_MODE set "CUDA_PTXAS_SAFE_MODE=0"
if not defined CUDA_PTXAS_PARTITION_BUILD set "CUDA_PTXAS_PARTITION_BUILD=1"
if not defined CUDA_SINGLE_TU_BUILD set "CUDA_SINGLE_TU_BUILD=0"
if not defined CUDA_PTXAS_MAXRREGCOUNT set "CUDA_PTXAS_MAXRREGCOUNT=0"
if not defined CUDA_PTXAS_VERBOSE set "CUDA_PTXAS_VERBOSE=0"
if not defined CUDA_PTXAS_KEEP_FILES set "CUDA_PTXAS_KEEP_FILES=0"
if not defined CUDA_AUTO_RETRY_PTXAS_CRASH set "CUDA_AUTO_RETRY_PTXAS_CRASH=1"
if not defined CUDA_SAFE_RETRY_CLEAN set "CUDA_SAFE_RETRY_CLEAN=1"
if not defined CUDA_CLEAR_NVCC_GLOBAL_FLAGS set "CUDA_CLEAR_NVCC_GLOBAL_FLAGS=1"
if not defined CUDA_NVCC_ADD_STD_FLAG set "CUDA_NVCC_ADD_STD_FLAG=0"
rem MSVC_TOOLSET=v143 keeps CUDA 13.0 away from VS2026/MSVC 14.5x, which can crash cudafe++.
if not defined MSVC_TOOLSET if defined MSVC_TOOLSET_VERSION set "MSVC_TOOLSET=%MSVC_TOOLSET_VERSION%"
if not defined MSVC_TOOLSET set "MSVC_TOOLSET=v143"
if not defined MSVC_VCVARS_VER set "MSVC_VCVARS_VER="
if not defined MSVC_VCVARS_ARGS set "MSVC_VCVARS_ARGS="
if not defined MSVC_STRICT_TOOLSET set "MSVC_STRICT_TOOLSET=1"
if not defined MSVC_VCVARS64_BAT if defined VCVARS64_BAT set "MSVC_VCVARS64_BAT=%VCVARS64_BAT%"
if not defined MSVC_VCVARS64_BAT set "MSVC_VCVARS64_BAT="

call :resolve_cuda_home
if errorlevel 1 (
    pause
    exit /b 1
)

set "BUILD_HELPER=%~dp0build_helper.py"
if not exist "%BUILD_HELPER%" (
    echo [Error] Missing build helper: %BUILD_HELPER%
    pause
    exit /b 1
)

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

rem Make CUDA DLLs visible before torch probes the local GPU architecture.
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

if /I "%TORCH_CUDA_ARCH_LIST%"=="auto" (
    for /f "tokens=*" %%A in ('%PYTHON_COMMAND% "%BUILD_HELPER%" detectarch 2^>nul') do set "TORCH_CUDA_ARCH_LIST=%%A"
)
if /I "%TORCH_CUDA_ARCH_LIST%"=="native" (
    for /f "tokens=*" %%A in ('%PYTHON_COMMAND% "%BUILD_HELPER%" detectarch 2^>nul') do set "TORCH_CUDA_ARCH_LIST=%%A"
)
if /I "%TORCH_CUDA_ARCH_LIST%"=="gpu" (
    for /f "tokens=*" %%A in ('%PYTHON_COMMAND% "%BUILD_HELPER%" detectarch 2^>nul') do set "TORCH_CUDA_ARCH_LIST=%%A"
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
echo [Info] CUDA_NATIVE_GPU_ONLY=%CUDA_NATIVE_GPU_ONLY%
echo [Info] CUDA_ARCH_FALLBACK=%CUDA_ARCH_FALLBACK%
echo [Info] CUDA_ARCH_KEEP_PTX=%CUDA_ARCH_KEEP_PTX%
echo [Info] CUDA_BUILD_MODE=%CUDA_BUILD_MODE%
echo [Info] CUDA_CXX_STANDARD=%CUDA_CXX_STANDARD%
echo [Info] CUDA_DISABLE_MSVC_GL=%CUDA_DISABLE_MSVC_GL%
echo [Info] CUDA_USE_FULL_TORCH_EXTENSION_HEADER=%CUDA_USE_FULL_TORCH_EXTENSION_HEADER%
echo [Info] CUDA_BINDING_OPT_LEVEL=%CUDA_BINDING_OPT_LEVEL%
echo [Info] CUDA_OPT_LEVEL=%CUDA_OPT_LEVEL%
echo [Info] CUDA_PTXAS_OPT_LEVEL=%CUDA_PTXAS_OPT_LEVEL%
echo [Info] CUDA_PTXAS_ALLOW_EXPENSIVE_OPT=%CUDA_PTXAS_ALLOW_EXPENSIVE_OPT%
echo [Info] CUDA_PTXAS_SAFE_MODE=%CUDA_PTXAS_SAFE_MODE%
echo [Info] CUDA_PTXAS_PARTITION_BUILD=%CUDA_PTXAS_PARTITION_BUILD%
echo [Info] CUDA_SINGLE_TU_BUILD=%CUDA_SINGLE_TU_BUILD%
echo [Info] CUDA_PTXAS_MAXRREGCOUNT=%CUDA_PTXAS_MAXRREGCOUNT%
echo [Info] CUDA_PTXAS_VERBOSE=%CUDA_PTXAS_VERBOSE%
echo [Info] CUDA_PTXAS_KEEP_FILES=%CUDA_PTXAS_KEEP_FILES%
echo [Info] CUDA_AUTO_RETRY_PTXAS_CRASH=%CUDA_AUTO_RETRY_PTXAS_CRASH%
echo [Info] CUDA_VALIDATE_NVCC_ARCH=%CUDA_VALIDATE_NVCC_ARCH%
echo [Info] CUDA_CLEAR_NVCC_GLOBAL_FLAGS=%CUDA_CLEAR_NVCC_GLOBAL_FLAGS%
echo [Info] CUDA_NVCC_ADD_STD_FLAG=%CUDA_NVCC_ADD_STD_FLAG%
echo [Info] CUDA_FAST_MATH=%CUDA_FAST_MATH%
echo [Info] CUDA_FAST_EQUIV_MATH=%CUDA_FAST_EQUIV_MATH%
echo [Info] CUDA_USE_ASYNC_MEMSET_CLEAR=%CUDA_USE_ASYNC_MEMSET_CLEAR%
echo [Info] CUDA_SPAWN_GRID_INSERT_FASTPATH=%CUDA_SPAWN_GRID_INSERT_FASTPATH%
echo [Info] SPEED_MIN_CRUISE_ENABLED=%SPEED_MIN_CRUISE_ENABLED%
echo [Info] SPEED_MIN_CRUISE_KMH=%SPEED_MIN_CRUISE_KMH%
echo [Info] CUDA_NVCC_THREADS=%CUDA_NVCC_THREADS%
echo [Info] CUDA_SPLIT_COMPILE=%CUDA_SPLIT_COMPILE%
echo [Info] CUDA_NVCC_TIME_LOG=%CUDA_NVCC_TIME_LOG%
echo [Info] CUDA_DEVICE_FORCEINLINE=%CUDA_DEVICE_FORCEINLINE%
echo [Info] CUDA_DEVICE_NOINLINE=%CUDA_DEVICE_NOINLINE%
echo [Info] CUDA_SHOW_WARNINGS=%CUDA_SHOW_WARNINGS%
echo [Info] CUDA_INCREMENTAL_BUILD=%CUDA_INCREMENTAL_BUILD%
echo [Info] CUDA_FORCE_REBUILD=%CUDA_FORCE_REBUILD%
echo [Info] MSVC_TOOLSET=%MSVC_TOOLSET%
echo [Info] MSVC_STRICT_TOOLSET=%MSVC_STRICT_TOOLSET%
echo [Info] MSVC_VCVARS_VER=%MSVC_VCVARS_VER%
echo [Info] MSVC_VCVARS_ARGS=%MSVC_VCVARS_ARGS%

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
call :cl_matches_requested_toolset
if errorlevel 1 (
    echo =======================================================
    echo [Error] cl.exe does not match MSVC_TOOLSET=%MSVC_TOOLSET%.
    echo [Error] Use VS2022/v143 ^(MSVC 14.3x or 14.4x^) or set MSVC_TOOLSET=latest explicitly.
    echo =======================================================
    pause
    exit /b 1
)
for /f "tokens=*" %%C in ('where cl.exe 2^>nul') do (
    echo [Info] cl.exe=%%C
    goto :after_cl_print
)
:after_cl_print

if /I "%CUDA_CLEAR_NVCC_GLOBAL_FLAGS%"=="1" (
    set "NVCC_PREPEND_FLAGS="
    set "NVCC_APPEND_FLAGS="
)

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

rem Syntax: Branch labels keep clean/check/incremental paths explicit.
rem Logic: If the fingerprint matches, skip setup.py and avoid a long nvcc rebuild.
if /I "%~1"=="hardclean" goto :hard_clean_rebuild
if /I "%~1"=="cleanall" goto :hard_clean_rebuild
if /I "%~1"=="clean-all" goto :hard_clean_rebuild
if /I "%CUDA_FORCE_REBUILD%"=="1" goto :full_clean_rebuild
if /I "%CUDA_SKIP_IF_UP_TO_DATE%"=="1" goto :maybe_skip_build
goto :prepare_incremental_build

:hard_clean_rebuild
echo [Info] Hard clean rebuild requested. Project outputs plus user-local PyTorch/CUDA caches will be removed.
%PYTHON_COMMAND% "%BUILD_HELPER%" hardclean
if errorlevel 1 (
    pause
    exit /b 1
)
goto :after_prepare_build

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
echo [Info] Native-GPU-only arch and performance-safe CUDA optimization profile are enabled.
%PYTHON_COMMAND% "%BUILD_HELPER%" runbuild
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
echo [Info] Build succeeded. Short log tail:
if exist "%BUILD_LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail 40"

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

:resolve_cuda_home
if not defined CUDA_HOME set "CUDA_HOME=auto"
if /I not "%CUDA_HOME%"=="auto" if /I not "%CUDA_HOME%"=="latest" if /I not "%CUDA_HOME%"=="detect" (
    if exist "%CUDA_HOME%\bin\nvcc.exe" exit /b 0
    echo [Warning] CUDA_HOME does not contain nvcc.exe: %CUDA_HOME%
)
if defined CUDA_PATH (
    call :try_cuda_home "%CUDA_PATH%"
    if not errorlevel 1 exit /b 0
)
for %%V in (v13.3 v13.2 v13.1 v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4) do (
    call :try_cuda_home "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%%V"
    if not errorlevel 1 exit /b 0
    call :try_cuda_home "%ProgramW6432%\NVIDIA GPU Computing Toolkit\CUDA\%%V"
    if not errorlevel 1 exit /b 0
)
echo =======================================================
echo [Error] nvcc.exe was not found. Install CUDA Toolkit or set CUDA_HOME in config.txt.
echo [Error] Example: CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
echo =======================================================
exit /b 1

:try_cuda_home
if "%~1"=="" exit /b 1
if exist "%~1\bin\nvcc.exe" (
    set "CUDA_HOME=%~1"
    echo [Info] CUDA_HOME auto-resolved to %~1
    exit /b 0
)
exit /b 1

:load_config
rem Unused fallback. Main config loading is inline near the top of this file.
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
    call :cl_matches_requested_toolset
    if not errorlevel 1 (
        echo [Info] MSVC cl.exe is already on PATH and matches requested toolset.
        exit /b 0
    )
    echo [Info] Existing cl.exe does not match MSVC_TOOLSET=%MSVC_TOOLSET%; reloading MSVC environment.
)

if defined MSVC_VCVARS64_BAT (
    call :try_vcvars_for_toolset "%MSVC_VCVARS64_BAT%" ""
    if not errorlevel 1 exit /b 0
    echo [Warning] MSVC_VCVARS64_BAT was set but did not provide the requested toolset: %MSVC_VCVARS64_BAT%
)

if /I "%MSVC_TOOLSET%"=="v143" goto :ensure_msvc_v143
if /I "%MSVC_TOOLSET%"=="latest" goto :ensure_msvc_latest

:ensure_msvc_latest
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL=%%I"
    )
)
if defined VS_INSTALL (
    call :try_vcvars_for_toolset "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" "%VS_INSTALL%"
    if not errorlevel 1 exit /b 0
)
goto :ensure_msvc_error

:ensure_msvc_v143
rem Prefer VS2022/v143 before vswhere -latest so VS2026/MSVC 14.5x is not selected accidentally.
call :try_vcvars_for_toolset "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" "%ProgramFiles%\Microsoft Visual Studio\2022\Community"
if not errorlevel 1 exit /b 0
call :try_vcvars_for_toolset "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" "%ProgramFiles%\Microsoft Visual Studio\2022\Professional"
if not errorlevel 1 exit /b 0
call :try_vcvars_for_toolset "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise"
if not errorlevel 1 exit /b 0
call :try_vcvars_for_toolset "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
if not errorlevel 1 exit /b 0
call :try_vcvars_for_toolset "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools"
if not errorlevel 1 exit /b 0

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL=%%I"
    )
)
if defined VS_INSTALL (
    call :try_vcvars_for_toolset "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" "%VS_INSTALL%"
    if not errorlevel 1 exit /b 0
)

rem Last chance: VS2026 can contain a side-by-side v143 toolset. Auto-pick newest 14.3x/14.4x if installed.
set "VS_INSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL=%%I"
    )
)
if defined VS_INSTALL (
    call :try_vcvars_for_toolset "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" "%VS_INSTALL%"
    if not errorlevel 1 exit /b 0
)
goto :ensure_msvc_error

:try_vcvars_for_toolset
set "_VCVARS_PATH=%~1"
set "_VS_ROOT=%~2"
if not exist "%_VCVARS_PATH%" exit /b 1
set "_VCVARS_ARGS=%MSVC_VCVARS_ARGS%"
if defined _VCVARS_ARGS goto :try_vcvars_have_args
if defined MSVC_VCVARS_VER set "_VCVARS_ARGS=-vcvars_ver=%MSVC_VCVARS_VER%"
:try_vcvars_have_args
if defined _VCVARS_ARGS goto :try_vcvars_call
if /I not "%MSVC_TOOLSET%"=="v143" goto :try_vcvars_call
call :find_v143_ver "%_VS_ROOT%"
if defined AVABM_FOUND_V143_VER set "_VCVARS_ARGS=-vcvars_ver=%AVABM_FOUND_V143_VER%"

:try_vcvars_call
set "_VCVARS_CALL_ARGS=%_VCVARS_ARGS%"
echo %_VCVARS_PATH% | findstr /I /C:"vcvarsall.bat" >nul
if errorlevel 1 goto :try_vcvars_echo
echo %_VCVARS_CALL_ARGS% | findstr /I /C:"x64" >nul
if not errorlevel 1 goto :try_vcvars_echo
set "_VCVARS_CALL_ARGS=x64 %_VCVARS_CALL_ARGS%"

:try_vcvars_echo
if defined _VCVARS_CALL_ARGS (
    echo [Info] Loading MSVC environment: %_VCVARS_PATH% %_VCVARS_CALL_ARGS%
    call "%_VCVARS_PATH%" %_VCVARS_CALL_ARGS% >nul
) else (
    echo [Info] Loading MSVC environment: %_VCVARS_PATH%
    call "%_VCVARS_PATH%" >nul
)
where cl.exe >nul 2>nul
if errorlevel 1 exit /b 1
call :cl_matches_requested_toolset
if errorlevel 1 (
    echo [Warning] Loaded cl.exe does not match MSVC_TOOLSET=%MSVC_TOOLSET%.
    exit /b 1
)
exit /b 0

:find_v143_ver
set "AVABM_FOUND_V143_VER="
set "_AVABM_VS_ROOT=%~1"
if not defined _AVABM_VS_ROOT exit /b 0
if not exist "%_AVABM_VS_ROOT%\VC\Tools\MSVC" exit /b 0
for /f "tokens=*" %%V in ('dir /b /ad "%_AVABM_VS_ROOT%\VC\Tools\MSVC\14.4*" 2^>nul ^| sort /R') do (
    for /f "tokens=1,2 delims=." %%A in ("%%V") do set "AVABM_FOUND_V143_VER=%%A.%%B"
    goto :find_v143_done
)
for /f "tokens=*" %%V in ('dir /b /ad "%_AVABM_VS_ROOT%\VC\Tools\MSVC\14.3*" 2^>nul ^| sort /R') do (
    for /f "tokens=1,2 delims=." %%A in ("%%V") do set "AVABM_FOUND_V143_VER=%%A.%%B"
    goto :find_v143_done
)
:find_v143_done
exit /b 0

:cl_matches_requested_toolset
if /I "%MSVC_TOOLSET%"=="latest" exit /b 0
if /I not "%MSVC_TOOLSET%"=="v143" exit /b 0
if /I "%MSVC_STRICT_TOOLSET%"=="0" exit /b 0
if /I "%MSVC_STRICT_TOOLSET%"=="false" exit /b 0
where cl.exe >nul 2>nul
if errorlevel 1 exit /b 1
for /f "tokens=*" %%C in ('where cl.exe 2^>nul') do (
    set "_AVABM_CL_PATH=%%C"
    goto :cl_toolset_path_check
)
exit /b 1
:cl_toolset_path_check
echo %_AVABM_CL_PATH% | findstr /I /C:"\MSVC\14.3" >nul
if not errorlevel 1 exit /b 0
echo %_AVABM_CL_PATH% | findstr /I /C:"\MSVC\14.4" >nul
if not errorlevel 1 exit /b 0
exit /b 1

:ensure_msvc_error
echo =======================================================
echo [Error] MSVC cl.exe was not found for MSVC_TOOLSET=%MSVC_TOOLSET%.
echo [Error] For CUDA Toolkit on Windows, use VS2022/v143 ^(MSVC 14.3x or 14.4x^) instead of VS2026/MSVC 14.5x.
echo [Error] Install Visual Studio 2022 Build Tools with Desktop development with C++,
echo [Error] or install the v143 side-by-side toolset and set this in config.txt:
echo [Error] MSVC_TOOLSET=v143
echo [Error] MSVC_VCVARS64_BAT=C:\path\to\VC\Auxiliary\Build\vcvars64.bat
echo [Error] MSVC_VCVARS_VER=14.44
echo =======================================================
exit /b 1
