@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

title AVABM Launcher
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "CUDA_DIR=%~dp0avabm_cuda"
set "CONFIG_FILE=%~dp0config.txt"
set "BUILD_HELPER=%CUDA_DIR%\build_helper.py"
set "ORIGINAL_ARGS=%*"
set "AVABM_EXIT_CODE=0"

if exist "%CONFIG_FILE%" (
    echo [Info] Loading config: %CONFIG_FILE%
    for /f "tokens=1,* delims==" %%A in ('findstr /R /C:"^[A-Za-z_][A-Za-z0-9_]*=" "%CONFIG_FILE%" 2^>nul') do (
        call set "%%~A=%%~B"
    )
) else (
    echo [Warning] config.txt not found. Built-in defaults will be used.
)

if not defined CONDA_ENV set "CONDA_ENV=avabm"
if not defined CONDA_ACTIVATE_PATH set "CONDA_ACTIVATE_PATH=%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined PYTHON_COMMAND set "PYTHON_COMMAND=python"
if not defined CUDA_HOME set "CUDA_HOME=auto"
if not defined TORCH_NVCC_FLAGS set "TORCH_NVCC_FLAGS=-allow-unsupported-compiler"
if not defined TORCH_DONT_CHECK_COMPILER_ABI set "TORCH_DONT_CHECK_COMPILER_ABI=1"
if not defined MAX_JOBS set "MAX_JOBS=1"
if not defined CUDA_BUILD_MAX_JOBS set "CUDA_BUILD_MAX_JOBS=1"
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
if not defined MSVC_TOOLSET if defined MSVC_TOOLSET_VERSION set "MSVC_TOOLSET=%MSVC_TOOLSET_VERSION%"
if not defined MSVC_TOOLSET set "MSVC_TOOLSET=v143"
if not defined MSVC_VCVARS_VER set "MSVC_VCVARS_VER="
if not defined MSVC_VCVARS_ARGS set "MSVC_VCVARS_ARGS="
if not defined MSVC_STRICT_TOOLSET set "MSVC_STRICT_TOOLSET=1"
if not defined MSVC_VCVARS64_BAT if defined VCVARS64_BAT set "MSVC_VCVARS64_BAT=%VCVARS64_BAT%"
if not defined MSVC_VCVARS64_BAT set "MSVC_VCVARS64_BAT="

if not exist "%BUILD_HELPER%" (
    echo [Error] Missing CUDA build helper: %BUILD_HELPER%
    set "AVABM_EXIT_CODE=1"
    goto :finish
)

if "%~1"=="" goto :menu
if /I "%~1"=="turbo" goto :cmd_turbo
if /I "%~1"=="--turbo" goto :cmd_turbo
if /I "%~1"=="headless" goto :cmd_turbo
if /I "%~1"=="--headless" goto :cmd_turbo
if /I "%~1"=="visual" goto :cmd_visual
if /I "%~1"=="--visual" goto :cmd_visual
if /I "%~1"=="gui" goto :cmd_visual
if /I "%~1"=="--gui" goto :cmd_visual
if /I "%~1"=="build" goto :cmd_build
if /I "%~1"=="rebuild" goto :cmd_rebuild
if /I "%~1"=="clean" goto :cmd_rebuild
if /I "%~1"=="hardclean" goto :cmd_hardclean
if /I "%~1"=="cleanall" goto :cmd_hardclean
if /I "%~1"=="clean-all" goto :cmd_hardclean
if /I "%~1"=="status" goto :cmd_status
if /I "%~1"=="check" goto :cmd_status

echo [Warning] Unknown command: %~1
echo [Info] Opening launcher menu instead.

goto :menu

:menu
echo.
echo =======================================================
echo AVABM Launcher
echo =======================================================
echo 1. Turbo   - headless CUDA batch run
echo 2. Visual  - OpenGL/Pygame window run
echo 3. Build CUDA only ^(skip if up to date^)
echo 4. Clean rebuild CUDA
echo 5. Hard clean + rebuild CUDA
echo 6. CUDA build status
echo 7. Exit
echo.
set "CHOICE="
set /p "CHOICE=Select [1-7]: "
if "%CHOICE%"=="1" goto :cmd_turbo
if "%CHOICE%"=="2" goto :cmd_visual
if "%CHOICE%"=="3" goto :cmd_build
if "%CHOICE%"=="4" goto :cmd_rebuild
if "%CHOICE%"=="5" goto :cmd_hardclean
if "%CHOICE%"=="6" goto :cmd_status
if "%CHOICE%"=="7" goto :finish
if /I "%CHOICE%"=="t" goto :cmd_turbo
if /I "%CHOICE%"=="turbo" goto :cmd_turbo
if /I "%CHOICE%"=="v" goto :cmd_visual
if /I "%CHOICE%"=="visual" goto :cmd_visual
echo [Warning] Invalid selection.
goto :menu

:cmd_turbo
call :ensure_cuda_built
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
if not "%AVABM_EXIT_CODE%"=="0" goto :finish
call :launch_mode turbo
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:cmd_visual
call :ensure_cuda_built
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
if not "%AVABM_EXIT_CODE%"=="0" goto :finish
call :launch_mode visual
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:cmd_build
call :build_cuda auto
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:cmd_rebuild
call :build_cuda clean
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:cmd_hardclean
call :build_cuda hardclean
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:cmd_status
call :activate_conda
if errorlevel 1 (
    set "AVABM_EXIT_CODE=1"
    goto :finish
)
%PYTHON_COMMAND% "%BUILD_HELPER%" status
set "AVABM_EXIT_CODE=%ERRORLEVEL%"
goto :finish

:finish
echo.
if "%AVABM_EXIT_CODE%"=="0" (
    echo [Info] Finished.
) else (
    echo [Error] Finished with exit code %AVABM_EXIT_CODE%.
)
pause
endlocal & exit /b %AVABM_EXIT_CODE%

:activate_conda
if defined AVABM_CONDA_READY exit /b 0
if exist "%CONDA_ACTIVATE_PATH%" (
    call "%CONDA_ACTIVATE_PATH%" base
    if errorlevel 1 (
        echo [Error] Failed to activate Conda base environment.
        exit /b 1
    )
    conda env list | findstr /R /C:"^%CONDA_ENV%[ ]" >nul 2>nul
    if errorlevel 1 (
        echo [Info] Creating Conda environment: %CONDA_ENV%
        call conda create -n "%CONDA_ENV%" python=3.12 ipykernel -y
        if errorlevel 1 (
            echo [Error] Failed to create Conda environment: %CONDA_ENV%
            exit /b 1
        )
    )
    echo [Info] Activating Conda environment: %CONDA_ENV%
    call "%CONDA_ACTIVATE_PATH%" "%CONDA_ENV%"
    if errorlevel 1 (
        echo [Error] Failed to activate Conda environment: %CONDA_ENV%
        exit /b 1
    )
) else (
    echo [Warning] Conda activate script not found: %CONDA_ACTIVATE_PATH%
    echo [Warning] Continuing with the current shell Python.
)
set "AVABM_CONDA_READY=1"
exit /b 0

:ensure_cuda_built
call :activate_conda
if errorlevel 1 exit /b 1
%PYTHON_COMMAND% "%BUILD_HELPER%" verify-run
if errorlevel 3 goto :ensure_cuda_check_failed
if errorlevel 2 goto :ensure_cuda_build_now
exit /b 0
:ensure_cuda_check_failed
echo [Error] CUDA build fingerprint check failed.
exit /b 1
:ensure_cuda_build_now
echo [Info] CUDA extension is missing or stale. Building now...
call :build_cuda auto
exit /b %ERRORLEVEL%

:launch_mode
set "RUN_MODE=%~1"
call :activate_conda
if errorlevel 1 exit /b 1
set "SIM_ENGINE=ecs_abm_cuda"
set "SIMULATION_ENGINE=ecs_abm_cuda"
if /I "%RUN_MODE%"=="turbo" goto :launch_turbo
if /I "%RUN_MODE%"=="visual" goto :launch_visual
echo [Error] Unknown run mode: %RUN_MODE%
exit /b 1
:launch_turbo
set "HEADLESS_MODE=1"
set "SIM_HEADLESS=1"
set "ABM_TURBO_PROFILE=1"
echo [Info] Starting AVABM in Turbo mode.
%PYTHON_COMMAND% main.py --ecs --turbo %ORIGINAL_ARGS%
exit /b %ERRORLEVEL%
:launch_visual
set "HEADLESS_MODE=0"
set "SIM_HEADLESS=0"
echo [Info] Starting AVABM in Visual mode.
%PYTHON_COMMAND% main.py --ecs --visual %ORIGINAL_ARGS%
exit /b %ERRORLEVEL%

:build_cuda
set "BUILD_REQUEST=%~1"
if not defined BUILD_REQUEST set "BUILD_REQUEST=auto"
call :activate_conda
if errorlevel 1 exit /b 1
call :resolve_cuda_home
if errorlevel 1 exit /b 1
call :cap_max_jobs

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
if errorlevel 1 exit /b 1

set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
set "USE_NINJA=1"
set "CMAKE_BUILD_PARALLEL_LEVEL=%MAX_JOBS%"
set "MAX_JOBS=%MAX_JOBS%"

where cl.exe >nul 2>nul
if errorlevel 1 (
    echo =======================================================
    echo [Error] cl.exe is not available after MSVC setup.
    echo [Error] Install Visual Studio 2022 Build Tools with Desktop C++ workload.
    echo [Error] Or set MSVC_VCVARS64_BAT in config.txt.
    echo =======================================================
    exit /b 1
)
call :cl_matches_requested_toolset
if errorlevel 1 (
    echo =======================================================
    echo [Error] cl.exe does not match MSVC_TOOLSET=%MSVC_TOOLSET%.
    echo [Error] Use VS2022/v143 ^(MSVC 14.3x or 14.4x^) or set MSVC_TOOLSET=latest explicitly.
    echo =======================================================
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
    set "CL=/wd4996 /wd4819 /wd4251 /wd4275 /wd4244 /wd4267 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 %CL%"
) else (
    set "CL=/wd4996 /wd4819 %CL%"
)

where nvcc.exe >nul 2>nul
if errorlevel 1 (
    echo =======================================================
    echo [Error] nvcc.exe was not found. Check CUDA_HOME in config.txt.
    echo =======================================================
    exit /b 1
)

where ninja.exe >nul 2>nul
if errorlevel 1 (
    echo [Info] ninja.exe not found. Installing ninja into the Conda environment...
    call conda install -n "%CONDA_ENV%" ninja -y
    if errorlevel 1 (
        echo [Error] Failed to install ninja.
        exit /b 1
    )
)

if /I "%BUILD_REQUEST%"=="hardclean" goto :build_hard_clean
if /I "%BUILD_REQUEST%"=="clean" goto :build_full_clean
if /I "%CUDA_FORCE_REBUILD%"=="1" goto :build_full_clean
if /I "%CUDA_SKIP_IF_UP_TO_DATE%"=="1" goto :build_maybe_skip
goto :build_prepare_incremental

:build_hard_clean
echo [Info] Hard clean rebuild requested. Project outputs plus user-local PyTorch/CUDA caches will be removed.
%PYTHON_COMMAND% "%BUILD_HELPER%" hardclean
if errorlevel 1 exit /b 1
goto :build_after_prepare

:build_full_clean
echo [Info] Full clean rebuild requested.
%PYTHON_COMMAND% "%BUILD_HELPER%" clean
if errorlevel 1 exit /b 1
goto :build_after_prepare

:build_maybe_skip
%PYTHON_COMMAND% "%BUILD_HELPER%" check
if errorlevel 3 (
    echo [Error] Build status check failed.
    exit /b 1
)
if errorlevel 2 goto :build_prepare_incremental
goto :build_skipped

:build_prepare_incremental
if /I "%CUDA_INCREMENTAL_BUILD%"=="1" (
    echo [Info] Preparing incremental build: remove stale extension binary only, keep build cache.
    %PYTHON_COMMAND% "%BUILD_HELPER%" prepare
) else (
    echo [Info] Incremental build disabled, cleaning full build cache.
    %PYTHON_COMMAND% "%BUILD_HELPER%" clean
)
if errorlevel 1 exit /b 1

:build_after_prepare
set "BUILD_LOG_DIR=%CUDA_DIR%\build_logs"
if not exist "%BUILD_LOG_DIR%" mkdir "%BUILD_LOG_DIR%"
set "BUILD_LOG=%BUILD_LOG_DIR%\build_last.log"

echo [Info] Building CUDA extension in place...
echo [Info] Full build log: %BUILD_LOG%
echo [Info] Repeated builds are skipped after the binary is stamped.
%PYTHON_COMMAND% "%BUILD_HELPER%" runbuild
if errorlevel 1 (
    echo =======================================================
    echo [Error] CUDA build failed. No old extension binary will be used.
    echo [Error] The last %CUDA_BUILD_LOG_TAIL% log lines are printed below.
    echo [Error] Full log: %BUILD_LOG%
    echo =======================================================
    if exist "%BUILD_LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail %CUDA_BUILD_LOG_TAIL%"
    exit /b 1
)
echo [Info] Build succeeded. Short log tail:
if exist "%BUILD_LOG%" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%BUILD_LOG%' -Tail 40"

%PYTHON_COMMAND% "%BUILD_HELPER%" mark
if errorlevel 1 exit /b 1

echo =======================================================
echo [Info] CUDA build completed successfully.
echo =======================================================
exit /b 0

:build_skipped
echo =======================================================
echo [Info] CUDA build skipped: compiled module already matches the source fingerprint.
echo [Info] Use "run.bat rebuild" only when you need a full rebuild.
echo =======================================================
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
echo [Error] Install Visual Studio 2022 Build Tools with Desktop development with C++.
echo [Error] Or set MSVC_VCVARS64_BAT, MSVC_TOOLSET, and MSVC_VCVARS_VER in config.txt.
echo =======================================================
exit /b 1
