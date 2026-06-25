@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

rem Force the most stable profile that still keeps ptxas optimization at O1.
set "CUDA_PTXAS_PARTITION_BUILD=1"
set "CUDA_SINGLE_TU_BUILD=0"
set "CUDA_OPT_LEVEL=2"
set "CUDA_PTXAS_OPT_LEVEL=1"
set "CUDA_PTXAS_ALLOW_EXPENSIVE_OPT=0"
set "CUDA_PTXAS_SAFE_MODE=0"
set "CUDA_DEVICE_FORCEINLINE=0"
set "CUDA_DEVICE_NOINLINE=1"
set "CUDA_FAST_MATH=0"
set "CUDA_NVCC_THREADS=0"
set "CUDA_SPLIT_COMPILE=0"
set "MAX_JOBS=1"
set "CUDA_BUILD_MAX_JOBS=1"
set "CUDA_AUTO_RETRY_PTXAS_CRASH=1"
set "CUDA_SAFE_RETRY_CLEAN=1"

call "%~dp0build.bat" clean
endlocal
