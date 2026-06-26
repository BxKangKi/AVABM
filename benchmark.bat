@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

if "%~1"=="" goto :menu
if /I "%~1"=="menu" goto :menu
if /I "%~1"=="select" goto :menu
if /I "%~1"=="architectures" goto :menu

goto :direct

:menu
call "%~dp0run.bat" benchmark-menu
exit /b %ERRORLEVEL%

:direct
call "%~dp0run.bat" benchmark %*
exit /b %ERRORLEVEL%
