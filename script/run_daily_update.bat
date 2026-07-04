@echo off
setlocal

set "BROKER_ID=%~1"
if not defined BROKER_ID set "BROKER_ID=8440"

:: Activate conda base and then run the PowerShell script under BMEM env
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" 2>nul
if errorlevel 1 (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
)

powershell.exe -ExecutionPolicy Bypass -File "%~dp0run_daily_update.ps1" -BrokerId "%BROKER_ID%"

if errorlevel 1 (
    echo [ERROR] run_daily_update.ps1 failed with exit code %errorlevel%
    exit /b %errorlevel%
)

endlocal
