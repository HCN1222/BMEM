@echo off
setlocal

:: Activate conda base and then run the PowerShell script under BMEM env
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" 2>nul
if errorlevel 1 (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
)

powershell.exe -ExecutionPolicy Bypass -File "C:\Users\hcn12\OneDrive\Desktop\NING\Github\BMEM\script\run_daily_update.ps1"

if errorlevel 1 (
    echo [ERROR] run_daily_update.ps1 failed with exit code %errorlevel%
    exit /b %errorlevel%
)

endlocal
