@echo off
REM start_riceguard_server.bat -- opens the RiceGuard server console.
REM Does NOT start the server automatically -- type 'start' once it's open.
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Could not find the project venv at %VENV_PY%
    echo Run this from the project's existing .venv setup.
    pause
    exit /b 1
)

"%VENV_PY%" "%ROOT%server\riceguard_console.py"
