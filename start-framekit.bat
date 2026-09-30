@echo off
rem Start the FrameKit web app and open it in the browser.
rem   Double-click this file, or run it from a terminal.
rem   Set FRAMEKIT_HOST=0.0.0.0 first to allow other machines on your network (there is no login).
cd /d "%~dp0"

set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
rem Host and port can also come from .env (FRAMEKIT_HOST=..., FRAMEKIT_PORT=...); a variable
rem already set in the environment wins. FrameKit itself reads the rest of .env.
if exist ".env" for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
  if /i "%%a"=="FRAMEKIT_HOST" if not defined FRAMEKIT_HOST set "FRAMEKIT_HOST=%%~b"
  if /i "%%a"=="FRAMEKIT_PORT" if not defined FRAMEKIT_PORT set "FRAMEKIT_PORT=%%~b"
)
if "%FRAMEKIT_HOST%"=="" set "FRAMEKIT_HOST=127.0.0.1"
if "%FRAMEKIT_PORT%"=="" set "FRAMEKIT_PORT=8082"

where ffmpeg >nul 2>nul || (
  echo ffmpeg was not found on PATH. Install it with:  winget install Gyan.FFmpeg
  echo then open a new terminal and run this again.
  pause
  exit /b 1
)

if /i not "%~1"=="--no-browser" start "" cmd /c "timeout /t 3 >nul & start http://localhost:%FRAMEKIT_PORT%/"
"%PY%" -m framekit serve --host %FRAMEKIT_HOST% --port %FRAMEKIT_PORT%
if errorlevel 1 pause
