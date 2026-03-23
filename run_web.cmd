@echo off
setlocal EnableExtensions
cd /d "%~dp0"
REM Do not put API keys here. Set OPENAI_API_KEY etc. in the environment first.
REM Use the sample *.cmd files in this folder (call them once before run_web).

set PORT=8770

where py >nul 2>&1 && goto run_py
where python >nul 2>&1 && goto run_python

echo [ERROR] Neither "py" nor "python" was found in PATH.
echo Install Python from https://www.python.org/downloads/ and enable "Add python.exe to PATH".
pause
exit /b 1

:run_py
echo Using: py -3
echo Browser opens in ~4s at http://127.0.0.1:%PORT%/
echo Keep this window open. Ctrl+C stops the server.
start /MIN cmd /c "ping -n 5 127.0.0.1>nul & start http://127.0.0.1:%PORT%/"
py -3 -m uvicorn people.webapp:app --host 127.0.0.1 --port %PORT%
goto uvicorn_done

:run_python
echo Using: python
echo Browser opens in ~4s at http://127.0.0.1:%PORT%/
echo Keep this window open. Ctrl+C stops the server.
start /MIN cmd /c "ping -n 5 127.0.0.1>nul & start http://127.0.0.1:%PORT%/"
python -m uvicorn people.webapp:app --host 127.0.0.1 --port %PORT%

:uvicorn_done
if errorlevel 1 (
  echo.
  echo Server exited with an error. If you did not press Ctrl+C, read messages above.
  echo If the port is in use, free it or edit PORT at the top of this file.
  pause
)
