@echo off
chcp 65001 >nul
REM 用法：双击默认释放 8765；或拖入 cmd 执行：结束端口占用.cmd 8770
set PORT=%~1
if "%PORT%"=="" set PORT=8765
echo 正在尝试结束监听端口 %PORT% 的进程（常见为上次未关的 uvicorn）...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
  echo   taskkill /PID %%P /F
  taskkill /F /PID %%P 2>nul
)
echo 完成。然后可重新运行 run_web.cmd
pause
