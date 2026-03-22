@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 请勿在本文件写入 API 密钥。请先在终端设置环境变量，例如：
REM   set OPENAI_API_KEY=你的密钥
REM   set OPENAI_BASE_URL=https://api.openai.com/v1
REM 密钥示例脚本（填好密钥后在 CMD 里 call 一次再运行本文件）：
REM   配置示例-OpenAI-ChatGPT.cmd  配置示例-Gemini.cmd  配置示例-Claude.cmd  配置示例-DeepSeek.cmd
REM 未设置 OPENAI_API_KEY 时，部分推理为 Dry-Run，仍可打开网页浏览。

REM 若提示端口占用，可先运行「结束端口占用.cmd」，或改下面 PORT 为其它数字
set PORT=8770
echo 浏览器访问: http://127.0.0.1:%PORT%（约 2 秒后自动打开）
echo 按 Ctrl+C 停止服务
start "" cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:%PORT%/"
python -m uvicorn pepole.webapp:app --host 127.0.0.1 --port %PORT%
