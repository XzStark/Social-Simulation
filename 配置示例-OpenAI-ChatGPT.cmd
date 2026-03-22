@echo off
REM OpenAI 官方 API（网页版 ChatGPT 与 API 计费/密钥是两套体系，此处为 API Key）
REM 控制台：https://platform.openai.com/api-keys

set OPENAI_API_KEY=请在此处填写你的_OpenAI_API_密钥
REM 官方默认地址，一般无需改：
set OPENAI_BASE_URL=https://api.openai.com/v1

REM 强模型 / 快模型（可按账户可用模型名调整）
set PEPOLE_MODEL_PRIMARY=openai:gpt-4o
set PEPOLE_MODEL_FAST=openai:gpt-4o-mini

echo 环境已设置。示例：
echo   python main.py run --scenario scenarios\default.yaml
echo   run_web.cmd
