@echo off
REM DeepSeek 走 OpenAI 兼容接口。请把下面密钥改为你的真实密钥（勿提交到 Git）。
REM 文档：https://api-docs.deepseek.com/

set OPENAI_API_KEY=你的_DeepSeek_API_密钥
set OPENAI_BASE_URL=https://api.deepseek.com/v1

REM 最强推理模型（适合关键人设；较慢、略贵）
set PEOPLE_MODEL_PRIMARY=openai:deepseek-reasoner

REM 轻量对话模型（cohort 批量等；省钱）
set PEOPLE_MODEL_FAST=openai:deepseek-chat

echo 环境已设置。运行示例：
echo   python main.py run --scenario scenarios\default.yaml
echo   python main.py run --scenario scenarios\policy_county_example.yaml
