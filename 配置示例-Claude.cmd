@echo off
REM Anthropic Claude API
REM https://console.anthropic.com/

set ANTHROPIC_API_KEY=请在此处填写你的_Anthropic_API_密钥

REM 强模型 + 轻量模型（仅依赖本密钥）
set PEPOLE_MODEL_PRIMARY=anthropic:claude-sonnet-4-20250514
set PEPOLE_MODEL_FAST=anthropic:claude-3-5-haiku-20241022

echo 环境已设置。若要与 OpenAI 混用，请自行增设 OPENAI_API_KEY 并把 FAST 改为 openai:gpt-4o-mini 等。
