@echo off
REM Google AI Studio / Gemini API 密钥（与「网页版 Gemini」登录可能不同，以 AI Studio 里创建的 Key 为准）
REM https://aistudio.google.com/apikey

set GOOGLE_API_KEY=请在此处填写你的_Gemini_API_密钥
REM 若你习惯用 GEMINI_API_KEY 变量名，二选一即可，代码里两者都认：
REM set GEMINI_API_KEY=同上密钥

REM 路由前缀必须是 google: 或 gemini:（见 people/providers/registry.py）
set PEOPLE_MODEL_PRIMARY=google:gemini-2.0-flash
set PEOPLE_MODEL_FAST=google:gemini-2.0-flash

echo 环境已设置。若 PRIMARY 与 FAST 用同一模型，成本低但略慢。
echo 示例：python main.py run --scenario scenarios\default.yaml
