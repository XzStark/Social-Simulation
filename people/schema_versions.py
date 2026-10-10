from __future__ import annotations

"""
外部接口与持久化对象的 schema 版本（v0）。

目标：
- 给 Web API 输入/输出、SSE 事件、checkpoint/pause 包提供稳定的版本号；
- 为向前兼容预留迁移入口（migrations）。
"""

PEOPLE_API_SCHEMA_VERSION = "0.1"
PEOPLE_CHECKPOINT_SCHEMA_VERSION = "0.1"
