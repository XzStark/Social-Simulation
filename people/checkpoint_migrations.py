from __future__ import annotations

from typing import Any

from people.schema_versions import PEOPLE_CHECKPOINT_SCHEMA_VERSION


def migrate_checkpoint_package(pkg: Any) -> Any:
    """
    最小向前兼容入口：
    - 允许旧 checkpoint 不带 checkpoint_schema_version
    - 未来可在此按版本号逐步迁移字段名/结构
    """
    if not isinstance(pkg, dict):
        return pkg
    ver = str(pkg.get("checkpoint_schema_version") or "").strip()
    if not ver:
        pkg = {**pkg, "checkpoint_schema_version": PEOPLE_CHECKPOINT_SCHEMA_VERSION}
        return pkg
    # v0.1 当前无结构迁移，仅保留钩子
    return pkg
