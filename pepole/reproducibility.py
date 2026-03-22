"""
企业级可复现：环境指纹（Python / 平台 / git / 包版本），供清单与导出 JSON 挂载。
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pepole


def _git_head(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _git_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment_fingerprint(*, repo_root: Path | None = None) -> dict[str, Any]:
    """与场景内容哈希互补：同一 YAML 在不同环境仍可对齐版本与 dirty 状态。"""
    ver = getattr(pepole, "__version__", "unknown")
    try:
        from importlib.metadata import version as pkg_version

        ver = pkg_version("pepole")
    except Exception:
        pass
    git = _git_head(repo_root)
    dirty = _git_dirty(repo_root) if git else None
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "pepole_package_version": ver,
        "git_commit": git,
        "git_worktree_dirty": dirty,
    }
