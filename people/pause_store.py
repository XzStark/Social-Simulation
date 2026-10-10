"""演练暂停快照存储（默认内存，可选持久化到磁盘）。

默认行为仍为进程内存存储（兼容原实现），但若设置环境变量
`PEOPLE_CHECKPOINT_DIR`，则会在该目录下按 token 写入 pickle 文件，
使得同一实例重启后仍可续跑或导出快照（不跨机器同步）。
"""

from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

_lock = Lock()
_STORE: dict[str, bytes] = {}
_CHECKPOINT_DIR_ENV = "PEOPLE_CHECKPOINT_DIR"


def _checkpoint_dir() -> Path | None:
    """返回磁盘持久化目录（若配置），否则 None 表示纯内存模式。"""
    d = (os.environ.get(_CHECKPOINT_DIR_ENV) or "").strip()
    if not d:
        return None
    path = Path(d).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # 若目录不可创建，则退化为纯内存模式，避免影响主流程
        return None
    return path


def _disk_path_for_token(token: str) -> Path | None:
    base = _checkpoint_dir()
    if base is None:
        return None
    # 每个 token 一个独立文件；不暴露真实结构给用户
    return base / f"{token}.pkl"


def put_checkpoint(obj: Any) -> str:
    token = str(uuid.uuid4())
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    path = _disk_path_for_token(token)
    with _lock:
        _STORE[token] = data
        if path is not None:
            try:
                path.write_bytes(data)
            except Exception:
                # 若磁盘写入失败，退化为仅内存；不抛出以避免影响演练主流程
                pass
    return token


def take_checkpoint(token: str) -> Any | None:
    path = _disk_path_for_token(token)
    with _lock:
        raw = _STORE.pop(token, None)
        # 优先使用磁盘内容，以便跨进程/重启后仍可恢复
        if path is not None and path.is_file():
            try:
                raw = path.read_bytes()
            except Exception:
                # 读取失败则退回内存版本（若有）
                pass
            try:
                path.unlink(missing_ok=True)
            except Exception:
                # 删除失败不影响使用
                pass
    if raw is None:
        return None
    return pickle.loads(raw)


def copy_checkpoint(token: str) -> Any | None:
    """读取快照但不消费（用于仅评估、或前端多次读取）。"""
    path = _disk_path_for_token(token)
    with _lock:
        raw = _STORE.get(token)
        if path is not None and path.is_file():
            try:
                raw = path.read_bytes()
            except Exception:
                # 读取失败则退回内存版本（若有）
                pass
    if raw is None:
        return None
    return pickle.loads(raw)
