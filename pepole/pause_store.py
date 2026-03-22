"""内存暂存：演练因危机暂停时的可续跑快照（单进程；重启即失效）。"""

from __future__ import annotations

import pickle
import uuid
from threading import Lock
from typing import Any

_lock = Lock()
_STORE: dict[str, bytes] = {}


def put_checkpoint(obj: Any) -> str:
    token = str(uuid.uuid4())
    with _lock:
        _STORE[token] = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return token


def take_checkpoint(token: str) -> Any | None:
    with _lock:
        raw = _STORE.pop(token, None)
    if raw is None:
        return None
    return pickle.loads(raw)


def copy_checkpoint(token: str) -> Any | None:
    """读取快照但不消费（用于仅评估、或前端多次读取）。"""
    with _lock:
        raw = _STORE.get(token)
    if raw is None:
        return None
    return pickle.loads(raw)
