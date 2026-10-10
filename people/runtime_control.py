from __future__ import annotations

import copy
import uuid
from threading import Lock
from typing import Any

_LOCK = Lock()
_RUNTIME_BLACK_SWANS: dict[str, list[dict[str, Any]]] = {}


def create_runtime_control_token() -> str:
    token = str(uuid.uuid4())
    with _LOCK:
        _RUNTIME_BLACK_SWANS[token] = []
    return token


def enqueue_black_swan(token: str, event: dict[str, Any]) -> bool:
    with _LOCK:
        if token not in _RUNTIME_BLACK_SWANS:
            return False
        _RUNTIME_BLACK_SWANS[token].append(copy.deepcopy(event))
    return True


def enqueue_pause_request(token: str, reason: str | None = None) -> bool:
    control_event: dict[str, Any] = {"_control": "pause"}
    if reason and str(reason).strip():
        control_event["reason"] = str(reason).strip()[:300]
    with _LOCK:
        if token not in _RUNTIME_BLACK_SWANS:
            return False
        _RUNTIME_BLACK_SWANS[token].append(control_event)
    return True


def pop_black_swans(token: str) -> list[dict[str, Any]]:
    with _LOCK:
        q = _RUNTIME_BLACK_SWANS.get(token)
        if not q:
            return []
        out = list(q)
        q.clear()
    return out


def release_runtime_control_token(token: str) -> None:
    with _LOCK:
        _RUNTIME_BLACK_SWANS.pop(token, None)
