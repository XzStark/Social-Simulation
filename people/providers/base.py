from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """统一：返回解析后的 JSON 对象。"""

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]: ...
