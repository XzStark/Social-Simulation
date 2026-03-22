from __future__ import annotations

import json
import os
from typing import Any

import httpx

from people.providers.dry_run import extract_json_object


class AnthropicClient:
    def __init__(self, model: str, *, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout_s = timeout_s

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 未设置")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # 强制 JSON：写在 system 末行；若失败则回退 extract
        sys = system + "\n\n你必须只输出一个 JSON 对象，不要 markdown。"
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "system": sys,
            "messages": [{"role": "user", "content": user}],
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        parts = data["content"]
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return extract_json_object(text)
