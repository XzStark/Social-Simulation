from __future__ import annotations

import json
import os
from typing import Any

import httpx

from pepole.providers.dry_run import extract_json_object


class OpenAICompatClient:
    """OpenAI 官方或兼容网关（DeepSeek / Together / Azure 等）：/v1/chat/completions"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s

    def _parse_message_json(self, data: dict[str, Any]) -> dict[str, Any]:
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        # DeepSeek deepseek-reasoner：最终答案在 content；偶发空时尝试从思维链尾部抽 JSON
        if not raw and msg.get("reasoning_content"):
            raw = str(msg.get("reasoning_content") or "").strip()
        if not raw:
            raise ValueError("empty model content")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return extract_json_object(raw)

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        url = f"{self.base_url}/chat/completions"
        is_reasoner = "reasoner" in self.model.lower()
        base: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 8192,
        }
        if not is_reasoner:
            base["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = max(self.timeout_s, 900.0) if is_reasoner else self.timeout_s
        last_exc: Exception | None = None
        with httpx.Client(timeout=timeout) as client:
            for use_json_object in (True, False):
                body = dict(base)
                if use_json_object:
                    body["response_format"] = {"type": "json_object"}
                r = client.post(url, json=body, headers=headers)
                if r.status_code >= 400:
                    detail = r.text[:800]
                    hint = ""
                    low = detail.lower()
                    if r.status_code == 400 and (
                        "model" in low and ("not exist" in low or "does not exist" in low or "invalid" in low)
                    ):
                        hint = (
                            f" [pepole 提示：本次请求的 model={self.model!r}，API 基座={self.base_url!r}。"
                            "模型名必须与该服务商一致：DeepSeek 一般为 deepseek-chat / deepseek-reasoner；"
                            "OpenAI 官方为 gpt-4o 等。请检查 PEPOLE_MODEL_PRIMARY / FAST 或网页里的 PRIMARY/FAST。]"
                        )
                    last_exc = RuntimeError(f"{r.status_code} {detail}{hint}")
                    continue
                try:
                    return self._parse_message_json(r.json())
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    last_exc = e
                    continue
        if last_exc:
            raise last_exc
        raise RuntimeError("chat completion failed")
