from __future__ import annotations

import json
import os
from typing import Any

import httpx

from people.providers.dry_run import extract_json_object


class GeminiClient:
    """Google AI Studio：gemini-* 模型，REST generateContent。"""

    def __init__(self, model: str, *, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self.model = model.removeprefix("models/")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        self.timeout_s = timeout_s

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY（或 GEMINI_API_KEY）未设置")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system + "\n只输出 JSON 对象。"}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, params={"key": self.api_key}, json=body)
            r.raise_for_status()
            data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return extract_json_object(text)
