"""
模拟前地区 grounding 可选的开放网页检索：抓取搜索引擎返回的标题/链接/摘要片段，
供快模型合成「情境先验」。非全文抓取、非官方法规库；结果可能偏题、过时或含推广，须人工核实。

网络说明：默认 **duckduckgo** 依赖第三方库访问 DuckDuckGo；在**中国大陆公网**上常出现
超时、空结果或被干扰，**不保证可用**。LLM（如 DeepSeek、OpenAI 兼容端点）与搜索提供商**无关**：
可照常调 DeepSeek 做合成，同时改用 **Brave**（`BRAVE_API_KEY` + `PEOPLE_WEB_SEARCH_PROVIDER=brave`）
或自备可访问的搜索 API / 代理。详见系统说明 §5.1。
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from people.config import RegionalGroundingSpec, Scenario


def _brave_api_key() -> str:
    return (os.environ.get("PEOPLE_BRAVE_API_KEY") or os.environ.get("BRAVE_API_KEY") or "").strip()


def _resolve_provider(rg: RegionalGroundingSpec) -> str:
    env = (os.environ.get("PEOPLE_WEB_SEARCH_PROVIDER") or "").strip().lower()
    if env in ("brave", "duckduckgo"):
        return env
    return str(rg.web_search_provider or "duckduckgo")


def _queries_for_scenario(scenario: Scenario, rg: RegionalGroundingSpec) -> list[str]:
    region = (rg.region_label or scenario.policy_context.jurisdiction_name or "").strip()
    pb = (scenario.player_brief or "").strip()
    pb_short = pb[:140].replace("\n", " ").strip()
    extra_raw = (rg.web_search_extra_queries or "").strip()
    extra = [x.strip() for x in re.split(r"[\n;；,，]+", extra_raw) if x.strip()]
    budget = max(1, min(5, int(rg.web_search_query_budget)))

    if scenario.exercise_type == "policy":
        base = [
            f"{region} 政府 政策" if region else "地方公共政策 执行",
            pb_short or (f"{region} 民生 治理" if region else "公共管理"),
        ]
    else:
        sec = (rg.business_sector_brief or "").strip()
        base = [
            f"{region} {sec or '营商环境'} 监管 执法" if region else f"{sec or '行业'} 监管",
            pb_short or (f"{region} 市场" if region else "消费投诉"),
        ]

    seen: set[str] = set()
    out: list[str] = []
    for q in base + extra:
        qn = q.strip()[:220]
        if len(qn) < 3 or qn in seen:
            continue
        seen.add(qn)
        out.append(qn)
        if len(out) >= budget:
            break
    return out


def _format_hits(hits: list[dict[str, str]], max_chars: int) -> str:
    if max_chars <= 0 or not hits:
        return ""
    parts: list[str] = []
    n = 0
    for h in hits:
        title = (h.get("title") or "").strip()
        url = (h.get("url") or "").strip()
        body = (h.get("body") or "").strip()[:400]
        line = f"· {title}\n  {url}\n  {body}\n"
        if n + len(line) > max_chars:
            break
        parts.append(line)
        n += len(line)
    return "\n".join(parts).strip()


def _search_duckduckgo(queries: list[str], max_results: int) -> tuple[list[dict[str, str]], str | None]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [], "未安装 duckduckgo-search：请 pip install duckduckgo-search，或改用 Brave API（BRAVE_API_KEY）。"

    hits: list[dict[str, str]] = []
    err: str | None = None
    try:
        with DDGS() as ddgs:
            for q in queries:
                if not q.strip():
                    continue
                try:
                    for r in ddgs.text(q, max_results=max_results):
                        hits.append(
                            {
                                "title": str(r.get("title") or ""),
                                "url": str(r.get("href") or r.get("url") or ""),
                                "body": str(r.get("body") or ""),
                            }
                        )
                except (OSError, httpx.HTTPError, TypeError, ValueError, RuntimeError) as e:
                    err = err or f"DuckDuckGo 检索异常：{type(e).__name__}"
    except (OSError, TypeError, ValueError, RuntimeError) as e:
        return [], f"DuckDuckGo 初始化失败：{type(e).__name__}"
    if not hits and err is None:
        err = "DuckDuckGo 未返回结果（可能被限流或查询过短）。"
    return hits, err


def _search_brave(queries: list[str], max_results: int, api_key: str) -> tuple[list[dict[str, str]], str | None]:
    if not api_key:
        return [], "未设置 BRAVE_API_KEY（或 PEOPLE_BRAVE_API_KEY）。"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    hits: list[dict[str, str]] = []
    err: str | None = None
    try:
        with httpx.Client(timeout=22.0, follow_redirects=True) as client:
            for q in queries:
                if not q.strip():
                    continue
                try:
                    r = client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": q, "count": max(1, min(20, max_results))},
                        headers=headers,
                    )
                except (OSError, httpx.HTTPError) as e:
                    err = err or f"Brave HTTP：{type(e).__name__}"
                    continue
                if r.status_code != 200:
                    err = err or f"Brave API HTTP {r.status_code}"
                    continue
                try:
                    data = r.json()
                except ValueError:
                    err = err or "Brave 响应非 JSON"
                    continue
                for item in (data.get("web") or {}).get("results") or []:
                    hits.append(
                        {
                            "title": str(item.get("title") or ""),
                            "url": str(item.get("url") or ""),
                            "body": str(item.get("description") or ""),
                        }
                    )
    except (OSError, httpx.HTTPError) as e:
        return [], f"Brave 请求失败：{type(e).__name__}"
    if not hits and err is None:
        err = "Brave 未返回网页结果。"
    return hits, err


def fetch_web_grounding_snippets(scenario: Scenario, rg: RegionalGroundingSpec) -> tuple[str, dict[str, Any]]:
    """
    返回 (拼入快模型上下文的纯文本, 复盘用元数据)。
    """
    meta: dict[str, Any] = {
        "used": False,
        "provider": "",
        "queries": [],
        "snippet_chars": 0,
        "error": None,
        "result_urls_sample": [],
    }
    if not rg.web_search_enabled:
        return "", meta

    provider = _resolve_provider(rg)
    meta["provider"] = provider
    queries = _queries_for_scenario(scenario, rg)
    meta["queries"] = list(queries)
    mr = max(1, min(15, int(rg.web_search_max_results)))
    mc = max(0, int(rg.web_search_max_chars))

    if provider == "brave":
        hits, err = _search_brave(queries, mr, _brave_api_key())
    else:
        hits, err = _search_duckduckgo(queries, mr)

    # 去重 URL，保留顺序
    seen_u: set[str] = set()
    deduped: list[dict[str, str]] = []
    for h in hits:
        u = (h.get("url") or "").strip()
        if u and u in seen_u:
            continue
        if u:
            seen_u.add(u)
        deduped.append(h)

    text = _format_hits(deduped, mc)
    meta["snippet_chars"] = len(text)
    meta["result_urls_sample"] = [h.get("url", "") for h in deduped[:8] if h.get("url")]
    meta["error"] = err
    meta["used"] = bool(text)
    if not text and err:
        meta["used"] = False
    return text, meta
