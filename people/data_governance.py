from __future__ import annotations

"""
数据治理（v0，最小可落地）：

- 缺失值处理：解析常见缺失标记，提供简单填充策略（forward-fill / mean / zero / drop）
- 采样偏差修正：提供“权重重标定”钩子（post-strat / raking 的占位接口）
- 脱敏与版权：提供可审计的文本脱敏钩子（不默认启用，避免破坏现有演练语义）

说明：这里的目标是让“规范”有具体可执行入口，而不是在文档里停留口号。
"""

import re
import statistics
from typing import Any, Literal


MISSING_MARKERS = frozenset({"", "na", "n/a", "null", "none", "nan", "NaN", "-", "--"})


def parse_float_maybe(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.lower() in MISSING_MARKERS:
            return None
        try:
            v = float(s)
        except Exception:
            return None
        if v != v:
            return None
        return v
    return None


ImputeStrategy = Literal["none", "ffill", "mean", "zero", "drop_row"]


def impute_missing_rows(
    rows: list[dict[str, Any]],
    *,
    keys: list[str],
    strategy: ImputeStrategy = "ffill",
) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    last: dict[str, float] = {}
    # pre-compute mean if needed
    means: dict[str, float] = {}
    if strategy == "mean":
        for k in keys:
            vals = [parse_float_maybe(r.get(k)) for r in rows]
            vals2 = [v for v in vals if v is not None]
            means[k] = float(statistics.mean(vals2)) if vals2 else 0.0
    for r in rows:
        rr = dict(r)
        drop = False
        for k in keys:
            v = parse_float_maybe(rr.get(k))
            if v is not None:
                rr[k] = float(v)
                last[k] = float(v)
                continue
            if strategy == "none":
                continue
            if strategy == "drop_row":
                drop = True
                break
            if strategy == "zero":
                rr[k] = 0.0
                continue
            if strategy == "mean":
                rr[k] = float(means.get(k, 0.0))
                continue
            if strategy == "ffill":
                if k in last:
                    rr[k] = float(last[k])
                else:
                    rr[k] = 0.0
        if not drop:
            out.append(rr)
    return out


_RE_PII = re.compile(r"(\b1[3-9]\d{9}\b)|(\b\d{15,18}[\dxX]\b)")


def redact_text(text: str) -> str:
    """
    非强制：仅提供最小脱敏钩子（手机号/疑似身份证），调用方按角色/场景决定是否启用。
    """
    return _RE_PII.sub("[REDACTED]", text or "")


def sampling_bias_reweight_stub(weights: list[float], *, target_sum: float = 1.0) -> list[float]:
    """
    采样偏差修正（占位）：把权重重标定到 target_sum。
    真正的 post-strat/raking 需要外部人群边际与约束表，本项目先提供入口。
    """
    w = [max(0.0, float(x)) for x in (weights or [])]
    s = sum(w)
    if s <= 1e-12:
        return [0.0 for _ in w]
    scale = float(target_sum) / s
    return [float(x * scale) for x in w]
