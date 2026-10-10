from __future__ import annotations

"""
动态可辨识性检验（v0，最小可落地）：

- 目的：在“可版本化因果骨架/边”存在时，给出运行期可审计的可辨识性提示：
  - excitation 不足（source 变化太小）
  - target 响应不足（delta 太小）
  - 多父节点共线性/不可区分（简单相关/条件数 proxy）

- 非目标：严格 SCM identifiability / do-calculus 证明；这里是工程化早期告警与证据摘要。
"""

import math
import statistics
from typing import Any


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    vx = statistics.pvariance(x)
    vy = statistics.pvariance(y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = statistics.mean([(x[i] - mx) * (y[i] - my) for i in range(len(x))])
    return float(cov / math.sqrt(vx * vy))


def assess_edges_identifiability(
    *,
    metrics_history: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    window: int = 16,
) -> dict[str, Any]:
    """
    输入为纯 dict 列表（便于从 Scenario/State dump 直接调用）。
    返回：warnings + per_edge 证据摘要（相关/方差/样本数）。
    """
    hist = [r for r in (metrics_history or []) if isinstance(r, dict)]
    if len(hist) < 6:
        return {"ok": True, "warnings": [], "per_edge": [], "window": int(window), "n": len(hist)}

    w = int(max(6, window))
    tail = hist[-w:]
    per_edge: list[dict[str, Any]] = []
    warnings: list[str] = []

    for e in edges or []:
        fm = str(e.get("from_metric") or "")
        tm = str(e.get("to_metric") or "")
        lag = int(e.get("lag_ticks") or 1)
        eid = str(e.get("id") or f"{fm}->{tm}@{lag}")
        weight = float(e.get("weight") or 0.0)
        if not fm or not tm or lag < 1:
            continue
        if len(tail) <= lag + 2:
            continue

        src = []
        tgt_delta = []
        for i in range(lag, len(tail)):
            prev = tail[i - 1].get(tm)
            cur = tail[i].get(tm)
            srow = tail[i - lag]
            s = srow.get(fm) if isinstance(srow, dict) else None
            if not isinstance(prev, (int, float)) or not isinstance(cur, (int, float)):
                continue
            if not isinstance(s, (int, float)):
                continue
            src.append(float(s))
            tgt_delta.append(float(cur) - float(prev))

        n = len(src)
        if n < 6:
            continue
        var_src = float(statistics.pvariance(src)) if n >= 2 else 0.0
        var_tgt = float(statistics.pvariance(tgt_delta)) if n >= 2 else 0.0
        corr = _pearson(src, tgt_delta)
        per_edge.append(
            {
                "id": eid,
                "from_metric": fm,
                "to_metric": tm,
                "lag_ticks": lag,
                "weight": weight,
                "n": n,
                "var_source": var_src,
                "var_target_delta": var_tgt,
                "corr_source_to_target_delta": corr,
            }
        )

        if var_src < 1e-4:
            warnings.append(f"edge {eid}: source {fm} 变化太小（excitation不足），难以辨识/校准")
        if var_tgt < 1e-5:
            warnings.append(f"edge {eid}: target {tm} 的增量几乎不动，边效应可能被淹没或被钳制")
        if abs(corr) < 0.06 and abs(weight) > 1e-9:
            warnings.append(f"edge {eid}: corr({fm}, Δ{tm})≈{corr:.2f}，证据弱（可能被共线/噪声覆盖）")

    # 多父节点不可区分（同一 target 下 source 之间高度相关）
    by_target: dict[str, list[dict[str, Any]]] = {}
    for r in per_edge:
        by_target.setdefault(str(r["to_metric"]), []).append(r)
    for tm, rows in by_target.items():
        if len(rows) < 2:
            continue
        # 取同 lag 的源序列做两两相关（简化）
        sources = [str(r["from_metric"]) for r in rows]
        # 构造窗口内各 source 的序列（lag 使用各自的 lag）
        series: dict[str, list[float]] = {}
        for r in rows:
            fm = str(r["from_metric"])
            lag = int(r["lag_ticks"])
            arr = []
            for i in range(lag, len(tail)):
                srow = tail[i - lag]
                v = srow.get(fm) if isinstance(srow, dict) else None
                if isinstance(v, (int, float)):
                    arr.append(float(v))
            if len(arr) >= 6:
                series[fm] = arr[-min(len(arr), 24) :]
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                a = sources[i]
                b = sources[j]
                xa = series.get(a)
                xb = series.get(b)
                if not xa or not xb:
                    continue
                m = min(len(xa), len(xb))
                if m < 6:
                    continue
                c = _pearson(xa[-m:], xb[-m:])
                if abs(c) > 0.92:
                    warnings.append(f"target {tm}: parents {a} 与 {b} 高相关(|r|≈{c:.2f})，存在共线/不可区分风险")

    return {
        "ok": True,
        "window": w,
        "n": len(hist),
        "warnings": warnings[:18],
        "per_edge": per_edge[:60],
    }
