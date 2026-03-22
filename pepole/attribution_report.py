"""
把 attribution_log 整理成 **可汇报、可存档** 的结构，并生成「用人话」说明。
"""

from __future__ import annotations

from typing import Any

from pepole.attribution import MACRO_ATTR_KEYS
from pepole.state import WorldState

_LAYER_CN: dict[str, str] = {
    "llm": "智能体与模型",
    "rules": "规则与系统汇总",
    "extension": "扩展插件",
    "audit": "系统审计",
}

_METRIC_CN: dict[str, str] = {
    "sentiment": "情绪",
    "economy_index": "经济景气",
    "policy_support": "支持度（政策或对产品的好感）",
    "rumor_level": "谣言强度",
    "unrest": "动荡",
    "issuer_trust_proxy": "对主体/品牌的信任",
    "supply_chain_stress": "供应链压力",
    "cash_balance_million": "账面现金（百万）",
    "fiscal_remaining_billion": "可用财力池（十亿）",
    "debt_balance_million": "债务本金余额（百万）",
}


def _metric_cn(k: str) -> str:
    return _METRIC_CN.get(k, k)


def _fmt_delta(k: str, v: float) -> str:
    direction = "抬高" if v > 0 else "压低"
    return f"{_metric_cn(k)} {direction}了约 {abs(v):.4f}"


def steps_for_tick(state: WorldState, tick: int) -> list[dict[str, Any]]:
    """某一 tick 内按时间顺序排列的溯源步骤（与写入顺序一致）。"""
    out: list[dict[str, Any]] = []
    for e in state.attribution_log:
        if int(e.get("tick", -1)) != int(tick):
            continue
        layer = str(e.get("layer") or "")
        comp = str(e.get("component") or "")
        deltas = dict(e.get("deltas") or {})
        out.append(
            {
                "环节类型": _LAYER_CN.get(layer, layer),
                "具体来源": comp,
                "指标变化": {_metric_cn(k): round(float(v), 6) for k, v in deltas.items()},
                "附加上下文": dict(e.get("meta") or {}),
            }
        )
    return out


def explain_metric_at_tick(state: WorldState, metric: str, tick: int) -> dict[str, Any]:
    """回答：这一轮里，这个数主要是谁动的。"""
    lines: list[str] = []
    total = 0.0
    for e in state.attribution_log:
        if int(e.get("tick", -1)) != int(tick):
            continue
        deltas = e.get("deltas") or {}
        if metric not in deltas:
            continue
        dv = float(deltas[metric])
        total += dv
        layer = _LAYER_CN.get(str(e.get("layer") or ""), str(e.get("layer")))
        comp = str(e.get("component") or "")
        lines.append(f"{layer} · {comp}：{_fmt_delta(metric, dv)}")
    summary = (
        f"第 {tick} 轮里，**{_metric_cn(metric)}** 合起来变化约 {total:+.4f}。"
        if lines
        else f"第 {tick} 轮里，溯源记录里没有直接出现 **{_metric_cn(metric)}** 的分项（可能来自本轮尚未单独打点的步骤）。"
    )
    return {
        "指标": _metric_cn(metric),
        "轮次": tick,
        "一句话": summary,
        "分步说明": lines,
    }


def build_why_quantitative_chain(
    state: WorldState,
    *,
    metrics: list[str] | None = None,
    max_ticks: int = 200,
) -> dict[str, Any]:
    """
    按 tick、按指标：净变化 + 各来源对「本轮该指标在归因中出现过的绝对变动」的份额。
    用于回答 why（数值责任分解）；不含未写入 attribution_log 的通道。
    """
    mlist = list(metrics) if metrics else list(MACRO_ATTR_KEYS)
    ticks = sorted({int(e.get("tick", 0)) for e in state.attribution_log if e.get("tick") is not None})
    truncated = len(ticks) > max_ticks
    if truncated:
        ticks = ticks[-max_ticks:]
    by_tick: dict[str, Any] = {}
    for t in ticks:
        per_m: dict[str, Any] = {}
        for m in mlist:
            contributors: list[dict[str, Any]] = []
            net = 0.0
            for e in state.attribution_log:
                if int(e.get("tick", -1)) != t:
                    continue
                deltas = e.get("deltas") or {}
                if m not in deltas:
                    continue
                dv = float(deltas[m])
                net += dv
                contributors.append(
                    {
                        "环节": _LAYER_CN.get(str(e.get("layer") or ""), str(e.get("layer"))),
                        "来源": str(e.get("component") or ""),
                        "delta": round(dv, 6),
                    }
                )
            if not contributors:
                continue
            abs_sum = sum(abs(float(c["delta"])) for c in contributors) or 1.0
            for c in contributors:
                c["占本轮该指标绝对变动比例"] = round(abs(float(c["delta"])) / abs_sum, 4)
            contributors.sort(key=lambda x: -abs(float(x["delta"])))
            per_m[_metric_cn(m)] = {
                "净变化": round(net, 6),
                "分责任序": contributors,
            }
        if per_m:
            by_tick[str(t)] = per_m
    return {
        "按轮次_按指标": by_tick,
        "指标集合": [_metric_cn(m) for m in mlist],
        "是否截断仅保留最后若干 tick": truncated,
        "说明": "份额按本轮归因记录中出现的各笔 delta 绝对值归一化，非模型内部真实梯度；"
        "未写入 attribution_log 的通道不会出现在此链。",
    }


def build_plain_report(state: WorldState) -> dict[str, Any]:
    """整局演练的「结果怎么来的」总览（通俗）。"""
    ticks = sorted({int(e.get("tick", 0)) for e in state.attribution_log if e.get("tick") is not None})
    if not ticks:
        return {
            "标题": "本局暂无溯源记录",
            "说明": "可能是旧存档或未跑完带归因的版本。请用当前版本重新跑并带上 --dump-full-state。",
        }
    by_tick: dict[str, Any] = {}
    for t in ticks:
        steps = steps_for_tick(state, t)
        if steps:
            by_tick[str(t)] = steps
    # 终局宏观值
    last_row = state.metrics_history[-1] if state.metrics_history else {}
    audit_hints: list[str] = []
    for e in state.attribution_log:
        if str(e.get("layer")) != "audit":
            continue
        meta = e.get("meta") or {}
        if meta.get("issue"):
            audit_hints.append(str(meta.get("issue")))

    closing = (
        "最后一轮结束时，主要公开指标大致是："
        + "；".join(
            f"{_metric_cn(k)}={last_row.get(k, '—')}"
            for k in MACRO_ATTR_KEYS
            if k in last_row
        )
        if last_row
        else ""
    )
    return {
        "标题": "本局指标变化 — 按轮次拆解（演练生成）",
        "有多少轮有记录": len(by_tick),
        "按轮次明细": by_tick,
        "完整归因链_why量化": build_why_quantitative_chain(state),
        "系统提醒": audit_hints or None,
        "收尾快照": closing or None,
    }


def narrative_for_final_metrics(state: WorldState) -> str:
    """两三句人话，适合贴进汇报附录。"""
    rep = build_plain_report(state)
    ticks_n = rep.get("有多少轮有记录", 0)
    audit = rep.get("系统提醒")
    parts = [
        f"本局共 {ticks_n} 个时间步留下了「谁改了哪个数」的记录。",
        (rep.get("收尾快照") or "").strip(),
    ]
    if audit:
        parts.append("系统审计提示：" + "；".join(audit))
    return "\n".join(p for p in parts if p)
