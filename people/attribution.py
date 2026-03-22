"""
事件级指标溯源（v0）：把「哪一层、哪一组件」对宏观量造成的增量记入 WorldState.attribution_log。
完整归因链、与叙事/检查点联合查询见系统说明 §16.4、§17。
"""

from __future__ import annotations

from typing import Any

from people.state import WorldState

_ATTRIBUTION_MAX = 3000

MACRO_ATTR_KEYS: tuple[str, ...] = (
    "sentiment",
    "economy_index",
    "policy_support",
    "rumor_level",
    "unrest",
    "issuer_trust_proxy",
    "supply_chain_stress",
)


def macro_metrics_snapshot(state: WorldState) -> dict[str, float]:
    return {k: float(getattr(state, k)) for k in MACRO_ATTR_KEYS if hasattr(state, k)}


def extended_metrics_snapshot(state: WorldState) -> dict[str, float]:
    """宏观七项 + 可选财务字段（用于交流扣款、经营台账等溯源）。"""
    d = macro_metrics_snapshot(state)
    if state.cash_balance_million is not None:
        d["cash_balance_million"] = float(state.cash_balance_million)
    if state.fiscal_remaining_billion is not None:
        d["fiscal_remaining_billion"] = float(state.fiscal_remaining_billion)
    if state.debt_balance_million is not None:
        d["debt_balance_million"] = float(state.debt_balance_million)
    return d


def delta_snapshots(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in before:
        if k not in after:
            continue
        d = float(after[k]) - float(before[k])
        if abs(d) > 1e-14:
            out[k] = d
    return out


def record_attribution(
    state: WorldState,
    *,
    layer: str,
    component: str,
    tick: int,
    deltas: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """追加一条溯源记录；deltas 为对各宏观键的加性变化（若已知）。"""
    entry: dict[str, Any] = {
        "tick": int(tick),
        "layer": str(layer),
        "component": str(component),
        "deltas": dict(deltas or {}),
        "meta": dict(meta or {}),
    }
    state.attribution_log.append(entry)
    if len(state.attribution_log) > _ATTRIBUTION_MAX:
        state.attribution_log = state.attribution_log[-_ATTRIBUTION_MAX:]
