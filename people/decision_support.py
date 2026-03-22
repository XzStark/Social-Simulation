"""
决策辅助：对指标时间序列做 **低成本粗检**（跳变、少见共变），输出通俗句。
不改变核心推演逻辑；默认开启，可在场景 YAML `decision_support.enabled: false` 关闭。
"""

from __future__ import annotations

from typing import Any

from people.attribution import MACRO_ATTR_KEYS
from people.config import Scenario
from people.state import WorldState

_DEFAULT_JUMP: dict[str, float] = {
    "sentiment": 0.35,
    "economy_index": 0.12,
    "policy_support": 0.09,
    "rumor_level": 0.14,
    "unrest": 0.12,
    "issuer_trust_proxy": 0.08,
    "supply_chain_stress": 0.14,
}

_METRIC_CN: dict[str, str] = {
    "sentiment": "情绪",
    "economy_index": "经济景气",
    "policy_support": "支持度",
    "rumor_level": "谣言",
    "unrest": "动荡",
    "issuer_trust_proxy": "信任",
    "supply_chain_stress": "供应链压力",
}


def _thr(scenario: Scenario, key: str) -> float:
    o = scenario.decision_support.jump_warn_threshold
    if key in o:
        return float(o[key])
    return _DEFAULT_JUMP.get(key, 0.12)


def build_tick_decision_hints(state: WorldState, scenario: Scenario) -> dict[str, Any]:
    """对比 `metrics_history` 倒数两行（本轮末 vs 上轮末）。"""
    if not scenario.decision_support.enabled:
        return {}
    mh = state.metrics_history
    if len(mh) < 2:
        return {}
    prev, cur = mh[-2], mh[-1]
    hints: list[str] = []

    for key in MACRO_ATTR_KEYS:
        if key not in prev or key not in cur:
            continue
        try:
            da = float(cur[key]) - float(prev[key])
        except (TypeError, ValueError):
            continue
        if abs(da) >= _thr(scenario, key):
            cn = _METRIC_CN.get(key, key)
            hints.append(
                f"本轮「{cn}」比上一轮变化约 {da:+.3f}，幅度偏大；若用于汇报请对照叙事与归因是否说得通。"
            )

    # 少见共变（启发式，非因果断言）
    try:
        dr = float(cur.get("rumor_level", 0)) - float(prev.get("rumor_level", 0))
        dp = float(cur.get("policy_support", 0)) - float(prev.get("policy_support", 0))
        du = float(cur.get("unrest", 0)) - float(prev.get("unrest", 0))
        if dr > 0.06 and dp > 0.05:
            hints.append(
                "本轮谣言与支持度同时明显上升，现实中可能出现但不常见；建议核对是否来自智能体偶然措辞或需加强「辟谣/沟通」叙事。"
            )
        if du > 0.07 and abs(dr) < 0.02:
            hints.append(
                "本轮动荡上升较明显，但谣言变化很小；可能来自人群聚合或执行摩擦，建议看归因里「汇总人群」与智能体环节。"
            )
    except (TypeError, ValueError):
        pass

    of = scenario.operational_finance
    if of.enabled:
        if state.cash_balance_million is not None and float(state.cash_balance_million) < 15.0:
            hints.append(
                f"账面现金已偏低（约 {float(state.cash_balance_million):.1f} 百万 proxy）；请关注运营支出、税费与债务本息是否在叙事上说得通。"
            )
        if state.debt_balance_million is not None and float(state.debt_balance_million) > 0:
            if state.cash_balance_million is not None and float(state.cash_balance_million) < float(
                state.debt_balance_million
            ) * 0.08:
                hints.append(
                    "现金相对债务本金偏紧，偿债与流动性压力在真实世界中会约束决策节奏；若曲线仍激进，建议核对智能体是否过度乐观。"
                )
        if state.fiscal_remaining_billion is not None and float(state.fiscal_remaining_billion) < 0.35:
            hints.append(
                f"可用财力池偏低（约 {float(state.fiscal_remaining_billion):.3f} 十亿 proxy）；大额支出与减税刺激等叙事需更谨慎。"
            )

    if not hints:
        return {}
    return {"本步提示": hints}


def build_decision_support_bundle(state: WorldState, scenario: Scenario) -> dict[str, Any]:
    """整局存档用：阅前说明 + 最后一轮提示 + 全程最陡单轮跳跃（每指标最多一条）。"""
    header = {
        "说明": "以下为规则层面的粗检，用于防「一眼不合理的曲线」被直接拿去当结论；不能代替业务、法务与线下数据。输出均为演练生成。",
        "建议": "重要结论请交叉：多随机种子（ensemble/stability）、对照真值（validate/calibrate）、阅读 explain 归因。",
    }
    if not scenario.decision_support.enabled:
        return {**header, "状态": "已在场景中关闭 decision_support.enabled"}

    tick_h = build_tick_decision_hints(state, scenario)
    mh = state.metrics_history
    worst: list[dict[str, Any]] = []
    if len(mh) >= 2:
        best_jump: dict[str, tuple[float, int]] = {}
        for i in range(1, len(mh)):
            a, b = mh[i - 1], mh[i]
            t = int(float(b.get("tick", i))) if b.get("tick") is not None else i
            for key in MACRO_ATTR_KEYS:
                if key not in a or key not in b:
                    continue
                try:
                    d = abs(float(b[key]) - float(a[key]))
                except (TypeError, ValueError):
                    continue
                prev_m = best_jump.get(key)
                if prev_m is None or d > prev_m[0]:
                    best_jump[key] = (d, t)
        for key, (mag, tt) in sorted(best_jump.items(), key=lambda x: -x[1][0])[:5]:
            if mag >= _thr(scenario, key) * 0.85:
                worst.append(
                    {
                        "指标": _METRIC_CN.get(key, key),
                        "最大单轮变化约": round(mag, 4),
                        "出现在约第几轮": tt,
                    }
                )

    out: dict[str, Any] = {**header}
    if tick_h:
        out["最后一轮粗检"] = tick_h
    if worst:
        out["全程最陡的几处跳跃"] = worst
    if not tick_h and not worst:
        out["粗检结果"] = "未发现超过默认阈值的单轮跳变（或历史太短）。"
    return out
