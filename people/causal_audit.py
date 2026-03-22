"""
因果治理：在「智能体仍能改数」的前提下，对指定指标做 **系统级提醒** ——
若本 tick 智能体推动幅度大，但因果层未记录对该指标的修正，则写入审计溯源并打日志。
"""

from __future__ import annotations

from people.attribution import MACRO_ATTR_KEYS, record_attribution
from people.config import Scenario
from people.state import WorldState


def _llm_key_actor_deltas_for_tick(state: WorldState, tick: int) -> dict[str, float]:
    acc: dict[str, float] = {}
    for e in state.attribution_log:
        if int(e.get("tick", -1)) != int(tick):
            continue
        if str(e.get("layer") or "") != "llm":
            continue
        comp = str(e.get("component") or "")
        if not comp.startswith("key_actor:"):
            continue
        for m, v in (e.get("deltas") or {}).items():
            acc[str(m)] = acc.get(str(m), 0.0) + float(v)
    return acc


def run_causal_governance_audit(state: WorldState, scenario: Scenario) -> None:
    cl = scenario.extensions.causal
    if not cl.enabled or cl.governance_mode != "warn_llm_without_causal":
        return
    t = int(state.tick)
    watch = list(cl.governance_metrics) if cl.governance_metrics else list(MACRO_ATTR_KEYS)
    touched = set(state.causal_metrics_this_tick)
    llm_d = _llm_key_actor_deltas_for_tick(state, t)
    thr = float(cl.governance_min_delta)
    flagged: list[str] = []
    for m in watch:
        if m not in llm_d:
            continue
        if abs(float(llm_d[m])) < thr:
            continue
        if m in touched:
            continue
        flagged.append(m)
    if not flagged:
        return
    record_attribution(
        state,
        layer="audit",
        component="causal_governance",
        tick=t,
        deltas={},
        meta={
            "issue": "智能体推动了指标，但本 tick 因果层未触及同名指标（请检查规则/边或关闭治理）",
            "metrics": flagged,
            "governance_mode": cl.governance_mode,
        },
    )
    state.log(
        "[因果治理] 下列指标本轮被智能体明显推动，但因果插件未写入修正："
        + ", ".join(flagged)
        + "（详见 attribution 中 layer=audit）"
    )
