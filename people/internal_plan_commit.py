"""
指挥台「仅评估」后确认执行：写入决策层摘要、沙盘级宏观/资金边际、约定时间节点（tick）并在到期时输出叙事。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from people.config import Scenario
from people.finance_ledger import effective_operating_cost_million_per_tick
from people.state import WorldState


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_commitment_tick_offset(digest: dict[str, Any], desk: dict[str, Any]) -> int:
    """从转化/桌面时间线文案中粗估「约定输出」落在未来第几个 tick（至少 1，上限 10）。"""
    blob = " ".join(
        str(x or "")
        for x in (
            digest.get("estimated_timeline_cn"),
            digest.get("feasibility_notes"),
            desk.get("estimated_timeline_desk_cn"),
            desk.get("desk_verdict_cn"),
            desk.get("recommended_internal_next_steps"),
        )
    )
    n = 2
    m = re.search(r"(\d+)\s*周", blob)
    if m:
        n = max(1, min(10, int(m.group(1))))
    else:
        m = re.search(r"(\d+)\s*[个個]?\s*月", blob)
        if m:
            n = max(1, min(10, int(m.group(1)) * 2))
        else:
            m = re.search(r"(\d+)\s*天", blob)
            if m:
                n = max(1, min(10, max(1, (int(m.group(1)) + 6) // 7)))
    return n


def _macro_deltas_from_feasibility(feasibility: str | None) -> tuple[float, float, float, float]:
    """sentiment, policy_support, issuer_trust, unrest 的边际（沙盘级、克制）。"""
    f = (feasibility or "").strip().lower()
    raw = feasibility or ""
    if "not_feasible" in f or "不可行" in raw:
        return -0.018, -0.022, -0.014, 0.012
    if "conditions" in f or "条件" in raw or "with_conditions" in f:
        return 0.006, 0.01, 0.005, 0.004
    if "feasible" in f or "可行" in raw:
        return 0.014, 0.018, 0.012, -0.006
    return 0.004, 0.006, 0.004, 0.0


def _combine_expected_cn(digest: dict[str, Any], desk: dict[str, Any]) -> str:
    parts: list[str] = []
    et = str(digest.get("estimated_timeline_cn") or "").strip()
    if et:
        parts.append("转化侧周期估计：" + et[:500])
    etd = str(desk.get("estimated_timeline_desk_cn") or "").strip()
    if etd:
        parts.append("桌面侧周期与里程碑：" + etd[:500])
    nxt = desk.get("recommended_internal_next_steps")
    if isinstance(nxt, list) and nxt:
        parts.append("建议动作：" + "；".join(str(x) for x in nxt[:8] if str(x).strip())[:500])
    return "\n".join(parts) if parts else "（未解析到显式时间线；按默认 tick 节点提醒）"


def apply_confirmed_internal_plan(
    state: WorldState,
    scenario: Scenario,
    *,
    plain: str,
    digest: dict[str, Any],
    desk: dict[str, Any],
) -> dict[str, Any]:
    """
    将「已确认」的内部草案写入状态：决策层 directive、叙事、记忆入口、宏观/资金边际、约定节点。
    返回供 API 展示的摘要字段。
    """
    from people.engine import _format_decision_layer_active

    plain = plain.strip()
    prof = str(digest.get("professional_execution_plan") or "").strip()
    chosen: dict[str, Any] = {
        "understood_intent": (digest.get("understood_intent") or "")[:900],
        "desk_verdict_cn": (desk.get("desk_verdict_cn") or desk.get("desk_verdict") or "")[:900],
        "estimated_timeline_cn": (digest.get("estimated_timeline_cn") or "")[:800],
        "estimated_timeline_desk_cn": (desk.get("estimated_timeline_desk_cn") or "")[:800],
        "professional_execution_plan_excerpt": (prof[:2000] if prof else ""),
    }
    state.decision_layer_active_summary = _format_decision_layer_active(chosen)
    state.decision_layer_history.append(
        {
            **chosen,
            "id": str(uuid.uuid4()),
            "status": "internal_eval_confirmed",
            "plain_full": plain[:4000],
            "confirmed_at_tick": int(state.tick),
        }
    )
    state.decision_layer_history = state.decision_layer_history[-48:]

    state.log(
        "[指挥台·已确认执行·内部草案] "
        + (plain[:260] + ("…" if len(plain) > 260 else ""))
    )
    state.push_detail(
        "internal_plan_confirmed",
        "已确认执行（内部评估草案）",
        (state.decision_layer_active_summary or "")[:360],
        {"digest_keys": list(digest.keys()), "desk_keys": list(desk.keys())},
    )

    state.company_memory_events.append(
        {
            "tick": state.tick,
            "role": "指挥台",
            "statement_excerpt": f"已确认执行内部草案（沙盘）：{plain[:200]}",
            "importance": 1.0,
        }
    )

    ds, dps, dit, dur = _macro_deltas_from_feasibility(
        str(digest.get("feasibility") or "") if digest else None
    )
    state.sentiment = _clamp(float(state.sentiment) + ds, -1.0, 1.0)
    state.policy_support = _clamp(float(state.policy_support) + dps, 0.0, 1.0)
    state.issuer_trust_proxy = _clamp(float(state.issuer_trust_proxy) + dit, 0.0, 1.0)
    state.unrest = _clamp(float(state.unrest) + dur, 0.0, 1.0)

    of = scenario.operational_finance
    if of.enabled and scenario.exercise_type == "product" and state.cash_balance_million is not None:
        eff = float(effective_operating_cost_million_per_tick(scenario))
        spend = min(float(state.cash_balance_million) * 0.1, max(eff * 1.5, 2.0))
        state.cash_balance_million = max(0.0, float(state.cash_balance_million) - spend)
    if of.enabled and scenario.exercise_type == "policy" and state.fiscal_remaining_billion is not None:
        sp = float(of.policy_spend_per_tick_billion or 0.0)
        if sp > 0:
            state.fiscal_remaining_billion = max(
                0.0, float(state.fiscal_remaining_billion) - min(sp * 0.8, 0.05)
            )

    off = estimate_commitment_tick_offset(digest, desk)
    expected = _combine_expected_cn(digest, desk)
    cid = str(uuid.uuid4())
    state.internal_draft_commitments.append(
        {
            "id": cid,
            "due_tick": int(state.tick) + off,
            "created_tick": int(state.tick),
            "expected_outputs_cn": expected,
            "plain_excerpt": plain[:320] + ("…" if len(plain) > 320 else ""),
            "fired": False,
        }
    )

    return {
        "decision_layer_excerpt": (state.decision_layer_active_summary or "")[:1200],
        "macro_deltas": {
            "sentiment": ds,
            "policy_support": dps,
            "issuer_trust_proxy": dit,
            "unrest": dur,
        },
        "commitment": {
            "id": cid,
            "due_tick": int(state.tick) + off,
            "due_in_ticks": off,
            "expected_outputs_cn": expected[:1200],
        },
    }


def fire_due_internal_commitments(state: WorldState, scenario: Scenario) -> list[str]:
    """在当前 tick 触发已到期的约定节点；返回供 SSE 展示的短句列表。"""
    out: list[str] = []
    for c in state.internal_draft_commitments:
        if c.get("fired"):
            continue
        due = int(c.get("due_tick", 999999))
        if due > int(state.tick):
            continue
        c["fired"] = True
        body = str(c.get("expected_outputs_cn") or "（无详情）")[:900]
        state.log(f"[内部草案·约定节点·t{state.tick}] {body[:280]}")
        state.push_detail(
            "internal_plan_milestone",
            "内部草案·时间节点到达（演练提醒）",
            body[:500],
            {"commitment_id": c.get("id")},
        )
        state.company_memory_events.append(
            {
                "tick": state.tick,
                "role": "战情室",
                "statement_excerpt": f"约定输出节点到达：{body[:200]}",
                "importance": 0.85,
            }
        )
        state.sentiment = _clamp(float(state.sentiment) + 0.005, -1.0, 1.0)
        out.append(f"t{state.tick} 应检视：{body[:420]}")
    return out
