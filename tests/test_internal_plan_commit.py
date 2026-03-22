from __future__ import annotations

import random

from people.config import OperationalFinance, Scenario
from people.engine import _hydrate_finance, build_initial_state
from people.internal_plan_commit import (
    apply_confirmed_internal_plan,
    estimate_commitment_tick_offset,
    fire_due_internal_commitments,
)


def test_estimate_commitment_prefers_weeks_in_text() -> None:
    assert estimate_commitment_tick_offset({"estimated_timeline_cn": "约 3 周内闭环"}, {}) == 3


def test_apply_confirmed_sets_decision_layer_and_commitment() -> None:
    s = Scenario(
        name="t",
        exercise_type="product",
        ticks=8,
        operational_finance=OperationalFinance(enabled=True, cash_balance_million=100.0),
    )
    st = build_initial_state(s, random.Random(0))
    _hydrate_finance(st, s)
    cash_before = float(st.cash_balance_million or 0)
    apply_confirmed_internal_plan(
        st,
        s,
        plain="加急公关与渠道补贴",
        digest={
            "feasibility": "feasible_with_conditions",
            "understood_intent": "稳住渠道",
            "estimated_timeline_cn": "2 周",
            "professional_execution_plan": "x" * 100,
        },
        desk={"desk_verdict_cn": "可做", "estimated_timeline_desk_cn": "2 周首轮"},
    )
    assert st.decision_layer_active_summary
    assert len(st.internal_draft_commitments) == 1
    c0 = st.internal_draft_commitments[0]
    assert c0["due_tick"] == st.tick + 2
    assert float(st.cash_balance_million or 0) < cash_before


def test_fire_due_emits_once() -> None:
    s = Scenario(name="t", exercise_type="product", ticks=5)
    st = build_initial_state(s, random.Random(0))
    st.tick = 2
    st.internal_draft_commitments = [
        {
            "id": "x",
            "due_tick": 2,
            "created_tick": 0,
            "expected_outputs_cn": "复盘投放 ROI",
            "fired": False,
        }
    ]
    lines = fire_due_internal_commitments(st, s)
    assert lines
    assert st.internal_draft_commitments[0]["fired"] is True
    assert fire_due_internal_commitments(st, s) == []
