from __future__ import annotations

from pepole.attribution_report import build_plain_report, explain_metric_at_tick
from pepole.attribution import record_attribution
from pepole.state import WorldState
from pepole.validation_tools import average_mae, compare_tick_series


def test_average_mae() -> None:
    r = compare_tick_series(
        [{"tick": 1, "policy_support": 0.5}, {"tick": 2, "policy_support": 0.6}],
        [{"tick": 1, "policy_support": 0.52}, {"tick": 2, "policy_support": 0.58}],
    )
    m = average_mae(r)
    assert m is not None
    assert m < 0.05


def test_plain_report_roundtrip() -> None:
    st = WorldState(tick=2, rumor_level=0.1, policy_support=0.5)
    record_attribution(
        st,
        layer="llm",
        component="key_actor:media_a",
        tick=1,
        deltas={"rumor_level": 0.05},
        meta={"role": "媒体"},
    )
    record_attribution(
        st,
        layer="rules",
        component="aggregate_from_cohorts",
        tick=2,
        deltas={"sentiment": 0.02},
        meta={},
    )
    st.metrics_history = [{"tick": 1, "policy_support": 0.5, "rumor_level": 0.15}]
    rep = build_plain_report(st)
    assert "按轮次明细" in rep
    exp = explain_metric_at_tick(st, "rumor_level", 1)
    assert "分步说明" in exp
    assert exp["分步说明"]
