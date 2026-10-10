from __future__ import annotations

import random

from people.black_swan import apply_black_swan_event, build_black_swan_event
from people.black_swan import maybe_pick_internal_issue_preset
from people.state import WorldState


def test_build_black_swan_event_from_preset_scales_by_severity() -> None:
    ev = build_black_swan_event(preset_id="supply_chain_cutoff", severity=1.5)
    assert ev["preset_id"] == "supply_chain_cutoff"
    assert ev["title"]
    assert ev["metric_deltas"]["supply_chain_stress"] > 0.3


def test_apply_black_swan_event_changes_state_and_clamps() -> None:
    st = WorldState(policy_support=0.04, rumor_level=0.9, unrest=0.3, issuer_trust_proxy=0.95)
    ev = build_black_swan_event(
        preset_id=None,
        title="custom",
        detail="x",
        severity=1.0,
        metric_deltas={
            "policy_support": -0.3,
            "rumor_level": +0.3,
            "unrest": +0.2,
            "issuer_trust_proxy": -0.1,
        },
    )
    d = apply_black_swan_event(st, ev, source="test")
    assert st.policy_support == 0.0
    assert st.rumor_level == 1.0
    assert st.unrest > 0.3
    assert st.issuer_trust_proxy < 0.95
    assert "policy_support" in d
    assert any("黑天鹅注入" in x for x in st.narrative)


def test_internal_issue_sampling_with_probability() -> None:
    rng = random.Random(42)
    pid = maybe_pick_internal_issue_preset(
        exercise_type="policy",
        issuer_archetype="large_group",
        rng=rng,
        probability=1.0,
    )
    assert isinstance(pid, str) and pid
