from __future__ import annotations

from people.engine import _append_timeline_uncertainty
from people.state import WorldState


def test_variance_decomposition_policy_support_sum_to_one() -> None:
    # tick_end = 6, window = 6 => tick_start = 1
    st = WorldState(tick=6)
    # policy_support series across ticks 1..6
    x = [0.50, 0.55, 0.52, 0.57, 0.54, 0.60]
    # Build metrics_history tail of length 6
    st.metrics_history = []
    for i, pv in enumerate(x):
        st.metrics_history.append(
            {
                "sentiment": 0.0,
                "economy_index": 0.5,
                "policy_support": float(pv),
                "rumor_level": 0.1,
                "unrest": 0.2,
                "issuer_trust_proxy": 0.4,
                "supply_chain_stress": 0.2,
            }
        )

    # Attribution deltas: only one component contributes to policy_support deltas per tick.
    st.attribution_log = []
    component = {"layer": "rules", "component": "test_component", "deltas": {}}
    # delta at tick 2..6 equals x[tick]-x[tick-1]; tick 1 contribution is 0 by default.
    for tick in range(2, 7):
        idx = tick - 1  # tick 2 -> idx 1, tick 6 -> idx 5
        prev = x[idx - 1]
        cur = x[idx]
        delta = cur - prev
        st.attribution_log.append(
            {
                "tick": tick,
                "layer": component["layer"],
                "component": component["component"],
                "deltas": {"policy_support": float(delta)},
                "meta": {},
            }
        )

    _append_timeline_uncertainty(st, window=6)
    row = st.timeline_uncertainty[-1]
    vd = row.get("variance_decomposition") or {}
    ps = vd.get("policy_support") or {}
    comps = ps.get("components") or []
    assert comps, "expected decomposition components"
    share_sum = sum(float(c.get("share") or 0.0) for c in comps)
    assert abs(share_sum - 1.0) < 1e-3
