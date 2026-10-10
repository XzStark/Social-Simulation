from __future__ import annotations

from people.causal_identifiability import assess_edges_identifiability


def test_identifiability_warns_on_low_excitation() -> None:
    # constant source => low excitation warning
    hist = []
    ps = 0.50
    for t in range(1, 21):
        # target moves a bit; source is constant
        ps = ps + (0.002 if t % 2 == 0 else -0.001)
        hist.append({"tick": t, "rumor_level": 0.2, "policy_support": ps})
    edges = [
        {"id": "e1", "from_metric": "rumor_level", "to_metric": "policy_support", "lag_ticks": 1, "weight": 0.2}
    ]
    rep = assess_edges_identifiability(metrics_history=hist, edges=edges, window=16)
    assert rep["ok"] is True
    assert any("excitation不足" in w for w in rep["warnings"])
