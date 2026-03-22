from __future__ import annotations

from pepole.attribution import MACRO_ATTR_KEYS
from pepole.config import DecisionSupportSpec, RealismConfig, Scenario
from pepole.decision_support import build_decision_support_bundle, build_tick_decision_hints
from pepole.rules import apply_macro_inertia_blend, finalize_tick
from pepole.state import WorldState


def _row(tick: int, **over: float) -> dict:
    r: dict = {"tick": float(tick)}
    for k in MACRO_ATTR_KEYS:
        r[k] = float(over.get(k, 0.45 if k != "sentiment" else 0.0))
    return r


def test_macro_inertia_blend_moves_toward_previous() -> None:
    st = WorldState(tick=1, policy_support=0.9, rumor_level=0.1)
    finalize_tick(st)
    st.policy_support = 0.2
    rc = RealismConfig(macro_inertia_blend=0.5)
    d = apply_macro_inertia_blend(st, rc)
    assert d
    assert abs(st.policy_support - 0.55) < 1e-6


def test_decision_support_large_jump() -> None:
    scen = Scenario.model_validate(
        {
            "name": "t",
            "ticks": 3,
            "decision_support": DecisionSupportSpec(enabled=True),
        }
    )
    st = WorldState(tick=2)
    st.metrics_history = [
        _row(1, policy_support=0.5, rumor_level=0.1, unrest=0.1),
        _row(2, policy_support=0.7, rumor_level=0.1, unrest=0.1),
    ]
    h = build_tick_decision_hints(st, scen)
    assert "本步提示" in h
    bundle = build_decision_support_bundle(st, scen)
    assert "说明" in bundle
