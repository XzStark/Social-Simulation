from __future__ import annotations

from people.agents.key_actor import _validate_and_normalize_actor_effect
from people.engine import _append_audit_chain
from people.engine import _update_layered_diffusion_proxy
from people.state import WorldState


def test_role_guard_clamps_and_records_violation() -> None:
    st = WorldState(tick=3)
    eff = {
        "public_statement": "ALL CLEAR, 无条件支持，零风险。",
        "sentiment_delta": 0.8,
        "rumor_delta": "bad",
        "policy_support_delta": -0.7,
        "unrest_delta": 0.2,
    }
    out = _validate_and_normalize_actor_effect(
        eff,
        actor_id="legal_center",
        role="法务合规中心",
        categories=["legal", "compliance"],
        state=st,
    )
    assert abs(float(out["sentiment_delta"])) <= 0.12
    assert abs(float(out["policy_support_delta"])) <= 0.12
    assert float(out["rumor_delta"]) == 0.0
    assert "不宜作绝对性结论" in str(out["public_statement"])
    assert st.role_guard_events, "expected violations to be recorded"


def test_audit_chain_hash_links() -> None:
    st = WorldState(tick=1, sentiment=0.2, policy_support=0.5)
    st.log("hello")
    _append_audit_chain(st, tick=1)
    assert len(st.audit_chain) == 1
    h1 = st.audit_chain[0]["hash"]
    st.tick = 2
    st.sentiment = 0.25
    st.log("next")
    _append_audit_chain(st, tick=2)
    assert len(st.audit_chain) == 2
    assert st.audit_chain[1]["prev_hash"] == h1
    assert st.audit_chain[1]["hash"] != h1


def test_layered_diffusion_proxy_updates() -> None:
    st = WorldState(tick=5, rumor_level=0.4)
    details = [
        {"kind": "media_unfavorable"},
        {"kind": "inter_agent_dialogue"},
        {"kind": "institution_pulse"},
    ]
    _update_layered_diffusion_proxy(st, details)
    dl = st.diffusion_layer_state
    assert "online_social" in dl and "mainstream_media" in dl
    assert dl["mainstream_media"] > 0
    assert dl["institutional_channel"] > 0
