from __future__ import annotations

from people.engine import _append_audit_chain
from people.state import WorldState
from people.webapp import _verify_audit_chain


def test_verify_audit_chain_prev_hash_ok() -> None:
    st = WorldState(tick=1, sentiment=0.1, policy_support=0.3)
    st.log("a")
    _append_audit_chain(st, tick=1)
    st.tick = 2
    st.sentiment = 0.15
    st.log("b")
    _append_audit_chain(st, tick=2)

    out = _verify_audit_chain(st.audit_chain)
    assert out["ok"] is True
    assert out["errors"] == []
