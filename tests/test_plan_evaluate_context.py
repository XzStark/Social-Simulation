from __future__ import annotations

import random

from pepole.config import Scenario
from pepole.context import build_decision_context, enrich_decision_context_for_plan_evaluate
from pepole.engine import _hydrate_finance, build_initial_state


def test_enrich_injects_plan_evaluate_live_context() -> None:
    s = Scenario.load("scenarios/default.yaml")
    rng = random.Random(1)
    st = build_initial_state(s, rng)
    _hydrate_finance(st, s)
    dctx = build_decision_context(s)
    enrich_decision_context_for_plan_evaluate(dctx, st, s)
    assert "plan_evaluate_live_context" in dctx
    live = dctx["plan_evaluate_live_context"]
    assert live["tick"] == st.tick
    assert "macro" in live
    assert "cohorts_snapshot" in live
    assert isinstance(live["cohorts_snapshot"], list)
