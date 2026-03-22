from __future__ import annotations

from people.config import MarketCompetitorSpec, Scenario
from people.context import build_decision_context


def test_market_competitors_in_decision_context_for_product() -> None:
    s = Scenario(
        name="t",
        exercise_type="product",
        ticks=3,
        market_competitors=[
            MarketCompetitorSpec(
                id="a",
                name="竞品甲",
                brief="生态绑定强",
                estimated_market_share_proxy=0.4,
            ),
            MarketCompetitorSpec(
                id="b",
                name="竞品乙",
                brief="低价带",
                linked_key_actor_id="rival_pr",
            ),
        ],
    )
    d = build_decision_context(s)
    assert "market_competitors" in d
    assert len(d["market_competitors"]) == 2
    assert d["market_competitors"][0]["id"] == "a"
    assert d["market_competitors"][1]["linked_key_actor_id"] == "rival_pr"


def test_market_competitors_omitted_for_policy() -> None:
    s = Scenario(
        name="t",
        exercise_type="policy",
        ticks=3,
        market_competitors=[
            MarketCompetitorSpec(id="x", name="不应出现", brief=""),
        ],
    )
    d = build_decision_context(s)
    assert "market_competitors" not in d
