from __future__ import annotations

import pytest

from pepole.config import Scenario
from pepole.engine import RunConfig
from pepole.providers.dry_run import DryRunClient
from pepole.regional_grounding import apply_regional_grounding, prebake_regional_grounding_for_ensemble


def test_apply_regional_grounding_prepends_digest() -> None:
    s = Scenario.load("scenarios/default.yaml")
    s = s.model_copy(
        update={
            "regional_grounding": s.regional_grounding.model_copy(
                update={
                    "enabled": True,
                    "region_label": "测试市",
                    "mode": "llm_only",
                }
            ),
            "player_brief": "试点一项新规",
        }
    )
    cfg = RunConfig(
        primary_client=DryRunClient(),
        fast_client=DryRunClient(),
        primary_model_slot="dry",
        fast_model_slot="dry",
    )
    s2, trace, _ws = apply_regional_grounding(s, cfg)
    assert trace and len(trace) == 3
    assert "地区情境" in s2.player_brief
    assert "指挥台" in s2.player_brief
    assert "试点一项新规" in s2.player_brief


def test_web_search_injects_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    from pepole import regional_grounding as rg_mod

    def _fake_fetch(_scenario: Scenario, _rg: object) -> tuple[str, dict]:
        return "· 标题\n  https://example.com\n  摘要", {"used": True, "provider": "test"}

    monkeypatch.setattr(rg_mod, "fetch_web_grounding_snippets", _fake_fetch)
    s = Scenario.load("scenarios/default.yaml")
    s = s.model_copy(
        update={
            "regional_grounding": s.regional_grounding.model_copy(
                update={
                    "enabled": True,
                    "region_label": "测试市",
                    "mode": "llm_only",
                    "web_search_enabled": True,
                }
            ),
            "player_brief": "试点",
        }
    )
    cfg = RunConfig(
        primary_client=DryRunClient(),
        fast_client=DryRunClient(),
        primary_model_slot="dry",
        fast_model_slot="dry",
    )
    s2, trace, ws = apply_regional_grounding(s, cfg)
    assert ws.get("used") is True
    assert "开放网页检索摘录" in s2.player_brief
    assert trace[0].get("web_search_used") is True


def test_prebake_disables_flag() -> None:
    s = Scenario.load("scenarios/default.yaml")
    s = s.model_copy(
        update={
            "regional_grounding": s.regional_grounding.model_copy(
                update={"enabled": True, "region_label": "X", "mode": "llm_only"}
            ),
        }
    )
    cfg = RunConfig(
        primary_client=DryRunClient(),
        fast_client=DryRunClient(),
        primary_model_slot="dry",
        fast_model_slot="dry",
    )
    s2 = prebake_regional_grounding_for_ensemble(s, cfg)
    assert s2.regional_grounding.enabled is False
    assert "地区情境" in s2.player_brief
