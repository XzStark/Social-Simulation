"""专业扩展：因果优先级/边、实验清单、性能预算字段存在性。"""

from __future__ import annotations

import pytest

from people.config import PEOPLE_SCENARIO_SCHEMA_VERSION, Scenario
from people.engine import RunConfig, _experiment_manifest_for_run, build_initial_state
from people.extension_stack import init_extensions_state, run_extension_plugins
from people.experiment_manifest import scenario_content_hash
from people.rules import finalize_tick
from people.state import CohortState, WorldState
from people.providers.registry import get_client


def _minimal_scenario(**kwargs: object) -> Scenario:
    base = {
        "name": "t",
        "ticks": 3,
        "random_seed": 1,
        "cohorts": [{"id": "a", "weight": 1.0}],
    }
    base.update(kwargs)
    return Scenario.model_validate(base)


def test_experiment_manifest_has_schema_and_performance() -> None:
    s = _minimal_scenario()
    cfg = RunConfig(
        primary_client=get_client("openai:gpt-4o", allow_dry=True),
        fast_client=get_client("openai:gpt-4o-mini", allow_dry=True),
        primary_model_slot="openai:gpt-4o",
        fast_model_slot="openai:gpt-4o-mini",
    )
    m = _experiment_manifest_for_run(s, seed=42, cfg=cfg)
    assert m["people_scenario_schema"] == PEOPLE_SCENARIO_SCHEMA_VERSION
    assert m["scenario_yaml_schema_version"] == s.scenario_schema_version
    assert "performance_budget" in m
    assert m["model_slots"]["primary"] == "openai:gpt-4o"
    h1 = scenario_content_hash(s)
    h2 = scenario_content_hash(s.model_copy(update={"name": "t2"}))
    assert h1 != h2


def test_causal_rule_priority_and_edges_use_history() -> None:
    s = _minimal_scenario(
        extensions={
            "causal": {
                "enabled": True,
                "rules": [
                    {
                        "id": "late",
                        "priority": 20,
                        "merge_mode": "last_wins",
                        "when_all": [],
                        "effects": [
                            {"metric": "rumor_level", "add": 0.9},
                            {"metric": "rumor_level", "add": 0.1},
                        ],
                    },
                    {
                        "id": "early",
                        "priority": 10,
                        "merge_mode": "additive",
                        "when_all": [],
                        "effects": [{"metric": "rumor_level", "add": 0.05}],
                    },
                ],
                "edges": [
                    {
                        "from_metric": "rumor_level",
                        "to_metric": "unrest",
                        "lag_ticks": 1,
                        "weight": 0.2,
                    }
                ],
            }
        }
    )
    import random

    rng = random.Random(0)
    state = build_initial_state(s, rng)
    state.rumor_level = 0.5
    state.unrest = 0.1
    finalize_tick(state)
    # 历史末行 rumor_level=0.5 → 边 lag=1 贡献 unrest += 0.1
    state.tick = 1
    run_extension_plugins("pre_finalize", state, s, rng, ctx={})
    # priority 先 early +0.05 → 0.55，再 late last_wins 仅末条 +0.1 → 0.65
    assert abs(state.rumor_level - 0.65) < 1e-5, state.rumor_level
    assert abs(state.unrest - 0.2) < 1e-5, state.unrest


def test_diffusion_sir_per_cohort_weights_global_i() -> None:
    s = _minimal_scenario(
        cohorts=[
            {"id": "x", "weight": 1.0},
            {"id": "y", "weight": 3.0},
        ],
        extensions={
            "diffusion": {
                "enabled": True,
                "mode": "sir_per_cohort",
                "seed_informed": 0.1,
                "beta": 0.01,
                "gamma": 0.05,
                "rumor_coupling": 0.0,
            }
        },
    )
    import random

    rng = random.Random(1)
    state = WorldState(
        tick=1,
        cohorts=[
            CohortState(id="x", weight=1.0, attitude=0.0, activation=0.3),
            CohortState(id="y", weight=3.0, attitude=0.0, activation=0.3),
        ],
    )
    init_extensions_state(state, s)
    run_extension_plugins("pre_finalize", state, s, rng, ctx={})
    assert "x" in state.diffusion_cohort_sir
    assert abs(state.diffusion_i - sum(state.diffusion_cohort_sir[c]["i"] * w for c, w in [("x", 1.0), ("y", 3.0)]) / 4.0) < 1e-5
