"""
可复现实验清单：场景/扩展栈/模式版本的规范摘要与内容哈希。
用于回答「结果漂移来自何处」；写入 run_start、pause 包、ensemble 导出。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pepole.config import PEPOLE_SCENARIO_SCHEMA_VERSION, Scenario
from pepole.reproducibility import environment_fingerprint


def _canonical_scenario_blob(scenario: Scenario) -> str:
    """稳定 JSON（排序键），用于 sha256。"""
    d = scenario.model_dump(mode="json", exclude_none=True)
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scenario_content_hash(scenario: Scenario) -> str:
    return hashlib.sha256(_canonical_scenario_blob(scenario).encode("utf-8")).hexdigest()


def build_experiment_manifest(
    scenario: Scenario,
    *,
    seed: int,
    primary_model_slot: str,
    fast_model_slot: str,
    pepole_package_version: str = "",
    include_environment: bool = True,
) -> dict[str, Any]:
    ch = scenario_content_hash(scenario)
    perf = scenario.performance
    out: dict[str, Any] = {
        "pepole_scenario_schema": PEPOLE_SCENARIO_SCHEMA_VERSION,
        "scenario_yaml_schema_version": scenario.scenario_schema_version,
        "scenario_name": scenario.name,
        "scenario_content_sha256": ch,
        "ticks": scenario.ticks,
        "exercise_type": scenario.exercise_type,
        "random_seed_yaml": scenario.random_seed,
        "run_seed": int(seed),
        "extensions_plugin_order": list(scenario.extensions.plugin_order),
        "extensions_enabled_flags": {
            "causal": scenario.extensions.causal.enabled,
            "ground_truth": scenario.extensions.ground_truth.enabled,
            "kpi": scenario.extensions.kpi.enabled,
            "diffusion": scenario.extensions.diffusion.enabled,
            "delay": scenario.extensions.delay.enabled,
            "resources": scenario.extensions.resources.enabled,
        },
        "performance_budget": {
            "enabled": perf.enabled,
            "max_tick_wall_seconds": perf.max_tick_wall_seconds,
            "skip_horizon_if_over_budget": perf.skip_horizon_if_over_budget,
            "max_primary_llm_calls_per_tick": perf.max_primary_llm_calls_per_tick,
        },
        "model_slots": {
            "primary": primary_model_slot,
            "fast": fast_model_slot,
        },
        "pepole_package_version": pepole_package_version or "unknown",
    }
    if include_environment:
        out["environment"] = environment_fingerprint()
    return out
