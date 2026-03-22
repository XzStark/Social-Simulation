from __future__ import annotations

from pathlib import Path

import yaml

from pepole.causal_consistency import audit_scenario_causal_consistency
from pepole.calibration_loop import merge_realism_into_scenario_yaml
from pepole.config import Scenario
from pepole.experiment_manifest import build_experiment_manifest
from pepole.reproducibility import environment_fingerprint


def test_environment_fingerprint_keys() -> None:
    fp = environment_fingerprint()
    assert "python" in fp
    assert "platform" in fp
    assert "pepole_package_version" in fp


def test_manifest_includes_environment() -> None:
    s = Scenario.load("scenarios/roadmap_demo.yaml")
    m = build_experiment_manifest(
        s, seed=1, primary_model_slot="x", fast_model_slot="y", pepole_package_version="test"
    )
    assert "environment" in m
    assert m["environment"]["pepole_package_version"]


def test_causal_cycle_warning() -> None:
    raw = yaml.safe_load(Path("scenarios/default.yaml").read_text(encoding="utf-8"))
    ext = raw.setdefault("extensions", {})
    c = ext.setdefault("causal", {})
    c["enabled"] = True
    c["edges"] = [
        {"id": "cyc_a", "from_metric": "rumor_level", "to_metric": "unrest", "lag_ticks": 1, "weight": 0.1},
        {"id": "cyc_b", "from_metric": "unrest", "to_metric": "rumor_level", "lag_ticks": 1, "weight": 0.1},
    ]
    s = Scenario.model_validate(raw)
    r = audit_scenario_causal_consistency(s)
    assert any("有向环" in w for w in r["warnings"])


def test_merge_realism_writes_yaml(tmp_path: Path) -> None:
    src = tmp_path / "in.yaml"
    src.write_text(
        "name: t\nrealism:\n  llm_effect_multiplier: 0.3\nticks: 2\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.yaml"
    merge_realism_into_scenario_yaml(src, {"llm_effect_multiplier": 0.99}, dest)
    out = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert float(out["realism"]["llm_effect_multiplier"]) == 0.99
