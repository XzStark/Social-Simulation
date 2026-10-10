from __future__ import annotations

from people.calibration_loop import run_calibration_bayes
from people.config import Scenario
from people.engine import RunConfig
from people.providers.dry_run import DryRunClient


def test_bayes_calibration_runs_in_dry_run() -> None:
    # Minimal scenario with a few ticks; DryRunClient makes engine deterministic for tests.
    sc = Scenario(name="t", ticks=6)
    cfg = RunConfig(primary_client=DryRunClient(), fast_client=DryRunClient())
    truth = [{"tick": i + 1, "policy_support": 0.5} for i in range(6)]
    out = run_calibration_bayes(
        sc,
        cfg,
        seed=42,
        truth_rows=truth,
        space={"llm_effect_multiplier": (0.2, 0.4)},
        n_trials=3,
        metric_weights={"policy_support": 1.0},
    )
    assert out.get("ok") is True
    assert "best_params" in out
