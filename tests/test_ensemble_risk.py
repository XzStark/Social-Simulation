"""Ensemble 风险简报与分位数扩展（无 LLM）。"""

from __future__ import annotations

from pepole.config import Scenario
from pepole.engine import summarize_ensemble
from pepole.risk_ensemble_report import build_standard_ensemble_briefing


def _fake_results() -> list[dict]:
    seeds_metrics = [
        (1, {"policy_support": 0.6, "unrest": 0.2, "rumor_level": 0.1, "sentiment": 0.5, "issuer_trust_proxy": 0.55}),
        (2, {"policy_support": 0.45, "unrest": 0.7, "rumor_level": 0.75, "sentiment": 0.3, "issuer_trust_proxy": 0.4}),
        (3, {"policy_support": 0.52, "unrest": 0.5, "rumor_level": 0.4, "sentiment": 0.45, "issuer_trust_proxy": 0.5}),
        (4, {"policy_support": 0.58, "unrest": 0.35, "rumor_level": 0.25, "sentiment": 0.48, "issuer_trust_proxy": 0.52}),
        (5, {"policy_support": 0.4, "unrest": 0.8, "rumor_level": 0.72, "sentiment": 0.25, "issuer_trust_proxy": 0.35}),
    ]
    out = []
    for seed, m in seeds_metrics:
        out.append({"seed": seed, "final_metrics": m, "narrative": [f"tick end {seed}"]})
    return out


def test_summarize_ensemble_tail_percentiles() -> None:
    s = summarize_ensemble(_fake_results(), threshold_key="policy_support", threshold=0.55, scenario=None)
    d = s["distributions"]["unrest"]
    assert d["n"] == 5
    assert "p95" in d and "p99" in d
    assert d["p99"] >= d["p95"] >= d["p50"]
    assert "std" in d and d["std"] >= 0


def test_briefing_structure() -> None:
    results = _fake_results()
    s = summarize_ensemble(results, scenario=Scenario.load("scenarios/roadmap_demo.yaml"))
    b = s["标准输出_决策简报"]
    assert "1_主路径_Most_Likely" in b
    assert b["1_主路径_Most_Likely"] is not None
    assert "风险评分_P×影响×不可逆" in b
    assert "事件表" in b["风险评分_P×影响×不可逆"]
    assert "分歧分析" in b
    assert "6_不确定性说明_Confidence" in b
    assert "不确定性来源拆解" in b


def test_build_briefing_without_scenario() -> None:
    results = _fake_results()
    base = {"n": 5, "p_estimate": 0.4, "distributions": {}, "mean_unrest": 0.5}
    b = build_standard_ensemble_briefing(base, results, scenario=None)
    assert b["3_风险触发点_Triggers"]["scenario_crisis_rules"] == []
