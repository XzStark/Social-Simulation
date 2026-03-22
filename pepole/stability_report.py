"""
多次运行后的 **稳定性** 与 **波动原因（通俗版）** —— 给汇报用，少术语。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pepole.config import Scenario


_METRIC_LABEL = {
    "policy_support": "支持度",
    "unrest": "动荡",
    "issuer_trust_proxy": "信任",
    "sentiment": "情绪",
    "rumor_level": "谣言",
    "supply_chain_stress": "供应链压力",
    "economy_index": "经济景气",
}


def enrich_ensemble_summary(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    threshold_key: str,
    scenario: Scenario | None = None,
    model_slots: dict[str, str] | None = None,
) -> dict[str, Any]:
    """在 summarize_ensemble 结果上追加「一眼能懂」的段落。"""
    from pepole.risk_ensemble_report import build_standard_ensemble_briefing

    out = dict(summary)
    dist = summary.get("distributions") or {}
    n = int(summary.get("n") or len(results) or 1)

    spread_lines: list[str] = []
    ranking: list[tuple[str, float]] = []
    for k, block in dist.items():
        if not isinstance(block, dict) or block.get("n", 0) < 2:
            continue
        lo = float(block["min"])
        hi = float(block["max"])
        span = hi - lo
        label = _METRIC_LABEL.get(k, k)
        spread_lines.append(
            f"{label}：多次跑下来，最低约 {lo:.3f}、最高约 {hi:.3f}，两头相差约 {span:.3f}。"
        )
        mean = float(block.get("mean") or 0)
        noise = span / max(abs(mean), 0.05) if mean != 0 else span
        ranking.append((k, noise))

    ranking.sort(key=lambda x: -x[1])
    top = ranking[:3]
    why = []
    if top:
        why.append(
            "波动相对更明显的是："
            + "、".join(_METRIC_LABEL.get(k, k) for k, _ in top)
            + "。常见原因包括：每轮智能体发言不同、抽样到的人设不同、随机环境噪声等（不是算错，是故意留的随机性）。"
        )
    else:
        why.append("当前只有 1 次运行或数据不足，看不出波动；建议把运行次数调到至少 5～8 次。")

    # 最好/最差 seed（按 threshold 指标）
    best_seed = worst_seed = None
    if results:
        key = threshold_key

        def _key_val(r: dict[str, Any]) -> float:
            m = r.get("final_metrics") or {}
            v = m.get(key)
            return float(v) if isinstance(v, (int, float)) else 0.0

        best = max(results, key=_key_val)
        worst = min(results, key=_key_val)
        best_seed = best.get("seed")
        worst_seed = worst.get("seed")

    out["用人话说"] = {
        "一共跑了几遍": n,
        "支持度过线的比例": f"约 {100.0 * float(summary.get('p_estimate', 0)):.1f}%（按您设的线 {summary.get('threshold')} 算）",
        "各指标两头差多少": spread_lines,
        "为什么数字会抖": why,
        "哪一次最好": f"种子 {best_seed}" if best_seed is not None else "—",
        "哪一次最差": f"种子 {worst_seed}" if worst_seed is not None else "—",
    }
    out["标准输出_决策简报"] = build_standard_ensemble_briefing(
        out, results, scenario=scenario, model_slots=model_slots
    )
    return out
