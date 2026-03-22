"""
多次运行终局聚合：风险分布尾部、P×影响×不可逆性、标准决策简报、分歧启发式说明。
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pepole.config import CrisisRule, EnsembleRiskSpec, Scenario, TriggerCondition

_METRIC_LABEL = {
    "policy_support": "支持度",
    "unrest": "动荡",
    "issuer_trust_proxy": "信任",
    "sentiment": "情绪",
    "rumor_level": "谣言",
    "supply_chain_stress": "供应链压力",
    "economy_index": "经济景气",
}


def _metrics(r: dict[str, Any]) -> dict[str, Any]:
    return r.get("final_metrics") or {}


def _condition_on_metrics(m: dict[str, Any], c: TriggerCondition) -> bool:
    v = float(m.get(c.metric, 0.0))
    if c.gte is not None and v < c.gte:
        return False
    if c.lte is not None and v > c.lte:
        return False
    if c.gt is not None and v <= c.gt:
        return False
    if c.lt is not None and v >= c.lt:
        return False
    return True


def _crisis_fires(m: dict[str, Any], rule: CrisisRule) -> bool:
    return all(_condition_on_metrics(m, c) for c in rule.when_all)


def _narrative_snip(r: dict[str, Any], max_chars: int = 420) -> str:
    nar = r.get("narrative") or []
    if not isinstance(nar, list) or not nar:
        return ""
    tail = nar[-5:]
    s = " ".join(str(x) for x in tail)
    return (s[:max_chars] + "…") if len(s) > max_chars else s


def _stress_score(m: dict[str, Any]) -> float:
    ps = float(m.get("policy_support", 0.5))
    ur = float(m.get("unrest", 0.0))
    rm = float(m.get("rumor_level", 0.0))
    return (1.0 - ps) * 0.52 + ur * 0.33 + rm * 0.15


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def _run_card(r: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    m = _metrics(r)
    card: dict[str, Any] = {
        "seed": r.get("seed"),
        "note": note,
        "policy_support": m.get("policy_support"),
        "unrest": m.get("unrest"),
        "rumor_level": m.get("rumor_level"),
        "narrative_tail": _narrative_snip(r),
    }
    return card


def build_standard_ensemble_briefing(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    scenario: Scenario | None,
    model_slots: dict[str, str] | None = None,
) -> dict[str, Any]:
    er: EnsembleRiskSpec = scenario.ensemble_risk if scenario else EnsembleRiskSpec()
    dist = summary.get("distributions") or {}
    n = max(len(results), 1)

    # —— 1. 主路径（最接近分布中位数的单次 run）——
    core = ("policy_support", "unrest", "rumor_level", "sentiment", "issuer_trust_proxy")
    best_run: dict[str, Any] | None = None
    best_cost = float("inf")
    for r in results:
        m = _metrics(r)
        cost = 0.0
        for k in core:
            block = dist.get(k)
            if not isinstance(block, dict) or "p50" not in block:
                continue
            target = float(block["p50"])
            v = float(m.get(k, target))
            scale = max(abs(target), 0.08)
            cost += ((v - target) / scale) ** 2
        if cost < best_cost:
            best_cost = cost
            best_run = r

    main_path = None
    if best_run is not None:
        main_path = _run_card(best_run, note="与各指标 p50 加权距离最小，视作「最像主路径」的单次样本")

    # —— 2. 高风险路径（综合压力分最高）——
    ranked = sorted(results, key=lambda r: -_stress_score(_metrics(r)))
    k_worst = min(er.worst_case_sample_n, len(ranked))
    worst_paths = [_run_card(ranked[i], note="综合压力分高（低支持+高动荡/谣言权重）") for i in range(k_worst)]

    # 极端尾部样本：最低支持、最高动荡
    by_support = sorted(results, key=lambda r: float(_metrics(r).get("policy_support", 0.5)))
    by_unrest = sorted(results, key=lambda r: -float(_metrics(r).get("unrest", 0.0)))
    extreme_samples = {
        "最低支持度样本": [_run_card(by_support[i], note="终局支持度最低档") for i in range(min(3, len(by_support)))],
        "最高动荡样本": [_run_card(by_unrest[i], note="终局动荡最高档") for i in range(min(3, len(by_unrest)))],
    }

    # —— 风险事件表：risk_score = P × I × R ——
    events: list[dict[str, Any]] = []

    def _add(eid: str, label: str, p: float, impact: float, irrev: float) -> None:
        events.append(
            {
                "id": eid,
                "label": label,
                "P": round(p, 4),
                "impact": impact,
                "irreversibility": irrev,
                "risk_score": round(p * impact * irrev, 4),
            }
        )

    cnt_low_s = sum(1 for r in results if float(_metrics(r).get("policy_support", 1.0)) < er.low_support_threshold)
    _add(
        "low_support",
        f"终局支持度 < {er.low_support_threshold}",
        cnt_low_s / n,
        er.impact_low_support,
        er.irreversibility_low_support,
    )
    cnt_hi_u = sum(1 for r in results if float(_metrics(r).get("unrest", 0.0)) > er.high_unrest_threshold)
    _add(
        "high_unrest",
        f"终局动荡 > {er.high_unrest_threshold}",
        cnt_hi_u / n,
        er.impact_high_unrest,
        er.irreversibility_high_unrest,
    )
    cnt_hi_r = sum(1 for r in results if float(_metrics(r).get("rumor_level", 0.0)) > er.high_rumor_threshold)
    _add(
        "high_rumor",
        f"终局谣言 > {er.high_rumor_threshold}",
        cnt_hi_r / n,
        er.impact_high_rumor,
        er.irreversibility_high_rumor,
    )
    if scenario:
        for cr in scenario.crisis_rules:
            cnt = sum(1 for r in results if _crisis_fires(_metrics(r), cr))
            _add(
                f"crisis:{cr.id}",
                f"危机规则终局满足：{cr.title}",
                cnt / n,
                er.crisis_rule_impact,
                er.crisis_rule_irreversibility,
            )

    events.sort(key=lambda e: -float(e["risk_score"]))

    high_ir = er.high_impact_low_prob_max_p
    hilp = [
        e
        for e in events
        if float(e["P"]) <= high_ir and float(e["impact"]) * float(e["irreversibility"]) >= 0.45
    ]
    hilp.sort(key=lambda e: -float(e["impact"]) * float(e["irreversibility"]) * (high_ir - float(e["P"])))

    med_floor = er.medium_high_prob_min_p
    med_band = er.medium_risk_score_band_max
    mhp = [
        e
        for e in events
        if float(e["P"]) >= med_floor and float(e["risk_score"]) <= med_band and float(e["risk_score"]) >= 0.05
    ]
    mhp.sort(key=lambda e: -float(e["P"]) * float(e["risk_score"]))

    collapse_n = sum(
        1
        for r in results
        if float(_metrics(r).get("unrest", 0.0)) >= er.collapse_unrest
        or float(_metrics(r).get("policy_support", 1.0)) <= er.collapse_support
    )
    collapse_triggers: list[dict[str, Any]] = [
        {
            "id": "metric_collapse_band",
            "label": f"动荡≥{er.collapse_unrest} 或 支持度≤{er.collapse_support}",
            "empirical_rate": round(collapse_n / n, 4),
        }
    ]
    if scenario:
        for cr in scenario.crisis_rules:
            cn = sum(1 for r in results if _crisis_fires(_metrics(r), cr))
            collapse_triggers.append(
                {
                    "id": f"crisis_rule:{cr.id}",
                    "title": cr.title,
                    "detail": cr.detail,
                    "empirical_rate": round(cn / n, 4),
                }
            )

    # —— 尾部一行话 ——
    tail_lines: list[str] = []
    for k in ("policy_support", "unrest", "rumor_level"):
        b = dist.get(k)
        if not isinstance(b, dict) or int(b.get("n") or 0) < 2:
            continue
        label = _METRIC_LABEL.get(k, k)
        p95 = b.get("p95")
        p99 = b.get("p99")
        if p95 is not None and p99 is not None:
            tail_lines.append(f"{label}：P95≈{float(p95):.3f}，P99≈{float(p99):.3f}（终局样本经验分位）")

    # —— 4. 脆弱点：按 cv 排序 ——
    fragility: list[dict[str, Any]] = []
    for k, b in dist.items():
        if not isinstance(b, dict):
            continue
        cv = b.get("cv")
        if cv is None:
            continue
        fragility.append(
            {
                "metric": k,
                "label": _METRIC_LABEL.get(k, k),
                "cv": round(float(cv), 4),
                "std": round(float(b.get("std") or 0), 4),
                "mean": round(float(b.get("mean") or 0), 4),
            }
        )
    fragility.sort(key=lambda x: -abs(float(x["cv"])))

    # —— 分歧分析（终局指标层面；非单智能体归因）——
    corr_keys = [k for k in ("policy_support", "unrest", "rumor_level", "sentiment", "issuer_trust_proxy") if k in dist]
    series: dict[str, list[float]] = {k: [] for k in corr_keys}
    for r in results:
        m = _metrics(r)
        for k in corr_keys:
            series[k].append(float(m.get(k, 0.0)))
    corr_pairs: list[dict[str, Any]] = []
    for i, ka in enumerate(corr_keys):
        for kb in corr_keys[i + 1 :]:
            pr = _pearson(series[ka], series[kb])
            if pr is not None:
                corr_pairs.append({"a": ka, "b": kb, "r": round(pr, 4)})
    corr_pairs.sort(key=lambda x: -abs(float(x["r"])))

    divergent_metrics: list[dict[str, Any]] = []
    for k, b in dist.items():
        if not isinstance(b, dict):
            continue
        nv = int(b.get("n") or 0)
        if nv < 2:
            continue
        std = float(b.get("std") or 0.0)
        divergent_metrics.append(
            {"metric": k, "label": _METRIC_LABEL.get(k, k), "std": round(std, 4), "n": nv}
        )
    divergent_metrics.sort(key=lambda x: -float(x["std"]))

    divergence_notes = [
        "下列排序仅基于各 run 的终局指标离散度（标准差），不区分随机种子、池化智能体发言或人群 LLM 批次中哪一条具体导致差异。",
        "若「动荡」与「谣言」跨 run 高度正相关，分歧可能较多来自舆情链路的共同放大；若二者脱钩，可能反映不同 run 中叙事路径分叉。",
        "更细粒度归因需单次 run 导出扩展栈/因果审计（本简报为 ensemble 终局视图）。",
    ]

    # —— 5. 决策建议（轻量规则，非策略枚举引擎）——
    p_est = float(summary.get("p_estimate") or 0.0)
    mean_u = float(summary.get("mean_unrest") or 0.0)
    unrest_p95 = None
    ub = dist.get("unrest")
    if isinstance(ub, dict):
        unrest_p95 = ub.get("p95")
    suggestions: dict[str, Any] = {
        "策略对比": (
            "场景 YAML 中若配置了多 desk/处置分支，可分别跑 ensemble 再对比本块风险表；"
            "当前为单次场景下的聚合建议。"
        ),
        "要点": [],
    }
    if p_est < 0.35:
        suggestions["要点"].append("支持度过线概率偏低：优先检视信任与叙事节奏（issuer_trust、rumor 相关因果/规则是否过松）。")
    if mean_u > 0.45 or (unrest_p95 is not None and float(unrest_p95) > 0.65):
        suggestions["要点"].append("动荡均值或 P95 偏高：建议预留降级沟通与节奏放慢的剧本，并关注 crisis_rules 是否易被触发。")
    if not suggestions["要点"]:
        suggestions["要点"].append("终局分布相对温和：仍建议结合 P99 尾部样本做压力预案，避免只盯均值。")

    # —— 6. 不确定性 + 多模型提示 ——
    p = p_est
    mc_se = math.sqrt(max(p * (1.0 - p) / n, 0.0)) if n > 1 else None
    slots = model_slots or {}
    confidence = {
        "runs": n,
        "threshold_hit_rate": p_est,
        "mc_stderr_hit_rate_approx": round(mc_se, 4) if mc_se is not None else None,
        "说明": "上述为过线比例的简单二项标准误近似；连续指标的分位见 distributions。",
    }
    multi_model_caveat = (
        "多模型（多 API）≠ 自动更真实：若各模型训练数据分布重叠，可能出现「集体偏见」；"
        "应优先保证模型/提示在风格与偏好上的多样性，并用规则层与因果约束限制离谱联合状态。"
    )
    if slots:
        confidence["model_slots"] = dict(slots)

    variance_rows: list[tuple[str, float]] = []
    for k, b in dist.items():
        if not isinstance(b, dict):
            continue
        if int(b.get("n") or 0) < 2:
            continue
        std = float(b.get("std") or 0.0)
        variance_rows.append((k, std * std))
    tot_var = sum(v for _, v in variance_rows) or 1e-12
    var_shares = [
        {
            "metric": k,
            "label": _METRIC_LABEL.get(k, k),
            "variance": round(v, 8),
            "占已追踪终局方差份额": round(v / tot_var, 4),
        }
        for k, v in sorted(variance_rows, key=lambda x: -x[1])
    ]
    coupling_hints: list[str] = []
    for p in corr_pairs[:6]:
        coupling_hints.append(
            f"{p['a']} 与 {p['b']} 的 r≈{p['r']}：若 |r| 大，终局波动可能部分来自同一潜因子（如舆情链路的联动）。"
        )
    uncertainty_breakdown = {
        "终局边际方差分解_启发式": var_shares[:15],
        "由相关性提示的耦合": coupling_hints,
        "无法仅从 ensemble 严格拆开": [
            "随机种子、池化智能体、人群 LLM 批次在一条 run 内纠缠，若无分源方差计量，不能把方差严格归因到 agent / cohort / policy。",
            "建议：关键结论用固定 seed 的 dump-full-state，配合「完整归因链_why量化」做单次剖析；ensemble 侧看分布、尾部与风险表。",
        ],
    }

    return {
        "风险分布与尾部": {
            "完整分位数与离散": "见 summary.distributions（p01/p05/p10/p50/p90/p95/p99、std、cv）",
            "尾部摘要": tail_lines,
            "极端路径样本": extreme_samples,
        },
        "风险评分_P×影响×不可逆": {
            "公式": "risk_score = P × impact × irreversibility（各事件 P 为本次 ensemble 经验频率）",
            "事件表": events,
            "高风险低概率_优先盯": hilp[:5],
            "中风险高概率": mhp[:5],
            "系统崩溃类触发_经验命中率": collapse_triggers,
        },
        "1_主路径_Most_Likely": main_path,
        "2_高风险路径_Worst_Case": worst_paths,
        "3_风险触发点_Triggers": {
            "scenario_crisis_rules": [cr.model_dump() for cr in scenario.crisis_rules] if scenario else [],
            "empirical_collapse_triggers": collapse_triggers,
        },
        "4_系统脆弱点_Fragility": fragility[:12],
        "5_决策建议_Strategies": suggestions,
        "6_不确定性说明_Confidence": confidence,
        "多模型与数据偏见提示": multi_model_caveat,
        "分歧分析": {
            "终局指标标准差排序": divergent_metrics[:12],
            "主要指标两两相关系数": corr_pairs[:10],
            "说明": divergence_notes,
        },
        "不确定性来源拆解": uncertainty_breakdown,
    }
