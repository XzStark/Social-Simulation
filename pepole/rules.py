from __future__ import annotations

import random
from typing import Any

from pepole.config import RealismConfig
from pepole.state import CohortState, WorldState

# 产品侧：强品牌缓冲负面支持度冲击；议题刚需高则负面新闻边际伤害更大（再乘品牌缓冲）

_DEFAULT_REALISM = RealismConfig()


def _stratum_rep_weight(layer: str, rc: RealismConfig) -> float:
    if layer == "lower":
        return rc.stratum_lower_representation_weight
    if layer == "middle":
        return rc.stratum_middle_representation_weight
    if layer == "upper":
        return rc.stratum_upper_representation_weight
    if layer == "mixed":
        return rc.stratum_mixed_representation_weight
    return rc.stratum_mixed_representation_weight


def _stratum_unrest_mult(layer: str, rc: RealismConfig) -> float:
    if layer == "lower":
        return rc.stratum_unrest_lower_mult
    if layer == "middle":
        return rc.stratum_unrest_middle_mult
    if layer == "upper":
        return rc.stratum_unrest_upper_mult
    if layer == "mixed":
        return rc.stratum_unrest_mixed_mult
    return rc.stratum_unrest_mixed_mult


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _scale(x: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x


def apply_environment_drift(
    state: WorldState,
    rng: random.Random,
    rc: RealismConfig | None = None,
) -> None:
    rc = rc or _DEFAULT_REALISM
    state.rumor_level = _clamp01(
        state.rumor_level + rng.uniform(rc.env_rumor_min_step, rc.env_rumor_max_step)
    )
    state.economy_index = _clamp01(
        state.economy_index + rng.uniform(rc.env_economy_min_step, rc.env_economy_max_step)
    )


def aggregate_from_cohorts(state: WorldState, rc: RealismConfig | None = None) -> None:
    rc = rc or _DEFAULT_REALISM
    if not state.cohorts:
        return
    eff_w = [c.weight * _stratum_rep_weight(c.class_layer, rc) for c in state.cohorts]
    tw = sum(eff_w) or 1.0
    att = sum(c.attitude * ew for c, ew in zip(state.cohorts, eff_w)) / tw
    act = sum(c.activation * ew for c, ew in zip(state.cohorts, eff_w)) / tw
    state.sentiment = _clamp(att)
    stress = 0.0
    for c, ew in zip(state.cohorts, eff_w):
        local = (1.0 - (c.attitude + 1) / 2) * c.activation
        stress += local * ew * _stratum_unrest_mult(c.class_layer, rc)
    stress /= tw
    state.unrest = _clamp01(
        state.unrest * rc.unrest_decay_factor
        + rc.unrest_stress_weight * stress
        + rc.unrest_rumor_coupling * state.rumor_level
    )
    state.policy_support = _clamp01(
        state.policy_support
        + rc.policy_from_sentiment_weight * att
        - rc.policy_unrest_penalty * state.unrest
    )
    # 信任与供应链：慢变量，向支持度回归并受谣言侵蚀
    state.issuer_trust_proxy = _clamp01(
        state.issuer_trust_proxy
        + rc.trust_mean_reversion * (state.policy_support - state.issuer_trust_proxy)
        - rc.trust_rumor_erosion * state.rumor_level
    )
    econ_soft = 1.0 - state.economy_index
    state.supply_chain_stress = _clamp01(
        state.supply_chain_stress * rc.supply_chain_persistence
        + rc.supply_chain_rumor_sensitivity * state.rumor_level * (0.45 + 0.55 * econ_soft)
        + rc.supply_chain_economy_sensitivity * econ_soft * 0.35
    )


def init_cohorts_from_spec(
    cohort_specs: list[Any],
    rng: random.Random,
) -> list[CohortState]:
    out: list[CohortState] = []
    for s in cohort_specs:
        base = rng.uniform(-0.08, 0.08)
        layer = getattr(s, "class_layer", None) or "mixed"
        if layer == "lower":
            base -= 0.045
        elif layer == "upper":
            base += 0.035
        elif layer == "middle":
            base += rng.uniform(-0.02, 0.02)
        out.append(
            CohortState(
                id=s.id,
                weight=float(s.weight),
                attitude=_clamp(base),
                activation=_clamp01(0.22 + rng.uniform(0, 0.2)),
                class_layer=str(layer),
                traits=dict(s.traits),
            )
        )
    return out


def apply_cohort_deltas(
    state: WorldState,
    deltas: dict[str, dict[str, float]],
    rng: random.Random,
    rc: RealismConfig | None = None,
) -> None:
    rc = rc or _DEFAULT_REALISM
    for c in state.cohorts:
        d = deltas.get(c.id, {})
        if "attitude_delta" in d:
            c.attitude = _clamp(c.attitude + float(d["attitude_delta"]))
        if "activation_delta" in d:
            c.activation = _clamp01(c.activation + float(d["activation_delta"]))
        c.attitude = _clamp(c.attitude + rng.uniform(-rc.cohort_micro_jitter, rc.cohort_micro_jitter))


def _modulate_product_deltas(
    eff: dict[str, Any],
    *,
    brand_equity: float,
    problem_salience: float,
    m: float,
    rc: RealismConfig,
) -> dict[str, Any]:
    """按品牌资产与议题重要性调整 LLM 产出 delta（产品演练）。"""
    be = _clamp01(float(brand_equity))
    ps = _clamp01(float(problem_salience))
    # 强品牌：负面「支持/好感」冲击打折；正面略放大。高 salience：负面冲击略放大（需求刚性强更伤）。
    neg_support_dampen = 1.0 - 0.58 * (be**1.12)
    neg_support_dampen = max(0.32, min(1.0, neg_support_dampen))
    salience_amplify_neg = 1.0 + 0.42 * ps
    pos_support_boost = 0.88 + 0.28 * be
    rumor_dampen_bad = 1.0 - 0.38 * be

    out = dict(eff)
    raw_pd = float(eff.get("policy_support_delta", 0.0)) * m
    if raw_pd < 0:
        pd = raw_pd * salience_amplify_neg * neg_support_dampen
    else:
        pd = raw_pd * pos_support_boost
    out["policy_support_delta"] = _scale(pd, rc.llm_cap_policy_support)

    raw_rd = float(eff.get("rumor_delta", 0.0)) * m
    if raw_rd > 0:
        out["rumor_delta"] = _scale(raw_rd * rumor_dampen_bad, rc.llm_cap_rumor)
    else:
        out["rumor_delta"] = _scale(raw_rd, rc.llm_cap_rumor)

    raw_ud = float(eff.get("unrest_delta", 0.0)) * m
    if raw_ud > 0:
        out["unrest_delta"] = _scale(
            raw_ud * (1.0 + 0.22 * ps) * (1.0 - 0.25 * be),
            rc.llm_cap_unrest,
        )
    else:
        out["unrest_delta"] = _scale(raw_ud, rc.llm_cap_unrest)

    raw_sd = float(eff.get("sentiment_delta", 0.0)) * m
    if raw_sd < 0:
        out["sentiment_delta"] = raw_sd * salience_amplify_neg * neg_support_dampen
    else:
        out["sentiment_delta"] = raw_sd * pos_support_boost
    return out


def _prepare_key_actor_eff_dict(
    raw: dict[str, Any],
    *,
    rng: random.Random,
    rc: RealismConfig,
    exercise_type: str,
    brand_equity: float | None,
    problem_salience: float | None,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    m = rc.llm_effect_multiplier
    use_product_curve = exercise_type == "product" and brand_equity is not None and problem_salience is not None
    if use_product_curve:
        return _modulate_product_deltas(
            raw,
            brand_equity=float(brand_equity),
            problem_salience=float(problem_salience),
            m=m,
            rc=rc,
        )
    return {
        **raw,
        "sentiment_delta": float(raw.get("sentiment_delta", 0.0)) * m,
        "rumor_delta": _scale(float(raw.get("rumor_delta", 0.0)) * m, rc.llm_cap_rumor),
        "policy_support_delta": _scale(
            float(raw.get("policy_support_delta", 0.0)) * m,
            rc.llm_cap_policy_support,
        ),
        "unrest_delta": _scale(float(raw.get("unrest_delta", 0.0)) * m, rc.llm_cap_unrest),
    }


def apply_key_actor_effect_one(
    state: WorldState,
    raw: dict[str, Any],
    rng: random.Random,
    rc: RealismConfig | None = None,
    *,
    exercise_type: str = "policy",
    brand_equity: float | None = None,
    problem_salience: float | None = None,
) -> None:
    """应用单个关键智能体 JSON 效果（供逐步归因）。"""
    rc = rc or _DEFAULT_REALISM
    eff = _prepare_key_actor_eff_dict(
        raw,
        rng=rng,
        rc=rc,
        exercise_type=exercise_type,
        brand_equity=brand_equity,
        problem_salience=problem_salience,
    )
    if eff is None:
        return
    m = rc.llm_effect_multiplier
    sd = float(eff.get("sentiment_delta", 0.0))
    if sd != 0 and state.cohorts:
        bump = sd * rc.llm_sentiment_to_cohort_scale
        for c in state.cohorts:
            c.attitude = _clamp(c.attitude + bump)
    rd = float(eff.get("rumor_delta", 0.0))
    state.rumor_level = _clamp01(state.rumor_level + rd)
    pd = float(eff.get("policy_support_delta", 0.0))
    state.policy_support = _clamp01(state.policy_support + pd)
    ud = float(eff.get("unrest_delta", 0.0))
    state.unrest = _clamp01(state.unrest + ud)
    cn = eff.get("cohort_nudges")
    if isinstance(cn, dict):
        scaled: dict[str, dict[str, float]] = {}
        for k, v in cn.items():
            if not isinstance(v, dict):
                continue
            scaled[k] = {
                kk: float(vv) * m
                for kk, vv in v.items()
                if kk in ("attitude_delta", "activation_delta")
            }
        apply_cohort_deltas(state, scaled, rng, rc)
    if state.cash_balance_million is not None and eff.get("cash_delta_million") is not None:
        cd = float(eff["cash_delta_million"]) * m
        state.cash_balance_million = max(0.0, float(state.cash_balance_million) + cd)
    if state.fiscal_remaining_billion is not None and eff.get("fiscal_delta_billion") is not None:
        fd = float(eff["fiscal_delta_billion"]) * m
        state.fiscal_remaining_billion = max(0.0, float(state.fiscal_remaining_billion) + fd)


def apply_key_actor_effects(
    state: WorldState,
    effects: list[dict[str, Any]],
    rng: random.Random,
    rc: RealismConfig | None = None,
    *,
    exercise_type: str = "policy",
    brand_equity: float | None = None,
    problem_salience: float | None = None,
) -> None:
    rc = rc or _DEFAULT_REALISM
    for raw in effects:
        apply_key_actor_effect_one(
            state,
            raw if isinstance(raw, dict) else {},
            rng,
            rc,
            exercise_type=exercise_type,
            brand_equity=brand_equity,
            problem_salience=problem_salience,
        )


def apply_macro_inertia_blend(
    state: WorldState,
    rc: RealismConfig | None = None,
) -> dict[str, float]:
    """
    在 finalize 前将宏观量向 metrics_history 末行（上一轮末快照）回拉，增强曲线「粘性」。
    blend=0 关闭；0.12～0.28 常用于试验（非实证标定）。
    """
    from pepole.attribution import MACRO_ATTR_KEYS

    rc = rc or _DEFAULT_REALISM
    blend = float(rc.macro_inertia_blend)
    if blend <= 0 or not state.metrics_history:
        return {}
    keys = list(rc.macro_inertia_keys) if rc.macro_inertia_keys else list(MACRO_ATTR_KEYS)
    last_row = state.metrics_history[-1]
    deltas: dict[str, float] = {}
    for k in keys:
        if k not in last_row:
            continue
        if not hasattr(state, k):
            continue
        cur = float(getattr(state, k))
        prev = float(last_row[k])
        nv = (1.0 - blend) * cur + blend * prev
        if k == "sentiment":
            nv = _clamp(nv, -1.0, 1.0)
        else:
            nv = _clamp01(nv)
        diff = nv - cur
        if abs(diff) > 1e-14:
            deltas[k] = diff
        setattr(state, k, nv)
    return deltas


def finalize_tick(state: WorldState) -> None:
    state.metrics_history.append(state.snapshot_metrics())
