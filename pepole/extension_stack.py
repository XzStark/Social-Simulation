from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from pepole.attribution import record_attribution
from pepole.config import CausalRuleSpec, Scenario
from pepole.state import WorldState
from pepole.triggers import condition_holds

Phase = str

PluginFn = Callable[[Phase, WorldState, Scenario, random.Random, dict[str, Any]], None]

_REGISTRY: dict[str, PluginFn] = {}


def register(name: str) -> Callable[[PluginFn], PluginFn]:
    def deco(fn: PluginFn) -> PluginFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def run_extension_plugins(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    *,
    ctx: dict[str, Any] | None = None,
) -> None:
    ctx = ctx if ctx is not None else {}
    order = list(scenario.extensions.plugin_order)
    for name in order:
        fn = _REGISTRY.get(name)
        if fn is None:
            continue
        fn(phase, state, scenario, rng, ctx)


def _sir_step(
    S: float,
    I: float,
    R: float,
    *,
    beta: float,
    gamma: float,
    rumor_level: float,
    rumor_coupling: float,
) -> tuple[float, float, float]:
    def _u(x: float) -> float:
        return max(0.0, min(1.0, x))

    tot = S + I + R
    if tot <= 1e-9:
        S, I, R = 0.94, 0.05, 0.01
    else:
        S, I, R = S / tot, I / tot, R / tot
    lam = float(rumor_coupling) * float(rumor_level)
    d_s = -beta * S * I - lam * S
    d_i = beta * S * I + lam * S - gamma * I
    d_r = gamma * I
    S = _u(S + d_s)
    I = _u(I + d_i)
    R = _u(R + d_r)
    t2 = S + I + R
    if t2 > 1e-9:
        S, I, R = S / t2, I / t2, R / t2
    return S, I, R


def _sync_global_diffusion_from_cohorts(state: WorldState, scenario: Scenario) -> None:
    cohorts = state.cohorts
    if not cohorts:
        return
    tw = sum(max(0.0, float(c.weight)) for c in cohorts) or 1.0
    S = I = R = 0.0
    for c in cohorts:
        w = max(0.0, float(c.weight))
        d = state.diffusion_cohort_sir.get(str(c.id)) or {"s": 1.0, "i": 0.0, "r": 0.0}
        S += w * float(d.get("s", 0.0))
        I += w * float(d.get("i", 0.0))
        R += w * float(d.get("r", 0.0))
    state.diffusion_s = S / tw
    state.diffusion_i = I / tw
    state.diffusion_r = R / tw


def _collapse_rule_effects(rule: CausalRuleSpec) -> dict[str, float]:
    if rule.merge_mode == "last_wins":
        out: dict[str, float] = {}
        for eff in rule.effects:
            out[str(eff.metric)] = float(eff.add)
        return out
    out: dict[str, float] = {}
    for eff in rule.effects:
        m = str(eff.metric)
        out[m] = out.get(m, 0.0) + float(eff.add)
    return out


def init_extensions_state(state: WorldState, scenario: Scenario) -> None:
    ex = scenario.extensions
    if ex.diffusion.enabled:
        i0 = max(0.0, min(1.0, float(ex.diffusion.seed_informed)))
        if ex.diffusion.mode == "sir_per_cohort":
            state.diffusion_cohort_sir = {}
            for c in state.cohorts:
                state.diffusion_cohort_sir[str(c.id)] = {
                    "s": max(0.0, 1.0 - i0),
                    "i": i0,
                    "r": 0.0,
                }
            _sync_global_diffusion_from_cohorts(state, scenario)
        else:
            state.diffusion_s = max(0.0, 1.0 - i0)
            state.diffusion_i = i0
            state.diffusion_r = 0.0
            state.diffusion_cohort_sir = {}
    if ex.delay.enabled and ex.delay.schedule:
        for ev in ex.delay.schedule:
            state.delayed_events.append(
                {
                    "due_tick": int(ev.due_tick),
                    "deltas": dict(ev.deltas),
                    "note": str(ev.note or ""),
                }
            )
    if ex.resources.enabled:
        state.resource_manpower = float(ex.resources.manpower_initial)
        state.resource_political_capital = float(ex.resources.political_capital_initial)


_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "sentiment": (-1.0, 1.0),
    "economy_index": (0.0, 1.0),
    "policy_support": (0.0, 1.0),
    "rumor_level": (0.0, 1.0),
    "unrest": (0.0, 1.0),
    "issuer_trust_proxy": (0.0, 1.0),
    "supply_chain_stress": (0.0, 1.0),
}


def _clamp_metric(metric: str, v: float) -> float:
    lo, hi = _METRIC_BOUNDS.get(metric, (-1e9, 1e9))
    return max(lo, min(hi, v))


def _register_causal_metric_touch(state: WorldState, metric: str) -> None:
    if metric not in state.causal_metrics_this_tick:
        state.causal_metrics_this_tick.append(metric)


def _apply_world_metric_delta(state: WorldState, metric: str, delta: float) -> None:
    if not hasattr(state, metric):
        return
    cur = getattr(state, metric)
    if not isinstance(cur, (int, float)):
        return
    setattr(state, metric, _clamp_metric(metric, float(cur) + float(delta)))


def _trace(state: WorldState, scenario: Scenario, line: str) -> None:
    if not scenario.extensions.validation_trace:
        return
    state.extension_trace.append(f"[t{state.tick}] {line}")
    state.extension_trace = state.extension_trace[-160:]


@register("delay")
def _plugin_delay(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "tick_start":
        return
    if not scenario.extensions.delay.enabled:
        return
    due = int(state.tick)
    kept: list[dict[str, Any]] = []
    fired: list[dict[str, Any]] = []
    for ev in state.delayed_events:
        if int(ev.get("due_tick", -1)) == due:
            fired.append(ev)
        else:
            kept.append(ev)
    state.delayed_events = kept
    for ev in fired:
        deltas = ev.get("deltas") if isinstance(ev.get("deltas"), dict) else {}
        for k, dv in deltas.items():
            _apply_world_metric_delta(state, str(k), float(dv))
        note = str(ev.get("note") or "")
        state.log(f"[延迟生效] tick {due} " + (note[:120] if note else str(deltas)))
        _trace(state, scenario, f"delay_flush {deltas}")
        if deltas:
            record_attribution(
                state,
                layer="extension",
                component="delay",
                tick=state.tick,
                deltas={str(k): float(v) for k, v in deltas.items()},
                meta={"note": note[:200]},
            )


@register("behavior_micro")
def _plugin_behavior_micro(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "after_cohort_llm":
        return
    bm = scenario.extensions.behavior_micro
    if not bm.enabled:
        return
    sig = float(bm.rating_jitter_sigma)
    for c in state.cohorts:
        base = 3.0 + float(c.attitude) * 1.85
        rating = base + rng.gauss(0.0, sig)
        rating = max(1.0, min(5.0, rating))
        state.behavior_micro_history.append(
            {
                "tick": state.tick,
                "cohort_id": c.id,
                "rating_proxy": round(rating, 3),
                "activation": round(float(c.activation), 4),
            }
        )
    state.behavior_micro_history = state.behavior_micro_history[-240:]
    _trace(state, scenario, f"behavior_micro n={len(state.cohorts)}")


@register("causal")
def _plugin_causal(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "pre_finalize":
        return
    cl = scenario.extensions.causal
    if not cl.enabled:
        return
    state.causal_rules_fired_tick = []
    state.causal_metrics_this_tick = []
    sorted_rules = sorted(cl.rules, key=lambda r: (int(r.priority), str(r.id or "")))
    for rule in sorted_rules:
        conds = list(rule.when_all)
        ok = True if not conds else all(condition_holds(state, c) for c in conds)
        if not ok:
            continue
        rid = str(rule.id or "rule")
        state.causal_rules_fired_tick.append(rid)
        collapsed = _collapse_rule_effects(rule)
        eff_d: dict[str, float] = {}
        for m, add in collapsed.items():
            if m not in _METRIC_BOUNDS:
                continue
            eff_d[m] = eff_d.get(m, 0.0) + float(add)
            _apply_world_metric_delta(state, m, float(add))
            _register_causal_metric_touch(state, m)
        _trace(state, scenario, f"causal {rid} effects={len(collapsed)}")
        if eff_d:
            record_attribution(
                state,
                layer="extension",
                component=f"causal:{rid}",
                tick=state.tick,
                deltas=eff_d,
                meta={"rule_id": rid, "priority": rule.priority, "merge_mode": rule.merge_mode},
            )

    hist = state.metrics_history
    edge_acc: dict[str, float] = {}
    for edge in cl.edges:
        lag = int(edge.lag_ticks)
        if lag < 1 or len(hist) < lag:
            continue
        row = hist[-lag]
        fm = str(edge.from_metric)
        tm = str(edge.to_metric)
        if fm not in row or tm not in _METRIC_BOUNDS:
            continue
        src = float(row[fm])
        raw_delta = src * float(edge.weight)
        cap = edge.max_abs_delta
        if cap is not None:
            cabs = float(cap)
            raw_delta = max(-cabs, min(cabs, raw_delta))
        _apply_world_metric_delta(state, tm, raw_delta)
        _register_causal_metric_touch(state, tm)
        edge_acc[tm] = edge_acc.get(tm, 0.0) + raw_delta
    if edge_acc:
        eid = ",".join(str(e.id or i) for i, e in enumerate(cl.edges))[:120]
        record_attribution(
            state,
            layer="extension",
            component="causal:edges",
            tick=state.tick,
            deltas=edge_acc,
            meta={"edge_ids_excerpt": eid},
        )
        _trace(state, scenario, f"causal edges n={len(cl.edges)} applied={len(edge_acc)}")


@register("diffusion")
def _plugin_diffusion(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "pre_finalize":
        return
    dl = scenario.extensions.diffusion
    if not dl.enabled:
        return

    beta = float(dl.beta)
    gamma = float(dl.gamma)
    rl = float(state.rumor_level)
    rc = float(dl.rumor_coupling)

    if dl.mode == "sir_per_cohort":
        if not state.cohorts:
            _trace(state, scenario, "diffusion sir_per_cohort skipped (no cohorts)")
            return
        I0_avg = float(state.diffusion_i)
        i0_seed = max(0.0, min(1.0, float(dl.seed_informed)))
        for c in state.cohorts:
            cid = str(c.id)
            if cid not in state.diffusion_cohort_sir:
                state.diffusion_cohort_sir[cid] = {
                    "s": max(0.0, 1.0 - i0_seed),
                    "i": i0_seed,
                    "r": 0.0,
                }
            tri = state.diffusion_cohort_sir[cid]
            S, I, R = float(tri["s"]), float(tri["i"]), float(tri["r"])
            S, I, R = _sir_step(S, I, R, beta=beta, gamma=gamma, rumor_level=rl, rumor_coupling=rc)
            state.diffusion_cohort_sir[cid] = {"s": S, "i": I, "r": R}
        _sync_global_diffusion_from_cohorts(state, scenario)
        S, I, R = float(state.diffusion_s), float(state.diffusion_i), float(state.diffusion_r)
        _trace(state, scenario, f"diffusion sir_per_cohort I_avg={I:.4f}")
        if abs(I - I0_avg) > 1e-12:
            record_attribution(
                state,
                layer="extension",
                component="diffusion",
                tick=state.tick,
                deltas={},
                meta={
                    "mode": "sir_per_cohort",
                    "diffusion_s": round(S, 6),
                    "diffusion_i": round(I, 6),
                    "diffusion_r": round(R, 6),
                    "delta_i_avg": round(I - I0_avg, 6),
                },
            )
        return

    S0, I0, R0 = float(state.diffusion_s), float(state.diffusion_i), float(state.diffusion_r)
    S, I, R = _sir_step(S0, I0, R0, beta=beta, gamma=gamma, rumor_level=rl, rumor_coupling=rc)
    state.diffusion_s, state.diffusion_i, state.diffusion_r = S, I, R
    _trace(state, scenario, f"diffusion SIR I={I:.4f}")
    if abs(I - I0) > 1e-12 or abs(S - S0) > 1e-12:
        record_attribution(
            state,
            layer="extension",
            component="diffusion",
            tick=state.tick,
            deltas={},
            meta={
                "mode": "sir_global",
                "diffusion_s": round(S, 6),
                "diffusion_i": round(I, 6),
                "diffusion_r": round(R, 6),
                "delta_i": round(I - I0, 6),
            },
        )


@register("anchor")
def _plugin_anchor(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "pre_finalize":
        return
    gt = scenario.extensions.ground_truth
    if not gt.enabled or not gt.series:
        return
    idx = int(state.tick) - 1
    if idx < 0:
        return
    a = float(gt.blend_alpha)
    anchor_deltas: dict[str, float] = {}
    for metric, series in gt.series.items():
        if not series or idx >= len(series):
            continue
        m = str(metric)
        if m not in _METRIC_BOUNDS:
            continue
        target = float(series[idx])
        cur = float(getattr(state, m))
        blended = (1.0 - a) * cur + a * target
        nv = _clamp_metric(m, blended)
        anchor_deltas[m] = nv - cur
        setattr(state, m, nv)
        _trace(state, scenario, f"anchor {m} -> mix {blended:.4f}")
    if anchor_deltas:
        record_attribution(
            state,
            layer="extension",
            component="ground_truth_anchor",
            tick=state.tick,
            deltas=anchor_deltas,
            meta={"blend_alpha": a, "series_index": idx},
        )


@register("kpi")
def _plugin_kpi(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "pre_finalize":
        return
    kpi_spec = scenario.extensions.kpi
    if not kpi_spec.enabled:
        state.kpi_values = {}
        state.kpi_by_tier = {}
        return
    if not state.cohorts:
        state.kpi_values = {
            "conversion_proxy": 0.0,
            "retention_proxy": float(state.issuer_trust_proxy),
            "churn_stress_proxy": float(state.unrest),
            "nps_proxy": float(state.sentiment),
        }
        if kpi_spec.hierarchy_enabled:
            tiers: dict[str, dict[str, float]] = {"outcome": {}, "process": {}, "resource": {}}
            for kk, vv in state.kpi_values.items():
                tier = kpi_spec.tier_by_key.get(kk, kpi_spec.tier_default)
                if tier not in tiers:
                    tier = kpi_spec.tier_default
                tiers[tier][kk] = float(vv)
            state.kpi_by_tier = tiers
        else:
            state.kpi_by_tier = {}
        record_attribution(
            state,
            layer="extension",
            component="kpi",
            tick=state.tick,
            deltas={},
            meta={
                "kpi_values": dict(state.kpi_values),
                "kpi_by_tier": dict(state.kpi_by_tier),
                "note": "no_cohorts",
            },
        )
        return
    tw = sum(max(0.0, float(c.weight)) for c in state.cohorts) or 1.0
    conv = 0.0
    w_act = 0.0
    for c in state.cohorts:
        w = max(0.0, float(c.weight))
        att = max(0.0, float(c.attitude))
        conv += w * att * float(c.activation)
        w_act += w * float(c.activation)
    conv /= tw
    activation_avg = w_act / tw
    trust = float(state.issuer_trust_proxy)
    ps = float(state.policy_support)
    state.kpi_values = {
        "conversion_proxy": max(0.0, min(1.0, conv * 0.5 + activation_avg * 0.12)),
        "retention_proxy": max(0.0, min(1.0, trust * 0.62 + ps * 0.38)),
        "churn_stress_proxy": max(0.0, min(1.0, float(state.unrest) * (1.05 - trust * 0.35))),
        "nps_proxy": max(-1.0, min(1.0, float(state.sentiment))),
        "activation_avg": max(0.0, min(1.0, activation_avg)),
    }
    if kpi_spec.hierarchy_enabled:
        tiers2: dict[str, dict[str, float]] = {"outcome": {}, "process": {}, "resource": {}}
        for kk, vv in state.kpi_values.items():
            tier = kpi_spec.tier_by_key.get(kk, kpi_spec.tier_default)
            if tier not in tiers2:
                tier = kpi_spec.tier_default
            tiers2[tier][kk] = float(vv)
        state.kpi_by_tier = tiers2
    else:
        state.kpi_by_tier = {}
    _trace(state, scenario, "kpi recomputed")
    record_attribution(
        state,
        layer="extension",
        component="kpi",
        tick=state.tick,
        deltas={},
        meta={"kpi_values": dict(state.kpi_values), "kpi_by_tier": dict(state.kpi_by_tier)},
    )


@register("resources")
def _plugin_resources(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    r = scenario.extensions.resources
    if not r.enabled:
        return
    if state.resource_manpower is None:
        state.resource_manpower = float(r.manpower_initial)
    if state.resource_political_capital is None:
        state.resource_political_capital = float(r.political_capital_initial)
    if phase == "after_interactions":
        n_int = int(ctx.get("interaction_count", 0))
        if n_int > 0:
            rm0 = float(state.resource_manpower or 0.0)
            state.resource_manpower = max(
                0.0,
                float(state.resource_manpower) - n_int * float(r.manpower_per_interaction),
            )
            record_attribution(
                state,
                layer="extension",
                component="resources:interactions",
                tick=state.tick,
                deltas={},
                meta={
                    "interaction_count": n_int,
                    "resource_manpower_delta": round(float(state.resource_manpower) - rm0, 6),
                },
            )
    if phase == "pre_finalize":
        rm_before = float(state.resource_manpower or 0.0)
        rp_before = float(state.resource_political_capital or 0.0)
        n_agents = int(ctx.get("n_primary_agents", 0))
        if n_agents > 0:
            state.resource_manpower = max(
                0.0,
                float(state.resource_manpower) - n_agents * float(r.manpower_per_primary_llm),
            )
        drain = float(state.unrest) * float(r.political_unrest_drain_scale)
        state.resource_political_capital = max(0.0, float(state.resource_political_capital) - drain)
        _trace(
            state,
            scenario,
            f"resources manpower={state.resource_manpower:.2f} political={state.resource_political_capital:.2f}",
        )
        rm_after = float(state.resource_manpower or 0.0)
        rp_after = float(state.resource_political_capital or 0.0)
        if abs(rm_after - rm_before) > 1e-12 or abs(rp_after - rp_before) > 1e-12:
            record_attribution(
                state,
                layer="extension",
                component="resources",
                tick=state.tick,
                deltas={},
                meta={
                    "resource_manpower_delta": round(rm_after - rm_before, 6),
                    "resource_political_delta": round(rp_after - rp_before, 6),
                    "resource_manpower": round(rm_after, 4),
                    "resource_political_capital": round(rp_after, 4),
                },
            )


@register("rl_stub")
def _plugin_rl_stub(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "after_finalize":
        return
    if not scenario.extensions.rl_stub.enabled:
        return
    line = "[rl_policy_stub] 占位：可接入离线 RL / 策略梯度，本 tick 无策略更新"
    state.log(line)
    _trace(state, scenario, line)


@register("multimodal_stub")
def _plugin_multimodal_stub(
    phase: Phase,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
    ctx: dict[str, Any],
) -> None:
    if phase != "after_finalize":
        return
    if not scenario.extensions.multimodal_stub.enabled:
        return
    line = "[multimodal_stub] 占位：图像/语音特征入口（未接模型）"
    state.log(line)
    _trace(state, scenario, line)


def metrics_for_validation_row(state: WorldState) -> dict[str, float]:
    row = dict(state.snapshot_metrics())
    row["tick"] = float(state.tick)
    return row
