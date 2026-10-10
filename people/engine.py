from __future__ import annotations

import math
import json
import random
import statistics
import time
import uuid
import hashlib
import hmac
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from people.agents.cohort_llm import run_cohort_batch_llm
from people.agents.horizon import run_horizon_forecast
from people.agents.interaction import run_social_interactions
from people.agents.key_actor import run_key_actor_turn
from people.agents.company_memory import augment_decision_context, merge_company_memory_tick
from people.agents.risk_milestone import (
    evaluate_internal_desk_feasibility,
    format_desk_feasibility_note,
    format_risk_inventory_for_pause,
    run_risk_inventory_llm,
)
from people.agents.user_resolution import format_resolution_note_for_agents, professionalize_plain_resolution
from people.black_swan import apply_black_swan_event, build_black_swan_event, maybe_pick_internal_issue_preset
from people.config import Scenario
from people.context import build_decision_context, enrich_decision_context_for_plan_evaluate
from people.causal_audit import run_causal_governance_audit
from people.decision_support import build_tick_decision_hints
from people.crisis import pick_crisis
from people.providers.base import LLMClient
from people.roster import agents_for_tick
from people.rules import (
    aggregate_from_cohorts,
    apply_cohort_deltas,
    apply_environment_drift,
    apply_key_actor_effect_one,
    apply_macro_inertia_blend,
    finalize_tick,
    init_cohorts_from_spec,
)
from people.attribution import (
    delta_snapshots,
    extended_metrics_snapshot,
    macro_metrics_snapshot,
    record_attribution,
)
from people.experiment_manifest import build_experiment_manifest
from people.extension_stack import init_extensions_state, run_extension_plugins
from people.state import WorldState
from people.causal_skeleton import materialize_causal_layer_from_skeleton
from people.causal_identifiability import assess_edges_identifiability


def _people_package_version() -> str:
    try:
        from importlib.metadata import version

        return version("people")
    except Exception:
        return ""


@dataclass
class RunConfig:
    primary_client: LLMClient
    fast_client: LLMClient
    key_actor_temperature: float = 0.48
    cohort_temperature: float = 0.35
    # 写入实验清单与 pause 包，便于对齐 API 路由字符串（如 openai:gpt-4o）
    primary_model_slot: str = ""
    fast_model_slot: str = ""


def _experiment_manifest_for_run(scenario: Scenario, seed: int, cfg: RunConfig) -> dict[str, Any]:
    return build_experiment_manifest(
        scenario,
        seed=seed,
        primary_model_slot=(cfg.primary_model_slot or "unknown"),
        fast_model_slot=(cfg.fast_model_slot or "unknown"),
        people_package_version=_people_package_version(),
    )


def _issuer_finance_proxy_million(archetype: str) -> tuple[float, float]:
    """写意现金与有息负债本金（百万 proxy），与 issuer 体量档位一致；非真实财报。"""
    m: dict[str, tuple[float, float]] = {
        "megacorp": (180_000.0, 220_000.0),
        "large_group": (22_000.0, 28_000.0),
        "sme": (520.0, 160.0),
        "startup": (42.0, 12.0),
    }
    return m.get(archetype, (800.0, 0.0))


def _hydrate_finance(state: WorldState, scenario: Scenario) -> None:
    of = scenario.operational_finance
    if not of.enabled:
        return
    if scenario.exercise_type == "product":
        cash = float(of.cash_balance_million)
        debt_p = float(of.debt_principal_million)
        if (
            of.scale_cash_debt_from_issuer_if_defaults
            and cash == 800.0
            and debt_p == 0.0
        ):
            cash, debt_p = _issuer_finance_proxy_million(scenario.issuer.archetype)
        state.cash_balance_million = cash
        state.debt_balance_million = debt_p
        if of.track_parallel_fiscal_pool:
            state.fiscal_remaining_billion = float(of.parallel_fiscal_pool_initial_billion)
    elif of.fiscal_pool_billion is not None:
        state.fiscal_remaining_billion = float(of.fiscal_pool_billion)


def build_initial_state(scenario: Scenario, rng: random.Random) -> WorldState:
    init = scenario.initial
    cohorts = init_cohorts_from_spec(scenario.cohorts, rng)
    ps = float(init.get("policy_support", 0.5))
    if scenario.exercise_type == "product":
        t0 = float(init.get("issuer_trust_proxy", scenario.issuer.brand_equity))
    else:
        t0 = float(init.get("issuer_trust_proxy", ps))
    t0 = max(0.0, min(1.0, t0))
    sc0 = float(
        init.get(
            "supply_chain_stress",
            0.13 + (1.0 - t0) * 0.14 + rng.uniform(-0.015, 0.015),
        )
    )
    sc0 = max(0.0, min(1.0, sc0))
    ws = WorldState(
        tick=0,
        sentiment=float(init.get("sentiment", 0.0)),
        economy_index=float(init.get("economy_index", 0.5)),
        policy_support=ps,
        rumor_level=float(init.get("rumor_level", 0.0)),
        unrest=float(init.get("unrest", 0.0)),
        issuer_trust_proxy=t0,
        supply_chain_stress=sc0,
        cohorts=cohorts,
    )
    init_extensions_state(ws, scenario)
    return ws


def _intervention_notes_block(state: WorldState) -> str:
    return "\n\n".join(list(state.user_resolution_notes[-24:]))


def _append_synthetic_details(state: WorldState, scenario: Scenario, rng: random.Random) -> None:
    if scenario.institutions:
        inst = rng.choice(scenario.institutions)
        tier_cn = "自媒体/小机构" if inst.tier == "small" else "权威机构/主流媒体"
        body = (inst.focus_brief or "围绕当前议题发表评论或二次传播。")[:220]
        state.push_detail("institution_pulse", f"{tier_cn} · {inst.name}", body)
    if scenario.cooperations and scenario.operational_finance.enabled:
        co = rng.choice(scenario.cooperations)
        cost = float(co.cost_million)
        tier_co = "权威侧合作" if co.partner_tier == "authority" else "轻量合作"
        if state.cash_balance_million is not None:
            cash = float(state.cash_balance_million)
            gap = cost - cash
            verdict = "账面可覆盖" if gap <= 0 else f"约缺 {gap:.1f} 百万"
            tail = (co.brief or "")[:120]
            state.push_detail(
                "cooperation_quote",
                co.name,
                f"{tier_co}；报价约 {cost:.1f} 百万；当前账面约 {cash:.1f} 百万 → {verdict}。"
                + (f" {tail}" if tail else ""),
            )
        elif state.fiscal_remaining_billion is not None:
            pool = float(state.fiscal_remaining_billion)
            need_b = cost / 1000.0
            verdict = "财力池可消化" if pool >= need_b else f"财力偏紧（约需 {need_b:.2f} 十亿等效）"
            state.push_detail(
                "cooperation_quote",
                co.name,
                f"{tier_co}；报价约 {cost:.1f} 百万（约合 {need_b:.3f} 十亿 proxy）；可用财力池约 {pool:.2f} 十亿 → {verdict}。",
            )


def _append_key_actor_recall(
    state: WorldState,
    actor_id: str,
    statement: str,
    *,
    max_snippets: int = 8,
    max_chars: int = 260,
) -> None:
    t = " ".join(statement.strip().split())[:max_chars]
    if not t:
        return
    prev = list(state.key_actor_recall.get(actor_id, []))
    prev.append(t)
    state.key_actor_recall = {**state.key_actor_recall, actor_id: prev[-max_snippets:]}


def _compact_actor_memory_if_needed(
    state: WorldState,
    *,
    actor_id: str,
    max_items: int = 8,
    soft_char_budget: int = 1200,
) -> None:
    arr = list(state.key_actor_recall.get(actor_id, []))
    if len(arr) <= max_items and sum(len(x) for x in arr) <= soft_char_budget:
        return
    keep_n = max(4, max_items // 2)
    old = arr[:-keep_n] if len(arr) > keep_n else []
    keep = arr[-keep_n:]
    if old:
        prior = state.key_actor_memory_rollups.get(actor_id, "")
        merged = (prior + "\n" if prior else "") + "；".join(x.strip() for x in old if x.strip())
        state.key_actor_memory_rollups = {
            **state.key_actor_memory_rollups,
            actor_id: merged[-2200:],
        }
        state.log(f"[记忆压缩] {actor_id} 旧表态已归档，后续由记忆摘要接力。")
    state.key_actor_recall = {**state.key_actor_recall, actor_id: keep}


def _maybe_media_detail(state: WorldState, agent_id: str, eff: dict[str, Any]) -> None:
    slant = eff.get("media_slant")
    hf = eff.get("headline_fragment")
    if not isinstance(hf, str) or not hf.strip():
        return
    text = hf.strip()[:280]
    if slant == "unfavorable":
        state.push_detail("media_unfavorable", f"不利报道 · {agent_id}", text, {"slant": slant})
    elif slant == "favorable":
        state.push_detail("media_favorable", f"正面报道 · {agent_id}", text, {"slant": slant})
    else:
        state.push_detail("media_neutral", f"中性报道 · {agent_id}", text, {"slant": slant or "neutral"})


def _append_timeline_uncertainty(state: WorldState, window: int = 6) -> None:
    hist = list(state.metrics_history or [])
    if not hist:
        return
    keys = ["sentiment", "policy_support", "unrest", "rumor_level", "issuer_trust_proxy"]
    row: dict[str, Any] = {
        "tick": state.tick,
        "window": int(window),
        "var": {},
        "std": {},
        "skew": {},
        "kurt_excess": {},
        "cv": {},
        "variance_decomposition": {},
    }
    tail = hist[-window:]
    for k in keys:
        vals = [float(x.get(k)) for x in tail if isinstance(x, dict) and isinstance(x.get(k), (int, float))]
        if len(vals) >= 3:
            mean = float(statistics.mean(vals))
            var = float(statistics.pvariance(vals))
            std = float(math.sqrt(var))
            row["var"][k] = var
            row["std"][k] = std
            if std > 1e-12:
                m3 = float(statistics.mean([(v - mean) ** 3 for v in vals]))
                m4 = float(statistics.mean([(v - mean) ** 4 for v in vals]))
                row["skew"][k] = float(m3 / (std**3))
                row["kurt_excess"][k] = float(m4 / (std**4) - 3.0)
                row["cv"][k] = float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0
            else:
                row["skew"][k] = 0.0
                row["kurt_excess"][k] = 0.0
                row["cv"][k] = 0.0
        elif len(vals) >= 2:
            row["var"][k] = float(statistics.pvariance(vals))
            row["std"][k] = float(statistics.pstdev(vals))
            row["skew"][k] = 0.0
            row["kurt_excess"][k] = 0.0
            row["cv"][k] = 0.0
        else:
            row["var"][k] = 0.0
            row["std"][k] = 0.0
            row["skew"][k] = 0.0
            row["kurt_excess"][k] = 0.0
            row["cv"][k] = 0.0

    # variance decomposition: cov(y_i, f) / var(f), where y_i are cumulative attributed contributions.
    tick_end = int(state.tick)
    tick_start = tick_end - int(window) + 1
    if len(hist) >= int(window):
        ticks = list(range(tick_start, tick_end + 1))
        x_rows = tail
        metric_start_values: dict[str, float] = {}
        for mk in keys:
            metric_vals = [
                float(r.get(mk))
                for r in x_rows
                if isinstance(r, dict) and mk in r and isinstance(r.get(mk), (int, float))
            ]
            if len(metric_vals) == len(x_rows):
                metric_start_values[mk] = metric_vals[0]

        attribution_tail = [
            e
            for e in getattr(state, "attribution_log", []) or []
            if int(e.get("tick", -1)) in ticks and isinstance(e.get("deltas"), dict)
        ]

        for mk in keys:
            if mk not in metric_start_values:
                continue
            f: list[float] = []
            for r in x_rows:
                v = r.get(mk)
                if isinstance(v, (int, float)):
                    f.append(float(v) - metric_start_values[mk])
            if len(f) != len(x_rows) or len(f) < 3:
                continue
            mean_f = float(statistics.mean(f))
            var_f = float(statistics.pvariance(f))
            if var_f <= 1e-12:
                row["variance_decomposition"][mk] = {"var_total": var_f, "components": []}
                continue

            comp_delta: dict[str, list[float]] = {}
            for e in attribution_tail:
                d = e.get("deltas") or {}
                if mk not in d:
                    continue
                t = int(e.get("tick", -1))
                idx = t - tick_start
                if idx < 0 or idx >= len(f):
                    continue
                comp_id = f"{e.get('layer','')}/{e.get('component','')}"
                comp_delta.setdefault(comp_id, [0.0] * len(f))[idx] += float(d.get(mk) or 0.0)

            comp_y: dict[str, list[float]] = {}
            for cid, del_arr in comp_delta.items():
                y = []
                csum = 0.0
                for dv in del_arr:
                    csum += float(dv)
                    y.append(csum)
                comp_y[cid] = y

            sum_y = [0.0] * len(f)
            for y in comp_y.values():
                for i in range(len(f)):
                    sum_y[i] += float(y[i])
            residual = [float(f[i] - sum_y[i]) for i in range(len(f))]
            comp_y["__residual__"] = residual

            shares: list[dict[str, Any]] = []
            for cid, y in comp_y.items():
                mean_y = float(statistics.mean(y))
                cov = float(statistics.mean([(y[i] - mean_y) * (f[i] - mean_f) for i in range(len(f))]))
                share = float(cov / var_f)
                shares.append({"id": cid, "share": share})

            max_keep = 14
            shares_sorted = sorted(shares, key=lambda x: abs(float(x.get("share") or 0.0)), reverse=True)
            keep = shares_sorted[:max_keep]
            drop = shares_sorted[max_keep:]
            other_share = sum(float(x.get("share") or 0.0) for x in drop)
            if abs(other_share) > 1e-12:
                keep.append({"id": "__other_components__", "share": float(other_share)})
            row["variance_decomposition"][mk] = {"var_total": var_f, "components": keep}
    state.timeline_uncertainty.append(row)
    state.timeline_uncertainty = state.timeline_uncertainty[-400:]


def _append_audit_chain(state: WorldState, *, tick: int) -> None:
    prev = (state.audit_chain[-1].get("hash") if state.audit_chain else "") or ""
    payload = {
        "tick": int(tick),
        "metrics": state.snapshot_metrics(),
        "narrative_tail": list(state.narrative[-6:]),
        "attribution_tail": [e for e in state.attribution_log if int(e.get("tick", -1)) == int(tick)][-12:],
        "prev_hash": prev,
    }
    ser = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(ser.encode("utf-8")).hexdigest()
    entry: dict[str, Any] = {"tick": int(tick), "hash": h, "prev_hash": prev}
    secret = (os.environ.get("PEOPLE_AUDIT_SECRET") or "").strip()
    if secret:
        msg = f"{tick}|{h}|{prev}".encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), msg, digestmod=hashlib.sha256).hexdigest()
        entry["sig"] = sig
    state.audit_chain.append(entry)
    state.audit_chain = state.audit_chain[-2000:]


def _update_layered_diffusion_proxy(state: WorldState, tick_details: list[dict[str, Any]]) -> None:
    base = dict(state.diffusion_layer_state or {})
    layers = {
        "online_social": float(base.get("online_social", 0.0)),
        "mainstream_media": float(base.get("mainstream_media", 0.0)),
        "offline_community": float(base.get("offline_community", 0.0)),
        "institutional_channel": float(base.get("institutional_channel", 0.0)),
    }
    for e in tick_details:
        kind = str(e.get("kind") or "")
        if "media_" in kind:
            layers["mainstream_media"] += 0.06
        elif kind in ("inter_agent_dialogue", "resolution_digest", "risk_milestone"):
            layers["institutional_channel"] += 0.05
        elif kind in ("cooperation_quote", "institution_pulse"):
            layers["offline_community"] += 0.04
        else:
            layers["online_social"] += 0.03
    # couple with macro rumor/activation proxy and decay.
    social_push = max(0.0, float(state.rumor_level)) * 0.12
    layers["online_social"] += social_push
    for k in list(layers.keys()):
        layers[k] = max(0.0, min(1.0, layers[k] * 0.92))
    state.diffusion_layer_state = layers


def _maybe_start_reg_investigations(state: WorldState, scenario: Scenario) -> None:
    """
    在高风险态势下启动简化版监管调查状态机（演练用，非真实监管流程）。
    触发条件：产品/政策演练中，谣言与动荡指标持续偏高且当前无进行中的调查。
    """
    # 仅在还有后续 tick 且为政策/产品情境时考虑启动
    if scenario.exercise_type not in ("policy", "product"):
        return
    # 若已有未结案调查，则不重复立案，只交给状态机推进
    open_exist = any(
        inv.get("status") in {"initiated", "pre_check", "in_depth", "rectification"}
        for inv in state.investigations or []
    )
    if open_exist:
        return
    # 简单触发规则：rumor_level 与 unrest 同时偏高，视为「可能触发监管关注」的 proxy
    rumor = float(state.rumor_level)
    unrest = float(state.unrest)
    support = float(state.policy_support)
    issuer_trust = float(state.issuer_trust_proxy)
    risk_score = max(rumor, unrest) * 0.6 + (1.0 - issuer_trust) * 0.25 + (0.4 - support) * 0.15
    if risk_score < 0.55:
        return
    severity = "high" if risk_score >= 0.8 else "medium"
    inv_id = str(uuid.uuid4())
    entry = {
        "id": inv_id,
        "trigger_decision_id": None,
        "status": "initiated",
        "severity": severity,
        "findings": [],
        "next_deadline_tick": int(state.tick) + 1,
        "created_tick": int(state.tick),
    }
    state.investigations.append(entry)
    state.push_detail(
        "reg_investigation_start",
        "监管部门启动合规调查（演练）",
        f"因谣言/动荡与信任 proxy 同时偏高，监管方在沙盘中启动一轮合规调查；当前风险级别：{severity}。",
        {
            "investigation_id": inv_id,
            "status": "initiated",
            "severity": severity,
            "trigger_kind": "macro_risk_proxy",
        },
    )


def _advance_reg_investigations(state: WorldState) -> None:
    """
    推进监管调查状态机阶段，并写入 detail_events。

    状态流转：
      initiated → pre_check → in_depth → rectification → closed
    """
    if not state.investigations:
        return
    phases = ["initiated", "pre_check", "in_depth", "rectification", "closed"]
    for inv in state.investigations:
        status = str(inv.get("status") or "")
        if status == "closed":
            continue
        due = int(inv.get("next_deadline_tick", state.tick + 1))
        if int(state.tick) < due:
            continue
        try:
            idx = phases.index(status)
        except ValueError:
            idx = 0
        if idx + 1 < len(phases):
            new_status = phases[idx + 1]
        else:
            new_status = "closed"
        inv["status"] = new_status
        inv["next_deadline_tick"] = int(state.tick) + (2 if new_status in {"in_depth", "rectification"} else 0)
        meta = {
            "investigation_id": inv.get("id"),
            "status": new_status,
            "severity": inv.get("severity"),
            "created_tick": inv.get("created_tick"),
            "next_deadline_tick": inv.get("next_deadline_tick"),
        }
        title = "监管调查进度更新"
        if new_status == "pre_check":
            body = "监管内部对本次投诉与线索进行形式审查与分办，尚未进入实地深查阶段。"
        elif new_status == "in_depth":
            body = "监管已进入实质性核查阶段，涉及取证、部门问询与多轮书面反馈；沙盘中后续 tick 将视为在调查阴影下运行。"
        elif new_status == "rectification":
            body = "监管初步形成整改要求，要求发行方/地方条线在限定时间内提交整改方案与回执，具体结论尚待评估。"
        else:
            body = "本轮监管调查在沙盘中视为阶段性结束；部分发现可能在记忆与决策层中长期保留影响。"
        state.push_detail("reg_investigation_progress", title, body, meta)


def _normalize_resume_plans(resume: dict[str, Any]) -> list[str]:
    raw = resume.get("latest_plain_resolutions")
    if isinstance(raw, list):
        out = [str(p).strip() for p in raw if str(p).strip()]
        if out:
            return out
    one = resume.get("latest_plain_resolution")
    if isinstance(one, str) and one.strip():
        return [one.strip()]
    return []


def _format_desk_review_detail(plan_evals: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"共 {len(plan_evals)} 套方案已完成「白话→专业转化」与「公司内部桌面可行性」研判（演练生成）。"
        "确认无误后再继续后续 tick；真实社会/监管反应可能一致也可能不一致。",
    ]
    for pe in plan_evals:
        i = int(pe.get("plan_index", 0))
        lines.append(f"\n—— 方案 {i + 1} ——")
        ex = str(pe.get("plain_excerpt") or "").strip()
        if ex:
            lines.append(f"摘要：{ex}")
        fe = pe.get("feasibility")
        if fe:
            lines.append(f"专业转化·可执行性：{fe}")
        et = str(pe.get("estimated_timeline_cn") or "").strip()
        if et:
            lines.append(f"阶段周期（演练估计·转化侧）：{et}")
        etd = str(pe.get("estimated_timeline_desk_cn") or "").strip()
        if etd:
            lines.append(f"阶段周期（演练估计·桌面侧）：{etd}")
        lines.append(str(pe.get("desk_verdict_cn") or pe.get("desk_verdict") or ""))
        wnf = pe.get("why_not_feasible")
        if isinstance(wnf, list) and wnf:
            lines.append("不可行/存疑点：" + "；".join(str(x) for x in wnf[:10] if str(x).strip()))
        cave = str(pe.get("caveat") or "").strip()
        if cave:
            lines.append(f"声明：{cave[:400]}")
    lines.append("\n请在前端点击「确认继续演练」，或「评估另一套方案」提交新文本重新研判。")
    return "\n".join(lines).strip()


def _archive_pending_desk_for_reeval(state: WorldState) -> None:
    """改评前将上一轮待确认快照记入历史，避免丢失。"""
    for row in state.pending_desk_review:
        h = {**row, "id": str(uuid.uuid4()), "status": "superseded_by_re_eval"}
        state.decision_layer_history.append(h)
    state.pending_desk_review = []


def _append_evaluated_plans_to_history(state: WorldState, plan_evals: list[dict[str, Any]]) -> None:
    for pe in plan_evals:
        h = {**pe, "id": str(uuid.uuid4()), "status": "evaluated_pending_confirm"}
        state.decision_layer_history.append(h)
    state.decision_layer_history = state.decision_layer_history[-48:]


def _format_decision_layer_active(chosen: dict[str, Any]) -> str:
    lines = ["【指挥台已确认·决策层有效指令（演练内参，注入后续智能体上下文）】"]
    ui = str(chosen.get("understood_intent") or "").strip()
    if ui:
        lines.append(f"意图：{ui[:900]}")
    dv = str(chosen.get("desk_verdict_cn") or chosen.get("desk_verdict") or "").strip()
    if dv:
        lines.append(f"桌面结论：{dv[:900]}")
    et = str(chosen.get("estimated_timeline_cn") or "").strip()
    etd = str(chosen.get("estimated_timeline_desk_cn") or "").strip()
    if et or etd:
        lines.append("阶段周期（演练估计）：" + f"{et} {etd}".strip())
    prof = str(chosen.get("professional_execution_plan_excerpt") or "").strip()
    if prof:
        lines.append(f"执行择要：{prof[:1200]}")
    return "\n".join(lines)[:4000]


def _apply_operating_ledger(state: WorldState, scenario: Scenario) -> None:
    of = scenario.operational_finance
    if not of.enabled:
        return
    if state.cash_balance_million is not None:
        state.cash_balance_million = max(
            0.0, float(state.cash_balance_million) - float(of.operating_cost_million_per_tick)
        )
    if state.fiscal_remaining_billion is not None:
        state.fiscal_remaining_billion = max(
            0.0,
            float(state.fiscal_remaining_billion) - float(of.policy_spend_per_tick_billion),
        )


def _apply_interaction_budget(state: WorldState, x: dict[str, Any]) -> None:
    if state.cash_balance_million is not None and x.get("cost_million") is not None:
        state.cash_balance_million = max(0.0, float(state.cash_balance_million) - float(x["cost_million"]))
    if state.fiscal_remaining_billion is not None and x.get("fiscal_billion") is not None:
        state.fiscal_remaining_billion = max(
            0.0,
            float(state.fiscal_remaining_billion) - float(x["fiscal_billion"]),
        )


def _actor_label(actor_id: str, tick_agents: list[Any], scenario: Scenario, state: WorldState) -> str:
    for a in tick_agents:
        if getattr(a, "id", None) == actor_id:
            role = str(getattr(a, "role", "") or "").strip()
            return role or actor_id
    if actor_id == "issuer_command_center" and getattr(scenario.issuer, "archetype", None) == "startup":
        return "创业公司指挥中心（创始人/核心团队）"
    fixed = {
        "issuer_command_center": "集团/产品指挥中心",
        "policy_command_center": "政策指挥中心",
        "frontline_execution_team": "基层执行团队",
        "department_manager_office": "中层管理办公室",
        "executive_steering_committee": "高层决策委员会",
        "user_frontline": "基层用户代表",
        "user_middle_group": "中层用户/组织客户",
        "user_upper_group": "高层用户代表",
        "customer_service_center": "客户服务中心",
        "public_opinion_ops": "舆情运营中心",
        "brand_pr_center": "品牌公关中心",
        "legal_compliance_center": "法务合规中心",
        "government_affairs_center": "政府事务中心",
        "regulatory_liaison_window": "监管联络窗口",
        "supply_chain_control_tower": "供应链控制塔",
        "operations_planning_center": "运营计划中心",
        "sales_channel_mgmt": "销售渠道管理部",
        "finance_treasury_center": "财务与资金中心",
        "township_frontline_team": "乡镇基层执行组",
        "county_policy_office": "县级政策办公室",
        "county_data_bureau_office": "县政务数据局",
        "county_justice_bureau": "县司法局",
        "county_public_security": "县公安条线",
        "county_emergency_bureau": "县应急管理局",
        "county_finance_bureau": "县财政局",
        "county_market_regulation": "县市场监管局",
        "county_cyberspace_office": "县网信条线",
        "county_convergence_media": "县融媒体中心",
        "county_petitions_office": "县信访办",
        "municipal_supervision_group": "市级督导组",
        "founder_ceo": "创始人/CEO",
        "core_build_team": "产品技术核心小队",
        "lean_growth_ops": "增长与运营（精简）",
        "customer_voice_channel": "早期用户与社群反馈通道",
        "fractional_cfo_advisor": "兼职/外包财务顾问",
        "external_counsel_pool": "外包律师池",
        "community_ops_volunteer": "社群志愿者运营",
        "external_pr_advisor": "外聘公关顾问",
    }
    if actor_id in fixed:
        return fixed[actor_id]
    # editor desk 等后缀给个可读名称
    if actor_id.endswith("_editorial_desk"):
        src = actor_id[: -len("_editorial_desk")]
        return _actor_label(src, tick_agents, scenario, state) + " 编辑台"
    cohort_ids = {c.id for c in state.cohorts}
    if actor_id in cohort_ids:
        nice = actor_id.replace("_", " ")
        return f"用户群体（{nice}）"
    return actor_id


def run_single_simulation(
    scenario: Scenario,
    cfg: RunConfig,
    *,
    seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
    internal_plan_commit: dict[str, Any] | None = None,
    pop_black_swan_events: Callable[[], list[dict[str, Any]]] | None = None,
) -> tuple[WorldState, dict[str, Any] | None]:
    """
    返回 (终态或暂停态, 暂停包)。
    暂停包为 None 表示正常跑完；非 None 时供网页序列化后续跑 `/api/run/resume/stream`。
    """

    def _emit(payload: dict[str, Any]) -> None:
        if progress_callback:
            progress_callback(payload)

    pause_package: dict[str, Any] | None = None
    runtime_manual_pause_reason: str | None = None

    def _apply_runtime_black_swans(source: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        nonlocal runtime_manual_pause_reason
        if not pop_black_swan_events:
            return out
        for ev in pop_black_swan_events():
            if str(ev.get("_control") or "") == "pause":
                runtime_manual_pause_reason = str(ev.get("reason") or "").strip()[:300] or None
                state.log("[手动暂停请求] 已接收，将在当前/下一轮安全点暂停等待新的处置输入。")
                continue
            d = apply_black_swan_event(state, ev, source=source)
            if d:
                record_attribution(
                    state,
                    layer="intervention",
                    component="black_swan",
                    tick=state.tick,
                    deltas=d,
                    meta={"title": ev.get("title"), "preset_id": ev.get("preset_id"), "source": source},
                )
            out.append(
                {
                    "title": ev.get("title"),
                    "preset_id": ev.get("preset_id"),
                    "severity": ev.get("severity"),
                    "deltas": d,
                }
            )
        return out

    if resume:
        rng = resume["rng"]
        state = resume["state"]
        if not isinstance(rng, random.Random) or not isinstance(state, WorldState):
            raise TypeError("resume 需包含 rng: Random 与 state: WorldState")
        stage = str(resume.get("resume_stage") or "submit_solutions")
        dctx0 = build_decision_context(scenario)
        augment_decision_context(dctx0, state)
        enrich_decision_context_for_plan_evaluate(dctx0, state, scenario)

        if stage == "confirm_proceed":
            state.simulation_outcome = "complete"
            state.log("[演练继续] 指挥台已确认桌面研判结果，后续 tick 开始执行。")
            _emit(
                {
                    "phase": "resume_start",
                    "tick_completed": state.tick,
                    "remaining_ticks": scenario.ticks - state.tick,
                    "scenario_name": scenario.name,
                    "finance_enabled": bool(scenario.operational_finance.enabled),
                    "exercise_type": scenario.exercise_type,
                }
            )
            injected = _apply_runtime_black_swans("resume_gate")
            if injected:
                _emit({"phase": "black_swan_injected", "tick": state.tick, "events": injected})
            sel = resume.get("selected_plan_index")
            si: int | None = None
            if sel is not None:
                try:
                    si = int(sel)
                except (TypeError, ValueError):
                    si = None
            pending = list(state.pending_desk_review)
            chosen: dict[str, Any] | None = None
            if pending:
                if si is not None and si >= 0:
                    chosen = next(
                        (p for p in pending if int(p.get("plan_index", -1)) == si),
                        None,
                    )
                if chosen is None:
                    chosen = pending[0]
            if chosen:
                state.decision_layer_active_summary = _format_decision_layer_active(chosen)
                state.decision_layer_history.append(
                    {**chosen, "id": str(uuid.uuid4()), "status": "confirmed"}
                )
                state.decision_layer_history = state.decision_layer_history[-48:]
                state.log("[决策层] 已固化指挥台确认指令，后续智能体上下文将引用 decision_layer_directive。")
            state.pending_desk_review = []
            if si is not None and si >= 0:
                state.user_resolution_notes.append(
                    f"[指挥台确认] 明示采用方案编号 {si + 1} 作为本轮主执行线。"
                )
                state.log(f"[指挥台确认] 采用方案编号 {si + 1}。")
        else:
            _archive_pending_desk_for_reeval(state)
            plans = _normalize_resume_plans(resume)
            if not plans:
                raise ValueError("submit_solutions 需要至少一条非空方案文本")
            state.simulation_outcome = "complete"
            state.log(
                f"[演练继续] 指挥台已录入 {len(plans)} 套方案，正在进行专业转化与桌面可行性研判…"
            )
            _emit(
                {
                    "phase": "resume_start",
                    "tick_completed": state.tick,
                    "remaining_ticks": scenario.ticks - state.tick,
                    "scenario_name": scenario.name,
                    "finance_enabled": bool(scenario.operational_finance.enabled),
                    "exercise_type": scenario.exercise_type,
                }
            )
            plan_evals: list[dict[str, Any]] = []
            for pi, plan in enumerate(plans):
                digest: dict[str, Any] | None = None
                try:
                    digest = professionalize_plain_resolution(
                        cfg.fast_client,
                        scenario=scenario,
                        state=state,
                        plain_solution=plan,
                        decision_context=dctx0,
                        temperature=0.25,
                    )
                except Exception:
                    digest = None
                if digest:
                    state.user_resolution_notes.append(
                        format_resolution_note_for_agents(digest, scenario=scenario)
                    )
                    preview = digest.get("professional_execution_plan", "") or ""
                    state.log(
                        f"[处置专业转化·方案{pi + 1}] "
                        + (preview[:380] + ("…" if len(preview) > 380 else ""))
                    )
                    state.push_detail(
                        "resolution_digest",
                        f"白话处置 → 专业方案（方案 {pi + 1}）",
                        f"可执行性：{digest.get('feasibility', '')}。{(digest.get('feasibility_notes') or '')[:200]}",
                        {"digest": digest, "plan_index": pi},
                    )
                    _emit(
                        {
                            "phase": "resolution_digest",
                            "plan_index": pi,
                            "tick": state.tick,
                            "understood_intent": (digest.get("understood_intent") or "")[:400],
                            "feasibility": digest.get("feasibility"),
                            "estimated_timeline_cn": (digest.get("estimated_timeline_cn") or "")[:600],
                            "professional_preview": preview[:500],
                            "policy_flow_preview": (digest.get("policy_equivalent_steps") or "")[:500],
                            "involved_functions": digest.get("involved_functions") or [],
                        }
                    )
                desk = evaluate_internal_desk_feasibility(
                    cfg.fast_client,
                    scenario=scenario,
                    state=state,
                    user_plain_solution=plan,
                    professional_digest=digest,
                    risk_inventory=state.last_risk_inventory,
                    decision_context=dctx0,
                    temperature=0.3,
                )
                wnf: list[str] = []
                if isinstance(desk, dict):
                    raw_wnf = desk.get("why_not_feasible")
                    if isinstance(raw_wnf, list):
                        wnf = [str(x).strip() for x in raw_wnf if str(x).strip()]
                et_prof = str(digest.get("estimated_timeline_cn") or "").strip()[:800] if digest else ""
                et_desk = (
                    str(desk.get("estimated_timeline_desk_cn") or "").strip()[:800]
                    if isinstance(desk, dict)
                    else ""
                )
                prev = (digest.get("professional_execution_plan") or "") if digest else ""
                pe_row: dict[str, Any] = {
                    "plan_index": pi,
                    "plain_full": plan,
                    "plain_excerpt": plan[:320] + ("…" if len(plan) > 320 else ""),
                    "understood_intent": (digest.get("understood_intent") or "")[:700] if digest else "",
                    "feasibility": digest.get("feasibility") if digest else None,
                    "feasibility_notes": (digest.get("feasibility_notes") or "")[:700] if digest else "",
                    "estimated_timeline_cn": et_prof,
                    "estimated_timeline_desk_cn": et_desk,
                    "professional_execution_plan_excerpt": (str(prev)[:2000] if prev else ""),
                    "desk_verdict": desk.get("desk_verdict") if isinstance(desk, dict) else None,
                    "desk_verdict_cn": (desk.get("desk_verdict_cn") or "")[:600]
                    if isinstance(desk, dict)
                    else "",
                    "safety_score": (
                        desk.get("safety_score")
                        if isinstance(desk, dict) and desk.get("safety_score") is not None
                        else (digest.get("safety_score") if isinstance(digest, dict) else None)
                    ),
                    "safety_summary": (
                        (desk.get("safety_summary") or "")[:600]
                        if isinstance(desk, dict)
                        else ((digest.get("safety_summary") or "")[:600] if isinstance(digest, dict) else "")
                    ),
                    "risk_items": (
                        desk.get("risk_items")
                        if isinstance(desk, dict) and isinstance(desk.get("risk_items"), list)
                        else (digest.get("risk_items") if isinstance(digest, dict) and isinstance(digest.get("risk_items"), list) else [])
                    ),
                    "why_not_feasible": wnf,
                    "caveat": (desk.get("caveat") or "")[:500] if isinstance(desk, dict) else "",
                }
                plan_evals.append(pe_row)
                if desk:
                    note = format_desk_feasibility_note(desk)
                    state.user_resolution_notes.append(note)
                    state.log(
                        "[公司内部桌面可行性·方案"
                        + str(pi + 1)
                        + "] "
                        + (str(desk.get("desk_verdict_cn") or "")[:360])
                    )
                    state.push_detail(
                        "internal_desk_feasibility",
                        f"公司内部桌面可行性（方案 {pi + 1}）",
                        str(desk.get("desk_verdict_cn") or desk.get("desk_verdict") or "")[:240],
                        {"desk": desk, "plan_index": pi},
                    )
                    _emit(
                        {
                            "phase": "internal_feasibility_digest",
                            "plan_index": pi,
                            "tick": state.tick,
                            "desk_verdict": desk.get("desk_verdict"),
                            "desk_verdict_cn": (desk.get("desk_verdict_cn") or "")[:500],
                            "safety_score": desk.get("safety_score"),
                            "safety_summary": (desk.get("safety_summary") or "")[:300],
                            "risk_items": (desk.get("risk_items") or [])[:6],
                            "why_not_feasible": wnf,
                            "social_linkage": (desk.get("social_linkage") or "")[:400],
                            "caveat": (desk.get("caveat") or "")[:400],
                            "estimated_timeline_desk_cn": et_desk[:500],
                        }
                    )
            _append_evaluated_plans_to_history(state, plan_evals)
            state.pending_desk_review = [dict(x) for x in plan_evals]
            state.last_risk_inventory = None
            if scenario.confirm_after_desk_feasibility:
                detail_body = _format_desk_review_detail(plan_evals)
                state.log("[待指挥台确认] 桌面研判已完成，需确认后方可继续演练。")
                pause_package = {
                    "scenario_dict": scenario.model_dump(),
                    "state_dict": state.model_dump(),
                    "rng_state": rng.getstate(),
                    "pause_kind": "desk_review",
                    "crisis_id": "__desk_confirm__",
                    "crisis_title": "方案可行性已研判 · 请确认后再继续演练",
                    "crisis_detail": detail_body,
                    "risk_inventory": None,
                    "plan_evaluations": plan_evals,
                    "seed": seed,
                    "decision_layer_active_summary": state.decision_layer_active_summary,
                    "decision_layer_history": state.decision_layer_history[-24:],
                    "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
                }
                state.simulation_outcome = "paused"
                return state, pause_package
    else:
        from people.regional_grounding import apply_regional_grounding, regional_grounding_artifact_dict

        scenario, rg_trace, rg_web = apply_regional_grounding(scenario, cfg)
        # materialize causal skeleton (if provided) into executable rules/edges
        try:
            cl = scenario.extensions.causal
            if getattr(cl, "skeleton", None):
                rules, edges, info = materialize_causal_layer_from_skeleton(cl.skeleton)  # type: ignore[arg-type]
                scenario = scenario.model_copy(
                    update={
                        "extensions": scenario.extensions.model_copy(
                            update={
                                "causal": scenario.extensions.causal.model_copy(
                                    update={"rules": rules, "edges": edges}
                                )
                            }
                        )
                    }
                )
                # best-effort: keep a trace for debugging/export
                try:
                    scenario.extensions.causal.skeleton = {**dict(cl.skeleton or {}), "_materialized": info}  # type: ignore[assignment]
                except Exception:
                    pass
        except Exception:
            # 若骨架无效：宁可继续跑（等同未启用 skeleton），避免影响既有场景；
            # 具体错误会在 causal_consistency / 上层校验中呈现。
            pass

        rng = random.Random(seed)
        state = build_initial_state(scenario, rng)
        state.regional_grounding_trace = list(rg_trace)
        rgn = (
            (scenario.regional_grounding.region_label or scenario.policy_context.jurisdiction_name or "")
            .strip()
        )
        if rg_trace or rg_web:
            state.regional_grounding_artifact = regional_grounding_artifact_dict(
                rg_trace, region=rgn, web_search=rg_web
            )
        _hydrate_finance(state, scenario)
        aggregate_from_cohorts(state, scenario.realism)
        finalize_tick(state)

        if internal_plan_commit and isinstance(internal_plan_commit, dict):
            from people.internal_plan_commit import apply_confirmed_internal_plan

            p = str(internal_plan_commit.get("plain_solution") or "").strip()
            if p:
                d = internal_plan_commit.get("evaluation_digest")
                k = internal_plan_commit.get("evaluation_desk")
                dk = internal_plan_commit.get("decision_kind")
                apply_confirmed_internal_plan(
                    state,
                    scenario,
                    plain=p,
                    digest=d if isinstance(d, dict) else {},
                    desk=k if isinstance(k, dict) else {},
                    decision_kind=str(dk or "general"),
                )

        pb = (scenario.player_brief or "").strip()
        if pb:
            label = "政策/措施" if scenario.exercise_type == "policy" else "产品/发行动作"
            state.log(f"[用户——{label}] {pb[:280]}{'…' if len(pb) > 280 else ''}")

        _emit(
            {
                "phase": "run_start",
                "scenario_name": scenario.name,
                "exercise_type": scenario.exercise_type,
                "total_ticks": scenario.ticks,
                "seed": seed,
                "cohort_llm": scenario.cohort_llm.enabled,
                "finance_enabled": scenario.operational_finance.enabled,
                "institution_count": len(scenario.institutions),
                "cooperation_count": len(scenario.cooperations),
                "crisis_rule_count": len(scenario.crisis_rules),
                "risk_milestone_primary_calls": scenario.risk_milestone_primary_calls,
                "confirm_after_desk_feasibility": scenario.confirm_after_desk_feasibility,
                "problem_salience": scenario.problem_salience,
                "regional_grounding_enabled": bool(scenario.regional_grounding.enabled),
                "regional_grounding_web_search": bool(scenario.regional_grounding.web_search_enabled),
                "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
            }
        )

    ticks_left = scenario.ticks - state.tick
    if ticks_left <= 0:
        state.simulation_outcome = "complete"
        return state, None

    for _ in range(ticks_left):
        state.tick += 1
        injected = _apply_runtime_black_swans("runtime")
        if injected:
            _emit({"phase": "black_swan_injected", "tick": state.tick, "events": injected})
        if runtime_manual_pause_reason is not None and state.tick < scenario.ticks:
            state.simulation_outcome = "paused"
            state.log("[待指挥台处置] 运行中手动暂停，等待输入新决策后续跑。")
            extra = (
                f"暂停理由：{runtime_manual_pause_reason}\n\n"
                if runtime_manual_pause_reason
                else ""
            )
            pause_package = {
                "scenario_dict": scenario.model_dump(),
                "state_dict": state.model_dump(),
                "rng_state": rng.getstate(),
                "pause_kind": "manual_pause",
                "crisis_id": "__manual_pause__",
                "crisis_title": "手动暂停（等待新增输入）",
                "crisis_detail": (
                    extra
                    + "已在运行中按请求暂停。请补充新的决策输入后继续推演；续跑会继承此前政策/运行态势，并叠加本次新增决策。"
                ),
                "risk_inventory": None,
                "seed": seed,
                "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
            }
            return state, pause_package
        from people.internal_plan_commit import fire_due_internal_commitments

        internal_ms = fire_due_internal_commitments(state, scenario)
        run_extension_plugins("tick_start", state, scenario, rng, ctx={})
        dctx = build_decision_context(scenario)
        augment_decision_context(dctx, state)
        intervention = _intervention_notes_block(state)
        macro_pre_env = macro_metrics_snapshot(state)
        apply_environment_drift(state, rng, scenario.realism)
        env_d = delta_snapshots(macro_pre_env, macro_metrics_snapshot(state))
        if env_d:
            record_attribution(
                state,
                layer="rules",
                component="environment_drift",
                tick=state.tick,
                deltas=env_d,
                meta={},
            )

        effects: list[dict[str, Any]] = []
        statements: list[str] = []
        tick_memory_entries: list[dict[str, Any]] = []
        tick_agents, fired_triggers = agents_for_tick(scenario, rng, state)
        if fired_triggers:
            state.log("[条件触发] " + ", ".join(fired_triggers))
        state.log("本轮 LLM 智能体 (" + str(len(tick_agents)) + "): " + ", ".join(a.id for a in tick_agents))

        _emit(
            {
                "phase": "tick_begin",
                "tick": state.tick,
                "total_ticks": scenario.ticks,
                "agent_count": len(tick_agents),
                "agent_ids": [a.id for a in tick_agents],
                "triggers_fired": list(fired_triggers),
                "internal_plan_milestones": internal_ms,
            }
        )

        tick_wall_start = time.perf_counter()
        perf = scenario.performance
        n_agents = len(tick_agents)
        ext_ctx: dict[str, Any] = {
            "n_primary_agents": 0,
            "interaction_count": 0,
            "primary_llm_calls_this_tick": 0,
        }
        primary_cap = int(perf.max_primary_llm_calls_per_tick) if perf.enabled else 0
        for idx, agent in enumerate(tick_agents):
            if primary_cap > 0 and int(ext_ctx.get("primary_llm_calls_this_tick", 0)) >= primary_cap:
                state.log(
                    f"[性能预算] 本 tick primary LLM 已达上限 {primary_cap}，跳过其余 {n_agents - idx} 名智能体"
                )
                break
            _emit(
                {
                    "phase": "agent_llm_start",
                    "tick": state.tick,
                    "index": idx + 1,
                    "total_in_tick": n_agents,
                    "agent_id": agent.id,
                    "role": agent.role,
                }
            )
            eff = run_key_actor_turn(
                cfg.primary_client,
                actor_id=agent.id,
                role=agent.role,
                goals=agent.goals,
                state=state,
                exercise_type=scenario.exercise_type,
                player_brief=scenario.player_brief,
                intervention_notes=intervention,
                persona=agent.persona,
                categories=list(agent.categories),
                simulation_context=dctx,
                temperature=cfg.key_actor_temperature,
            )
            state.primary_llm_calls_total += 1
            ext_ctx["primary_llm_calls_this_tick"] = int(ext_ctx.get("primary_llm_calls_this_tick", 0)) + 1
            effects.append(eff)
            st = eff.get("public_statement") if isinstance(eff, dict) else None
            if isinstance(st, str):
                if st.strip():
                    _append_key_actor_recall(state, agent.id, st)
                    _compact_actor_memory_if_needed(state, actor_id=agent.id)
                statements.append(f"{agent.id}: {st}")
            preview = (st[:220] + "…") if isinstance(st, str) and len(st) > 220 else (st if isinstance(st, str) else "")
            _maybe_media_detail(state, agent.id, eff if isinstance(eff, dict) else {})
            slant = eff.get("media_slant") if isinstance(eff, dict) else None
            hf = eff.get("headline_fragment") if isinstance(eff, dict) else None
            # 关键智能体本轮博弈动作（主体视角事件）
            try:
                impact_keys = []
                for key in ("sentiment_delta", "policy_support_delta", "rumor_delta", "unrest_delta"):
                    if isinstance(eff, dict) and isinstance(eff.get(key), (int, float)):
                        impact_keys.append(key.replace("_delta", ""))
                state.push_detail(
                    "actor_action",
                    f"{agent.role or agent.id}",
                    (st[:260] if isinstance(st, str) else "") or "本轮有动作（详见叙事与归因）。",
                    {
                        "actor_id": agent.id,
                        "actor_role": agent.role,
                        "categories": list(agent.categories),
                        "phase": "decide",
                        "channel": "primary_llm",
                        "impact_metrics": impact_keys,
                    },
                )
            except Exception:
                # 不因博弈视图失败影响主流程
                pass
            tick_memory_entries.append(
                {
                    "agent_id": agent.id,
                    "role": agent.role,
                    "statement_excerpt": (st[:500] if isinstance(st, str) else ""),
                    "headline_fragment": hf if isinstance(hf, str) else None,
                    "media_slant": slant,
                }
            )
            _emit(
                {
                    "phase": "agent_llm_done",
                    "tick": state.tick,
                    "agent_id": agent.id,
                    "agent_label": agent.role,
                    "statement_preview": preview,
                    "media_slant": slant,
                    "headline_fragment": hf if isinstance(hf, str) else None,
                }
            )

        ext_ctx["n_primary_agents"] = int(ext_ctx.get("primary_llm_calls_this_tick", 0))

        be = scenario.issuer.brand_equity if scenario.exercise_type == "product" else None
        ps = scenario.problem_salience if scenario.exercise_type == "product" else None
        for agent, eff in zip(tick_agents[: len(effects)], effects):
            macro_pre_one = macro_metrics_snapshot(state)
            apply_key_actor_effect_one(
                state,
                eff if isinstance(eff, dict) else {},
                rng,
                scenario.realism,
                exercise_type=scenario.exercise_type,
                brand_equity=be,
                problem_salience=ps,
            )
            one_d = delta_snapshots(macro_pre_one, macro_metrics_snapshot(state))
            if one_d:
                record_attribution(
                    state,
                    layer="llm",
                    component=f"key_actor:{agent.id}",
                    tick=state.tick,
                    deltas=one_d,
                    meta={"role": agent.role, "agent_id": agent.id},
                )
        for line in statements:
            state.log(line)
        if tick_memory_entries:
            for e in tick_memory_entries:
                state.company_memory_events.append({**e, "tick": state.tick})
            state.company_memory_events = state.company_memory_events[-120:]
            merge_company_memory_tick(
                cfg.fast_client,
                scenario=scenario,
                state=state,
                tick_entries=tick_memory_entries,
                temperature=0.25,
            )

        if scenario.cohort_llm.enabled:
            _emit({"phase": "cohort_batch_start", "tick": state.tick})
            deltas_map = run_cohort_batch_llm(
                cfg.fast_client,
                state=state,
                exercise_type=scenario.exercise_type,
                player_brief=scenario.player_brief,
                simulation_context=dctx,
                temperature=cfg.cohort_temperature,
            )
            macro_pre_coh = macro_metrics_snapshot(state)
            apply_cohort_deltas(state, deltas_map, rng, scenario.realism)
            coh_d = delta_snapshots(macro_pre_coh, macro_metrics_snapshot(state))
            record_attribution(
                state,
                layer="llm",
                component="cohort_batch",
                tick=state.tick,
                deltas=coh_d,
                meta={"cohort_ids": list(deltas_map.keys())[:24]},
            )
            _emit({"phase": "cohort_batch_done", "tick": state.tick})

        run_extension_plugins("after_cohort_llm", state, scenario, rng, ctx=ext_ctx)

        # 本轮跨主体交流：监管/政府/媒体/企业等之间的真实沟通动作
        interactions = run_social_interactions(
            cfg.fast_client,
            state=state,
            scenario=scenario,
            tick_agents=tick_agents,
            simulation_context=dctx,
            rng=rng,
            intervention_notes=intervention,
            temperature=0.3,
        )
        macro_pre_interactions = extended_metrics_snapshot(state)
        for it in interactions:
            _apply_interaction_budget(state, it)
            channel = str(it.get("channel") or "工作群")
            from_id = str(it.get("from_id") or "")
            to_id = str(it.get("to_id") or "")
            from_label = _actor_label(from_id, tick_agents, scenario, state)
            to_label = _actor_label(to_id, tick_agents, scenario, state)
            line = (
                f"{from_label} -> {to_label} [{channel}] "
                f"{it.get('summary')}"
            )
            state.log("[跨主体交流] " + line)
            state.push_detail(
                "inter_agent_dialogue",
                f"{from_label} ↔ {to_label}",
                f"{channel}：{it.get('summary')}",
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "from_label": from_label,
                    "to_label": to_label,
                    "kind": it.get("kind"),
                    "cost_million": it.get("cost_million"),
                    "fiscal_billion": it.get("fiscal_billion"),
                },
            )
            _emit(
                {
                    "phase": "interaction_event",
                    "tick": state.tick,
                    "from_id": from_id,
                    "to_id": to_id,
                    "from_label": from_label,
                    "to_label": to_label,
                    "channel": channel,
                    "kind": it.get("kind"),
                    "summary": it.get("summary"),
                    "cost_million": it.get("cost_million"),
                    "fiscal_billion": it.get("fiscal_billion"),
                }
            )

        int_d = delta_snapshots(macro_pre_interactions, extended_metrics_snapshot(state))
        if int_d:
            record_attribution(
                state,
                layer="llm",
                component="interaction_budgets",
                tick=state.tick,
                deltas=int_d,
                meta={"interaction_count": len(interactions)},
            )

        ext_ctx["interaction_count"] = len(interactions)
        run_extension_plugins("after_interactions", state, scenario, rng, ctx=ext_ctx)

        macro_pre_agg = macro_metrics_snapshot(state)
        aggregate_from_cohorts(state, scenario.realism)
        agg_deltas = delta_snapshots(macro_pre_agg, macro_metrics_snapshot(state))
        if agg_deltas:
            record_attribution(
                state,
                layer="rules",
                component="aggregate_from_cohorts",
                tick=state.tick,
                deltas=agg_deltas,
                meta={},
            )
        macro_pre_ledger = extended_metrics_snapshot(state)
        ledger_meta = _apply_operating_ledger(state, scenario)
        led_d = delta_snapshots(macro_pre_ledger, extended_metrics_snapshot(state))
        if led_d or ledger_meta:
            record_attribution(
                state,
                layer="rules",
                component="operating_ledger",
                tick=state.tick,
                deltas=led_d,
                meta=ledger_meta,
            )
        run_extension_plugins("pre_finalize", state, scenario, rng, ctx=ext_ctx)
        run_causal_governance_audit(state, scenario)
        inertia_d = apply_macro_inertia_blend(state, scenario.realism)
        if inertia_d:
            record_attribution(
                state,
                layer="rules",
                component="macro_inertia_blend",
                tick=state.tick,
                deltas=inertia_d,
                meta={"macro_inertia_blend": float(scenario.realism.macro_inertia_blend)},
            )
        finalize_tick(state)
        run_extension_plugins("after_finalize", state, scenario, rng, ctx=ext_ctx)

        skip_horizon = False
        if perf.enabled and perf.max_tick_wall_seconds > 0:
            elapsed = time.perf_counter() - tick_wall_start
            if elapsed > perf.max_tick_wall_seconds and perf.skip_horizon_if_over_budget:
                skip_horizon = True
                if perf.log_perf_warning:
                    state.log(
                        f"[性能预算] 本 tick wall {elapsed:.2f}s > {perf.max_tick_wall_seconds}s，跳过 horizon"
                    )
        if skip_horizon:
            state.horizon_forecast = {}
        else:
            try:
                state.horizon_forecast = run_horizon_forecast(
                    cfg.fast_client,
                    state=state,
                    scenario=scenario,
                    simulation_context=dctx,
                    temperature=0.35,
                )
            except Exception:
                state.horizon_forecast = {}

        _append_synthetic_details(state, scenario, rng)
        tick_details = [e for e in state.detail_events if e.get("tick") == state.tick]
        # 监管调查状态机：基于当前态势尝试新立案，并推进既有调查阶段
        try:
            _maybe_start_reg_investigations(state, scenario)
            _advance_reg_investigations(state)
        except Exception:
            # 不因监管调查视图失败影响主流程
            pass
        _append_timeline_uncertainty(state)
        _append_audit_chain(state, tick=state.tick)
        _update_layered_diffusion_proxy(state, tick_details)
        # causal identifiability (v0): periodic windowed checks, stored on state for audit/report
        try:
            cl = scenario.extensions.causal
            if cl.enabled and list(getattr(cl, "edges", []) or []) and state.tick % 3 == 0:
                rep = assess_edges_identifiability(
                    metrics_history=list(state.metrics_history or []),
                    edges=[e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in (cl.edges or [])],
                    window=18,
                )
                state.causal_identifiability.append({"tick": state.tick, **rep})
                state.causal_identifiability = state.causal_identifiability[-120:]
        except Exception:
            pass
        # 多主体博弈视图用的轻量级时间线摘要（只带当 tick 的关键主体事件）
        actor_timelines_tick: list[dict[str, Any]] = []
        for e in tick_details:
            kind = str(e.get("kind") or "")
            if kind not in ("actor_action", "inter_agent_dialogue", "reg_investigation_start", "reg_investigation_progress", "competition_action"):
                continue
            meta = e.get("meta") or {}
            actor_timelines_tick.append(
                {
                    "tick": state.tick,
                    "kind": kind,
                    "title": e.get("title"),
                    "body": e.get("body"),
                    "meta": meta,
                }
            )

        ongoing_investigations = [
            {
                "id": inv.get("id"),
                "status": inv.get("status"),
                "severity": inv.get("severity"),
                "next_deadline_tick": inv.get("next_deadline_tick"),
                "created_tick": inv.get("created_tick"),
            }
            for inv in (state.investigations or [])
            if inv.get("status") != "closed"
        ]

        tick_end_payload: dict[str, Any] = {
            "phase": "tick_end",
            "tick": state.tick,
            "metrics": state.snapshot_metrics(),
            "horizon": state.horizon_forecast,
            "detail_events_tick": tick_details,
            "primary_llm_calls_total": state.primary_llm_calls_total,
            "causal_rules_fired": list(state.causal_rules_fired_tick),
            "extension_trace_tail": list(state.extension_trace[-8:]),
            "attribution_tick": [e for e in state.attribution_log if int(e.get("tick", -1)) == state.tick][
                -48:
            ],
            "kpi_by_tier": dict(state.kpi_by_tier),
            "tick_wall_seconds": round(time.perf_counter() - tick_wall_start, 4),
            "timeline_uncertainty": (state.timeline_uncertainty[-1] if state.timeline_uncertainty else {}),
            "audit_hash": (state.audit_chain[-1].get("hash") if state.audit_chain else ""),
            "diffusion_layers": dict(state.diffusion_layer_state or {}),
            "actor_timelines_tick": actor_timelines_tick,
            "investigations": ongoing_investigations,
        }
        if scenario.decision_support.enabled:
            ds_hints = build_tick_decision_hints(state, scenario)
            if ds_hints:
                tick_end_payload["decision_support"] = ds_hints
        _emit(tick_end_payload)

        # 概率触发的「内部问题」：强调现实治理摩擦，并为用户创造决策介入点（非自动通关）。
        ii = scenario.internal_issues
        can_fire_internal_issue = (
            bool(ii.enabled)
            and state.tick < scenario.ticks
            and (state.tick - int(getattr(state, "internal_issue_last_tick", 0))) > int(ii.cooldown_ticks)
        )
        if can_fire_internal_issue:
            scope = str(getattr(ii, "sampling_scope", "auto") or "auto")
            if scope == "policy":
                p_issue = ii.probability_policy_per_tick
                issue_ex = "policy"
                issue_arc = "large_group"
            elif scope == "enterprise":
                p_issue = ii.probability_enterprise_per_tick
                issue_ex = "product"
                issue_arc = "large_group"
            elif scope == "startup":
                p_issue = ii.probability_startup_per_tick
                issue_ex = "product"
                issue_arc = "startup"
            elif scope == "shop":
                p_issue = ii.probability_shop_per_tick
                issue_ex = "product"
                issue_arc = "sme"
            elif scenario.exercise_type == "policy":
                p_issue = ii.probability_policy_per_tick
                issue_ex = scenario.exercise_type
                issue_arc = getattr(scenario.issuer, "archetype", "large_group")
            else:
                arc = getattr(scenario.issuer, "archetype", "large_group")
                if arc == "startup":
                    p_issue = ii.probability_startup_per_tick
                elif arc == "sme":
                    p_issue = ii.probability_shop_per_tick
                else:
                    p_issue = ii.probability_enterprise_per_tick
                issue_ex = scenario.exercise_type
                issue_arc = arc
            pid = maybe_pick_internal_issue_preset(
                exercise_type=issue_ex,
                issuer_archetype=issue_arc,
                rng=rng,
                probability=float(p_issue),
            )
            if pid:
                ev = build_black_swan_event(preset_id=pid, severity=1.0)
                d_issue = apply_black_swan_event(state, ev, source="internal_probabilistic")
                state.internal_issue_last_tick = state.tick
                record_attribution(
                    state,
                    layer="intervention",
                    component="internal_issue_probabilistic",
                    tick=state.tick,
                    deltas=d_issue,
                    meta={"preset_id": pid},
                )
                _emit(
                    {
                        "phase": "black_swan_injected",
                        "tick": state.tick,
                        "events": [
                            {
                                "title": ev.get("title"),
                                "preset_id": ev.get("preset_id"),
                                "severity": ev.get("severity"),
                                "deltas": d_issue,
                            }
                        ],
                    }
                )
                if ii.trigger_pause:
                    state.simulation_outcome = "paused"
                    state.log(f"[待指挥台处置] 内部问题触发：{ev.get('title')}")
                    pause_package = {
                        "scenario_dict": scenario.model_dump(),
                        "state_dict": state.model_dump(),
                        "rng_state": rng.getstate(),
                        "pause_kind": "internal_issue",
                        "crisis_id": f"__internal_issue__{pid}",
                        "crisis_title": f"内部问题触发（概率事件）· {ev.get('title')}",
                        "crisis_detail": str(ev.get("detail") or "内部治理风险已触发，请给出处置方案后续跑。"),
                        "risk_inventory": None,
                        "seed": seed,
                        "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
                    }
                    return state, pause_package

        crisis = pick_crisis(scenario, state)
        # 仅在「后面还有 tick」时暂停，否则续跑无迭代空间
        if crisis is not None and state.tick < scenario.ticks:
            state.simulation_outcome = "paused"
            state.log(f"[待指挥台处置] {crisis.title} — {crisis.detail}")
            pause_package = {
                "scenario_dict": scenario.model_dump(),
                "state_dict": state.model_dump(),
                "rng_state": rng.getstate(),
                "pause_kind": "crisis_rule",
                "crisis_id": crisis.id,
                "crisis_title": crisis.title,
                "crisis_detail": crisis.detail,
                "risk_inventory": None,
                "seed": seed,
                "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
            }
            return state, pause_package

        ms = scenario.risk_milestone_primary_calls
        if (
            ms > 0
            and state.primary_llm_calls_total >= ms
            and not state.risk_milestone_shown
            and state.tick < scenario.ticks
        ):
            inv = run_risk_inventory_llm(
                cfg.fast_client,
                scenario=scenario,
                state=state,
                decision_context=dctx,
                primary_calls_so_far=state.primary_llm_calls_total,
                temperature=0.35,
            )
            state.last_risk_inventory = inv
            state.risk_milestone_shown = True
            detail_text = format_risk_inventory_for_pause(inv)
            state.log("[内部风险盘点] 关键推演次数已达阈值，暂停等待指挥台应对方案。")
            state.push_detail(
                "risk_milestone",
                f"内部风险盘点（已满 {ms} 次关键推演）",
                str(inv.get("summary") or "")[:220],
                {"inventory": inv},
            )
            state.simulation_outcome = "paused"
            pause_package = {
                "scenario_dict": scenario.model_dump(),
                "state_dict": state.model_dump(),
                "rng_state": rng.getstate(),
                "pause_kind": "risk_milestone",
                "crisis_id": "__risk_milestone__",
                "crisis_title": f"关键推演已满 {ms} 次 · 内部风险盘点（演练生成）",
                "crisis_detail": detail_text,
                "risk_inventory": inv,
                "seed": seed,
                "experiment_manifest": _experiment_manifest_for_run(scenario, seed, cfg),
            }
            return state, pause_package

    state.simulation_outcome = "complete"
    return state, None


def _worker_run(payload: tuple[str, int, str, str]) -> dict[str, Any]:
    from people.providers.registry import get_client

    scenario_json, seed, primary_spec, fast_spec = payload
    scenario = Scenario.model_validate_json(scenario_json)
    cfg = RunConfig(
        primary_client=get_client(primary_spec, allow_dry=True),
        fast_client=get_client(fast_spec, allow_dry=True),
        primary_model_slot=primary_spec,
        fast_model_slot=fast_spec,
    )
    final, _pause = run_single_simulation(scenario, cfg, seed=seed)
    return {
        "seed": seed,
        "final_metrics": final.snapshot_metrics(),
        "narrative": final.narrative[-16:],
    }


def run_ensemble_parallel(
    scenario: Scenario,
    *,
    n_runs: int,
    base_seed: int,
    primary_spec: str,
    fast_spec: str,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    scenario_json = scenario.model_dump_json()
    seeds = [base_seed + i for i in range(n_runs)]
    args_list = [(scenario_json, s, primary_spec, fast_spec) for s in seeds]
    if n_runs == 1:
        return [_worker_run(args_list[0])]

    results: list[dict[str, Any]] = []
    # Windows spawn：需可导入 people
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker_run, a) for a in args_list]
        for fu in as_completed(futs):
            results.append(fu.result())
    results.sort(key=lambda r: r["seed"])
    return results


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
    """p∈[0,1]，线性插值分位（与常见 numpy percentile 行为接近）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    x = p * (n - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return sorted_vals[lo]
    w = x - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def summarize_ensemble(
    results: list[dict[str, Any]],
    *,
    threshold_key: str = "policy_support",
    threshold: float = 0.55,
    scenario: Scenario | None = None,
    model_slots: dict[str, str] | None = None,
) -> dict[str, Any]:
    hits = sum(1 for r in results if r["final_metrics"].get(threshold_key, 0) >= threshold)
    n = max(len(results), 1)
    avg_support = sum(r["final_metrics"].get("policy_support", 0) for r in results) / n
    avg_unrest = sum(r["final_metrics"].get("unrest", 0) for r in results) / n

    dist_keys = [
        "policy_support",
        "unrest",
        "issuer_trust_proxy",
        "sentiment",
        "rumor_level",
        "supply_chain_stress",
        "economy_index",
        "diffusion_i_share",
        "resource_manpower",
        "resource_political_capital",
    ]
    distributions: dict[str, Any] = {}
    for k in dist_keys:
        vals = []
        for r in results:
            m = r.get("final_metrics") or {}
            v = m.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if len(vals) < 2:
            if vals:
                v0 = vals[0]
                distributions[k] = {
                    "n": 1,
                    "mean": v0,
                    "std": 0.0,
                    "cv": None,
                    "p01": v0,
                    "p05": v0,
                    "p10": v0,
                    "p50": v0,
                    "p90": v0,
                    "p95": v0,
                    "p99": v0,
                    "min": v0,
                    "max": v0,
                }
            continue
        vals.sort()
        nv = len(vals)
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if nv > 1 else 0.0
        cv = (std / abs(mean)) if abs(mean) > 1e-9 else None

        distributions[k] = {
            "n": nv,
            "mean": mean,
            "std": std,
            "cv": cv,
            "p01": _percentile_linear(vals, 0.01),
            "p05": _percentile_linear(vals, 0.05),
            "p10": _percentile_linear(vals, 0.10),
            "p50": _percentile_linear(vals, 0.50),
            "p90": _percentile_linear(vals, 0.90),
            "p95": _percentile_linear(vals, 0.95),
            "p99": _percentile_linear(vals, 0.99),
            "min": vals[0],
            "max": vals[-1],
        }

    base = {
        "n": len(results),
        "p_estimate": hits / n,
        "threshold_key": threshold_key,
        "threshold": threshold,
        "mean_policy_support": avg_support,
        "mean_unrest": avg_unrest,
        "distributions": distributions,
    }
    from people.stability_report import enrich_ensemble_summary

    return enrich_ensemble_summary(
        base,
        results,
        threshold_key=threshold_key,
        scenario=scenario,
        model_slots=model_slots,
    )
