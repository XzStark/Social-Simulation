from __future__ import annotations

from typing import Any

from pepole.config import Scenario
from pepole.state import WorldState


def build_decision_context(scenario: Scenario) -> dict[str, Any]:
    out: dict[str, Any] = {
        "policy_context": scenario.policy_context.model_dump(),
        "issuer": scenario.issuer.model_dump(),
        "problem_salience": scenario.problem_salience,
    }
    ref = (scenario.reference_cases_brief or "").strip()
    if ref:
        out["reference_cases_brief"] = ref
    of = scenario.operational_finance
    if of.enabled:
        from pepole.finance_ledger import effective_operating_cost_million_per_tick

        out["operational_finance"] = {
            "enabled": True,
            "exercise_type": scenario.exercise_type,
            "cash_balance_million": of.cash_balance_million if scenario.exercise_type == "product" else None,
            "operating_cost_million_per_tick": of.operating_cost_million_per_tick
            if scenario.exercise_type == "product"
            else None,
            "effective_operating_cost_million_per_tick": effective_operating_cost_million_per_tick(scenario)
            if scenario.exercise_type == "product"
            else None,
            "revenue_proxy_million_per_tick": of.revenue_proxy_million_per_tick
            if scenario.exercise_type == "product"
            else None,
            "effective_tax_rate_on_revenue": of.effective_tax_rate_on_revenue
            if scenario.exercise_type == "product"
            else None,
            "debt_principal_million": of.debt_principal_million if scenario.exercise_type == "product" else None,
            "debt_interest_annual_rate_proxy": of.debt_interest_annual_rate_proxy
            if scenario.exercise_type == "product"
            else None,
            "fiscal_pool_billion": of.fiscal_pool_billion if scenario.exercise_type == "policy" else None,
            "policy_spend_per_tick_billion": of.policy_spend_per_tick_billion
            if scenario.exercise_type == "policy"
            else None,
            "fiscal_tax_inflow_from_economy_billion_per_tick": of.fiscal_tax_inflow_from_economy_billion_per_tick
            if scenario.exercise_type == "policy"
            else None,
            "track_parallel_fiscal_pool": of.track_parallel_fiscal_pool
            if scenario.exercise_type == "product"
            else None,
        }
    if scenario.institutions:
        out["institutions"] = [i.model_dump() for i in scenario.institutions]
    if scenario.cooperations:
        out["cooperations"] = [c.model_dump() for c in scenario.cooperations]
    if scenario.exercise_type == "product" and scenario.market_competitors:
        out["market_competitors"] = [m.model_dump() for m in scenario.market_competitors]
    return out


def enrich_decision_context_for_plan_evaluate(
    dctx: dict[str, Any],
    state: WorldState,
    scenario: Scenario,
    *,
    max_narrative_lines: int = 28,
    max_cohort_rows: int = 24,
) -> None:
    """
    将当前 WorldState 的实盘资金、资源池、宏观、cohort 与叙事并入 decision_context，
    供白话处置专业转化与内部桌面研判**综合**引用（与 YAML 静态 operational_finance 互补）。
    """
    finance_live: dict[str, float] = {}
    if state.cash_balance_million is not None:
        finance_live["cash_balance_million"] = float(state.cash_balance_million)
    if state.fiscal_remaining_billion is not None:
        finance_live["fiscal_remaining_billion"] = float(state.fiscal_remaining_billion)
    if state.debt_balance_million is not None:
        finance_live["debt_balance_million"] = float(state.debt_balance_million)
    pools: dict[str, float] = {}
    if state.resource_manpower is not None:
        pools["resource_manpower"] = float(state.resource_manpower)
    if state.resource_political_capital is not None:
        pools["resource_political_capital"] = float(state.resource_political_capital)

    live: dict[str, Any] = {
        "tick": int(state.tick),
        "macro": state.snapshot_metrics(),
        "narrative_tail": list(state.narrative[-max_narrative_lines:]) if state.narrative else [],
        "cohorts_snapshot": [
            {
                "id": c.id,
                "class_layer": c.class_layer,
                "attitude": float(c.attitude),
                "activation": float(c.activation),
                "weight": float(c.weight),
            }
            for c in state.cohorts[:max_cohort_rows]
        ],
        "operational_finance_yaml_enabled": bool(scenario.operational_finance.enabled),
        "simulation_outcome": getattr(state, "simulation_outcome", "complete"),
    }
    if finance_live:
        live["finance_state_live"] = finance_live
    if pools:
        live["resource_pools_live"] = pools
    dctx["plan_evaluate_live_context"] = live
