"""
经营台账：按 tick 扣运营支出（可按行业×体量缩放）、写意营收税费、债务本息；
政策沙盘：财力池支出 + 与经济景气挂钩的写意税收入库。
"""

from __future__ import annotations

from typing import Any

from people.config import Scenario
from people.state import WorldState

# 叙事上每 tick ≈ 1～2 周，写意按 26 tick ≈ 1 年折息
TICKS_PER_YEAR = 26.0

_ISSUER_OPEX_MULT: dict[str, float] = {
    "startup": 0.2,
    "sme": 0.5,
    "large_group": 1.0,
    "megacorp": 1.22,
}


def effective_operating_cost_million_per_tick(scenario: Scenario) -> float:
    of = scenario.operational_finance
    base = float(of.operating_cost_million_per_tick) * float(of.industry_operating_cost_multiplier)
    if of.auto_scale_operating_cost_from_issuer:
        base *= float(_ISSUER_OPEX_MULT.get(scenario.issuer.archetype, 1.0))
    return max(0.0, base)


def apply_tick_finance_ledger(state: WorldState, scenario: Scenario) -> dict[str, Any]:
    """
    就地修改 state 的现金/财力/债务；返回写入 attribution meta 用的摘要。
    """
    of = scenario.operational_finance
    if not of.enabled:
        return {}

    meta: dict[str, Any] = {"component": "tick_finance_ledger"}

    if scenario.exercise_type == "policy":
        if state.fiscal_remaining_billion is not None:
            spend = float(of.policy_spend_per_tick_billion)
            econ = max(0.0, min(1.0, float(state.economy_index)))
            tax_in = float(of.fiscal_tax_inflow_from_economy_billion_per_tick) * econ
            before = float(state.fiscal_remaining_billion)
            state.fiscal_remaining_billion = max(0.0, before - spend + tax_in)
            meta["policy_spend_billion"] = round(spend, 6)
            meta["fiscal_tax_inflow_billion"] = round(tax_in, 6)
            meta["fiscal_net_billion"] = round(float(state.fiscal_remaining_billion) - before, 6)
        return meta

    # product
    if state.cash_balance_million is None:
        return meta

    opex = effective_operating_cost_million_per_tick(scenario)
    interest_m = 0.0
    if state.debt_balance_million is not None and float(of.debt_interest_annual_rate_proxy) > 0:
        interest_m = float(state.debt_balance_million) * (
            float(of.debt_interest_annual_rate_proxy) / TICKS_PER_YEAR
        )
    repay = float(of.debt_principal_repay_million_per_tick)
    revenue = max(0.0, float(of.revenue_proxy_million_per_tick))
    tax_m = revenue * max(0.0, min(0.6, float(of.effective_tax_rate_on_revenue)))

    total_out = opex + interest_m + repay + tax_m
    before_cash = float(state.cash_balance_million)
    state.cash_balance_million = max(0.0, before_cash - total_out)

    if state.debt_balance_million is not None:
        state.debt_balance_million = max(0.0, float(state.debt_balance_million) - repay)

    mirror_b = 0.0
    if (
        state.fiscal_remaining_billion is not None
        and of.track_parallel_fiscal_pool
        and tax_m > 0.0
    ):
        mirror_b = (tax_m / 1000.0) * max(0.0, min(1.0, float(of.company_tax_mirror_to_fiscal_fraction)))
        state.fiscal_remaining_billion = float(state.fiscal_remaining_billion) + mirror_b

    meta.update(
        {
            "operating_cost_million": round(opex, 4),
            "debt_interest_million": round(interest_m, 4),
            "debt_principal_repay_million": round(repay, 4),
            "tax_on_revenue_million": round(tax_m, 4),
            "revenue_proxy_million": round(revenue, 4),
            "cash_out_total_million": round(total_out, 4),
            "tax_mirrored_to_fiscal_billion": round(mirror_b, 6),
        }
    )
    return meta
