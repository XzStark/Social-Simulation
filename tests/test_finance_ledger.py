from __future__ import annotations

import random

from pepole.config import Scenario
from pepole.engine import build_initial_state
from pepole.finance_ledger import apply_tick_finance_ledger, effective_operating_cost_million_per_tick


def _product_finance_scenario(**of_kw) -> Scenario:
    raw = Scenario.load("scenarios/startup_example.yaml")
    of = raw.operational_finance.model_copy(
        update={
            "enabled": True,
            "cash_balance_million": 100.0,
            "operating_cost_million_per_tick": 10.0,
            "revenue_proxy_million_per_tick": 20.0,
            "effective_tax_rate_on_revenue": 0.1,
            "debt_principal_million": 40.0,
            "debt_interest_annual_rate_proxy": 0.13,
            "debt_principal_repay_million_per_tick": 1.0,
            **of_kw,
        }
    )
    return raw.model_copy(update={"operational_finance": of})


def test_effective_opex_issuer_scale() -> None:
    s = _product_finance_scenario(auto_scale_operating_cost_from_issuer=True)
    assert effective_operating_cost_million_per_tick(s) < 10.0
    s2 = s.model_copy(
        update={
            "issuer": s.issuer.model_copy(update={"archetype": "megacorp"}),
            "operational_finance": s.operational_finance.model_copy(
                update={"auto_scale_operating_cost_from_issuer": True}
            ),
        }
    )
    assert effective_operating_cost_million_per_tick(s2) > effective_operating_cost_million_per_tick(s)


def test_product_ledger_tax_interest_repay() -> None:
    s = _product_finance_scenario(track_parallel_fiscal_pool=True, parallel_fiscal_pool_initial_billion=0.1)
    rng = random.Random(1)
    st = build_initial_state(s, rng)
    st.cash_balance_million = 100.0
    st.debt_balance_million = 40.0
    st.fiscal_remaining_billion = 0.1
    cash0 = float(st.cash_balance_million)
    debt0 = float(st.debt_balance_million)
    fis0 = float(st.fiscal_remaining_billion or 0)
    meta = apply_tick_finance_ledger(st, s)
    assert st.cash_balance_million < cash0
    assert st.debt_balance_million == debt0 - 1.0
    assert st.fiscal_remaining_billion is not None and float(st.fiscal_remaining_billion) > fis0
    assert "tax_on_revenue_million" in meta


def test_policy_fiscal_tax_inflow() -> None:
    s = Scenario.load("scenarios/policy_county_example.yaml")
    rng = random.Random(0)
    st = build_initial_state(s, rng)
    st.fiscal_remaining_billion = 10.0
    st.economy_index = 0.8
    before = float(st.fiscal_remaining_billion)
    meta = apply_tick_finance_ledger(st, s)
    assert st.fiscal_remaining_billion is not None
    assert float(st.fiscal_remaining_billion) != before
    assert "fiscal_tax_inflow_billion" in meta
