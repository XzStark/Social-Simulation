"""operational_finance 默认现金/负债与 issuer 体量写意对齐。"""

from __future__ import annotations

import random

from pepole.config import OperationalFinance, Scenario
from pepole.engine import _hydrate_finance, build_initial_state


def test_hydrate_scales_default_cash_debt_for_startup() -> None:
    s = Scenario(
        name="t",
        exercise_type="product",
        ticks=5,
        issuer={"archetype": "startup"},
        operational_finance=OperationalFinance(
            enabled=True,
            cash_balance_million=800.0,
            debt_principal_million=0.0,
            scale_cash_debt_from_issuer_if_defaults=True,
        ),
    )
    st = build_initial_state(s, random.Random(0))
    _hydrate_finance(st, s)
    assert st.cash_balance_million == 42.0
    assert st.debt_balance_million == 12.0


def test_explicit_cash_not_overridden() -> None:
    s = Scenario(
        name="t",
        exercise_type="product",
        ticks=5,
        issuer={"archetype": "startup"},
        operational_finance=OperationalFinance(
            enabled=True,
            cash_balance_million=500.0,
            debt_principal_million=0.0,
            scale_cash_debt_from_issuer_if_defaults=True,
        ),
    )
    st = build_initial_state(s, random.Random(0))
    _hydrate_finance(st, s)
    assert st.cash_balance_million == 500.0
    assert st.debt_balance_million == 0.0


def test_nonzero_debt_disables_default_scale() -> None:
    s = Scenario(
        name="t",
        exercise_type="product",
        ticks=5,
        issuer={"archetype": "megacorp"},
        operational_finance=OperationalFinance(
            enabled=True,
            cash_balance_million=800.0,
            debt_principal_million=10.0,
            scale_cash_debt_from_issuer_if_defaults=True,
        ),
    )
    st = build_initial_state(s, random.Random(0))
    _hydrate_finance(st, s)
    assert st.cash_balance_million == 800.0
    assert st.debt_balance_million == 10.0
