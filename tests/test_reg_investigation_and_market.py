from __future__ import annotations

import random

from people.config import CohortSpec, Scenario
from people.engine import _update_layered_diffusion_proxy, build_initial_state
from people.extension_stack import init_extensions_state, run_extension_plugins
from people.state import WorldState


def _make_state_for_product() -> tuple[WorldState, Scenario]:
    s = Scenario(name="t", exercise_type="product", ticks=6, cohorts=[CohortSpec(id="customers", weight=1.0)])
    st = build_initial_state(s, random.Random(0))
    init_extensions_state(st, s)
    return st, s


def test_market_finance_proxies_populated_via_kpi_plugin() -> None:
    st, s = _make_state_for_product()
    # Extensions are opt-in; exercise the KPI plugin with its switch enabled.
    s.extensions.kpi.enabled = True
    # 模拟一次 pre_finalize 阶段，驱动 KPI 插件与市场 proxy
    rng = random.Random(0)
    run_extension_plugins("pre_finalize", st, s, rng, ctx={})
    assert st.price_proxy is not None
    assert st.volume_proxy is not None
    assert st.market_share_proxy is not None
    assert st.roi_proxy is not None


def test_reg_investigation_fsm_creates_and_advances(tmp_path) -> None:
    # 高 risk proxy：强制触发一次监管调查
    st, s = _make_state_for_product()
    st.tick = 1
    st.rumor_level = 0.9
    st.unrest = 0.85
    # 通过一次 tick_end 附近的调用链：更新 diffusion 后，调用风险与监管逻辑
    _update_layered_diffusion_proxy(st, [])
    # 手动调用内部 helper：通过 import 规避循环
    from people.engine import _advance_reg_investigations, _maybe_start_reg_investigations

    _maybe_start_reg_investigations(st, s)
    assert any(inv.get("status") == "initiated" for inv in st.investigations)
    # 推进到下一 tick，触发状态机前进
    st.tick = 2
    _advance_reg_investigations(st)
    statuses = {inv.get("status") for inv in st.investigations}
    assert "pre_check" in statuses or "in_depth" in statuses
