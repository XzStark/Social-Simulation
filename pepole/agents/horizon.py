from __future__ import annotations

import json
from typing import Any

from pepole.config import Scenario
from pepole.providers.base import LLMClient
from pepole.state import WorldState

from pepole.agents.realism_layer import REALISM_SOCIAL_LAYER

HORIZON_SYSTEM = (
    """
你是政策与企业内参写手，用克制、可核查的表述写**情境分支下的走势讨论**（不是对未来的精准预测，不得写死结局）。
必须输出一个 JSON 对象，键如下（字符串可为 1～3 句中文；不需要的键填 null）：
{
  "today": "string",
  "next_month": "string",
  "next_year": "string",
  "two_years": "string | null",
  "three_years": "string | null"
}
规则：
- today：对应「当前这一两周」最可能出现的舆论/市场/执行层面动态。
- next_month、next_year：宏观走势与主要风险，避免具体股价与未证实爆料。
- two_years、three_years：仅当 issuer_trust_proxy 与 policy_support（或政策场景下对施政接受度）均处于「明显偏强」时填写实质性中长期展望；否则必须 null。
- 全文多用「若维持当前摩擦强度」「在信息不进一步恶化的前提下」等**条件句**；避免「一年后一定如何」。
禁止游戏隐喻；禁止把未证实信息写成定论；禁用元宇宙/科幻叙事形容日常政策与消费议题。
展望须与**部门协作迟滞、人群分化**相容：可写「若执行层仍不同步则…」「若基层信息仍碎片化则…」，勿写单一结局时间表。
"""
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()


def run_horizon_forecast(
    client: LLMClient,
    *,
    state: WorldState,
    scenario: Scenario,
    simulation_context: dict[str, Any],
    temperature: float = 0.35,
) -> dict[str, str | None]:
    ps = float(state.policy_support)
    tr = float(state.issuer_trust_proxy)
    allow_long = ps >= 0.54 and tr >= 0.52
    payload = {
        "horizon_axes": True,
        "exercise_type": scenario.exercise_type,
        "tick": state.tick,
        "macro": state.snapshot_metrics(),
        "allow_two_and_three_year_outlook": allow_long,
        "player_brief_excerpt": (scenario.player_brief or "")[:1200],
        "decision_context": simulation_context,
    }
    user = "当前态势 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = client.complete_json(system=HORIZON_SYSTEM, user=user, temperature=temperature)
    if not isinstance(raw, dict):
        return {}
    keys = ("today", "next_month", "next_year", "two_years", "three_years")
    out: dict[str, str | None] = {}
    for k in keys:
        v = raw.get(k)
        out[k] = v if isinstance(v, str) and v.strip() else None
    if not allow_long:
        out["two_years"] = None
        out["three_years"] = None
    return out
