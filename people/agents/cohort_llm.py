from __future__ import annotations

import json
from typing import Any

from people.agents.key_actor import GROUNDING
from people.providers.base import LLMClient
from people.state import WorldState


COHORT_BATCH_SYSTEM_POLICY = """你是舆情建模助手（**社会调查式粗粒度 proxy**，非个体预测）。输入含政策制定者的 player_brief、decision_context（行政层级/地域民风/参考案例）与各 cohort。
须区分 class_layer（基层/中产/高层 proxy）的反应逻辑与信息半径。真实人群中态度变化多为缓慢、嘈杂、区域差异大；根据政策与宏观变量，为每个 cohort 输出态度与激活度增量。
若 decision_context 含 regional_grounding：摘要为**既有地方政策与基层常态**；本轮新政策见 player_brief 后部「指挥台」——请输出**在旧常态之上的增量**，勿写成从未有过政策环境。
只输出 JSON：{ "cohorts": [ { "id": "...", "attitude_delta": number, "activation_delta": number }, ... ] }
id 必须与输入一致。增量范围建议 attitude_delta ∈ [-0.035,0.035]，activation_delta ∈ [-0.028,0.028]；除非输入指标已极端恶化，否则避免单周剧烈翻转。
""" + GROUNDING

COHORT_BATCH_SYSTEM_PRODUCT = """你是市场反响建模助手（**人群统计倾向**，不预测个人购买决策）。输入含产品发行方 player_brief、decision_context.issuer（体量/品牌资产/声誉/供应链位置）与各 cohort（含 class_layer）。
同一舆情对基层、中产、高层的含义不同；品牌资产高也不等于基层自动买单。真实卖场与社群里多为碎嘴、跟风与沉默大多数并存。据此输出每个 cohort 对品牌/产品态度与发声意愿的增量。
若 decision_context 含 regional_grounding：先验含地方监管与中央框架；player_brief 后部为**新动作**（如新开火锅店/扩店）——人群反应须体现**对熟悉营商环境上的边际变化**。
只输出 JSON：{ "cohorts": [ { "id": "...", "attitude_delta": number, "activation_delta": number }, ... ] }
id 必须与输入一致。增量范围建议 attitude_delta ∈ [-0.035,0.035]，activation_delta ∈ [-0.028,0.028]。
""" + GROUNDING


def run_cohort_batch_llm(
    client: LLMClient,
    *,
    state: WorldState,
    exercise_type: str = "policy",
    player_brief: str = "",
    simulation_context: dict[str, Any] | None = None,
    temperature: float = 0.35,
) -> dict[str, dict[str, float]]:
    sys = COHORT_BATCH_SYSTEM_POLICY if exercise_type != "product" else COHORT_BATCH_SYSTEM_PRODUCT
    ids = [c.id for c in state.cohorts]
    brief = (player_brief or "").strip() or "（未提供 player_brief。）"
    payload = {
        "exercise_type": exercise_type,
        "player_brief": brief,
        "tick": state.tick,
        "macro": state.snapshot_metrics(),
        "cohorts": [
            {
                "id": c.id,
                "class_layer": c.class_layer,
                "weight": c.weight,
                "attitude": c.attitude,
                "activation": c.activation,
                "traits": c.traits,
            }
            for c in state.cohorts
        ],
        "decision_context": simulation_context or {},
    }
    user = "输入：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = client.complete_json(system=sys, user=user, temperature=temperature)
    out: dict[str, dict[str, float]] = {}
    items = raw.get("cohorts")
    if not isinstance(items, list):
        return out
    valid = set(ids)
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("id")
        if cid not in valid:
            continue
        out[str(cid)] = {
            "attitude_delta": float(it.get("attitude_delta", 0.0)),
            "activation_delta": float(it.get("activation_delta", 0.0)),
        }
    return out
