"""
网页/API 传入的可选字段与已加载的 Scenario 合并（不写回 YAML）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pepole.config import AdminLevel, IssuerArchetype, Scenario, WebSearchProvider


class ScenarioOverridePayload(BaseModel):
    """与 RunRequest / PlanEvaluateRequest 共用；仅非空字段覆盖底稿。"""

    exercise_type: Literal["policy", "product"] | None = None
    ticks: int | None = Field(default=None, ge=1)
    problem_salience: float | None = Field(default=None, ge=0.0, le=1.0)
    issuer_archetype: IssuerArchetype | None = None
    brand_equity: float | None = Field(default=None, ge=0.0, le=1.0)
    reputation_brief: str | None = None
    supply_chain_position: str | None = None
    policy_admin_level: AdminLevel | None = None
    policy_jurisdiction_name: str | None = None
    local_norms_brief: str | None = None
    media_environment_brief: str | None = None
    # —— 地区情境检索（模拟前多轮摘要）——
    regional_grounding_enabled: bool | None = None
    regional_grounding_region: str | None = None
    regional_grounding_mode: Literal["llm_only", "wikipedia_then_llm"] | None = None
    regional_grounding_web_search: bool | None = None
    web_search_provider: WebSearchProvider | None = None
    business_scale: Literal["unset", "street_shop", "sme_chain", "regional_group", "national_group"] | None = None
    business_sector_brief: str | None = None
    user_known_local_policy_brief: str | None = None
    user_known_central_policy_brief: str | None = None


def merge_scenario_overrides(
    scenario: Scenario,
    overrides: ScenarioOverridePayload,
    *,
    player_brief: str | None = None,
) -> Scenario:
    """player_brief 来自请求顶层 brief 字段；与 overrides 一并应用。"""
    updates: dict[str, Any] = {}
    if player_brief is not None and player_brief.strip():
        updates["player_brief"] = player_brief.strip()

    o = overrides
    if o.exercise_type is not None:
        updates["exercise_type"] = o.exercise_type
    if o.ticks is not None:
        updates["ticks"] = o.ticks
    if o.problem_salience is not None:
        updates["problem_salience"] = o.problem_salience

    iss_up: dict[str, Any] = {}
    if o.issuer_archetype is not None:
        iss_up["archetype"] = o.issuer_archetype
    if o.brand_equity is not None:
        iss_up["brand_equity"] = o.brand_equity
    if o.reputation_brief is not None:
        iss_up["reputation_brief"] = o.reputation_brief.strip()
    if o.supply_chain_position is not None:
        iss_up["supply_chain_position"] = o.supply_chain_position.strip()
    if iss_up:
        updates["issuer"] = scenario.issuer.model_copy(update=iss_up)

    pc_up: dict[str, Any] = {}
    if o.policy_admin_level is not None:
        pc_up["admin_level"] = o.policy_admin_level
    if o.policy_jurisdiction_name is not None:
        pc_up["jurisdiction_name"] = o.policy_jurisdiction_name.strip()
    if o.local_norms_brief is not None:
        pc_up["local_norms_brief"] = o.local_norms_brief.strip()
    if o.media_environment_brief is not None:
        pc_up["media_environment_brief"] = o.media_environment_brief.strip()
    if pc_up:
        updates["policy_context"] = scenario.policy_context.model_copy(update=pc_up)

    rg = scenario.regional_grounding
    rg_up: dict[str, Any] = {}
    if o.regional_grounding_enabled is not None:
        rg_up["enabled"] = o.regional_grounding_enabled
    if o.regional_grounding_region is not None and o.regional_grounding_region.strip():
        rg_up["region_label"] = o.regional_grounding_region.strip()
    if o.regional_grounding_mode is not None:
        rg_up["mode"] = o.regional_grounding_mode
    if o.web_search_provider is not None:
        rg_up["web_search_provider"] = o.web_search_provider
        if o.regional_grounding_web_search is not False:
            rg_up["web_search_enabled"] = True
    if o.regional_grounding_web_search is not None:
        rg_up["web_search_enabled"] = o.regional_grounding_web_search
    if o.business_scale is not None:
        rg_up["business_scale"] = o.business_scale
    if o.business_sector_brief is not None:
        rg_up["business_sector_brief"] = o.business_sector_brief.strip()
    if o.user_known_local_policy_brief is not None:
        rg_up["user_known_local_policy_brief"] = o.user_known_local_policy_brief.strip()
    if o.user_known_central_policy_brief is not None:
        rg_up["user_known_central_policy_brief"] = o.user_known_central_policy_brief.strip()
    if rg_up:
        updates["regional_grounding"] = rg.model_copy(update=rg_up)

    if updates:
        return scenario.model_copy(update=updates)
    return scenario


def override_payload_from_model(model: Any) -> ScenarioOverridePayload:
    """从 RunRequest / PlanEvaluateRequest 等同构模型提取覆盖块。"""
    kw = {name: getattr(model, name, None) for name in ScenarioOverridePayload.model_fields}
    return ScenarioOverridePayload(**kw)
