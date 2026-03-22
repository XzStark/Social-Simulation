from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field


class CohortState(BaseModel):
    id: str
    weight: float
    attitude: float = 0.0  # -1..1 对当前议题
    activation: float = 0.3  # 0..1 参与/发声倾向
    class_layer: str = "mixed"  # lower | middle | upper | mixed
    traits: dict[str, float] = Field(default_factory=dict)


class WorldState(BaseModel):
    tick: int = 0
    sentiment: float = 0.0
    economy_index: float = 0.5
    policy_support: float = 0.5
    rumor_level: float = 0.0
    unrest: float = 0.0
    # 产品：品牌/主体社会信任 proxy；政策：对政策执行与公权力信任 proxy
    issuer_trust_proxy: float = 0.5
    supply_chain_stress: float = 0.2
    cohorts: list[CohortState] = Field(default_factory=list)
    narrative: list[str] = Field(default_factory=list)
    metrics_history: list[dict[str, float]] = Field(default_factory=list)
    # 可选：财务与叙事细节（由 YAML operational_finance 等开启）
    cash_balance_million: float | None = None
    fiscal_remaining_billion: float | None = None
    debt_balance_million: float | None = None
    detail_events: list[dict[str, Any]] = Field(default_factory=list)
    horizon_forecast: dict[str, str | None] = Field(default_factory=dict)
    user_resolution_notes: list[str] = Field(default_factory=list)
    resolved_crisis_ids: list[str] = Field(default_factory=list)
    simulation_outcome: Literal["complete", "paused"] = "complete"
    # 累计「关键智能体」primary LLM 调用次数（跨 tick）；用于风险里程碑暂停
    primary_llm_calls_total: int = 0
    risk_milestone_shown: bool = False
    last_risk_inventory: dict[str, Any] | None = None
    # 公司内部「记忆」：每 tick 用快模型压缩，注入后续研判与关键智能体上下文，减少重复踩坑
    company_memory_synthesis: str = ""
    company_memory_events: list[dict[str, Any]] = Field(default_factory=list)
    # 各关键智能体本人前几轮「公开陈述」摘录，避免下一轮 LLM 复述同一段话
    key_actor_recall: dict[str, list[str]] = Field(default_factory=dict)
    # 指挥台确认后的「决策层」摘要，长期注入 decision_context，影响机关/公司内部推演口径
    decision_layer_active_summary: str = ""
    # 待确认桌面研判快照（与 pause 包 plan_evaluations 对齐）；确认或改评后清空/归档
    pending_desk_review: list[dict[str, Any]] = Field(default_factory=list)
    # 历史方案（含已改评、已确认），供前端与复盘；条目为 dict，含 id/status/plain_text 等
    decision_layer_history: list[dict[str, Any]] = Field(default_factory=list)
    # 「仅评估→确认执行」约定的未来 tick 节点（到期写叙事/细节，供智能体记忆流通）
    internal_draft_commitments: list[dict[str, Any]] = Field(default_factory=list)
    # —— Scenario.extensions：延迟队列、扩散、KPI、资源池、校准跟踪（见系统说明 §12）——
    delayed_events: list[dict[str, Any]] = Field(default_factory=list)
    diffusion_s: float = 1.0
    diffusion_i: float = 0.0
    diffusion_r: float = 0.0
    kpi_values: dict[str, float] = Field(default_factory=dict)
    # extensions.kpi.hierarchy_enabled 时：按 outcome / process / resource 分桶（供 tick_end 与复盘）
    kpi_by_tier: dict[str, dict[str, float]] = Field(default_factory=dict)
    # sir_per_cohort：各 cohort 的 S/I/R 份额（0~1，三者归一）
    diffusion_cohort_sir: dict[str, dict[str, float]] = Field(default_factory=dict)
    resource_manpower: float | None = None
    resource_political_capital: float | None = None
    causal_rules_fired_tick: list[str] = Field(default_factory=list)
    # 本 tick 因果层（规则+边）动过的宏观指标名，供系统级因果审计（见 causal_audit）
    causal_metrics_this_tick: list[str] = Field(default_factory=list)
    extension_trace: list[str] = Field(default_factory=list)
    behavior_micro_history: list[dict[str, Any]] = Field(default_factory=list)
    # 事件级溯源 v0：每层对宏观量的增量（见 pepole/attribution.py、系统说明 §16.4）
    attribution_log: list[dict[str, Any]] = Field(default_factory=list)
    # 地区情境 grounding：模拟前多轮检索 trace + 注入 decision_context 的结构化摘要
    regional_grounding_trace: list[dict[str, Any]] = Field(default_factory=list)
    regional_grounding_artifact: dict[str, Any] = Field(default_factory=dict)

    def snapshot_metrics(self) -> dict[str, float]:
        out = {
            "sentiment": self.sentiment,
            "economy_index": self.economy_index,
            "policy_support": self.policy_support,
            "rumor_level": self.rumor_level,
            "unrest": self.unrest,
            "issuer_trust_proxy": self.issuer_trust_proxy,
            "supply_chain_stress": self.supply_chain_stress,
        }
        if self.cash_balance_million is not None:
            out["cash_balance_million"] = float(self.cash_balance_million)
        if self.fiscal_remaining_billion is not None:
            out["fiscal_remaining_billion"] = float(self.fiscal_remaining_billion)
        if self.debt_balance_million is not None:
            out["debt_balance_million"] = float(self.debt_balance_million)
        out["diffusion_i_share"] = float(self.diffusion_i)
        if self.resource_manpower is not None:
            out["resource_manpower"] = float(self.resource_manpower)
        if self.resource_political_capital is not None:
            out["resource_political_capital"] = float(self.resource_political_capital)
        for k, v in self.kpi_values.items():
            out[f"kpi_{k}"] = float(v)
        return out

    def clone(self) -> WorldState:
        return self.model_validate(copy.deepcopy(self.model_dump()))

    def log(self, line: str) -> None:
        self.narrative.append(f"[t{self.tick}] {line}")

    def push_detail(
        self,
        kind: str,
        title: str,
        body: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.detail_events.append(
            {
                "tick": self.tick,
                "kind": kind,
                "title": title,
                "body": body,
                "meta": meta or {},
            }
        )

    def cohort_proxy_for_prompts(self) -> list[dict[str, Any]]:
        """供 LLM 推断「哪类用户更可能发声、会反映什么问题」，勿仅复述 YAML 设定。"""
        return [
            {
                "id": c.id,
                "class_layer": c.class_layer,
                "attitude": round(float(c.attitude), 4),
                "activation": round(float(c.activation), 4),
                "weight": round(float(c.weight), 4),
                "traits": dict(c.traits),
            }
            for c in self.cohorts
        ]
