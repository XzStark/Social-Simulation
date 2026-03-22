from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# 场景 YAML 与 checkpoint 的语义版本（ bump 时需迁移说明与兼容层）
PEPOLE_SCENARIO_SCHEMA_VERSION = "2.3"

ClassLayer = Literal["lower", "middle", "upper", "mixed"]


class CohortSpec(BaseModel):
    id: str
    weight: float = Field(ge=0.0)
    # 社会阶层 proxy：影响聚合时「谁的声音更像真实舆论权重」「动荡来自哪里」
    class_layer: ClassLayer = "mixed"
    traits: dict[str, float] = Field(default_factory=dict)


class KeyActorSpec(BaseModel):
    id: str
    role: str
    goals: str
    persona: str = ""
    categories: list[str] = Field(default_factory=list)


class PersonaSpec(BaseModel):
    """单个人设；可准备 80+ 条，由引擎每 tick 抽样一部分上 LLM。"""

    id: str
    role: str
    goals: str = ""
    persona: str = ""
    categories: list[str] = Field(default_factory=list)
    # 空 = 不限制；否则须与 simulation 有交集才进入池
    markets: list[Literal["domestic", "export"]] = Field(default_factory=list)
    product_kinds: list[Literal["software", "hardware", "hybrid", "general"]] = Field(default_factory=list)
    llm_each_tick: bool = False
    sampling_weight: float = Field(default=1.0, ge=0.0)


AdminLevel = Literal["unset", "central", "province", "city", "county"]


class PolicyContext(BaseModel):
    """政策演练：哪一级政府、何地、民风与媒体环境（供模型推理，非替真实统计）。"""

    admin_level: AdminLevel = "unset"
    jurisdiction_name: str = ""
    local_norms_brief: str = ""
    media_environment_brief: str = ""


IssuerArchetype = Literal["megacorp", "large_group", "sme", "startup"]


class IssuerProfile(BaseModel):
    """产品演练：集团体量、品牌资产与社会评价先验（可对照不同品牌国民度差异）。"""

    archetype: IssuerArchetype = "large_group"
    brand_equity: float = Field(default=0.55, ge=0.0, le=1.0)
    reputation_brief: str = ""
    supply_chain_position: str = ""


class MarketCompetitorSpec(BaseModel):
    """产品盘：与 playable 发行方形成有效竞争的其它主体或产品线（可多条；演练用 proxy）。"""

    id: str
    name: str
    brief: str = ""
    estimated_market_share_proxy: float | None = Field(default=None, ge=0.0, le=1.0)
    linked_key_actor_id: str | None = Field(
        default=None,
        description="可选：与 key_actors[].id 对齐，便于叙事与关系图一致",
    )


MetricName = Literal[
    "sentiment",
    "economy_index",
    "policy_support",
    "rumor_level",
    "unrest",
    "issuer_trust_proxy",
    "supply_chain_stress",
]


class TriggerCondition(BaseModel):
    """单条条件，与其它条 AND。"""

    metric: MetricName
    gte: float | None = None
    lte: float | None = None
    gt: float | None = None
    lt: float | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> TriggerCondition:
        if all(x is None for x in (self.gte, self.lte, self.gt, self.lt)):
            raise ValueError("TriggerCondition 至少需要 gte/lte/gt/lt 之一")
        return self


InstitutionTier = Literal["small", "authority"]


class InstitutionSpec(BaseModel):
    """舆论场/合作方：小机构 vs 权威机构，供叙事与界面展示。"""

    id: str
    name: str
    tier: InstitutionTier = "small"
    focus_brief: str = ""


class CooperationOfferSpec(BaseModel):
    """可谈判合作项：报价与档位（不自动成交，仅作账面压力测试叙事）。"""

    id: str
    name: str
    cost_million: float = Field(ge=0.0)
    partner_tier: InstitutionTier = "small"
    brief: str = ""


class CrisisRule(BaseModel):
    """
    当 when_all 满足时，本 tick 结束后暂停演练，要求用户在网页端输入处置方案后再继续。
    CLI 单次 run 会在暂停点结束并提示使用网页续跑。
    """

    id: str
    when_all: list[TriggerCondition] = Field(default_factory=list)
    title: str
    detail: str = ""
    once: bool = True

    @model_validator(mode="after")
    def _when_nonempty(self) -> CrisisRule:
        if not self.when_all:
            raise ValueError(f"crisis_rules[{self.id!r}]: when_all 不能为空")
        return self


class OperationalFinance(BaseModel):
    """
    产品：账面现金、每 tick 运营消耗、写意营收与税费、债务本息；可按行业与 issuer 体量缩放 opex。
    政策：可用财力池、每 tick 支出，以及与经济景气挂钩的写意税收入库（入库增加财力）。
    产品侧可选 parallel fiscal pool：企业缴税（从现金扣）同时按 mirror 比例增加「地方财政入库 proxy」。
    """

    enabled: bool = False
    cash_balance_million: float = Field(default=800.0, ge=0.0)
    operating_cost_million_per_tick: float = Field(default=35.0, ge=0.0)
    fiscal_pool_billion: float | None = Field(default=None, ge=0.0)
    policy_spend_per_tick_billion: float = Field(default=0.04, ge=0.0)
    # —— 行业 / 体量（仅影响运营现金消耗）——
    industry_operating_cost_multiplier: float = Field(default=1.0, ge=0.0)
    auto_scale_operating_cost_from_issuer: bool = False
    # —— 写意营收与税负（产品）；税费从现金扣 ——
    revenue_proxy_million_per_tick: float = Field(default=0.0, ge=0.0)
    effective_tax_rate_on_revenue: float = Field(default=0.0, ge=0.0, le=0.6)
    # —— 产品 + 同时跟踪地方财力 proxy（十亿）——
    track_parallel_fiscal_pool: bool = False
    parallel_fiscal_pool_initial_billion: float = Field(default=0.0, ge=0.0)
    company_tax_mirror_to_fiscal_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    # —— 债务（百万）；年化利率写意，按 ~26 tick/年 摊息 ——
    debt_principal_million: float = Field(default=0.0, ge=0.0)
    debt_interest_annual_rate_proxy: float = Field(default=0.0, ge=0.0, le=0.5)
    debt_principal_repay_million_per_tick: float = Field(default=0.0, ge=0.0)
    # 产品：仍为模板默认「现金 800、负债 0」时，按 issuer 体量自动写意现金与负债（YAML 显式改写后不覆盖）
    scale_cash_debt_from_issuer_if_defaults: bool = True
    # —— 政策：经济景气越高，写意税收入库越多（十亿/tick × economy_index）——
    fiscal_tax_inflow_from_economy_billion_per_tick: float = Field(default=0.0, ge=0.0)


class TriggerRule(BaseModel):
    """
    当 when_all 全满足时，本 tick 将 inject_pool_priority_ids 并入「优先占池化名额」集合
    （与 always_sample_ids 效果相同，只是条件触发）。
    """

    id: str
    when_all: list[TriggerCondition] = Field(default_factory=list)
    inject_pool_priority_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _when_all_nonempty(self) -> TriggerRule:
        if not self.when_all:
            raise ValueError(f"触发器 {self.id!r}: when_all 不能为空")
        return self


class SimulationScope(BaseModel):
    """本次演练的「世界范围」：用来过滤海量人设里谁可能登场。"""

    product_kind: Literal["general", "software", "hardware", "hybrid"] = "general"
    markets_active: list[Literal["domestic", "export"]] = Field(default_factory=lambda: ["domestic", "export"])
    # 每 tick 从池化人设中抽几个上 LLM（红楼梦式 80 人里每轮只醒几个）
    pooled_llm_calls_per_tick: int = Field(default=4, ge=0)
    # 池化人设中这些 id 尽量优先占用抽样名额（如首席竞品、出口负责人）
    always_sample_ids: list[str] = Field(default_factory=list)


class RealismConfig(BaseModel):
    """
    让世界线更「钝」、更像真实组织与市场：小步噪声、高惯性、压制单 tick 极端跳变。
    tick 在叙事上可理解为约 1～2 周。
    """

    env_rumor_min_step: float = -0.018
    env_rumor_max_step: float = 0.022
    env_economy_min_step: float = -0.012
    env_economy_max_step: float = 0.012
    unrest_decay_factor: float = 0.91
    unrest_stress_weight: float = 0.08
    unrest_rumor_coupling: float = 0.025
    policy_from_sentiment_weight: float = 0.028
    policy_unrest_penalty: float = 0.022
    # 略压低：单轮智能体表态对宏观指标的边际冲击更接近「真实舆论惯性」
    llm_effect_multiplier: float = 0.38
    llm_cap_rumor: float = 0.028
    llm_cap_policy_support: float = 0.02
    llm_cap_unrest: float = 0.015
    llm_sentiment_to_cohort_scale: float = 0.22
    cohort_micro_jitter: float = 0.006
    # 阶层在「舆情加权」中的代表权（非道德判断，仅模拟可见度与动员性）
    stratum_lower_representation_weight: float = 1.08
    stratum_middle_representation_weight: float = 1.0
    stratum_upper_representation_weight: float = 0.94
    stratum_mixed_representation_weight: float = 1.0
    # 各阶层对「动荡压力项」的贡献系数（基层更易受生计/安全冲击动员）
    stratum_unrest_lower_mult: float = 1.14
    stratum_unrest_middle_mult: float = 1.0
    stratum_unrest_upper_mult: float = 0.88
    stratum_unrest_mixed_mult: float = 1.0
    # 主体信任、供应链压力缓慢随宏观漂移
    trust_mean_reversion: float = 0.038
    trust_rumor_erosion: float = 0.014
    supply_chain_persistence: float = 0.93
    supply_chain_rumor_sensitivity: float = 0.028
    supply_chain_economy_sensitivity: float = 0.022
    # 回合末在写入 metrics_history 前，将宏观量向「上一轮快照」线性回拉，模拟民调/舆论粘性（0=关闭）
    macro_inertia_blend: float = Field(default=0.0, ge=0.0, le=0.95)
    # 空列表表示对 MACRO 七项全部生效（见 attribution.MACRO_ATTR_KEYS）
    macro_inertia_keys: list[str] = Field(default_factory=list)


class DecisionSupportSpec(BaseModel):
    """决策辅助输出：指标序列粗检（不改变默认推演，仅多一层提示）。"""

    enabled: bool = True
    # 单轮变化绝对值超过阈值则生成提示；键可只写部分指标，其余用代码内默认
    jump_warn_threshold: dict[str, float] = Field(default_factory=dict)


RegionalGroundingMode = Literal["llm_only", "wikipedia_then_llm"]
WebSearchProvider = Literal["duckduckgo", "brave"]


class RegionalGroundingSpec(BaseModel):
    """
    模拟前「地区情境检索」：在 player_brief 新政策/新动作之前，先合成或拉取
    「当地已有执行口径、中央上位框架、基层/商户常态」等多轮摘要，注入决策上下文。
    wikipedia_then_llm：先尝试中文维基 REST 摘要（失败则仅 LLM）；均非官方法规检索，须标免责声明。
    web_search_enabled：可选再拉开放网页检索摘要（DuckDuckGo 或 Brave），snippet 级，非全文、非法规库。
    """

    enabled: bool = False
    region_label: str = ""
    mode: RegionalGroundingMode = "llm_only"
    business_scale: Literal["unset", "street_shop", "sme_chain", "regional_group", "national_group"] = "unset"
    business_sector_brief: str = ""
    user_known_local_policy_brief: str = ""
    user_known_central_policy_brief: str = ""
    sync_issuer_archetype_from_scale: bool = True
    pass_temperature: float = Field(default=0.28, ge=0.0, le=1.0)
    max_wiki_chars: int = Field(default=1800, ge=0, le=8000)
    web_search_enabled: bool = False
    web_search_provider: WebSearchProvider = "duckduckgo"
    web_search_max_results: int = Field(default=5, ge=1, le=15)
    web_search_max_chars: int = Field(default=3500, ge=0, le=12000)
    web_search_query_budget: int = Field(default=2, ge=1, le=5)
    web_search_extra_queries: str = ""


class EnsembleRiskSpec(BaseModel):
    """
    多次蒙特卡洛/集成运行汇总为「风险简报」时的阈值与 P×影响×不可逆性 权重。
    可在场景 YAML 的 `ensemble_risk:` 下覆盖。
    """

    low_support_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    high_unrest_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    high_rumor_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    collapse_unrest: float = Field(default=0.85, ge=0.0, le=1.0)
    collapse_support: float = Field(default=0.25, ge=0.0, le=1.0)
    # 归类「高影响低概率」：P 不超过该值且 I×R 足够大时进入该桶
    high_impact_low_prob_max_p: float = Field(default=0.22, ge=0.0, le=1.0)
    # 「中高概率、中等风险」桶的下界
    medium_high_prob_min_p: float = Field(default=0.38, ge=0.0, le=1.0)
    medium_risk_score_band_max: float = Field(default=0.42, ge=0.0, le=1.0)
    impact_low_support: float = Field(default=0.85, ge=0.0, le=1.0)
    impact_high_unrest: float = Field(default=0.8, ge=0.0, le=1.0)
    impact_high_rumor: float = Field(default=0.55, ge=0.0, le=1.0)
    irreversibility_low_support: float = Field(default=0.75, ge=0.0, le=1.0)
    irreversibility_high_unrest: float = Field(default=0.7, ge=0.0, le=1.0)
    irreversibility_high_rumor: float = Field(default=0.4, ge=0.0, le=1.0)
    crisis_rule_impact: float = Field(default=0.92, ge=0.0, le=1.0)
    crisis_rule_irreversibility: float = Field(default=0.88, ge=0.0, le=1.0)
    worst_case_sample_n: int = Field(default=3, ge=1, le=20)


class CohortLLMConfig(BaseModel):
    enabled: bool = False
    batch_all_cohorts: bool = True  # true: 1 call/tick；false: 每 cohort 1 call


class PerformanceBudgetSpec(BaseModel):
    """
    运行时性能预算（非叙事内 manpower/political 模拟）。
    用于超时跳过非关键 LLM（如 horizon）或记录告警。
    """

    enabled: bool = False
    max_tick_wall_seconds: float = Field(default=0.0, ge=0.0)  # 0 = 不启用 wall 限制
    skip_horizon_if_over_budget: bool = True
    max_primary_llm_calls_per_tick: int = Field(default=0, ge=0)  # 0 = 不截断
    log_perf_warning: bool = True


class CausalEffectSpec(BaseModel):
    """因果层：条件满足时对宏观指标做加性调整（在 cohort 聚合之前执行）。"""

    metric: MetricName
    add: float = 0.0


class CausalRuleSpec(BaseModel):
    id: str = ""
    when_all: list[TriggerCondition] = Field(default_factory=list)
    effects: list[CausalEffectSpec] = Field(default_factory=list)
    # 同 tick 内多规则：priority 升序先执行；merge_mode 控制同指标多 effect 的合并
    priority: int = Field(default=100, ge=0)
    merge_mode: Literal["additive", "last_wins"] = "additive"

    @model_validator(mode="after")
    def _when_nonempty(self) -> CausalRuleSpec:
        if self.when_all and not self.effects:
            raise ValueError(f"causal rule {self.id!r}: 有 when_all 时 effects 不能为空")
        return self


class CausalEdgeSpec(BaseModel):
    """因果图滞后边：end-of-tick(t−lag) 的源指标值 × weight 加到目标（加性）。"""

    id: str = ""
    from_metric: MetricName
    to_metric: MetricName
    lag_ticks: int = Field(default=1, ge=0)
    weight: float = 0.0
    max_abs_delta: float | None = Field(default=None, ge=0.0)


class CausalLayerSpec(BaseModel):
    enabled: bool = False
    rules: list[CausalRuleSpec] = Field(default_factory=list)
    edges: list[CausalEdgeSpec] = Field(default_factory=list)
    # 系统级因果治理（非「全方程替代 LLM」，而是审计与告警）
    governance_mode: Literal["off", "warn_llm_without_causal"] = "off"
    governance_metrics: list[MetricName] = Field(default_factory=list)
    governance_min_delta: float = Field(default=0.015, ge=0.0)


class GroundTruthAnchorSpec(BaseModel):
    """
    外部真值/先验时间序列锚定：每 tick 将指标向 series 中对应点线性混合。
    series 键为指标名，值为长度 >= ticks 的列表（下标 tick-1 对应当前 tick）。
    """

    enabled: bool = False
    blend_alpha: float = Field(default=0.18, ge=0.0, le=1.0)
    series: dict[str, list[float]] = Field(default_factory=dict)


class KpiLayerSpec(BaseModel):
    """业务 KPI 代理：由 cohort + 宏观量推导 conversion / retention 等（写入 state.kpi_values）。"""

    enabled: bool = False
    hierarchy_enabled: bool = False
    tier_default: Literal["outcome", "process", "resource"] = "outcome"
    tier_by_key: dict[str, Literal["outcome", "process", "resource"]] = Field(default_factory=dict)


class DiffusionLayerSpec(BaseModel):
    """认知/情绪扩散 proxy：SIR 式 informed 份额 + 与谣言耦合。"""

    enabled: bool = False
    mode: Literal["sir_global", "sir_per_cohort"] = "sir_global"
    beta: float = Field(default=0.14, ge=0.0)
    gamma: float = Field(default=0.065, ge=0.0)
    rumor_coupling: float = Field(default=0.32, ge=0.0)
    seed_informed: float = Field(default=0.06, ge=0.0, le=1.0)


class DelayEventSpec(BaseModel):
    due_tick: int = Field(ge=1)
    deltas: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class DelayLayerSpec(BaseModel):
    """延迟生效：在 tick 开始时冲刷 due_tick==当前 tick 的增量。"""

    enabled: bool = False
    schedule: list[DelayEventSpec] = Field(default_factory=list)


class ResourceLayerSpec(BaseModel):
    """人力/政治资本池：与 LLM 人数、跨主体交流、动荡挂钩（可选）。"""

    enabled: bool = False
    manpower_initial: float = Field(default=100.0, ge=0.0)
    political_capital_initial: float = Field(default=100.0, ge=0.0)
    manpower_per_primary_llm: float = Field(default=0.07, ge=0.0)
    manpower_per_interaction: float = Field(default=0.28, ge=0.0)
    political_unrest_drain_scale: float = Field(default=0.14, ge=0.0)


class BehaviorMicroSpec(BaseModel):
    """更细用户行为 proxy：按 cohort 生成评分/留存噪声（写入 state.behavior_micro_history）。"""

    enabled: bool = False
    rating_jitter_sigma: float = Field(default=0.055, ge=0.0)


class RlPolicyStubSpec(BaseModel):
    """可选：决策层强化学习占位（记录诊断，不接训练）。"""

    enabled: bool = False


class MultimodalStubSpec(BaseModel):
    """可选：多模态特征入口占位。"""

    enabled: bool = False


class ScenarioExtensions(BaseModel):
    """
    扩展能力开关与参数（默认全关，旧 YAML 无需改动；说明见系统说明.md §12）。
    执行顺序由 plugin_order 控制；未识别的名字会被忽略。
    """

    causal: CausalLayerSpec = Field(default_factory=CausalLayerSpec)
    ground_truth: GroundTruthAnchorSpec = Field(default_factory=GroundTruthAnchorSpec)
    kpi: KpiLayerSpec = Field(default_factory=KpiLayerSpec)
    diffusion: DiffusionLayerSpec = Field(default_factory=DiffusionLayerSpec)
    delay: DelayLayerSpec = Field(default_factory=DelayLayerSpec)
    resources: ResourceLayerSpec = Field(default_factory=ResourceLayerSpec)
    behavior_micro: BehaviorMicroSpec = Field(default_factory=BehaviorMicroSpec)
    rl_stub: RlPolicyStubSpec = Field(default_factory=RlPolicyStubSpec)
    multimodal_stub: MultimodalStubSpec = Field(default_factory=MultimodalStubSpec)
    validation_trace: bool = False
    plugin_order: list[str] = Field(
        default_factory=lambda: [
            "delay",
            "behavior_micro",
            "causal",
            "diffusion",
            "anchor",
            "kpi",
            "resources",
            "rl_stub",
            "multimodal_stub",
        ]
    )


class Scenario(BaseModel):
    name: str = "unnamed"
    # policy = 政策制定者演练；product = 集团/产品发行方观测市场与社会反响
    exercise_type: Literal["policy", "product"] = "policy"
    # 用户下达的具体内容：条文要点、产品定位、定价与渠道、宣传话术等（会注入所有智能体提示）
    player_brief: str = ""
    reference_cases_brief: str = ""
    policy_context: PolicyContext = Field(default_factory=PolicyContext)
    issuer: IssuerProfile = Field(default_factory=IssuerProfile)
    ticks: int = Field(ge=1, default=10)
    random_seed: int = 0
    initial: dict[str, float] = Field(default_factory=dict)
    cohorts: list[CohortSpec] = Field(default_factory=list)
    key_actors: list[KeyActorSpec] = Field(default_factory=list)
    market_competitors: list[MarketCompetitorSpec] = Field(default_factory=list)
    personas: list[PersonaSpec] = Field(default_factory=list)
    simulation: SimulationScope = Field(default_factory=SimulationScope)
    triggers: list[TriggerRule] = Field(default_factory=list)
    realism: RealismConfig = Field(default_factory=RealismConfig)
    cohort_llm: CohortLLMConfig = Field(default_factory=CohortLLMConfig)
    operational_finance: OperationalFinance = Field(default_factory=OperationalFinance)
    institutions: list[InstitutionSpec] = Field(default_factory=list)
    cooperations: list[CooperationOfferSpec] = Field(default_factory=list)
    crisis_rules: list[CrisisRule] = Field(default_factory=list)
    # 0=关闭（默认）。>0 时：累计关键智能体(primary) LLM 达此次数后暂停并生成内部风险盘点。若要「只有指标过线才暂停」，请用 crisis_rules，勿依赖本项。
    risk_milestone_primary_calls: int = Field(default=0, ge=0)
    # 多方案研判后是否必须再点「确认」才进入后续 tick（默认开）
    confirm_after_desk_feasibility: bool = True
    # 0~1：产品/议题对用户或社会的「刚需、不可替代」程度；高则负面舆情对支持度/信任的边际伤害更大（仍受品牌缓冲调制）
    problem_salience: float = Field(default=0.5, ge=0.0, le=1.0)
    extensions: ScenarioExtensions = Field(default_factory=ScenarioExtensions)
    # 与引擎/清单对齐的语义版本（YAML 可覆盖以强制旧行为；默认随代码 bump）
    scenario_schema_version: str = Field(default=PEPOLE_SCENARIO_SCHEMA_VERSION)
    performance: PerformanceBudgetSpec = Field(default_factory=PerformanceBudgetSpec)
    decision_support: DecisionSupportSpec = Field(default_factory=DecisionSupportSpec)
    ensemble_risk: EnsembleRiskSpec = Field(default_factory=EnsembleRiskSpec)
    regional_grounding: RegionalGroundingSpec = Field(default_factory=RegionalGroundingSpec)

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        p = Path(path)
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class ModelRouting(BaseModel):
    """Which logical model slot calls which API."""

    primary: str = "openai:gpt-4o"
    fast: str = "openai:gpt-4o-mini"
    embedding: str | None = None


class ProviderEnv(BaseModel):
    """环境变量名约定；也可在代码里直接传 base_url / key。"""

    openai_api_key: str = "OPENAI_API_KEY"
    openai_base_url: str = "OPENAI_BASE_URL"
    anthropic_api_key: str = "ANTHROPIC_API_KEY"
    google_api_key: str = "GOOGLE_API_KEY"
