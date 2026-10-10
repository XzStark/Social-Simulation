from __future__ import annotations

import random
from typing import Any

from people.state import WorldState

_CLAMP_KEYS = {
    "sentiment",
    "economy_index",
    "policy_support",
    "rumor_level",
    "unrest",
    "issuer_trust_proxy",
    "supply_chain_stress",
}


BLACK_SWAN_PRESETS: dict[str, dict[str, Any]] = {
    # Media / public opinion shocks
    "anchor_kol_scandal": {
        "id": "anchor_kol_scandal",
        "title": "头部主播翻车",
        "detail": "核心主播或代言人突发负面，舆情情绪急转，品牌承压。",
        "metric_deltas": {"policy_support": -0.12, "rumor_level": +0.22, "unrest": +0.1, "issuer_trust_proxy": -0.1},
    },
    "competitor_smear": {
        "id": "competitor_smear",
        "title": "竞品恶意抹黑",
        "detail": "竞争对手发起针对性负面传播，引发信任与舆情波动。",
        "metric_deltas": {"policy_support": -0.07, "rumor_level": +0.2, "issuer_trust_proxy": -0.08, "unrest": +0.06},
    },
    "viral_negative_clip": {
        "id": "viral_negative_clip",
        "title": "负面短视频爆发",
        "detail": "单条负面内容跨平台扩散，带动跟风评论与误读。",
        "metric_deltas": {"policy_support": -0.08, "rumor_level": +0.24, "unrest": +0.07},
    },
    "rumor_fabrication_campaign": {
        "id": "rumor_fabrication_campaign",
        "title": "谣言造势攻击",
        "detail": "系统性造谣并投放传播，导致辟谣成本陡升。",
        "metric_deltas": {"policy_support": -0.09, "rumor_level": +0.3, "issuer_trust_proxy": -0.08},
    },
    "boycott_wave": {
        "id": "boycott_wave",
        "title": "抵制潮",
        "detail": "核心社群发起抵制并形成线下联动，动员强度走高。",
        "metric_deltas": {"policy_support": -0.13, "unrest": +0.15, "issuer_trust_proxy": -0.11},
    },
    "celebrity_withdrawal": {
        "id": "celebrity_withdrawal",
        "title": "代言人撤约",
        "detail": "关键代言合作终止，外部解读偏向负面。",
        "metric_deltas": {"policy_support": -0.08, "issuer_trust_proxy": -0.09, "rumor_level": +0.1},
    },
    # Regulatory / policy / legal shocks
    "regulatory_shock": {
        "id": "regulatory_shock",
        "title": "监管政策突变",
        "detail": "监管口径临时收紧，审批与执行不确定性上升。",
        "metric_deltas": {"policy_support": -0.08, "economy_index": -0.06, "unrest": +0.08, "supply_chain_stress": +0.12},
    },
    "license_revocation_risk": {
        "id": "license_revocation_risk",
        "title": "牌照/资质风险",
        "detail": "核心资质审查趋严，业务连续性面临冲击。",
        "metric_deltas": {"policy_support": -0.09, "economy_index": -0.05, "supply_chain_stress": +0.1},
    },
    "enforcement_intensification": {
        "id": "enforcement_intensification",
        "title": "执法密度骤增",
        "detail": "检查频次上升，运营灵活性下降且违规成本上行。",
        "metric_deltas": {"policy_support": -0.06, "unrest": +0.06, "economy_index": -0.04},
    },
    "lawsuit_cluster": {
        "id": "lawsuit_cluster",
        "title": "集体诉讼/多案并发",
        "detail": "法律纠纷集中爆发，管理层与现金流同时承压。",
        "metric_deltas": {"policy_support": -0.08, "issuer_trust_proxy": -0.09, "rumor_level": +0.09},
        "cash_delta_million": -60.0,
    },
    # Platform / channel / market shocks
    "platform_rule_shift": {
        "id": "platform_rule_shift",
        "title": "平台规则突改",
        "detail": "流量平台规则突变，获客与曝光效率骤降。",
        "metric_deltas": {"policy_support": -0.05, "economy_index": -0.04, "rumor_level": +0.06},
    },
    "channel_ban": {
        "id": "channel_ban",
        "title": "核心渠道封禁",
        "detail": "关键投放或分发渠道受限，转化链路被截断。",
        "metric_deltas": {"policy_support": -0.08, "economy_index": -0.07, "supply_chain_stress": +0.06},
    },
    "algorithm_demotion": {
        "id": "algorithm_demotion",
        "title": "推荐权重下调",
        "detail": "算法下调导致自然流量断崖式下滑。",
        "metric_deltas": {"policy_support": -0.05, "economy_index": -0.05, "rumor_level": +0.05},
    },
    "price_war_escalation": {
        "id": "price_war_escalation",
        "title": "价格战升级",
        "detail": "行业陷入非理性降价，利润与现金回笼恶化。",
        "metric_deltas": {"economy_index": -0.08, "policy_support": -0.04, "unrest": +0.04},
        "cash_delta_million": -75.0,
    },
    # Supply chain / operations shocks
    "supply_chain_cutoff": {
        "id": "supply_chain_cutoff",
        "title": "供应链断供",
        "detail": "关键部件或渠道突然中断，交付和履约风险显著放大。",
        "metric_deltas": {"economy_index": -0.08, "policy_support": -0.06, "supply_chain_stress": +0.28, "unrest": +0.04},
    },
    "logistics_disruption": {
        "id": "logistics_disruption",
        "title": "物流主干受阻",
        "detail": "跨区域配送中断或时延严重，库存与履约压力上升。",
        "metric_deltas": {"economy_index": -0.06, "supply_chain_stress": +0.18, "policy_support": -0.05},
    },
    "quality_defect_recall": {
        "id": "quality_defect_recall",
        "title": "质量缺陷召回",
        "detail": "批量质量问题触发召回，合规与品牌双重受损。",
        "metric_deltas": {"policy_support": -0.1, "issuer_trust_proxy": -0.12, "unrest": +0.07},
        "cash_delta_million": -90.0,
    },
    "core_supplier_bankruptcy": {
        "id": "core_supplier_bankruptcy",
        "title": "核心供应商破产",
        "detail": "上游主体停摆，替代周期与成本远超预期。",
        "metric_deltas": {"economy_index": -0.09, "supply_chain_stress": +0.3, "policy_support": -0.06},
    },
    "facility_shutdown": {
        "id": "facility_shutdown",
        "title": "关键产线/门店停摆",
        "detail": "核心设施停工停业，交付与服务连续性受损。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.07, "supply_chain_stress": +0.2},
    },
    # Cyber / data / tech shocks
    "data_leak_exposure": {
        "id": "data_leak_exposure",
        "title": "数据泄露曝光",
        "detail": "数据安全事件被披露，社会信任与监管压力同步上升。",
        "metric_deltas": {"policy_support": -0.1, "rumor_level": +0.16, "issuer_trust_proxy": -0.14, "unrest": +0.08},
    },
    "ransomware_attack": {
        "id": "ransomware_attack",
        "title": "勒索软件攻击",
        "detail": "核心系统被加密瘫痪，运营与恢复成本飙升。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.07, "supply_chain_stress": +0.12},
        "cash_delta_million": -70.0,
    },
    "model_hallucination_incident": {
        "id": "model_hallucination_incident",
        "title": "模型重大幻觉事故",
        "detail": "AI 输出严重失实并造成外部负面后果。",
        "metric_deltas": {"policy_support": -0.08, "issuer_trust_proxy": -0.1, "rumor_level": +0.1},
    },
    "ai_safety_redline_breach": {
        "id": "ai_safety_redline_breach",
        "title": "AI 安全红线触发",
        "detail": "输出触及敏感边界，监管与社会关注急剧升高。",
        "metric_deltas": {"policy_support": -0.1, "unrest": +0.09, "rumor_level": +0.11},
    },
    "core_system_outage": {
        "id": "core_system_outage",
        "title": "核心系统长时间宕机",
        "detail": "交易/服务主链路不可用，恢复窗口超过预期。",
        "metric_deltas": {"economy_index": -0.06, "policy_support": -0.07, "issuer_trust_proxy": -0.07},
    },
    # Finance / macro shocks
    "capital_liquidity_freeze": {
        "id": "capital_liquidity_freeze",
        "title": "融资与流动性冻结",
        "detail": "融资窗口关闭或现金流紧张，运营韧性受挤压。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.05, "unrest": +0.05},
        "cash_delta_million": -80.0,
        "debt_delta_million": +50.0,
    },
    "bank_run_or_credit_crunch": {
        "id": "bank_run_or_credit_crunch",
        "title": "信用收缩/抽贷",
        "detail": "信贷环境突变，短期周转承压并触发连锁违约风险。",
        "metric_deltas": {"economy_index": -0.1, "policy_support": -0.06, "unrest": +0.06},
        "cash_delta_million": -120.0,
        "debt_delta_million": +80.0,
    },
    "fx_shock": {
        "id": "fx_shock",
        "title": "汇率剧烈波动",
        "detail": "进口成本与出口回款偏离预期，定价体系失衡。",
        "metric_deltas": {"economy_index": -0.06, "supply_chain_stress": +0.1, "policy_support": -0.03},
    },
    "interest_rate_spike": {
        "id": "interest_rate_spike",
        "title": "利率急升",
        "detail": "融资成本上行，资产负债表压力外溢到经营侧。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.04},
        "debt_delta_million": +40.0,
    },
    # Geopolitics / external environment shocks
    "sanction_or_export_control": {
        "id": "sanction_or_export_control",
        "title": "制裁/出口管制升级",
        "detail": "关键技术与市场准入受限，替代成本陡增。",
        "metric_deltas": {"economy_index": -0.11, "policy_support": -0.07, "supply_chain_stress": +0.2},
    },
    "trade_barrier_surge": {
        "id": "trade_barrier_surge",
        "title": "贸易壁垒上升",
        "detail": "关税与非关税壁垒抬升，跨境业务受阻。",
        "metric_deltas": {"economy_index": -0.08, "supply_chain_stress": +0.12, "policy_support": -0.04},
    },
    "cross_border_payment_block": {
        "id": "cross_border_payment_block",
        "title": "跨境结算受限",
        "detail": "回款链路中断，外贸或海外业务现金回流放缓。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.04, "supply_chain_stress": +0.08},
        "cash_delta_million": -65.0,
    },
    # Social / safety / incident shocks
    "offline_incident": {
        "id": "offline_incident",
        "title": "线下安全事故",
        "detail": "线下事件发酵，风险认知上行并触发问责预期。",
        "metric_deltas": {"policy_support": -0.09, "unrest": +0.12, "issuer_trust_proxy": -0.08},
    },
    "workplace_safety_event": {
        "id": "workplace_safety_event",
        "title": "生产/施工安全事故",
        "detail": "安全管理缺口暴露，社会关注与监管压力同步上升。",
        "metric_deltas": {"policy_support": -0.08, "unrest": +0.1, "issuer_trust_proxy": -0.07},
    },
    "labor_dispute_strike": {
        "id": "labor_dispute_strike",
        "title": "劳资冲突与停工",
        "detail": "劳资矛盾升级，人员流失与舆论风险并发。",
        "metric_deltas": {"policy_support": -0.06, "unrest": +0.13, "economy_index": -0.05},
    },
    "public_health_wave": {
        "id": "public_health_wave",
        "title": "公共卫生波动",
        "detail": "突发公共卫生事件影响客流、人员到岗和线下经营。",
        "metric_deltas": {"economy_index": -0.09, "unrest": +0.05, "supply_chain_stress": +0.08},
    },
    # Natural / climate / infrastructure shocks
    "extreme_weather": {
        "id": "extreme_weather",
        "title": "极端天气冲击",
        "detail": "连续极端天气导致物流、门店与产线承压。",
        "metric_deltas": {"economy_index": -0.08, "supply_chain_stress": +0.14, "unrest": +0.04},
    },
    "major_earthquake_flood": {
        "id": "major_earthquake_flood",
        "title": "重大地震/洪灾",
        "detail": "区域基础设施损坏，恢复周期显著拉长。",
        "metric_deltas": {"economy_index": -0.12, "supply_chain_stress": +0.22, "unrest": +0.09},
    },
    "power_grid_failure": {
        "id": "power_grid_failure",
        "title": "电网级故障",
        "detail": "大范围供电不稳影响生产、冷链与线上服务。",
        "metric_deltas": {"economy_index": -0.07, "supply_chain_stress": +0.12, "policy_support": -0.03},
    },
    "telecom_backbone_failure": {
        "id": "telecom_backbone_failure",
        "title": "通信骨干故障",
        "detail": "通信主链路故障导致业务与应急协同受阻。",
        "metric_deltas": {"economy_index": -0.06, "unrest": +0.06, "policy_support": -0.04},
    },
    # Startup / small shop specific shocks
    "founder_reputation_crash": {
        "id": "founder_reputation_crash",
        "title": "创始人声誉危机",
        "detail": "创始人个人争议外溢到品牌与融资合作面。",
        "metric_deltas": {"policy_support": -0.1, "issuer_trust_proxy": -0.13, "rumor_level": +0.11},
    },
    "runway_below_quarter": {
        "id": "runway_below_quarter",
        "title": "现金跑道不足一季度",
        "detail": "企业进入紧急控支状态，战略动作被迫收缩。",
        "metric_deltas": {"economy_index": -0.07, "policy_support": -0.05},
        "cash_delta_million": -45.0,
        "debt_delta_million": +25.0,
    },
    "store_food_safety_issue": {
        "id": "store_food_safety_issue",
        "title": "门店食品/卫生事件",
        "detail": "本地口碑快速恶化，复购与到店率下滑。",
        "metric_deltas": {"policy_support": -0.11, "rumor_level": +0.14, "issuer_trust_proxy": -0.12, "unrest": +0.05},
    },
    "store_landlord_dispute": {
        "id": "store_landlord_dispute",
        "title": "房东/物业冲突",
        "detail": "租赁纠纷导致经营不稳定并触发临时停业风险。",
        "metric_deltas": {"economy_index": -0.05, "policy_support": -0.04, "unrest": +0.03},
        "cash_delta_million": -18.0,
    },
    "franchise_channel_conflict": {
        "id": "franchise_channel_conflict",
        "title": "加盟渠道冲突",
        "detail": "总部与门店利益错配，执行口径与价格体系失控。",
        "metric_deltas": {"policy_support": -0.07, "issuer_trust_proxy": -0.08, "unrest": +0.05},
    },
}

BLACK_SWAN_ALIASES: dict[str, str] = {
    "kol_scandal": "anchor_kol_scandal",
    "influencer_scandal": "anchor_kol_scandal",
    "regulation_shock": "regulatory_shock",
    "policy_shock": "regulatory_shock",
    "supply_cutoff": "supply_chain_cutoff",
    "smear_campaign": "competitor_smear",
    "data_breach": "data_leak_exposure",
    "liquidity_crisis": "capital_liquidity_freeze",
    "offline_accident": "offline_incident",
    "shop_food_safety": "store_food_safety_issue",
}

BLACK_SWAN_PRESETS.update(
    {
        # Government internal issues
        "official_corruption_probe": {
            "id": "official_corruption_probe",
            "title": "官员贪腐被查",
            "detail": "关键岗位人员被立案调查，历史项目与审批链同步被复核。",
            "metric_deltas": {"policy_support": -0.13, "rumor_level": +0.18, "unrest": +0.09},
        },
        "approval_process_violation_exposed": {
            "id": "approval_process_violation_exposed",
            "title": "不合规审批被集中曝光",
            "detail": "多项审批流程被质疑程序不完备，执行公信力受损。",
            "metric_deltas": {"policy_support": -0.11, "rumor_level": +0.16, "unrest": +0.07},
        },
        "superior_investigation_spillover": {
            "id": "superior_investigation_spillover",
            "title": "上级被查引发连带审查",
            "detail": "条线上级被查，相关经办人员与既往事项被穿透核查。",
            "metric_deltas": {"policy_support": -0.1, "rumor_level": +0.14, "unrest": +0.06},
        },
        "data_fabrication_internal": {
            "id": "data_fabrication_internal",
            "title": "内部数据造假暴露",
            "detail": "关键报送数据被证伪，决策依据可信度下降。",
            "metric_deltas": {"policy_support": -0.09, "issuer_trust_proxy": -0.08, "rumor_level": +0.1},
        },
        "procurement_irregularity": {
            "id": "procurement_irregularity",
            "title": "采购招标违规",
            "detail": "采购流程存在瑕疵，预算与合法性双重承压。",
            "metric_deltas": {"policy_support": -0.08, "unrest": +0.05, "economy_index": -0.03},
            "fiscal_delta_billion": -0.06,
        },
        "discipline_commission_intervention": {
            "id": "discipline_commission_intervention",
            "title": "纪检介入调查",
            "detail": "纪检监察介入后，流程节奏显著放缓并触发组织震荡。",
            "metric_deltas": {"policy_support": -0.07, "rumor_level": +0.09, "unrest": +0.05},
        },
        "grassroots_execution_resistance": {
            "id": "grassroots_execution_resistance",
            "title": "基层执行阻滞",
            "detail": "基层条线出现消极执行或信息回流失真，政策落地偏差扩大。",
            "metric_deltas": {"policy_support": -0.06, "unrest": +0.08, "economy_index": -0.03},
        },
        "cross_department_infighting": {
            "id": "cross_department_infighting",
            "title": "部门协同失灵",
            "detail": "部门间责任划分冲突，联合行动效率下降。",
            "metric_deltas": {"policy_support": -0.05, "unrest": +0.06, "rumor_level": +0.05},
        },
        "public_petition_escalation": {
            "id": "public_petition_escalation",
            "title": "信访集中升级",
            "detail": "投诉集中上行并触发跨层级关注。",
            "metric_deltas": {"policy_support": -0.08, "unrest": +0.12, "rumor_level": +0.08},
        },
        # Enterprise internal issues
        "executive_compliance_probe": {
            "id": "executive_compliance_probe",
            "title": "高管合规调查",
            "detail": "核心高管被调查，战略执行与外部信任同步受损。",
            "metric_deltas": {"policy_support": -0.09, "issuer_trust_proxy": -0.12, "rumor_level": +0.09},
        },
        "financial_misstatement_exposure": {
            "id": "financial_misstatement_exposure",
            "title": "财务失实披露",
            "detail": "财务披露被质疑失真，投融资与合作方信心下降。",
            "metric_deltas": {"policy_support": -0.1, "issuer_trust_proxy": -0.13, "economy_index": -0.04},
            "cash_delta_million": -95.0,
        },
        "internal_control_failure": {
            "id": "internal_control_failure",
            "title": "内控失效",
            "detail": "关键审批和风控环节失灵，经营风险累积。",
            "metric_deltas": {"policy_support": -0.07, "issuer_trust_proxy": -0.08, "rumor_level": +0.08},
        },
        "middle_management_rent_seeking": {
            "id": "middle_management_rent_seeking",
            "title": "中层寻租曝光",
            "detail": "中层管理寻租行为曝光，组织公允性受损。",
            "metric_deltas": {"policy_support": -0.07, "unrest": +0.06, "issuer_trust_proxy": -0.07},
        },
        "channel_rebate_fraud": {
            "id": "channel_rebate_fraud",
            "title": "渠道返利舞弊",
            "detail": "渠道返利链条出现舞弊，价格体系与渠道关系受冲击。",
            "metric_deltas": {"policy_support": -0.08, "economy_index": -0.05, "supply_chain_stress": +0.07},
            "cash_delta_million": -55.0,
        },
        "hr_mass_turnover": {
            "id": "hr_mass_turnover",
            "title": "关键团队集中离职",
            "detail": "核心岗位离职潮导致执行能力短期坍塌。",
            "metric_deltas": {"policy_support": -0.06, "economy_index": -0.04, "supply_chain_stress": +0.05},
        },
        "boardroom_power_struggle": {
            "id": "boardroom_power_struggle",
            "title": "董事会权力冲突",
            "detail": "治理层分歧公开化，组织决策效率明显下滑。",
            "metric_deltas": {"policy_support": -0.06, "issuer_trust_proxy": -0.09, "rumor_level": +0.07},
        },
        "whistleblower_report_burst": {
            "id": "whistleblower_report_burst",
            "title": "吹哨人集中举报",
            "detail": "内部举报集中爆发，外部质询与审计压力上升。",
            "metric_deltas": {"policy_support": -0.08, "issuer_trust_proxy": -0.1, "rumor_level": +0.12},
        },
        "supplier_collusion_inside": {
            "id": "supplier_collusion_inside",
            "title": "内外勾连采购异常",
            "detail": "内部采购与外部供应商勾连，成本与质量双失控。",
            "metric_deltas": {"policy_support": -0.07, "supply_chain_stress": +0.12, "issuer_trust_proxy": -0.08},
        },
        # Startup internal issues
        "founder_conflict_split": {
            "id": "founder_conflict_split",
            "title": "创始团队分裂",
            "detail": "联合创始人路线冲突公开化，组织稳定性骤降。",
            "metric_deltas": {"policy_support": -0.08, "issuer_trust_proxy": -0.1, "rumor_level": +0.08},
        },
        "cofounder_departure": {
            "id": "cofounder_departure",
            "title": "联合创始人离职",
            "detail": "关键岗位离开导致产品与融资叙事受损。",
            "metric_deltas": {"policy_support": -0.07, "issuer_trust_proxy": -0.09, "economy_index": -0.03},
        },
        "runway_misreporting": {
            "id": "runway_misreporting",
            "title": "跑道数据被高估",
            "detail": "内部现金跑道被误报，经营策略被迫急转弯。",
            "metric_deltas": {"policy_support": -0.06, "economy_index": -0.05},
            "cash_delta_million": -35.0,
            "debt_delta_million": +20.0,
        },
        "option_dispute_core_team": {
            "id": "option_dispute_core_team",
            "title": "核心团队期权争议",
            "detail": "激励分配争议引发关键成员不稳定。",
            "metric_deltas": {"policy_support": -0.05, "unrest": +0.04, "issuer_trust_proxy": -0.06},
        },
        "outsourced_partner_betrayal": {
            "id": "outsourced_partner_betrayal",
            "title": "外包伙伴违约",
            "detail": "关键外包方违约或拖期，交付链条断裂。",
            "metric_deltas": {"economy_index": -0.05, "supply_chain_stress": +0.12, "policy_support": -0.05},
        },
        "quick_growth_control_breakdown": {
            "id": "quick_growth_control_breakdown",
            "title": "快速增长导致失控",
            "detail": "增长速度超过治理能力，投诉与失误并发。",
            "metric_deltas": {"policy_support": -0.06, "rumor_level": +0.08, "unrest": +0.05},
        },
        "tax_compliance_lag_startup": {
            "id": "tax_compliance_lag_startup",
            "title": "税务合规滞后",
            "detail": "财税规范化滞后带来潜在处罚与补缴压力。",
            "metric_deltas": {"policy_support": -0.05, "issuer_trust_proxy": -0.06},
            "cash_delta_million": -22.0,
        },
        # Shop internal issues
        "store_cashier_leakage": {
            "id": "store_cashier_leakage",
            "title": "门店收银跑冒滴漏",
            "detail": "收银管理松散导致现金流失。",
            "metric_deltas": {"economy_index": -0.03, "policy_support": -0.03},
            "cash_delta_million": -8.0,
        },
        "inventory_shrinkage_internal": {
            "id": "inventory_shrinkage_internal",
            "title": "库存内耗异常",
            "detail": "库存损耗与管理偏差持续累积。",
            "metric_deltas": {"economy_index": -0.04, "supply_chain_stress": +0.08, "policy_support": -0.03},
        },
        "store_manager_absconding": {
            "id": "store_manager_absconding",
            "title": "店长失联/离岗",
            "detail": "关键负责人突然失联，门店运营连续性中断。",
            "metric_deltas": {"policy_support": -0.05, "unrest": +0.04, "economy_index": -0.04},
        },
        "family_partner_dispute": {
            "id": "family_partner_dispute",
            "title": "合伙人纠纷",
            "detail": "合伙关系冲突引发经营方向摇摆。",
            "metric_deltas": {"policy_support": -0.04, "issuer_trust_proxy": -0.05, "unrest": +0.03},
        },
        "local_license_noncompliance": {
            "id": "local_license_noncompliance",
            "title": "本地证照合规缺口",
            "detail": "证照办理或变更不及时，存在停业整改风险。",
            "metric_deltas": {"policy_support": -0.06, "unrest": +0.03, "economy_index": -0.03},
        },
        "kitchen_hygiene_coverup": {
            "id": "kitchen_hygiene_coverup",
            "title": "后厨问题隐瞒",
            "detail": "后厨卫生问题内部未及时上报，外泄后冲击扩大。",
            "metric_deltas": {"policy_support": -0.09, "issuer_trust_proxy": -0.1, "rumor_level": +0.1},
        },
        "staff_group_resignation": {
            "id": "staff_group_resignation",
            "title": "员工集中离职",
            "detail": "关键班次人手骤减，服务质量与营业稳定性下滑。",
            "metric_deltas": {"policy_support": -0.05, "economy_index": -0.04, "unrest": +0.03},
        },
    }
)

# Internal governance/discipline issues that should emerge with probability,
# then force decision-making pauses instead of auto-resolving.
INTERNAL_ISSUE_PRESET_IDS: dict[str, list[str]] = {
    "policy": [
        "official_corruption_probe",
        "approval_process_violation_exposed",
        "superior_investigation_spillover",
        "data_fabrication_internal",
        "procurement_irregularity",
        "discipline_commission_intervention",
        "grassroots_execution_resistance",
        "cross_department_infighting",
        "public_petition_escalation",
    ],
    "enterprise": [
        "executive_compliance_probe",
        "financial_misstatement_exposure",
        "internal_control_failure",
        "middle_management_rent_seeking",
        "channel_rebate_fraud",
        "hr_mass_turnover",
        "boardroom_power_struggle",
        "whistleblower_report_burst",
        "supplier_collusion_inside",
    ],
    "startup": [
        "founder_conflict_split",
        "cofounder_departure",
        "runway_misreporting",
        "option_dispute_core_team",
        "outsourced_partner_betrayal",
        "quick_growth_control_breakdown",
        "tax_compliance_lag_startup",
    ],
    "shop": [
        "store_cashier_leakage",
        "inventory_shrinkage_internal",
        "store_manager_absconding",
        "family_partner_dispute",
        "local_license_noncompliance",
        "kitchen_hygiene_coverup",
        "staff_group_resignation",
    ],
}


def list_black_swan_presets() -> list[dict[str, Any]]:
    return [dict(v) for v in BLACK_SWAN_PRESETS.values()]


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def build_black_swan_event(
    *,
    preset_id: str | None,
    title: str | None = None,
    detail: str | None = None,
    severity: float = 1.0,
    metric_deltas: dict[str, float] | None = None,
    cash_delta_million: float | None = None,
    fiscal_delta_billion: float | None = None,
    debt_delta_million: float | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {}
    if preset_id:
        preset_id = BLACK_SWAN_ALIASES.get(preset_id, preset_id)
        if preset_id not in BLACK_SWAN_PRESETS:
            raise ValueError(f"未知黑天鹅预设: {preset_id}")
        base = dict(BLACK_SWAN_PRESETS[preset_id])
    sev = max(0.1, min(3.0, float(severity)))
    raw_deltas = metric_deltas if isinstance(metric_deltas, dict) else base.get("metric_deltas", {})
    out_deltas: dict[str, float] = {}
    for k, v in (raw_deltas or {}).items():
        if k in _CLAMP_KEYS:
            out_deltas[k] = float(v) * sev
    out: dict[str, Any] = {
        "preset_id": preset_id or "custom",
        "title": (title or base.get("title") or "黑天鹅事件").strip(),
        "detail": (detail or base.get("detail") or "极端外部冲击触发，需快速评估并应对。").strip(),
        "severity": sev,
        "metric_deltas": out_deltas,
    }
    c = cash_delta_million if cash_delta_million is not None else base.get("cash_delta_million")
    f = fiscal_delta_billion if fiscal_delta_billion is not None else base.get("fiscal_delta_billion")
    d = debt_delta_million if debt_delta_million is not None else base.get("debt_delta_million")
    if c is not None:
        out["cash_delta_million"] = float(c) * sev
    if f is not None:
        out["fiscal_delta_billion"] = float(f) * sev
    if d is not None:
        out["debt_delta_million"] = float(d) * sev
    return out


def apply_black_swan_event(state: WorldState, event: dict[str, Any], *, source: str) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for k, dv in dict(event.get("metric_deltas") or {}).items():
        if k not in _CLAMP_KEYS:
            continue
        old = float(getattr(state, k))
        new = _clip01(old + float(dv))
        setattr(state, k, new)
        deltas[k] = new - old

    if state.cash_balance_million is not None and event.get("cash_delta_million") is not None:
        old = float(state.cash_balance_million)
        state.cash_balance_million = old + float(event["cash_delta_million"])
        deltas["cash_balance_million"] = float(state.cash_balance_million) - old
    if state.fiscal_remaining_billion is not None and event.get("fiscal_delta_billion") is not None:
        old = float(state.fiscal_remaining_billion)
        state.fiscal_remaining_billion = old + float(event["fiscal_delta_billion"])
        deltas["fiscal_remaining_billion"] = float(state.fiscal_remaining_billion) - old
    if state.debt_balance_million is not None and event.get("debt_delta_million") is not None:
        old = float(state.debt_balance_million)
        state.debt_balance_million = old + float(event["debt_delta_million"])
        deltas["debt_balance_million"] = float(state.debt_balance_million) - old

    title = str(event.get("title") or "黑天鹅事件")
    detail = str(event.get("detail") or "").strip()
    source_tag = source.strip() or "manual"
    line = f"[黑天鹅注入/{source_tag}] {title}"
    if detail:
        line += f"：{detail[:180]}"
    state.log(line)
    state.push_detail(
        "black_swan",
        title,
        detail or "外部极端冲击已注入本轮推演。",
        {
            "source": source_tag,
            "preset_id": event.get("preset_id") or "custom",
            "severity": event.get("severity"),
            "deltas": deltas,
        },
    )
    state.user_resolution_notes.append(
        f"[黑天鹅事件] {title}（source={source_tag}）已触发，请在后续策略中优先评估二阶与尾部风险。"
    )
    return deltas


def maybe_pick_internal_issue_preset(
    *,
    exercise_type: str,
    issuer_archetype: str,
    rng: random.Random,
    probability: float,
) -> str | None:
    p = max(0.0, min(1.0, float(probability)))
    if rng.random() > p:
        return None
    if exercise_type == "policy":
        pool = INTERNAL_ISSUE_PRESET_IDS["policy"]
    else:
        if issuer_archetype == "startup":
            pool = INTERNAL_ISSUE_PRESET_IDS["startup"]
        elif issuer_archetype == "sme":
            pool = INTERNAL_ISSUE_PRESET_IDS["shop"]
        else:
            pool = INTERNAL_ISSUE_PRESET_IDS["enterprise"]
    if not pool:
        return None
    return rng.choice(pool)
