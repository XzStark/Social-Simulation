from __future__ import annotations

import json
import random
from typing import Any

from people.config import Scenario
from people.providers.base import LLMClient
from people.roster import ResolvedAgent
from people.state import WorldState

from people.agents.realism_layer import REALISM_INTERACTION_APPEND, REALISM_SOCIAL_LAYER

INTERACTION_SYSTEM = (
    """
你在做**真实组织与社会网络**里会发生的沟通回合生成器（演练用）：像电话、微信工作群、采访求证、内部会签，而不是网游组队广播。
必须输出 JSON：
{
  "interactions": [
    {
      "from_id": "string",
      "to_id": "string",
      "channel": "电话|闭门会|记者会|约谈|函件|工作群",
      "kind": "regulator_contact|gov_media_coordination|enterprise_media_response|cross_actor_coordination",
      "summary": "1-2句，写清谁联系谁、在谈什么、下一步怎么做",
      "cost_million": number|null,
      "fiscal_billion": number|null
    }
  ]
}
规则：
- 只写真实组织会发生的联系，禁止夸张。
- 优先覆盖监管/政府↔企业、政府↔媒体、企业↔媒体。
- **产品/消费类演练**（exercise_type=product）：必须覆盖「一线用户/渠道/社群侧 → 媒体或自媒体」的**报料、求证、投稿、跟进**（kind 可用 cross_actor_coordination）；summary 须点出**具体新问题类型**（如交付、品控、续航、软件缺陷、售后、价格落差、替代品类比、职场禁用争议等），**不要**每一轮交流都重复同一条隐私/摄像头叙事，除非 macro 与 cohort 强烈指向该线。
- 若 user JSON 的 decision_context 含 **market_competitors**（多家竞品/替代方案），交流摘要须承认**竞争格局**：份额挤压、对比评测、渠道抢位、公关互啄、白牌与生态绑定等；勿把市场写成只有发行方一家在做同类产品。
- 若 user JSON 含 cohort_social_feedback_proxy，据此推断哪类人群更可能先发声、媒体如何跟进或降温。
- summary 要体现**具体动作**（谁找谁、要什么材料、卡在什么环节、下次节点），而非口号或剧情旁白。
- 可适当体现**现实摩擦**：法务未回、热线占线、爆料人拒实名、稿件待二审、跨时区回复慢等（点到为止，勿编具体案号）。
- 本输出**不预测**真实案件结果或股价；只描述「在此情境下常见的沟通类型」。
- cost/fiscal 只有明确发生预算动作时填写，其他为 null。

"""
    + REALISM_SOCIAL_LAYER
    + "\n"
    + REALISM_INTERACTION_APPEND
    + "\n"
).strip()

PRODUCT_DEPARTMENT_CHAIN: list[tuple[str, str, str, str]] = [
    ("customer_service_center", "public_opinion_ops", "工单系统", "客服汇总投诉与高频误解，转舆情运营复核。"),
    ("public_opinion_ops", "brand_pr_center", "晨会", "舆情团队整理传播路径，提交品牌公关拟定回应框架。"),
    ("brand_pr_center", "legal_compliance_center", "会签", "公关稿件进入法务合规会签，校对风险表述与证据边界。"),
    ("legal_compliance_center", "government_affairs_center", "函件", "法务与政务事务同步口径，准备监管沟通材料。"),
    ("government_affairs_center", "regulatory_liaison_window", "约谈", "政务事务对接监管窗口，说明处置进度与下一步节点。"),
    ("regulatory_liaison_window", "executive_steering_committee", "简报", "监管反馈回流高层委员会，触发资源调度与时序重排。"),
    ("supply_chain_control_tower", "operations_planning_center", "调度会", "供应链控制塔更新到货与缺料风险，运营计划调整排产。"),
    ("operations_planning_center", "sales_channel_mgmt", "工作群", "运营计划下发渠道节奏与交付承诺修订。"),
    ("sales_channel_mgmt", "executive_steering_committee", "周报", "渠道反馈转高层，评估销量、口碑与补贴策略。"),
    ("finance_treasury_center", "executive_steering_committee", "预算会", "财务报告现金消耗与预算压力，请求费用优先级裁决。"),
]

POLICY_DEPARTMENT_CHAIN: list[tuple[str, str, str, str]] = [
    ("township_frontline_team", "county_policy_office", "工作群", "乡镇一线反馈执行障碍与群众疑问，提交县级政策办汇总。"),
    ("county_policy_office", "county_data_bureau_office", "联席会", "县级政策办与数据局核对台账口径与流程节点。"),
    ("county_data_bureau_office", "county_justice_bureau", "会签", "司法条线审看文本合法性、程序与救济路径。"),
    ("county_justice_bureau", "county_public_security", "联动会", "司法与公安研判线下稳控风险与秩序保障边界。"),
    ("county_public_security", "county_emergency_bureau", "应急会商", "公安与应急会商突发预案，明确响应级别。"),
    ("county_finance_bureau", "county_policy_office", "预算评审", "财政核算经费承受能力与支出节奏。"),
    ("county_market_regulation", "county_convergence_media", "通气会", "市监与融媒体对齐消费者告知与信息发布口径。"),
    ("county_cyberspace_office", "county_convergence_media", "选题沟通", "网信与融媒体校准网络传播风险点和辟谣节奏。"),
    ("county_petitions_office", "county_policy_office", "专报", "信访专报重点诉求与聚集风险，建议调整窗口服务。"),
    ("county_policy_office", "municipal_supervision_group", "专报", "县级向市级督导组上报进度与问题清单，申请协调资源。"),
]

STARTUP_DEPARTMENT_CHAIN: list[tuple[str, str, str, str]] = [
    ("customer_voice_channel", "founder_ceo", "社群/工单", "早期用户与志愿者渠道汇总投诉与误读，直达创始人拍板。"),
    ("founder_ceo", "core_build_team", "站会", "创始人与产品技术对齐止血范围、排期砍切与可演示里程碑。"),
    ("core_build_team", "lean_growth_ops", "工作群", "研发输出可解释的修复说明，增长/运营承接话术与推送实验。"),
    ("lean_growth_ops", "fractional_cfo_advisor", "电话", "活动与投放消耗拉齐兼职财务，核对现金与应付边界。"),
    ("fractional_cfo_advisor", "founder_ceo", "闭门会", "财务回报现金流红线与回款节奏，请创始人拍板费用与外包。"),
    ("founder_ceo", "external_counsel_pool", "邮件", "外包律师审阅对外声明与证据留存，规避过度承诺。"),
    ("community_ops_volunteer", "customer_voice_channel", "群接龙", "志愿者整理 FAQ 与苗头舆情，回填官方渠道。"),
    ("external_pr_advisor", "lean_growth_ops", "简报", "外聘 PR 与内部增长对齐口径，避免两层说法打架。"),
]


def _pick_pairs(rng: random.Random, agents: list[ResolvedAgent], scenario: Scenario) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    def add(a: str, b: str) -> None:
        if a == b:
            return
        p = (a, b)
        if p not in pairs:
            pairs.append(p)

    # 虚拟组织中枢：保证监管/媒体等能与决策主体对话
    command_id = "issuer_command_center" if scenario.exercise_type == "product" else "policy_command_center"
    for a in agents:
        cs = set(a.categories)
        role = a.role.lower()
        is_gov = ("regulator" in cs) or ("政府" in a.role) or ("监管" in a.role) or ("执法" in a.role)
        is_media = ("media" in cs) or ("媒体" in a.role) or ("主编" in a.role) or ("记者" in a.role)
        if is_gov:
            add(a.id, command_id)
        if is_media:
            add(command_id, a.id)
        if is_gov and is_media:
            add(a.id, a.id + "_editorial_desk")
        if ("legal" in cs) and ("media" in cs or "记者" in role):
            add(a.id, command_id)

    # 产品侧：增加「非媒体角色 → 媒体」候选边，便于模型生成用户反馈上报料链
    if scenario.exercise_type == "product":
        media_agents = [
            a
            for a in agents
            if ("media" in set(a.categories))
            or ("媒体" in a.role)
            or ("记者" in a.role)
            or ("博主" in a.role)
            or ("主编" in a.role)
        ]
        voice_kw = ("用户", "渠道", "店", "采购", "行政", "论坛", "社群", "维权", "买家", "店主", "店长", "运营", "电商")
        voice_agents = [
            a
            for a in agents
            if a not in media_agents
            and any(k in a.role for k in voice_kw)
            and ("competitor" not in set(a.categories))
        ]
        if media_agents and voice_agents:
            m = rng.choice(media_agents)
            rng.shuffle(voice_agents)
            for v in voice_agents[:2]:
                add(v.id, m.id)
        for a in agents:
            if "competitor" in set(a.categories):
                add(command_id, a.id)

    # 常规智能体之间再补充 1-2 组
    cand = [(a.id, b.id) for i, a in enumerate(agents) for b in agents[i + 1 :]]
    rng.shuffle(cand)
    for p in cand[:2]:
        add(*p)

    return pairs[:6]


def _cohort_by_layer(state: WorldState) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"lower": [], "middle": [], "upper": [], "mixed": []}
    for c in state.cohorts:
        out.setdefault(c.class_layer, []).append(c.id)
    return out


def _baseline_hierarchy_interactions(
    *,
    state: WorldState,
    scenario: Scenario,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    每 tick 至少给出一组真实组织链路：
    基层 -> 中层 -> 高层，以及用户反馈 -> 组织中枢。
    """
    layers = _cohort_by_layer(state)
    command_id = "issuer_command_center" if scenario.exercise_type == "product" else "policy_command_center"

    # 若场景有显式用户群体，优先把群体反馈放进沟通链
    lower_user = rng.choice(layers["lower"]) if layers["lower"] else (layers["middle"][0] if layers["middle"] else "user_frontline")
    middle_user = rng.choice(layers["middle"]) if layers["middle"] else "user_middle_group"
    upper_user = rng.choice(layers["upper"]) if layers["upper"] else "user_upper_group"

    is_startup_product = (
        scenario.exercise_type == "product" and getattr(scenario.issuer, "archetype", None) == "startup"
    )

    # 组织内部层级（可被 LLM 回合补充/覆盖）
    if is_startup_product:
        internal_chain = [
            {
                "from_id": "customer_voice_channel",
                "to_id": "founder_ceo",
                "channel": "工作群",
                "kind": "cross_actor_coordination",
                "summary": "社群与工单汇总一线误读与投诉样本，创始人直接消化优先级与对外口径边界。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": "founder_ceo",
                "to_id": "core_build_team",
                "channel": "站会",
                "kind": "cross_actor_coordination",
                "summary": "创始人与产品技术小队对齐止血范围：能砍什么、必须保什么、何时能给用户可验证说明。",
                "cost_million": None,
                "fiscal_billion": None,
            },
        ]
        user_loop = [
            {
                "from_id": lower_user,
                "to_id": "customer_voice_channel",
                "channel": "电话",
                "kind": "cross_actor_coordination",
                "summary": "基层用户把具体痛点与谣言来源捅进反馈通道，要求短周期内有人认领与回复。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": middle_user,
                "to_id": "lean_growth_ops",
                "channel": "函件",
                "kind": "cross_actor_coordination",
                "summary": "中层用户/小 B 客户关注交付承诺与退款边界，增长运营承接话术与个案升级路径。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": upper_user,
                "to_id": "founder_ceo",
                "channel": "闭门会",
                "kind": "cross_actor_coordination",
                "summary": "投资人或关键合作方代表施压时间表与现金约束，要求创始人明确取舍与下一步节点。",
                "cost_million": None,
                "fiscal_billion": None,
            },
        ]
        chain_src = STARTUP_DEPARTMENT_CHAIN
    else:
        internal_chain = [
            {
                "from_id": "frontline_execution_team",
                "to_id": "department_manager_office",
                "channel": "工作群",
                "kind": "cross_actor_coordination",
                "summary": "基层执行团队回传一线阻力、误读点与投诉样本，请中层统一口径并分派处置。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": "department_manager_office",
                "to_id": "executive_steering_committee",
                "channel": "闭门会",
                "kind": "cross_actor_coordination",
                "summary": "中层汇总风险矩阵与资源缺口，提交高层拍板：节奏调整、预算边界与对外发声窗口。",
                "cost_million": None,
                "fiscal_billion": None,
            },
        ]
        user_loop = [
            {
                "from_id": lower_user,
                "to_id": "frontline_execution_team",
                "channel": "电话",
                "kind": "cross_actor_coordination",
                "summary": "基层用户反馈具体痛点与误解来源，要求给出可执行说明而非口号。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": middle_user,
                "to_id": "department_manager_office",
                "channel": "函件",
                "kind": "cross_actor_coordination",
                "summary": "中层用户/组织客户提交正式意见，关注合同条款、服务承诺与责任边界。",
                "cost_million": None,
                "fiscal_billion": None,
            },
            {
                "from_id": upper_user,
                "to_id": command_id,
                "channel": "闭门会",
                "kind": "cross_actor_coordination",
                "summary": "高层用户代表或关键利益相关方提出约束条件，要求管理层明确时间表与问责链。",
                "cost_million": None,
                "fiscal_billion": None,
            },
        ]
        chain_src = PRODUCT_DEPARTMENT_CHAIN if scenario.exercise_type == "product" else POLICY_DEPARTMENT_CHAIN
    # 每个 tick 按窗口切片，确保“部门很多”且长期运行可覆盖更多真实条线
    k = min(6, len(chain_src))
    start = max(0, (state.tick - 1) % len(chain_src))
    dept_edges: list[dict[str, Any]] = []
    for i in range(k):
        a, b, ch, summary = chain_src[(start + i) % len(chain_src)]
        dept_edges.append(
            {
                "from_id": a,
                "to_id": b,
                "channel": ch,
                "kind": "cross_actor_coordination",
                "summary": summary,
                "cost_million": None,
                "fiscal_billion": None,
            }
        )

    return internal_chain + user_loop + dept_edges


def run_social_interactions(
    client: LLMClient,
    *,
    state: WorldState,
    scenario: Scenario,
    tick_agents: list[ResolvedAgent],
    simulation_context: dict[str, Any],
    rng: random.Random,
    intervention_notes: str = "",
    temperature: float = 0.3,
) -> list[dict[str, Any]]:
    out = _baseline_hierarchy_interactions(
        state=state,
        scenario=scenario,
        rng=rng,
    )
    pairs = _pick_pairs(rng, tick_agents, scenario)
    if not pairs:
        return out
    payload = {
        "interaction_round": True,
        "exercise_type": scenario.exercise_type,
        "tick": state.tick,
        "macro": state.snapshot_metrics(),
        "cohort_social_feedback_proxy": state.cohort_proxy_for_prompts(),
        "intervention_notes": intervention_notes,
        "pairs": [{"from_id": a, "to_id": b} for a, b in pairs],
        "agents": [
            {
                "id": a.id,
                "role": a.role,
                "categories": list(a.categories),
            }
            for a in tick_agents
        ],
        "decision_context": simulation_context,
    }
    user = "请生成本 tick 的跨主体交流：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = client.complete_json(system=INTERACTION_SYSTEM, user=user, temperature=temperature)
    data = raw.get("interactions") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return []
    model_out: list[dict[str, Any]] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        a = str(it.get("from_id") or "").strip()
        b = str(it.get("to_id") or "").strip()
        s = str(it.get("summary") or "").strip()
        if not a or not b or not s:
            continue
        model_out.append(
            {
                "from_id": a,
                "to_id": b,
                "channel": str(it.get("channel") or "工作群"),
                "kind": str(it.get("kind") or "cross_actor_coordination"),
                "summary": s[:240],
                "cost_million": float(it["cost_million"]) if it.get("cost_million") is not None else None,
                "fiscal_billion": float(it["fiscal_billion"]) if it.get("fiscal_billion") is not None else None,
            }
        )
    return (out + model_out)[:8]
