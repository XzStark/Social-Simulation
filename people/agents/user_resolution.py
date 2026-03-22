"""
指挥台白话处置 → 执行层/政策条线专业转化（快模型一次调用）。
用于续跑前 enrich 上下文，供后续关键智能体与部门交流链使用。
"""

from __future__ import annotations

import json
from typing import Any

from people.config import Scenario
from people.providers.base import LLMClient
from people.state import WorldState

from people.agents.realism_layer import REALISM_SOCIAL_LAYER

_NO_GAME = """
【禁用隐喻】严禁游戏式措辞。用机关公文、企业内参、会议纪要式表述。
【严肃】输出为演练辅助，非真实法律意见或行政决定。
【校准】这是情境推演下的处置备忘，**不是**对真实市场或监管结果的预测；避免「必胜/必败」式措辞，可写前提与条件。
""".strip()

_PLAN_EVAL_SYNTHESIS = """
【综合研判·必读】`decision_context` 中若含 **plan_evaluate_live_context**，表示当前推演的**实盘快照**（tick、宏观指标曲线、实盘账面现金/财力/债务、人力与政治资本池、cohort 态度·激活度·权重、叙事尾部、财务模块是否开启等）。须与同一 context 内的 **company_memory_synthesis**、**decision_layer_directive**、**regional_grounding**、**operational_finance**（YAML 参数）、**institutions** / **cooperations** 等**一并纳入**可执行性判断。
撰写 **feasibility** 与 **feasibility_notes** 时：**不得忽视资金与人力现实**——若草案与实盘记账、运营成本/债务压力、财力池余量或 cohort 动员难度明显不匹配，须写明**缺口、前提**（如增资、外协、压缩范围、分阶段）或将 **feasibility** 定为 `likely_infeasible` / `needs_more_info`；若信息不足亦须 `needs_more_info` 并列出待核实项。
""".strip()

SYSTEM_PRODUCT = (
    f"""
你是大型企业集团「执行层联合办公会」记录员。用户（指挥台）用白话描述危机处置意图，你要：
1) 准确概括其真实意图（不添油加醋）；
2) 转化为可分工执行的专业表述（公关口径边界、法务合规要点、客服/渠道动作、供应链与交付、财务与预算等，择要覆盖）；
3) 给出可执行性判断与关键前提/风险。

必须输出 JSON：
{{
  "understood_intent": "string",
  "intent_bullets": ["string", ...],
  "professional_execution_plan": "string",
  "feasibility": "feasible_with_conditions" | "likely_infeasible" | "needs_more_info",
  "feasibility_notes": "string",
  "involved_functions": ["string", ...],
  "policy_equivalent_steps": "",
  "estimated_timeline_cn": "string"
}}
intent_bullets 最多 5 条；involved_functions 用职能部门简称（如法务合规、品牌公关、客服、供应链、财务资金、政府事务等）。
feasibility 必须三选一；policy_equivalent_steps 产品场景填空字符串 ""。
estimated_timeline_cn：分阶段给出**大致**周期（如 0～2 周口径冻结 / 1～3 月功能迭代 / 视监管反馈再定等），标明**演练用排序与量级**，非合同或对外承诺。
转化方案须**显式覆盖**各职能/外包方的利害与可能抵制点（与下文「真实世界条线」一致），勿写成单一部门包办。
{_NO_GAME}
"""
    + "\n\n"
    + _PLAN_EVAL_SYNTHESIS
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()

SYSTEM_STARTUP = (
    f"""
你是早期创业公司的「创始人核心会议」记录员。用户（指挥台）往往用白话下达危机处置意图；公司人少、职能合并、现金与带宽都紧，常依赖外包律师/兼职财务/志愿者社群。
1) 概括真实意图，避免替用户编造没说的承诺；
2) 转化为**可落地**的专业动作：哪些创始人亲自盯、哪些可外包、哪些必须砍范围或延期；公关口径与证据链、用户沟通渠道、产品/技术止血、融资/IR 沟通择要；
3) 明确可执行性：人力/现金/时间窗口是否够；若不够，写清「最小可行处置」与「必须放弃或延后」的部分。须点出**外包/兼职响应滞后、社群口径与官方不一致**等典型摩擦。

必须输出 JSON（键名与大厂产品端相同，便于引擎解析）：
{{
  "understood_intent": "string",
  "intent_bullets": ["string", ...],
  "professional_execution_plan": "string",
  "feasibility": "feasible_with_conditions" | "likely_infeasible" | "needs_more_info",
  "feasibility_notes": "string",
  "involved_functions": ["string", ...],
  "policy_equivalent_steps": "",
  "estimated_timeline_cn": "string"
}}
involved_functions 可用：创始人、产品技术、增长运营、社群、外包律师、兼职财务、投资人沟通 等；policy_equivalent_steps 填空字符串 ""。
estimated_timeline_cn：同上，突出人手紧、外包排期与融资沟通等约束下的**阶段周期（演练估计）**。
{_NO_GAME}
"""
    + "\n\n"
    + _PLAN_EVAL_SYNTHESIS
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()

SYSTEM_POLICY = (
    f"""
你是地方政府政策推演中的「政策办牵头协调纪要」撰写助手。用户（指挥台）用白话描述应对思路，你要：
1) 概括其政策意图与稳控/民生/程序关切；
2) 映射到常见行政流程节点（不必声称对应某一具体法规条文号，写机制与顺序即可）：议题与目标 → 调研与部门会商 → 合法性/合规审查 → 社会稳定与舆情风险评估 → 征求意见或听证（如适用）→ 集体审议决策 → 公布与解读 → 执行与督查评估；
3) 转化为可供各条线执行的专业表述（数据口径、乡镇动员、宣传节奏、财政与采购边界等择要），并兼顾**县乡条块、财政就业、网信融媒体、信访督导**等常见利害（与下文「真实世界条线」一致）；
4) 评估在「当前演练态势」下是否可推进及前提条件。

必须输出 JSON：
{{
  "understood_intent": "string",
  "intent_bullets": ["string", ...],
  "professional_execution_plan": "string",
  "feasibility": "feasible_with_conditions" | "likely_infeasible" | "needs_more_info",
  "feasibility_notes": "string",
  "involved_functions": ["string", ...],
  "policy_equivalent_steps": "string",
  "estimated_timeline_cn": "string"
}}
policy_equivalent_steps 用 2～4 句说明白话意图对应到上述流程的哪些环节、先后关系。
estimated_timeline_cn：按行政程序常见节奏给出**阶段周期（演练估计）**（如征求意见窗口、会签、试点扩围等），非对外承诺。
{_NO_GAME}
"""
    + "\n\n"
    + _PLAN_EVAL_SYNTHESIS
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()


def professionalize_plain_resolution(
    client: LLMClient,
    *,
    scenario: Scenario,
    state: WorldState,
    plain_solution: str,
    decision_context: dict[str, Any],
    temperature: float = 0.25,
) -> dict[str, Any] | None:
    if scenario.exercise_type == "policy":
        sys = SYSTEM_POLICY
    elif getattr(scenario.issuer, "archetype", None) == "startup":
        sys = SYSTEM_STARTUP
    else:
        sys = SYSTEM_PRODUCT
    payload = {
        "plain_language_resolution": True,
        "exercise_type": scenario.exercise_type,
        "issuer_archetype": scenario.issuer.archetype,
        "tick_when_submitted": state.tick,
        "user_plain_text": plain_solution.strip(),
        "player_brief_excerpt": (scenario.player_brief or "")[:2000],
        "macro": state.snapshot_metrics(),
        "company_memory_synthesis": (state.company_memory_synthesis or "")[:3200],
        "decision_context": decision_context,
    }
    user = "输入 JSON 上下文：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = client.complete_json(system=sys, user=user, temperature=temperature)
    if not isinstance(raw, dict):
        return None
    feas = raw.get("feasibility")
    if feas not in ("feasible_with_conditions", "likely_infeasible", "needs_more_info"):
        raw["feasibility"] = "needs_more_info"
    bullets = raw.get("intent_bullets")
    if not isinstance(bullets, list):
        raw["intent_bullets"] = []
    funcs = raw.get("involved_functions")
    if not isinstance(funcs, list):
        raw["involved_functions"] = []
    for k in ("understood_intent", "professional_execution_plan", "feasibility_notes", "estimated_timeline_cn"):
        if raw.get(k) is None:
            raw[k] = ""
        raw[k] = str(raw[k]).strip()
    if scenario.exercise_type == "product":
        raw["policy_equivalent_steps"] = ""
    elif raw.get("policy_equivalent_steps") is None:
        raw["policy_equivalent_steps"] = ""
    else:
        raw["policy_equivalent_steps"] = str(raw["policy_equivalent_steps"]).strip()
    return raw


def format_resolution_note_for_agents(d: dict[str, Any], *, scenario: Scenario | None = None) -> str:
    """拼成一段注入 key_actor 上下文的附录。"""
    if scenario and scenario.exercise_type == "policy":
        head = "【指挥台白话处置·政策条线专业转化（演练生成，非真实批复）】"
    elif scenario and getattr(scenario.issuer, "archetype", None) == "startup":
        head = "【指挥台白话处置·创业公司专业转化（演练生成，非真实批复）】"
    else:
        head = "【指挥台白话处置·执行层专业转化（演练生成，非真实批复）】"
    lines = [
        head,
        f"意图概括：{d.get('understood_intent', '')}",
    ]
    bs = d.get("intent_bullets") or []
    if bs:
        lines.append("要点：" + "；".join(str(b) for b in bs[:5]))
    lines.append(f"专业方案与分工择要：{d.get('professional_execution_plan', '')}")
    lines.append(f"可执行性：{d.get('feasibility', '')} — {d.get('feasibility_notes', '')}")
    fs = d.get("involved_functions") or []
    if fs:
        lines.append("涉及职能：" + "、".join(str(x) for x in fs[:12]))
    pe = (d.get("policy_equivalent_steps") or "").strip()
    if pe:
        lines.append(f"政策流程映射：{pe}")
    et = (d.get("estimated_timeline_cn") or "").strip()
    if et:
        lines.append(f"阶段周期（演练估计）：{et}")
    return "\n".join(lines)
