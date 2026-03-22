"""
累计关键智能体(primary) LLM 达到阈值后：生成「内部风险盘点」；
用户提交方案续跑后：追加「公司内部桌面可行性评估」（不声称代表社会真实）。
"""

from __future__ import annotations

import json
from typing import Any

from people.config import Scenario
from people.providers.base import LLMClient
from people.state import WorldState

from people.agents.realism_layer import REALISM_SOCIAL_LAYER

_SYSTEM_RISK = (
    """
你是企业/政策机关内部的「态势与风险台账」撰写助手（演练用）。
根据当前宏观指标、叙事摘录、player_brief 与 company_memory_synthesis（公司内部记忆，若有），列出**需要指挥台尽快知情并准备应对方案**的风险点；不要编造未在上下文中出现的具体人名、文件编号或「已发生」的虚假事实。若记忆中已有未闭环事项，不要重复罗列相同措辞，可写「延续性风险」。
台账风格须像**真实内控/PR 联席会**：具体、可核查、分轻重；**不是**对未来的精准预测清单。盘点时兼顾**各条线/各主体的利害**（职能 KPI、财政与就业约束、媒体流量、竞品动作等），勿只写「抽象舆情」。

必须输出 JSON：
{
  "risk_bullets": [
    { "title": "string", "detail": "string", "severity": "high" | "medium" | "low" }
  ],
  "summary": "string（2-4句，总括当前最紧迫的摩擦面）"
}
risk_bullets 3～7 条为宜；severity 仅相对本场演练的紧迫度。
【禁用游戏措辞】用内参/会议纪要式表述。
"""
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()

_SYSTEM_DESK = (
    """
你是**公司内部**「战研 + 相关职能」的**桌面可行性研判**记录员（演练用，非真实批复）。

任务：指挥台已提交一版处置/应对思路（可能经过系统前一步的「专业转化」摘要）。你要判断：在**公司内部**能否调研清楚、资源是否够做最小可行包、跨部门谁要牵头——可**点到为止地关联**社会舆论、监管、供应链等外生变量，但必须写明这只是 desk 层假设，**不代表**真实社会或监管反馈。措辞像**真实跨部门会议纪要**，避免「剧情选项」式结论。研判时显式考虑**各职能自利与扯皮点**（法务免责、公关要节奏、渠道要承诺、财务要现金、政务要监管闭环等），政策场景则考虑**县乡条线与财政/稳控**。

**资源与资金（内部 desk 专用）**：必须给出**粗量级**估计——`resource_cash_band_cn`（资金/费用：一次性+首季等，用「约 ××～×× 百万人民币等效 proxy」或「订单/采购量级」表述）、`resource_manpower_band_cn`（全职/外包人月、专班编制、驻场人数等区间）、`resource_other_cn`（可选）。
**综合实盘**：须完整阅读 `decision_context` 中的 **plan_evaluate_live_context**（实盘宏观、**finance_state_live** 账面/财力/债务、**resource_pools_live** 人力与政治资本、cohort 快照、**narrative_tail**）以及 **company_memory**、**决策层指令**、**地区情境**、**operational_finance** 静态参数、机构与合作清单等。**resource_*** 粗估必须与上述信息**逻辑自洽**；若草案明显超出沙盘可承受的资金或动员能力，须在 `desk_verdict` / `desk_verdict_cn` / `why_not_feasible` 中明确写出冲突与补救前提（增资、外协、压缩范围、分阶段、政策盘财力调度等）。若用户草案隐含突破记账约束，须标注所需**组织决策或例外机制**，不得无视实盘数字。

硬性要求：
- 明确写出：桌面结论与真实情况**可能一致也可能不一致**，不得当作对外承诺或法务意见。
- 输出 JSON：
{
  "desk_verdict": "go" | "conditional" | "no_go" | "needs_more_info",
  "desk_verdict_cn": "string（一两句人话）",
  "resource_cash_band_cn": "string（资金/费用粗量级，百万 proxy 区间或采购量级）",
  "resource_manpower_band_cn": "string（人力/外包人月/专班粗量级）",
  "resource_other_cn": "string（可选：场地、算力、政务窗口、第三方审计等其他资源要点）",
  "internal_gaps": ["string", ...],
  "social_linkage": "string（与社会侧关联的 desk 假设，简短）",
  "recommended_internal_next_steps": ["string", ...],
  "why_not_feasible": ["string", ...],
  "caveat": "string（再次强调仅限内部桌面推演）",
  "estimated_timeline_desk_cn": "string（与专业转化中的阶段周期对齐或细化内部里程碑；演练估计，非承诺）"
}
why_not_feasible：若 verdict 为 go 仍可能存在的内部短板；若为 no_go/conditional/needs_more_info 则写清**卡点与不可行原因**（条目式）。
"""
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()


def run_risk_inventory_llm(
    client: LLMClient,
    *,
    scenario: Scenario,
    state: WorldState,
    decision_context: dict[str, Any],
    primary_calls_so_far: int,
    temperature: float = 0.35,
) -> dict[str, Any]:
    payload = {
        "risk_milestone_inventory": True,
        "exercise_type": scenario.exercise_type,
        "primary_llm_calls_so_far": primary_calls_so_far,
        "tick": state.tick,
        "macro": state.snapshot_metrics(),
        "player_brief_excerpt": (scenario.player_brief or "")[:2400],
        "narrative_tail": state.narrative[-24:],
        "company_memory_synthesis": (state.company_memory_synthesis or "")[:3200],
        "decision_context": decision_context,
    }
    user = "请生成本场演练的内部风险盘点：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        raw = client.complete_json(system=_SYSTEM_RISK, user=user, temperature=temperature)
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return _fallback_inventory(state, primary_calls_so_far)
    bullets = raw.get("risk_bullets")
    if not isinstance(bullets, list) or not bullets:
        return _fallback_inventory(state, primary_calls_so_far)
    cleaned: list[dict[str, Any]] = []
    for b in bullets[:10]:
        if not isinstance(b, dict):
            continue
        t = str(b.get("title") or "").strip()
        d = str(b.get("detail") or "").strip()
        if not t and not d:
            continue
        sev = str(b.get("severity") or "medium").strip().lower()
        if sev not in ("high", "medium", "low"):
            sev = "medium"
        cleaned.append({"title": t or "（未命名风险）", "detail": d, "severity": sev})
    if not cleaned:
        return _fallback_inventory(state, primary_calls_so_far)
    summary = str(raw.get("summary") or "").strip() or "请结合指标与叙事，由指挥台逐项给出应对。"
    return {"risk_bullets": cleaned, "summary": summary}


def _fallback_inventory(state: WorldState, primary_calls_so_far: int) -> dict[str, Any]:
    m = state.snapshot_metrics()
    bullets: list[dict[str, Any]] = [
        {
            "title": "指标摩擦",
            "detail": f"当前 rumor_level={m.get('rumor_level', 0):.3f}、unrest={m.get('unrest', 0):.3f}，需核对是否与对外口径一致。",
            "severity": "medium",
        },
        {
            "title": "推演深度",
            "detail": f"关键智能体 LLM 已调用 {primary_calls_so_far} 次，叙事中可能已积累多条未闭环表态。",
            "severity": "low",
        },
    ]
    return {
        "risk_bullets": bullets,
        "summary": "（模型不可用或解析失败）以下为基于指标的占位盘点；请指挥台据实补充。",
    }


def format_risk_inventory_for_pause(inv: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = str(inv.get("summary") or "").strip()
    if summary:
        lines.append(summary)
        lines.append("")
    bullets = inv.get("risk_bullets")
    if isinstance(bullets, list):
        for i, b in enumerate(bullets, 1):
            if not isinstance(b, dict):
                continue
            t = str(b.get("title") or "").strip()
            d = str(b.get("detail") or "").strip()
            sev = str(b.get("severity") or "").strip()
            head = f"{i}. [{sev}] {t}" if sev else f"{i}. {t}"
            lines.append(head)
            if d:
                lines.append(f"   {d}")
    lines.append("")
    lines.append(
        "—— 以上为演练生成的内部台账摘要。请指挥台在下方填写应对方案；续跑后将追加「公司内部桌面可行性评估」（不代表现实社会或监管结论）。"
    )
    return "\n".join(lines).strip()


def evaluate_internal_desk_feasibility(
    client: LLMClient,
    *,
    scenario: Scenario,
    state: WorldState,
    user_plain_solution: str,
    professional_digest: dict[str, Any] | None,
    risk_inventory: dict[str, Any] | None,
    decision_context: dict[str, Any],
    temperature: float = 0.3,
) -> dict[str, Any] | None:
    payload = {
        "internal_desk_feasibility": True,
        "exercise_type": scenario.exercise_type,
        "tick": state.tick,
        "macro": state.snapshot_metrics(),
        "user_command_plain": user_plain_solution.strip()[:4000],
        "professional_digest_excerpt": professional_digest if isinstance(professional_digest, dict) else None,
        "prior_risk_inventory_excerpt": risk_inventory if isinstance(risk_inventory, dict) else None,
        "company_memory_synthesis": (state.company_memory_synthesis or "")[:3200],
        "decision_context": decision_context,
    }
    user = "请做公司内部桌面可行性研判：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        raw = client.complete_json(system=_SYSTEM_DESK, user=user, temperature=temperature)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def format_desk_feasibility_note(d: dict[str, Any]) -> str:
    verdict = str(d.get("desk_verdict_cn") or d.get("desk_verdict") or "").strip()
    cave = str(d.get("caveat") or "").strip()
    soc = str(d.get("social_linkage") or "").strip()
    gaps = d.get("internal_gaps")
    steps = d.get("recommended_internal_next_steps")
    rcash = str(d.get("resource_cash_band_cn") or "").strip()
    rman = str(d.get("resource_manpower_band_cn") or "").strip()
    roth = str(d.get("resource_other_cn") or "").strip()
    lines = [
        "【公司内部桌面可行性（演练生成，与真实社会反馈可能一致也可能不一致）】",
        f"研判结论：{verdict}" if verdict else "研判结论：（空）",
    ]
    if rcash:
        lines.append(f"资金/费用量级（desk）：{rcash}")
    if rman:
        lines.append(f"人力与组织量级（desk）：{rman}")
    if roth:
        lines.append(f"其他资源（desk）：{roth}")
    if isinstance(gaps, list) and gaps:
        lines.append("内部缺口：" + "；".join(str(x) for x in gaps[:8] if str(x).strip()))
    if soc:
        lines.append(f"与社会侧关联（desk 假设）：{soc}")
    if isinstance(steps, list) and steps:
        lines.append("建议内部下一步：" + "；".join(str(x) for x in steps[:8] if str(x).strip()))
    etd = str(d.get("estimated_timeline_desk_cn") or "").strip()
    if etd:
        lines.append(f"内部里程碑周期（desk 估计）：{etd}")
    wnf = d.get("why_not_feasible")
    if isinstance(wnf, list) and wnf:
        lines.append(
            "不可行/存疑点："
            + "；".join(str(x) for x in wnf[:10] if str(x).strip()),
        )
    if cave:
        lines.append(f"声明：{cave}")
    return "\n".join(lines)
