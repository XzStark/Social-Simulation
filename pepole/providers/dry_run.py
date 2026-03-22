from __future__ import annotations

import json
import re
from typing import Any


class DryRunClient:
    """无 API Key 时使用：从 user 文本里抽数字做伪随机扰动，产出合法 JSON 结构。"""

    def __init__(self, label: str = "dry") -> None:
        self.label = label

    def complete_json(self, *, system: str, user: str, temperature: float) -> dict[str, Any]:
        seed = sum(ord(c) for c in user[:200]) % 997
        jitter = (seed % 17) / 100.0 * max(0.1, temperature)
        if "regional_grounding_pass" in user:
            return {
                "bullet_points": [
                    f"[{self.label}] 地方常见执行口径与部门协同习惯（占位）",
                    f"[{self.label}] 上位法律/行政法规类型级关切（勿当真文号）",
                    f"[{self.label}] 基层信息渠道：熟人、业主群、短视频切片（占位）",
                ],
                "summary_100": f"[{self.label}] 合成情境：在既有地方与中央框架下的常态 baseline。",
                "caveats": ["需核实具体条文", "仅为演练先验"],
            }
        if '"plain_language_resolution"' in user:
            is_policy = '"exercise_type": "policy"' in user
            is_startup = '"issuer_archetype": "startup"' in user
            if is_startup and not is_policy:
                return {
                    "understood_intent": f"[{self.label}] 理解为少人手、紧现金下先止血再解释，避免过度承诺。",
                    "intent_bullets": ["砍范围或延期非核心", "创始人盯对外口径与证据", "外包律师/兼职财务把关"],
                    "professional_execution_plan": f"[{self.label}] 转化：社群与工单统一回复模板；产品小队出可验证修复节点；增长侧承接话术；财务核对现金与应付；必要时外聘声明审阅。",
                    "feasibility": "feasible_with_conditions",
                    "feasibility_notes": "前提：外包与兼职可即时响应；若现金不足需明确最小可行处置与放弃项。",
                    "involved_functions": ["创始人", "产品技术", "增长运营", "社群", "外包律师", "兼职财务"],
                    "policy_equivalent_steps": "",
                    "estimated_timeline_cn": f"[{self.label}] 0～1 周止血与口径；2～4 周迭代可验证修复；外包与融资沟通并行视排期。",
                }
            return {
                "understood_intent": f"[{self.label}] 理解为稳节奏、讲清楚事实边界、留出核查与沟通窗口。",
                "intent_bullets": ["先稳情绪与误读", "再给可验证信息", "明确下一步责任与时间"],
                "professional_execution_plan": f"[{self.label}] 转化：统一口径与证据链；法务/合规会签；客服与渠道同步；必要时第三方说明会。",
                "feasibility": "feasible_with_conditions",
                "feasibility_notes": "前提：人力与预算可调配、监管/政务沟通渠道畅通。",
                "involved_functions": ["品牌公关", "法务合规", "客服", "政府事务", "供应链"],
                "policy_equivalent_steps": (
                    "对应流程：部门会商→合法性把关→社会稳定与舆情风险评估→征求意见或说明→集体审议→公布解读→执行督查。"
                    if is_policy
                    else ""
                ),
                "estimated_timeline_cn": f"[{self.label}] 0～2 周口径与会签；1～3 月试点与督查节奏视程序推进。",
            }
        if '"risk_milestone_inventory"' in user:
            return {
                "risk_bullets": [
                    {
                        "title": "舆情节奏",
                        "detail": "猜疑链在社交平台易被截断引用，需统一口径与证据边界。",
                        "severity": "high",
                    },
                    {
                        "title": "资源边界",
                        "detail": "并行沟通与外包排期是否覆盖当前摩擦面。",
                        "severity": "medium",
                    },
                ],
                "summary": f"[{self.label}] 占位：关键推演次数已达标，请指挥台逐条给出应对方案。",
            }
        if '"internal_desk_feasibility"' in user:
            return {
                "desk_verdict": "conditional",
                "desk_verdict_cn": f"[{self.label}] 内部可先形成最小可行方案包，但关键数据需职能回填验证。",
                "resource_cash_band_cn": f"[{self.label}] 约 15～60 百万 proxy（一次性响应+首季运营，内部 desk 不按沙盘账面封顶）",
                "resource_manpower_band_cn": f"[{self.label}] 峰值约 6～12 人月（公关/法务/客服/产研拆并，视外包比例）",
                "resource_other_cn": f"[{self.label}] 云资源与监测工具、必要时外聘声明审阅",
                "internal_gaps": ["数据口径未冻结", "跨部门责任人未对齐"],
                "social_linkage": "社会侧反应仅为 desk 假设，真实舆论需另做监测。",
                "recommended_internal_next_steps": ["48h 内口径会", "列出依赖项与 Owner"],
                "why_not_feasible": [
                    "预算未锁定",
                    "外包律师排期未确认",
                    "与记忆中上周对外口径存在潜在打架点",
                ],
                "caveat": "纯属公司内部桌面推演，与真实监管或公众态度可能一致也可能不一致。",
                "estimated_timeline_desk_cn": f"[{self.label}] desk：48h 内对齐口径；1～2 周形成最小可行包；关键数据回填后再评估是否扩围。",
            }
        if '"company_memory_merge"' in user:
            return {
                "synthesis": f"[{self.label}] 记忆更新：本 tick 关键表态已并入备忘录；重复承诺与未闭环风险已去重占位。",
            }
        if '"interaction_round"' in user:
            return {
                "interactions": [
                    {
                        "from_id": "regulator_data_security",
                        "to_id": "issuer_command_center",
                        "channel": "约谈",
                        "kind": "regulator_contact",
                        "summary": f"[{self.label}] 监管要求补充日志留存与默认关闭跨境副本的书面说明，企业承诺48小时提交。",
                        "cost_million": None,
                        "fiscal_billion": None,
                    },
                    {
                        "from_id": "issuer_command_center",
                        "to_id": "tech_media_chief_editor",
                        "channel": "记者会",
                        "kind": "enterprise_media_response",
                        "summary": f"[{self.label}] 企业组织媒体沟通会，公开测试边界并回应误读视频来源。",
                        "cost_million": 2.0 if seed % 2 == 0 else None,
                        "fiscal_billion": None,
                    },
                ]
            }
        if (
            '"actor_id"' in user
            or "关键角色" in user
            or "structured" in system.lower()
            or "effect" in user.lower()
        ):
            media = ["neutral", "favorable", "unfavorable"][seed % 3]
            return {
                "public_statement": f"[{self.label}] 占位陈述：呼吁平衡各方利益。",
                "sentiment_delta": round((seed % 5 - 2) / 50 + jitter, 4),
                "rumor_delta": round(-0.01 if seed % 3 == 0 else 0.01, 4),
                "policy_support_delta": round((seed % 7 - 3) / 200, 4),
                "unrest_delta": round(-0.02 if seed % 2 == 0 else 0.015, 4),
                "cohort_nudges": {},
                "media_slant": media,
                "headline_fragment": f"[{self.label}] 报道角度：{media}",
                "cash_delta_million": None,
                "fiscal_delta_billion": None,
            }
        if '"horizon_axes"' in user:
            allow = "allow_two_and_three_year_outlook" in user and "true" in user
            return {
                "today": f"[{self.label}] 本周：舆情维持震荡，执行层关注合规与供应链措辞。",
                "next_month": f"[{self.label}] 下月：若沟通节奏稳定，争议或缓慢降温；反之监管问询可能上升。",
                "next_year": f"[{self.label}] 明年：格局取决于连续兑现与第三方背书，不宜线性外推。",
                "two_years": "中长期展望需态势持续偏强方有意义。" if allow else None,
                "three_years": "更长周期仅作情景假设，不作承诺。" if allow else None,
            }
        if '"cohorts"' in user and "输入" in user:
            # 从 user JSON 解析 cohort id，避免 product 场景 id 与默认不符
            try:
                start = user.index("{")
                data = json.loads(user[start:])
                ids = [c["id"] for c in data.get("cohorts", []) if isinstance(c, dict) and "id" in c]
            except (ValueError, json.JSONDecodeError, KeyError):
                ids = []
            if not ids:
                ids = ["urban_youth", "suburban_family", "rural_older", "business_owners"]
            cohorts = []
            for i, cid in enumerate(ids):
                cohorts.append(
                    {
                        "id": cid,
                        "attitude_delta": round(0.015 * ((-1) ** i), 4),
                        "activation_delta": round(0.005 * (i % 3 - 1), 4),
                    }
                )
            return {"cohorts": cohorts}
        if "cohort" in user.lower() or "人群" in user:
            return {
                "cohorts": [
                    {"id": "urban_youth", "attitude_delta": 0.02, "activation_delta": 0.01},
                    {"id": "suburban_family", "attitude_delta": -0.01, "activation_delta": 0.0},
                    {"id": "rural_older", "attitude_delta": 0.0, "activation_delta": -0.01},
                    {"id": "business_owners", "attitude_delta": 0.015, "activation_delta": 0.0},
                ]
            }
        return {"raw": "dry_run", "jitter": jitter}


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        raise ValueError("no json object in model output")
    return json.loads(m.group(0))
