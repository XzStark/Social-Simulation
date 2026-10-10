from __future__ import annotations

import json
from typing import Any

from people.providers.base import LLMClient

_SYSTEM = """
你是企业战略评估助手。请基于输入场景、检索片段、资源约束做专项评估。
必须输出 JSON，结构：
{
  "focus": "new_product_launch|acquisition_mna",
  "success_probability_range": "string",
  "key_prerequisites": ["string", ...],
  "integration_timeline_weeks": "string",
  "cash_occupancy": "string",
  "manpower_occupancy": "string",
  "regulatory_resistance": "string",
  "antitrust_resistance": "string",
  "local_execution_resistance": "string",
  "mna_card": {
    "integration_timeline": "string",
    "cash_occupancy": "string",
    "approval_risk": "string",
    "org_conflict_score": 0
  },
  "launch_card": {
    "launch_window": "string",
    "channel_capacity": "string",
    "customer_service_peak": "string",
    "manpower_and_rollback": "string"
  },
  "conflict_matrix": [
    {
      "department_a": "法务|公关|财务|供应链|基层执行",
      "department_b": "法务|公关|财务|供应链|基层执行",
      "conflict_point": "string",
      "compromise_path": "string",
      "severity": "high|medium|low"
    }
  ]
}
要求：
- 成功概率用区间表达（如 35%~50%），不要伪精确。
- org_conflict_score 为 0~100，越高冲突越大。
- conflict_matrix 至少 5 条，覆盖法务/公关/财务/供应链/基层执行。
- 结论必须考虑现金、人力、合规、节奏与本地执行摩擦。
- 若输入或检索涉及竞品，需体现“至少两家对手”的差异化威胁；可补充公司简介/产品线差异。
- 若输入涉及专利/IP，需给出专利纠纷与规避路径风险提示。
- 若输入涉及开店/同城，需考虑“当地门店密度、评价口碑、平台评分、客流可能性”的不确定区间。
""".strip()


def _fallback(focus: str) -> dict[str, Any]:
    return {
        "focus": focus,
        "success_probability_range": "40%~55%",
        "key_prerequisites": ["预算与权限边界先锁定", "跨部门统一口径并设每周复盘", "先小范围试点再扩面"],
        "integration_timeline_weeks": "8~16 周（视审批与组织协同）",
        "cash_occupancy": "中高（需预留应急池与试错预算）",
        "manpower_occupancy": "中高（法务/公关/运营/供应链均需投入）",
        "regulatory_resistance": "中等，需提前准备合规说明与报备材料",
        "antitrust_resistance": "中等偏低（并购规模与行业集中度决定）",
        "local_execution_resistance": "中等，基层执行节奏与资源约束可能导致偏差",
        "mna_card": {
            "integration_timeline": "12~24 周",
            "cash_occupancy": "较高（并购款+整合成本）",
            "approval_risk": "中等（合规/审批/数据迁移）",
            "org_conflict_score": 62,
        },
        "launch_card": {
            "launch_window": "4~10 周（先灰度再全面上线）",
            "channel_capacity": "需按渠道承载分层投放",
            "customer_service_peak": "上线后 1~2 周压力峰值",
            "manpower_and_rollback": "需临时客服与技术值守，并设置回滚阈值",
        },
        "conflict_matrix": [
            {
                "department_a": "法务",
                "department_b": "公关",
                "conflict_point": "披露边界与响应速度冲突",
                "compromise_path": "先定红线句库，再分层发布",
                "severity": "high",
            },
            {
                "department_a": "财务",
                "department_b": "公关",
                "conflict_point": "预算收缩与投放需求冲突",
                "compromise_path": "采用阶段预算闸门与ROI复盘",
                "severity": "medium",
            },
            {
                "department_a": "供应链",
                "department_b": "基层执行",
                "conflict_point": "交付节奏与一线承诺不一致",
                "compromise_path": "承诺口径绑定库存与补货计划",
                "severity": "high",
            },
            {
                "department_a": "法务",
                "department_b": "基层执行",
                "conflict_point": "合规流程复杂导致执行延迟",
                "compromise_path": "关键流程模板化并设快速审批通道",
                "severity": "medium",
            },
            {
                "department_a": "财务",
                "department_b": "供应链",
                "conflict_point": "现金占用与安全库存冲突",
                "compromise_path": "按品类分级库存并动态周转",
                "severity": "medium",
            },
        ],
    }


def run_special_evaluation(
    client: LLMClient,
    *,
    focus: str,
    scenario_text: str,
    decision_context: dict[str, Any],
    search_snippets: str = "",
    temperature: float = 0.2,
) -> dict[str, Any]:
    payload = {
        "focus": focus,
        "scenario_text": scenario_text[:5000],
        "decision_context": decision_context,
        "external_search_snippets": search_snippets[:4000],
    }
    user = "专项评估输入：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        raw = client.complete_json(system=_SYSTEM, user=user, temperature=temperature)
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return _fallback(focus)
    fb = _fallback(focus)
    out = {**fb, **raw}
    # normalize
    m = out.get("mna_card")
    if not isinstance(m, dict):
        out["mna_card"] = fb["mna_card"]
    else:
        try:
            m["org_conflict_score"] = int(max(0, min(100, round(float(m.get("org_conflict_score", 60))))))
        except Exception:
            m["org_conflict_score"] = 60
    l = out.get("launch_card")
    if not isinstance(l, dict):
        out["launch_card"] = fb["launch_card"]
    c = out.get("conflict_matrix")
    if not isinstance(c, list) or not c:
        out["conflict_matrix"] = fb["conflict_matrix"]
    return out
