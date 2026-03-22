"""
每 tick 将关键智能体公开表态压缩进「公司内部记忆」，供后续研判与 LLM 上下文引用，减少重复失误。
"""

from __future__ import annotations

import json
from typing import Any

from people.config import Scenario
from people.providers.base import LLMClient
from people.state import WorldState

from people.agents.realism_layer import REALISM_SOCIAL_LAYER

_SYSTEM = (
    """
你是公司内部「态势记忆库」编辑（演练用）。把本 tick 新增的关键对外表态与指标变化，合并进「既有记忆」：
- 去重：与旧记忆重复的信息压缩成一条，不要堆抄。
- 保留：承诺口径、未闭环风险、已打过的脸、监管/媒体敏感点、用户群摩擦；**若涉及多部门/县域节点/外包顾问**，用短标签记下**谁关心什么**（利害摘要），便于下轮不写成单一声音。
- 语气像**真实战情室白板/纪要**：谁说过什么、尚缺什么证据、下一步找谁核实；不要写成游戏任务日志或世界观设定。
- 输出 JSON：{ "synthesis": "string" }
synthesis 总长不超过 2800 字；用条目式短句。
【禁用游戏措辞】
"""
    + "\n\n"
    + REALISM_SOCIAL_LAYER
    + "\n"
).strip()


def merge_company_memory_tick(
    client: LLMClient,
    *,
    scenario: Scenario,
    state: WorldState,
    tick_entries: list[dict[str, Any]],
    temperature: float = 0.25,
) -> None:
    if not tick_entries:
        return
    payload = {
        "company_memory_merge": True,
        "exercise_type": scenario.exercise_type,
        "tick": state.tick,
        "prior_synthesis": (state.company_memory_synthesis or "")[:3200],
        "new_entries": tick_entries,
        "macro_after_tick": state.snapshot_metrics(),
    }
    user = "请更新公司内部记忆：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        raw = client.complete_json(system=_SYSTEM, user=user, temperature=temperature)
    except Exception:
        raw = {}
    if isinstance(raw, dict):
        syn = str(raw.get("synthesis") or "").strip()
        if syn:
            state.company_memory_synthesis = syn[:8000]
            return
    # 回退：纯追加摘录
    tail = state.company_memory_synthesis[-3500:] if state.company_memory_synthesis else ""
    lines = [tail] if tail else []
    for e in tick_entries:
        who = e.get("role") or e.get("agent_id")
        st = (e.get("statement_excerpt") or "")[:200]
        if st:
            lines.append(f"[t{state.tick}] {who}: {st}")
    state.company_memory_synthesis = "\n".join(lines)[-8000:]


def augment_decision_context(dctx: dict[str, Any], state: WorldState) -> None:
    syn = (state.company_memory_synthesis or "").strip()
    if syn:
        dctx["company_memory_synthesis"] = syn[:4500]
    dl = (state.decision_layer_active_summary or "").strip()
    if dl:
        dctx["decision_layer_directive"] = dl[:2500]
    rga = state.regional_grounding_artifact
    if isinstance(rga, dict) and rga:
        dctx["regional_grounding"] = rga
