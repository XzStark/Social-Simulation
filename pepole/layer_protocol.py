"""
层冲突与流水线协议（草案，与实现对照用）。

当前 **实际执行顺序** 以 `pepole/engine.py` 与 `extension_stack.run_extension_plugins` 为准。
本模块提供可导入的常量，供测试与后续「协议强制校验」使用；详细文字约定见 **系统说明.md §16.5、§17**。
"""

from __future__ import annotations

# 单 tick 内主要阶段（与文档 §4 对齐；细粒度子步骤见引擎源码）
TICK_PIPELINE_STAGES: tuple[str, ...] = (
    "tick_start_extensions",
    "decision_context",
    "environment_drift",
    "roster",
    "key_actor_llm",
    "key_actor_effects",
    "cohort_llm",
    "after_cohort_extensions",
    "social_interactions",
    "after_interactions_extensions",
    "aggregate_from_cohorts",
    "operating_ledger",
    "pre_finalize_extensions",
    "finalize_tick",
    "after_finalize_extensions",
    "horizon_forecast",
    "crisis_and_milestone",
)

# 建议的「同指标多写者」合并优先级：序号小的阶段先于序号大的阶段生效于状态，
# 后写覆盖先写时，应由显式冲突策略处理（当前实现多为顺序叠加，见 §16.5）。
EXTENSION_PRE_FINALIZE_ORDER_HINT: tuple[str, ...] = (
    "causal",
    "diffusion",
    "anchor",
    "kpi",
    "resources",
)
