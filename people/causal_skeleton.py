from __future__ import annotations

"""
可版本化因果骨架（v0）：

- 目标：把因果层从“散落 rules/edges 列表”升级为可版本化、可依赖管理的 DSL，
  同时保持现有执行器（extension_stack._plugin_causal）不变——运行前编译为 CausalRuleSpec / CausalEdgeSpec。
- 取舍：先落最小闭环（结构化 + 依赖拓扑 + 编译 + 一致性检查），不引入外部依赖。
"""

from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, Field, model_validator

from people.config import CausalEdgeSpec, CausalEffectSpec, CausalRuleSpec, MetricName, TriggerCondition


SKELETON_SCHEMA_VERSION = "0.1"


class CausalSkeletonRule(BaseModel):
    id: str
    depends_on: list[str] = Field(default_factory=list)
    when_all: list[TriggerCondition] = Field(default_factory=list)
    effects: list[CausalEffectSpec] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0)
    merge_mode: str = "additive"

    @model_validator(mode="after")
    def _basic(self) -> "CausalSkeletonRule":
        if not self.id.strip():
            raise ValueError("skeleton.rules[].id 不能为空")
        if self.when_all and not self.effects:
            raise ValueError(f"skeleton rule {self.id!r}: 有 when_all 时 effects 不能为空")
        return self


class CausalSkeletonEdge(BaseModel):
    id: str = ""
    from_metric: MetricName
    to_metric: MetricName
    lag_ticks: int = Field(default=1, ge=0)
    weight: float = 0.0
    max_abs_delta: float | None = Field(default=None, ge=0.0)


class CausalSkeleton(BaseModel):
    schema_version: str = SKELETON_SCHEMA_VERSION
    # 显式声明：便于做依赖/一致性审计
    metrics: list[MetricName] = Field(default_factory=list)
    rules: list[CausalSkeletonRule] = Field(default_factory=list)
    edges: list[CausalSkeletonEdge] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ids_unique_and_deps_valid(self) -> "CausalSkeleton":
        seen: dict[str, int] = {}
        for r in self.rules:
            seen[r.id] = seen.get(r.id, 0) + 1
        dups = [k for k, c in seen.items() if c > 1]
        if dups:
            raise ValueError("skeleton.rules id 重复：" + ", ".join(dups[:8]))
        ids = set(seen.keys())
        for r in self.rules:
            for dep in r.depends_on:
                if dep not in ids:
                    raise ValueError(f"skeleton rule {r.id!r}: depends_on 引用不存在的规则 {dep!r}")
                if dep == r.id:
                    raise ValueError(f"skeleton rule {r.id!r}: depends_on 不能引用自身")
        return self


def topo_sort_rule_ids(rules: list[CausalSkeletonRule]) -> list[str]:
    """
    拓扑排序：用于把“依赖图”映射为稳定执行顺序（仍会在同 priority 下排序）。
    若有环则抛 ValueError。
    """
    ids = [r.id for r in rules]
    idset = set(ids)
    indeg: dict[str, int] = {i: 0 for i in ids}
    g: dict[str, list[str]] = defaultdict(list)
    for r in rules:
        for dep in r.depends_on:
            if dep not in idset:
                continue
            g[dep].append(r.id)
            indeg[r.id] += 1
    q = deque([i for i in ids if indeg[i] == 0])
    out: list[str] = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in g.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(ids):
        # 找一个环的提示
        left = [i for i in ids if indeg[i] > 0][:8]
        raise ValueError("skeleton rules depends_on 存在环或不可解依赖，涉及：" + ", ".join(left))
    return out


def materialize_causal_layer_from_skeleton(skeleton_dict: dict[str, Any]) -> tuple[list[CausalRuleSpec], list[CausalEdgeSpec], dict[str, Any]]:
    """
    输入：Scenario.extensions.causal.skeleton（dict）
    输出：可直接赋值给 Scenario.extensions.causal.rules/edges 的对象列表。
    """
    sk = CausalSkeleton.model_validate(skeleton_dict)
    order = topo_sort_rule_ids(sk.rules)
    by_id: dict[str, CausalSkeletonRule] = {r.id: r for r in sk.rules}

    # 先按依赖拓扑排，再用 priority 二次排序（同 priority 下保持依赖顺序稳定）
    dep_rank = {rid: i for i, rid in enumerate(order)}
    rules_sorted = sorted(
        sk.rules,
        key=lambda r: (int(r.priority), dep_rank.get(r.id, 10**9), str(r.id)),
    )
    out_rules: list[CausalRuleSpec] = []
    for r in rules_sorted:
        out_rules.append(
            CausalRuleSpec(
                id=r.id,
                when_all=list(r.when_all),
                effects=list(r.effects),
                priority=int(r.priority),
                merge_mode=("last_wins" if str(r.merge_mode) == "last_wins" else "additive"),
            )
        )

    out_edges: list[CausalEdgeSpec] = []
    for e in sk.edges:
        out_edges.append(
            CausalEdgeSpec(
                id=str(e.id or ""),
                from_metric=e.from_metric,
                to_metric=e.to_metric,
                lag_ticks=int(e.lag_ticks),
                weight=float(e.weight),
                max_abs_delta=e.max_abs_delta,
            )
        )

    info = {
        "skeleton_schema_version": sk.schema_version,
        "rules_count": len(out_rules),
        "edges_count": len(out_edges),
        "topo_order": order,
        "meta": dict(sk.meta or {}),
    }
    return out_rules, out_edges, info
