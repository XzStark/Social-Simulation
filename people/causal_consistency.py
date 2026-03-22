"""
全局因果一致性（静态）：在跑仿真前检查因果层边、规则与指标域是否自洽。
不替代运行时 governance_audit，而是补「图结构 / 语义」层面的硬错误与软告警。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, get_args

from people.config import MetricName, Scenario

_VALID_METRICS: frozenset[str] = frozenset(get_args(MetricName))


def _collect_causal_edges(scenario: Scenario) -> list[tuple[str, str, str, int]]:
    """(edge_id, from_metric, to_metric, lag_ticks)"""
    cl = scenario.extensions.causal
    out: list[tuple[str, str, str, int]] = []
    for e in cl.edges:
        out.append((e.id or f"edge:{e.from_metric}->{e.to_metric}", e.from_metric, e.to_metric, int(e.lag_ticks)))
    return out


def _detect_cycle(edges: list[tuple[str, str]]) -> list[str] | None:
    """若存在有向环，返回环上节点序列；否则 None。"""
    graph: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for a, b in edges:
        graph[a].append(b)
        nodes.add(a)
        nodes.add(b)
    visited: set[str] = set()
    in_stack: set[str] = set()
    stack: list[str] = []

    def dfs(u: str) -> list[str] | None:
        visited.add(u)
        in_stack.add(u)
        stack.append(u)
        for v in graph[u]:
            if v not in visited:
                cyc = dfs(v)
                if cyc:
                    return cyc
            elif v in in_stack:
                i = stack.index(v)
                return stack[i:] + [v]
        stack.pop()
        in_stack.remove(u)
        return None

    for n in sorted(nodes):
        if n not in visited:
            cyc = dfs(n)
            if cyc:
                return cyc
    return None


def audit_scenario_causal_consistency(scenario: Scenario) -> dict[str, Any]:
    """
    返回 errors（应阻止上线或强制修复）、warnings（建议复查）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    cl = scenario.extensions.causal

    trig = _audit_triggers_and_crisis(scenario)
    if not cl.enabled:
        err = list(trig["errors"])
        return {
            "causal_enabled": False,
            "ok": len(err) == 0,
            "errors": err,
            "warnings": ["因果扩展未启用；未检查因果边/规则图结构。"],
            "trigger_and_crisis": trig,
        }

    seen_rule_ids: dict[str, int] = {}
    for r in cl.rules:
        rid = r.id or ""
        if rid:
            seen_rule_ids[rid] = seen_rule_ids.get(rid, 0) + 1
        for c in r.when_all:
            if c.metric not in _VALID_METRICS:
                errors.append(f"因果规则 {rid!r} 条件引用了未知指标 {c.metric!r}")
        for eff in r.effects:
            if eff.metric not in _VALID_METRICS:
                errors.append(f"因果规则 {rid!r} effect 引用了未知指标 {eff.metric!r}")
    for rid, cnt in seen_rule_ids.items():
        if cnt > 1:
            warnings.append(f"因果规则 id 重复：{rid!r} 出现 {cnt} 次（执行顺序依赖 priority）")

    edge_pairs: list[tuple[str, str]] = []
    for eid, fm, tm, lag in _collect_causal_edges(scenario):
        if fm not in _VALID_METRICS:
            errors.append(f"因果边 {eid!r} from_metric 无效：{fm!r}")
        if tm not in _VALID_METRICS:
            errors.append(f"因果边 {eid!r} to_metric 无效：{tm!r}")
        if fm == tm:
            warnings.append(
                f"因果边 {eid!r} 为自环（{fm}→{fm}，lag={lag}）；"
                "请确认滞后与 finalize 顺序是否仍符合叙事，否则可能数值不稳定。"
            )
        edge_pairs.append((fm, tm))

    cyc = _detect_cycle(edge_pairs)
    if cyc:
        warnings.append(
            "因果边在无滞后展开图上形成有向环："
            + " → ".join(cyc)
            + "。同 tick 内多条边仍可能叠加；若与叙事时间方向冲突，请拆环或改为更大 lag。"
        )

    for m in cl.governance_metrics:
        if m not in _VALID_METRICS:
            errors.append(f"governance_metrics 含未知指标：{m!r}")

    trig = _audit_triggers_and_crisis(scenario)
    errors.extend(trig["errors"])
    return {
        "causal_enabled": True,
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metric_domain": sorted(_VALID_METRICS),
        "trigger_and_crisis": trig,
    }


def _audit_triggers_and_crisis(scenario: Scenario) -> dict[str, Any]:
    te: list[str] = []
    for rule in scenario.triggers:
        for c in rule.when_all:
            if c.metric not in _VALID_METRICS:
                te.append(f"triggers[{rule.id}]: 未知指标 {c.metric!r}")
    for cr in scenario.crisis_rules:
        for c in cr.when_all:
            if c.metric not in _VALID_METRICS:
                te.append(f"crisis_rules[{cr.id}]: 未知指标 {c.metric!r}")
    return {"errors": te, "ok": len(te) == 0}
