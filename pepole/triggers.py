from __future__ import annotations

from pepole.config import Scenario, TriggerCondition
from pepole.state import WorldState


def _value(state: WorldState, metric: str) -> float:
    return float(getattr(state, metric))


def condition_holds(state: WorldState, c: TriggerCondition) -> bool:
    v = _value(state, c.metric)
    if c.gte is not None and v < c.gte:
        return False
    if c.lte is not None and v > c.lte:
        return False
    if c.gt is not None and v <= c.gt:
        return False
    if c.lt is not None and v >= c.lt:
        return False
    return True


def evaluate_triggers(scenario: Scenario, state: WorldState) -> tuple[list[str], list[str]]:
    """返回 (优先入池的 persona id 列表, 触发的规则 id 列表)。"""
    forced: list[str] = []
    fired: list[str] = []
    for rule in scenario.triggers:
        if not rule.when_all:
            continue
        if all(condition_holds(state, c) for c in rule.when_all):
            fired.append(rule.id)
            forced.extend(rule.inject_pool_priority_ids)
    return forced, fired
