from __future__ import annotations

from people.config import CrisisRule, Scenario
from people.state import WorldState
from people.triggers import condition_holds


def pick_crisis(scenario: Scenario, state: WorldState) -> CrisisRule | None:
    for rule in scenario.crisis_rules:
        if rule.once and rule.id in state.resolved_crisis_ids:
            continue
        if all(condition_holds(state, c) for c in rule.when_all):
            return rule
    return None
