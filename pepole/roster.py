from __future__ import annotations

import random
from dataclasses import dataclass

from pepole.config import KeyActorSpec, PersonaSpec, Scenario, SimulationScope
from pepole.state import WorldState
from pepole.triggers import evaluate_triggers


@dataclass(frozen=True)
class ResolvedAgent:
    id: str
    role: str
    goals: str
    persona: str
    categories: tuple[str, ...]


def _persona_eligible(p: PersonaSpec, scope: SimulationScope) -> bool:
    if p.markets and not (set(p.markets) & set(scope.markets_active)):
        return False
    if p.product_kinds and scope.product_kind != "general":
        pk = scope.product_kind
        kinds = set(p.product_kinds)
        if pk not in kinds and "general" not in kinds:
            return False
    return True


def _from_key_actor(a: KeyActorSpec) -> ResolvedAgent:
    return ResolvedAgent(
        id=a.id,
        role=a.role,
        goals=a.goals,
        persona=(a.persona or "").strip(),
        categories=tuple(a.categories) if getattr(a, "categories", None) else (),
    )


def _from_persona(p: PersonaSpec) -> ResolvedAgent:
    return ResolvedAgent(
        id=p.id,
        role=p.role,
        goals=p.goals,
        persona=(p.persona or "").strip(),
        categories=tuple(p.categories),
    )


def _weighted_sample_no_replace(
    rng: random.Random,
    pool: list[PersonaSpec],
    k: int,
) -> list[PersonaSpec]:
    if k <= 0 or not pool:
        return []
    bag = list(pool)
    out: list[PersonaSpec] = []
    draws = min(k, len(bag))
    for _ in range(draws):
        weights = [max(p.sampling_weight, 0.0) for p in bag]
        if sum(weights) <= 0:
            i = rng.randrange(len(bag))
        else:
            i = rng.choices(range(len(bag)), weights=weights, k=1)[0]
        out.append(bag.pop(i))
    return out


def agents_for_tick(
    scenario: Scenario,
    rng: random.Random,
    state: WorldState | None = None,
) -> tuple[list[ResolvedAgent], list[str]]:
    """
    每 tick 参与 LLM 的智能体列表：
    - 全部 key_actors（每 tick 必跑，兼容旧场景）
    - personas 中 llm_each_tick=true 的
    - 再从「池化人设」中按权重无放回抽样 pooled_llm_calls_per_tick 个
    - 若提供 state，合并 triggers 触发的 inject_pool_priority_ids

    返回 (智能体列表, 本 tick 触发的 trigger id 列表)。
    """
    scope = scenario.simulation
    standing: list[ResolvedAgent] = [_from_key_actor(a) for a in scenario.key_actors]

    pool_candidates: list[PersonaSpec] = []
    for p in scenario.personas:
        if not _persona_eligible(p, scope):
            continue
        if p.llm_each_tick:
            standing.append(_from_persona(p))
        else:
            pool_candidates.append(p)

    trigger_fired: list[str] = []
    extra_forced: list[str] = []
    if state is not None and scenario.triggers:
        extra_forced, trigger_fired = evaluate_triggers(scenario, state)

    # 强制每 tick 出现的池化人设 + 条件触发优先入场
    forced_ids = set(scope.always_sample_ids) | set(extra_forced)
    forced: list[PersonaSpec] = []
    rest_pool: list[PersonaSpec] = []
    for p in pool_candidates:
        if p.id in forced_ids:
            forced.append(p)
        else:
            rest_pool.append(p)

    k = scope.pooled_llm_calls_per_tick
    # 先放入强制，再从 rest 抽剩余名额
    take_forced = forced[:k]
    remaining = max(0, k - len(take_forced))
    sampled_rest = _weighted_sample_no_replace(rng, rest_pool, remaining)
    pooled_agents = [_from_persona(p) for p in take_forced + sampled_rest]

    # 去重 id（standing 优先）
    seen = {a.id for a in standing}
    merged = list(standing)
    for a in pooled_agents:
        if a.id not in seen:
            seen.add(a.id)
            merged.append(a)
    return merged, trigger_fired


def count_llm_agents_upper_bound(scenario: Scenario) -> tuple[int, int]:
    """(每 tick 固定人数, 每 tick 固定 + 池化抽满时人数)。"""
    scope = scenario.simulation
    standing_n = len(scenario.key_actors) + sum(
        1 for p in scenario.personas if p.llm_each_tick and _persona_eligible(p, scope)
    )
    pool_n = sum(
        1 for p in scenario.personas if not p.llm_each_tick and _persona_eligible(p, scope)
    )
    k = scope.pooled_llm_calls_per_tick
    return standing_n, standing_n + min(k, pool_n)
