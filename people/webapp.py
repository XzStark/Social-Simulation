"""
本地网页界面：选择场景、运行演练、查看指标曲线与叙事。
启动：在项目根目录执行
  python -m uvicorn people.webapp:app --host 127.0.0.1 --port 8770
或双击 run_web.cmd
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from people.config import WebSearchProvider
from people.scenario_overrides import merge_scenario_overrides, override_payload_from_model

ROOT = Path(__file__).resolve().parent.parent


def _safe_scenario_file(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="无效的场景文件名")
    base = (ROOT / "scenarios").resolve()
    p = (base / name).resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(status_code=400, detail="路径非法")
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"未找到场景: {name}")
    return p


@contextlib.contextmanager
def _env_overlay(updates: dict[str, str | None]) -> Any:
    saved: dict[str, str | None] = {}
    try:
        for k, v in updates.items():
            if not v:
                continue
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _maybe_overlay_deepseek_base_url(body: Any, overlay: dict[str, str | None]) -> None:
    """
    网页只填密钥、不填 Base URL 时，OpenAI 兼容客户端会默认走 https://api.openai.com/v1，
    DeepSeek 密钥会被发到错误主机而表现为 401 / 无效密钥。
    若当前选用的模型 id 含 deepseek，则强制把本次请求的 Base 指向 DeepSeek 兼容端点。
    """
    if "OPENAI_BASE_URL" in overlay:
        return
    if not (overlay.get("OPENAI_API_KEY") or "").strip():
        return
    primary = getattr(body, "primary", None)
    fast = getattr(body, "fast", None)
    blob = " ".join(
        str(x)
        for x in (
            primary,
            fast,
            os.environ.get("PEOPLE_MODEL_PRIMARY"),
            os.environ.get("PEOPLE_MODEL_FAST"),
        )
        if x
    ).lower()
    if "deepseek" not in blob:
        return
    overlay["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"


def _openai_env_overlay_from_body(body: Any) -> dict[str, str | None]:
    o: dict[str, str | None] = {}
    key = getattr(body, "openai_api_key", None)
    base = getattr(body, "openai_base_url", None)
    if key and str(key).strip():
        o["OPENAI_API_KEY"] = str(key).strip()
    if base and str(base).strip():
        o["OPENAI_BASE_URL"] = str(base).strip()
    _maybe_overlay_deepseek_base_url(body, o)
    return o


class RunRequest(BaseModel):
    scenario_file: str = Field(..., description="scenarios 目录下 yaml 文件名，如 default.yaml")
    seed: int = 42
    primary: str | None = None
    fast: str | None = None
    brief: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    # —— 以下为可选覆盖（不必改 YAML；由「场景向导」写入）——
    exercise_type: Literal["policy", "product"] | None = None
    ticks: int | None = Field(default=None, ge=1)
    problem_salience: float | None = Field(default=None, ge=0.0, le=1.0)
    issuer_archetype: Literal["megacorp", "large_group", "sme", "startup"] | None = None
    brand_equity: float | None = Field(default=None, ge=0.0, le=1.0)
    reputation_brief: str | None = None
    supply_chain_position: str | None = None
    policy_admin_level: Literal["unset", "central", "province", "city", "county"] | None = None
    policy_jurisdiction_name: str | None = None
    local_norms_brief: str | None = None
    media_environment_brief: str | None = None
    regional_grounding_enabled: bool | None = None
    regional_grounding_region: str | None = None
    regional_grounding_mode: Literal["llm_only", "wikipedia_then_llm"] | None = None
    regional_grounding_web_search: bool | None = None
    web_search_provider: WebSearchProvider | None = None
    business_scale: Literal["unset", "street_shop", "sme_chain", "regional_group", "national_group"] | None = None
    business_sector_brief: str | None = None
    user_known_local_policy_brief: str | None = None
    user_known_central_policy_brief: str | None = None
    # 无 checkpoint 时「确认执行」暂存于此，随下次「开始运行」一次性写入 tick0 态势
    internal_plan_commit: dict[str, Any] | None = None


class EnsembleRequest(RunRequest):
    """并行多次演练（蒙特卡洛 / 分布汇总），不写 checkpoint。"""

    runs: int = Field(default=8, ge=1, le=256)
    workers: int = Field(default=4, ge=1, le=64)
    threshold_key: str = "policy_support"
    threshold: float = 0.55


class PlanEvaluateRequest(BaseModel):
    """不推进 tick：仅对白板方案做专业转化 + 桌面可行性（可选绑定 checkpoint 当前态势）。"""

    scenario_file: str = "default.yaml"
    seed: int = 42
    plain_solution: str = Field(..., min_length=1)
    checkpoint_token: str | None = None
    brief: str | None = None
    primary: str | None = None
    fast: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    exercise_type: Literal["policy", "product"] | None = None
    ticks: int | None = Field(default=None, ge=1)
    problem_salience: float | None = Field(default=None, ge=0.0, le=1.0)
    issuer_archetype: Literal["megacorp", "large_group", "sme", "startup"] | None = None
    brand_equity: float | None = Field(default=None, ge=0.0, le=1.0)
    reputation_brief: str | None = None
    supply_chain_position: str | None = None
    policy_admin_level: Literal["unset", "central", "province", "city", "county"] | None = None
    policy_jurisdiction_name: str | None = None
    local_norms_brief: str | None = None
    media_environment_brief: str | None = None
    regional_grounding_enabled: bool | None = None
    regional_grounding_region: str | None = None
    regional_grounding_mode: Literal["llm_only", "wikipedia_then_llm"] | None = None
    regional_grounding_web_search: bool | None = None
    web_search_provider: WebSearchProvider | None = None
    business_scale: Literal["unset", "street_shop", "sme_chain", "regional_group", "national_group"] | None = None
    business_sector_brief: str | None = None
    user_known_local_policy_brief: str | None = None
    user_known_central_policy_brief: str | None = None


class PlanConfirmExecuteRequest(PlanEvaluateRequest):
    """在「仅评估」结果基础上确认执行：写入决策层/记忆/宏观与资金边际，并登记约定 tick 节点。"""

    evaluation_digest: dict[str, Any] = Field(default_factory=dict)
    evaluation_desk: dict[str, Any] = Field(default_factory=dict)


class ResumeStreamRequest(BaseModel):
    """暂停后续跑：提交方案（可多套）→ 研判；若场景开启确认门闩，需再发 confirm_proceed。"""

    checkpoint_token: str = Field(..., min_length=8)
    resume_stage: Literal["submit_solutions", "confirm_proceed"] = "submit_solutions"
    user_solution: str | None = None
    user_solutions: list[str] | None = None
    selected_plan_index: int | None = Field(default=None, ge=0)
    primary: str | None = None
    fast: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    @model_validator(mode="after")
    def _resume_plans(self) -> "ResumeStreamRequest":
        if self.resume_stage == "submit_solutions":
            from_list = bool(
                self.user_solutions and any(str(x).strip() for x in self.user_solutions)
            )
            from_one = bool(self.user_solution and str(self.user_solution).strip())
            if not from_list and not from_one:
                raise ValueError("提交方案阶段需要至少一条非空方案（user_solution 或 user_solutions）")
        return self


def _header_from_scenario(scenario: Any, *, seed: int) -> dict[str, Any]:
    pc = scenario.policy_context
    iss = scenario.issuer
    return {
        "name": scenario.name,
        "exercise_type": scenario.exercise_type,
        "ticks": scenario.ticks,
        "seed": seed,
        "admin_level": pc.admin_level,
        "jurisdiction_name": pc.jurisdiction_name,
        "issuer_archetype": iss.archetype,
        "brand_equity": iss.brand_equity,
        "problem_salience": getattr(scenario, "problem_salience", 0.5),
        "confirm_after_desk_feasibility": getattr(scenario, "confirm_after_desk_feasibility", True),
    }


def _execute_simulation(
    body: RunRequest, progress_callback: Any
) -> tuple[Any, Any, dict[str, Any], dict[str, Any] | None]:
    from people.config import Scenario
    from people.engine import RunConfig, run_single_simulation
    from people.providers.registry import get_client

    path = _safe_scenario_file(body.scenario_file)
    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        scenario = Scenario.load(path)
        scenario = merge_scenario_overrides(
            scenario,
            override_payload_from_model(body),
            player_brief=body.brief,
        )

        primary_spec = body.primary or os.environ.get("PEOPLE_MODEL_PRIMARY") or "openai:gpt-4o"
        fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
        primary = get_client(primary_spec, allow_dry=True)
        fast = get_client(fast_spec, allow_dry=True)
        cfg = RunConfig(
            primary_client=primary,
            fast_client=fast,
            primary_model_slot=primary_spec,
            fast_model_slot=fast_spec,
        )
        state, pause_pkg = run_single_simulation(
            scenario,
            cfg,
            seed=body.seed,
            progress_callback=progress_callback,
            internal_plan_commit=body.internal_plan_commit,
        )
        header = _header_from_scenario(scenario, seed=body.seed)
    return state, scenario, header, pause_pkg


def _execute_resume_simulation(
    body: "ResumeStreamRequest", progress_callback: Any
) -> tuple[Any, Any, dict[str, Any], dict[str, Any] | None]:
    import random

    from people.config import Scenario
    from people.engine import RunConfig, run_single_simulation
    from people.pause_store import take_checkpoint
    from people.providers.registry import get_client
    from people.state import WorldState

    pkg = take_checkpoint(body.checkpoint_token)
    if pkg is None:
        raise HTTPException(status_code=404, detail="checkpoint 不存在或已使用（请重新开始演练）")

    pause_kind = str(pkg.get("pause_kind") or "crisis_rule")
    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        scenario = Scenario.model_validate(pkg["scenario_dict"])
        state = WorldState.model_validate(pkg["state_dict"])
        rng = random.Random()
        rng.setstate(pkg["rng_state"])
        seed = int(pkg.get("seed", 42))

        primary_spec = body.primary or os.environ.get("PEOPLE_MODEL_PRIMARY") or "openai:gpt-4o"
        fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
        primary = get_client(primary_spec, allow_dry=True)
        fast = get_client(fast_spec, allow_dry=True)
        cfg = RunConfig(
            primary_client=primary,
            fast_client=fast,
            primary_model_slot=primary_spec,
            fast_model_slot=fast_spec,
        )

        if body.resume_stage == "confirm_proceed":
            if pause_kind != "desk_review":
                raise HTTPException(
                    status_code=400,
                    detail="当前 checkpoint 不是「待确认可行性」状态，请用提交方案流程",
                )
            state, pause_pkg = run_single_simulation(
                scenario,
                cfg,
                seed=seed,
                progress_callback=progress_callback,
                resume={
                    "state": state,
                    "rng": rng,
                    "resume_stage": "confirm_proceed",
                    "selected_plan_index": body.selected_plan_index,
                },
            )
            header = _header_from_scenario(scenario, seed=seed)
            return state, scenario, header, pause_pkg

        plans: list[str] = []
        if body.user_solutions:
            plans = [str(p).strip() for p in body.user_solutions if str(p).strip()]
        if not plans and body.user_solution and body.user_solution.strip():
            one = body.user_solution.strip()
            if "\n---\n" in one:
                plans = [p.strip() for p in one.split("\n---\n") if p.strip()]
            else:
                plans = [one]
        if not plans:
            raise HTTPException(status_code=400, detail="请至少提交一条方案")

        if pause_kind == "desk_review":
            if len(plans) == 1:
                state.user_resolution_notes.append(f"[t{state.tick} 指挥台改评方案] {plans[0]}")
            else:
                for i, p in enumerate(plans):
                    state.user_resolution_notes.append(
                        f"[t{state.tick} 指挥台改评备选{i + 1}] {p}"
                    )
            tail = (plans[0] if len(plans) == 1 else f"共 {len(plans)} 套方案")[:400]
            state.log(
                f"[指挥台改评 t{state.tick}] {tail}{'…' if (len(plans) == 1 and len(plans[0]) > 400) else ''}"
            )
            state, pause_pkg = run_single_simulation(
                scenario,
                cfg,
                seed=seed,
                progress_callback=progress_callback,
                resume={
                    "state": state,
                    "rng": rng,
                    "resume_stage": "submit_solutions",
                    "latest_plain_resolutions": plans,
                },
            )
            header = _header_from_scenario(scenario, seed=seed)
            return state, scenario, header, pause_pkg

        if len(plans) == 1:
            state.user_resolution_notes.append(f"[t{state.tick} 指挥台处置] {plans[0]}")
        else:
            for i, p in enumerate(plans):
                state.user_resolution_notes.append(f"[t{state.tick} 指挥台方案备选{i + 1}] {p}")
        tail = (plans[0] if len(plans) == 1 else f"共 {len(plans)} 套方案")[:400]
        state.log(f"[指挥台处置 t{state.tick}] {tail}{'…' if (len(plans) == 1 and len(plans[0]) > 400) else ''}")

        state.resolved_crisis_ids.append(str(pkg["crisis_id"]))

        state, pause_pkg = run_single_simulation(
            scenario,
            cfg,
            seed=seed,
            progress_callback=progress_callback,
            resume={
                "state": state,
                "rng": rng,
                "resume_stage": "submit_solutions",
                "latest_plain_resolutions": plans,
            },
        )
        header = _header_from_scenario(scenario, seed=seed)
    return state, scenario, header, pause_pkg


def _execute_plan_evaluate(body: PlanEvaluateRequest) -> dict[str, Any]:
    import random

    from people.agents.company_memory import augment_decision_context
    from people.agents.risk_milestone import evaluate_internal_desk_feasibility
    from people.agents.user_resolution import professionalize_plain_resolution
    from people.config import Scenario
    from people.context import build_decision_context, enrich_decision_context_for_plan_evaluate
    from people.engine import RunConfig, _hydrate_finance, build_initial_state
    from people.pause_store import copy_checkpoint
    from people.providers.registry import get_client
    from people.rules import aggregate_from_cohorts, finalize_tick
    from people.state import WorldState

    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        if body.checkpoint_token and body.checkpoint_token.strip():
            pkg = copy_checkpoint(body.checkpoint_token.strip())
            if pkg is None:
                raise HTTPException(
                    status_code=404,
                    detail="checkpoint 无效或已失效（仅内存快照；已消费、服务重启或未暂停过均会失败）。请取消勾选「绑定 checkpoint」后仅评估，或重新运行至暂停以生成新快照。",
                )
            scenario = Scenario.model_validate(pkg["scenario_dict"])
            scenario = merge_scenario_overrides(
                scenario,
                override_payload_from_model(body),
                player_brief=body.brief,
            )
            state = WorldState.model_validate(pkg["state_dict"])
        else:
            path = _safe_scenario_file(body.scenario_file)
            scenario = Scenario.load(path)
            scenario = merge_scenario_overrides(
                scenario,
                override_payload_from_model(body),
                player_brief=body.brief,
            )
            rng = random.Random(body.seed)
            state = build_initial_state(scenario, rng)
            _hydrate_finance(state, scenario)
            aggregate_from_cohorts(state, scenario.realism)
            finalize_tick(state)

        dctx = build_decision_context(scenario)
        augment_decision_context(dctx, state)
        enrich_decision_context_for_plan_evaluate(dctx, state, scenario)

        primary_spec = body.primary or os.environ.get("PEOPLE_MODEL_PRIMARY") or "openai:gpt-4o"
        fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
        primary = get_client(primary_spec, allow_dry=True)
        fast = get_client(fast_spec, allow_dry=True)
        cfg = RunConfig(
            primary_client=primary,
            fast_client=fast,
            primary_model_slot=primary_spec,
            fast_model_slot=fast_spec,
        )

        plain = body.plain_solution.strip()
        digest = professionalize_plain_resolution(
            cfg.fast_client,
            scenario=scenario,
            state=state,
            plain_solution=plain,
            decision_context=dctx,
            temperature=0.25,
        )
        desk = evaluate_internal_desk_feasibility(
            cfg.fast_client,
            scenario=scenario,
            state=state,
            user_plain_solution=plain,
            professional_digest=digest,
            risk_inventory=state.last_risk_inventory,
            decision_context=dctx,
            temperature=0.3,
        )

    def _trim(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(d, dict):
            return None
        return {
            k: d.get(k)
            for k in (
                "understood_intent",
                "feasibility",
                "feasibility_notes",
                "estimated_timeline_cn",
                "professional_execution_plan",
                "involved_functions",
                "policy_equivalent_steps",
                "intent_bullets",
            )
            if d.get(k) is not None
        }

    def _trim_desk(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(d, dict):
            return None
        keys = (
            "desk_verdict",
            "desk_verdict_cn",
            "resource_cash_band_cn",
            "resource_manpower_band_cn",
            "resource_other_cn",
            "internal_gaps",
            "social_linkage",
            "recommended_internal_next_steps",
            "why_not_feasible",
            "caveat",
            "estimated_timeline_desk_cn",
        )
        return {k: d.get(k) for k in keys if d.get(k) is not None}

    return {
        "ok": True,
        "tick_context": state.tick,
        "digest": _trim(digest),
        "desk": _trim_desk(desk),
    }


def _execute_plan_confirm_execute(body: PlanConfirmExecuteRequest) -> dict[str, Any]:
    import random

    from people.config import Scenario
    from people.engine import _hydrate_finance, build_initial_state
    from people.internal_plan_commit import apply_confirmed_internal_plan
    from people.pause_store import copy_checkpoint, put_checkpoint
    from people.rules import aggregate_from_cohorts, finalize_tick
    from people.state import WorldState

    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        pkg0: dict[str, Any] | None = None
        rng: random.Random
        if body.checkpoint_token and body.checkpoint_token.strip():
            pkg0 = copy_checkpoint(body.checkpoint_token.strip())
            if pkg0 is None:
                raise HTTPException(
                    status_code=404,
                    detail="checkpoint 无效或已失效（仅内存快照；请取消绑定或重跑至暂停）。",
                )
            scenario = Scenario.model_validate(pkg0["scenario_dict"])
            scenario = merge_scenario_overrides(
                scenario,
                override_payload_from_model(body),
                player_brief=body.brief,
            )
            state = WorldState.model_validate(pkg0["state_dict"])
            rng = random.Random()
            rng.setstate(pkg0["rng_state"])
        else:
            path = _safe_scenario_file(body.scenario_file)
            scenario = Scenario.load(path)
            scenario = merge_scenario_overrides(
                scenario,
                override_payload_from_model(body),
                player_brief=body.brief,
            )
            rng = random.Random(body.seed)
            state = build_initial_state(scenario, rng)
            _hydrate_finance(state, scenario)
            aggregate_from_cohorts(state, scenario.realism)
            finalize_tick(state)

        applied = apply_confirmed_internal_plan(
            state,
            scenario,
            plain=body.plain_solution.strip(),
            digest=body.evaluation_digest if isinstance(body.evaluation_digest, dict) else {},
            desk=body.evaluation_desk if isinstance(body.evaluation_desk, dict) else {},
        )

        out: dict[str, Any] = {
            "ok": True,
            "tick_context": state.tick,
            "applied": applied,
        }

        if pkg0 is not None:
            new_pkg = dict(pkg0)
            new_pkg["scenario_dict"] = scenario.model_dump()
            new_pkg["state_dict"] = state.model_dump()
            new_pkg["rng_state"] = rng.getstate()
            new_pkg["decision_layer_active_summary"] = state.decision_layer_active_summary
            new_pkg["decision_layer_history"] = state.decision_layer_history[-24:]
            out["checkpoint_token"] = put_checkpoint(new_pkg)
        else:
            out["internal_plan_commit"] = {
                "plain_solution": body.plain_solution.strip(),
                "evaluation_digest": body.evaluation_digest,
                "evaluation_desk": body.evaluation_desk,
            }

    return out


app = FastAPI(title="people", docs_url=None, redoc_url=None)


@app.get("/")
def index_page() -> FileResponse:
    html = ROOT / "web" / "index.html"
    if not html.is_file():
        raise HTTPException(status_code=500, detail="缺少 web/index.html")
    return FileResponse(html)


@app.get("/api/scenarios")
def api_scenarios() -> dict[str, list[str]]:
    d = ROOT / "scenarios"
    names = sorted(p.name for p in d.glob("*.yaml") if p.is_file())
    return {"scenarios": names}


class FinanceCheckpointAdjustRequest(BaseModel):
    """暂停态下手工改写账面现金、财力池、债务本金（演练 proxy）；返回新 checkpoint token。"""

    checkpoint_token: str = Field(..., min_length=8)
    set_cash_balance_million: float | None = None
    set_fiscal_remaining_billion: float | None = None
    set_debt_balance_million: float | None = None
    note: str = ""


@app.post("/api/finance/adjust-checkpoint", response_model=None)
def api_finance_adjust_checkpoint(body: FinanceCheckpointAdjustRequest) -> dict[str, Any]:
    from people.pause_store import copy_checkpoint, put_checkpoint
    from people.state import WorldState

    pkg = copy_checkpoint(body.checkpoint_token)
    if not isinstance(pkg, dict):
        raise HTTPException(status_code=404, detail="checkpoint 不存在或已失效（仅内存、未消费快照）")
    nt = (body.note or "").strip()
    if (
        body.set_cash_balance_million is None
        and body.set_fiscal_remaining_billion is None
        and body.set_debt_balance_million is None
        and not nt
    ):
        raise HTTPException(status_code=400, detail="至少需要一项 set_* 或非空备注")
    st = WorldState.model_validate(pkg["state_dict"])
    if body.set_cash_balance_million is not None:
        st.cash_balance_million = max(0.0, float(body.set_cash_balance_million))
    if body.set_fiscal_remaining_billion is not None:
        st.fiscal_remaining_billion = max(0.0, float(body.set_fiscal_remaining_billion))
    if body.set_debt_balance_million is not None:
        st.debt_balance_million = max(0.0, float(body.set_debt_balance_million))
    if nt:
        st.log(f"[财务手工调整] {nt[:900]}")
    pkg["state_dict"] = st.model_dump()
    new_tok = put_checkpoint(pkg)
    return {"ok": True, "checkpoint_token": new_tok, "metrics": st.snapshot_metrics()}


@app.post("/api/ensemble", response_model=None)
def api_ensemble(body: EnsembleRequest) -> JSONResponse | dict[str, Any]:
    """并行 N 次完整演练，返回终局分布（均值/分位等）与各 seed 终局指标。"""
    from people.engine import RunConfig, run_ensemble_parallel, summarize_ensemble
    from people.providers.registry import get_client
    from people.regional_grounding import prebake_regional_grounding_for_ensemble

    path = _safe_scenario_file(body.scenario_file)
    overlay = _openai_env_overlay_from_body(body)
    try:
        with _env_overlay(overlay):
            from people.config import Scenario

            scenario = Scenario.load(path)
            scenario = merge_scenario_overrides(
                scenario,
                override_payload_from_model(body),
                player_brief=body.brief,
            )
            primary_spec = body.primary or os.environ.get("PEOPLE_MODEL_PRIMARY") or "openai:gpt-4o"
            fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
            if scenario.regional_grounding.enabled:
                cfg0 = RunConfig(
                    primary_client=get_client(primary_spec, allow_dry=True),
                    fast_client=get_client(fast_spec, allow_dry=True),
                    primary_model_slot=primary_spec,
                    fast_model_slot=fast_spec,
                )
                scenario = prebake_regional_grounding_for_ensemble(scenario, cfg0)
            results = run_ensemble_parallel(
                scenario,
                n_runs=body.runs,
                base_seed=body.seed,
                primary_spec=primary_spec,
                fast_spec=fast_spec,
                max_workers=body.workers,
            )
            summary = summarize_ensemble(
                results,
                threshold_key=body.threshold_key,
                threshold=body.threshold,
                scenario=scenario,
                model_slots={"primary": primary_spec, "fast": fast_spec},
            )
        return {
            "ok": True,
            "header": _header_from_scenario(scenario, seed=body.seed),
            "summary": summary,
            "runs": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/api/plan/evaluate", response_model=None)
def api_plan_evaluate(body: PlanEvaluateRequest) -> JSONResponse | dict[str, Any]:
    try:
        return _execute_plan_evaluate(body)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/api/plan/confirm-execute", response_model=None)
def api_plan_confirm_execute(body: PlanConfirmExecuteRequest) -> JSONResponse | dict[str, Any]:
    try:
        return _execute_plan_confirm_execute(body)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/api/run", response_model=None)
def api_run(body: RunRequest) -> JSONResponse | dict[str, Any]:
    from people.pause_store import put_checkpoint

    try:
        state, _scenario, header, pause_pkg = _execute_simulation(body, progress_callback=None)
        out: dict[str, Any] = {
            "ok": True,
            "paused": pause_pkg is not None,
            "header": header,
            "final_metrics": state.snapshot_metrics(),
            "metrics_history": state.metrics_history,
            "narrative": state.narrative,
            "detail_events": state.detail_events,
            "horizon_forecast": state.horizon_forecast,
            "decision_layer_active_summary": getattr(state, "decision_layer_active_summary", "") or "",
            "decision_layer_history": list(getattr(state, "decision_layer_history", []) or [])[-24:],
            "attribution_log": list(getattr(state, "attribution_log", []) or [])[-400:],
        }
        if pause_pkg is not None:
            out["checkpoint_token"] = put_checkpoint(pause_pkg)
            out["crisis"] = {
                "id": pause_pkg["crisis_id"],
                "title": pause_pkg["crisis_title"],
                "detail": pause_pkg["crisis_detail"],
            }
        return out
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


def _sse_stream_from_worker(worker: Any) -> StreamingResponse:
    q: queue.Queue[Any] = queue.Queue()

    def run_worker() -> None:
        worker(q)

    def event_iter():
        threading.Thread(target=run_worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield ("data: " + json.dumps(item, ensure_ascii=False) + "\n\n").encode("utf-8")

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/run/stream")
def api_run_stream(body: RunRequest) -> StreamingResponse:
    """Server-Sent Events：逐 tick 推送；遇危机则 phase=crisis_pause（含 checkpoint_token）；否则 complete。"""

    from people.pause_store import put_checkpoint

    def worker(q: queue.Queue[Any]) -> None:
        try:

            def cb(ev: dict[str, Any]) -> None:
                q.put(ev)

            t0 = time.perf_counter()
            state, scenario, header, pause_pkg = _execute_simulation(body, progress_callback=cb)
            elapsed = time.perf_counter() - t0
            if pause_pkg is not None:
                tok = put_checkpoint(pause_pkg)
                q.put(
                    {
                        "phase": "crisis_pause",
                        "checkpoint_token": tok,
                        "elapsed_seconds_so_far": round(elapsed, 2),
                        "pause_kind": pause_pkg.get("pause_kind") or "crisis_rule",
                        "crisis_id": pause_pkg["crisis_id"],
                        "title": pause_pkg["crisis_title"],
                        "detail": pause_pkg["crisis_detail"],
                        "risk_inventory": pause_pkg.get("risk_inventory"),
                        "plan_evaluations": pause_pkg.get("plan_evaluations"),
                        "tick": state.tick,
                        "header": header,
                        "metrics_history": state.metrics_history,
                        "narrative": state.narrative,
                        "detail_events": state.detail_events[-64:],
                        "horizon_forecast": state.horizon_forecast,
                        "finance_enabled": bool(scenario.operational_finance.enabled),
                        "exercise_type": scenario.exercise_type,
                        "decision_layer_active_summary": getattr(
                            state, "decision_layer_active_summary", ""
                        )
                        or "",
                        "decision_layer_history": list(
                            getattr(state, "decision_layer_history", []) or []
                        )[-24:],
                    }
                )
            else:
                q.put(
                    {
                        "phase": "complete",
                        "elapsed_seconds": round(elapsed, 2),
                        "header": header,
                        "final_metrics": state.snapshot_metrics(),
                        "metrics_history": state.metrics_history,
                        "narrative": state.narrative,
                        "detail_events": state.detail_events,
                        "horizon_forecast": state.horizon_forecast,
                        "finance_enabled": bool(scenario.operational_finance.enabled),
                        "exercise_type": scenario.exercise_type,
                        "decision_layer_active_summary": getattr(
                            state, "decision_layer_active_summary", ""
                        )
                        or "",
                        "decision_layer_history": list(
                            getattr(state, "decision_layer_history", []) or []
                        )[-24:],
                    }
                )
        except HTTPException as he:
            q.put(
                {
                    "phase": "error",
                    "error": he.detail if isinstance(he.detail, str) else str(he.detail),
                    "traceback": "",
                }
            )
        except Exception as e:
            q.put(
                {
                    "phase": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            q.put(None)

    return _sse_stream_from_worker(worker)


@app.post("/api/run/resume/stream")
def api_run_resume_stream(body: ResumeStreamRequest) -> StreamingResponse:
    from people.pause_store import put_checkpoint

    def worker(q: queue.Queue[Any]) -> None:
        try:

            def cb(ev: dict[str, Any]) -> None:
                q.put(ev)

            t0 = time.perf_counter()
            state, scenario, header, pause_pkg = _execute_resume_simulation(body, progress_callback=cb)
            elapsed = time.perf_counter() - t0
            if pause_pkg is not None:
                tok = put_checkpoint(pause_pkg)
                q.put(
                    {
                        "phase": "crisis_pause",
                        "checkpoint_token": tok,
                        "elapsed_seconds_so_far": round(elapsed, 2),
                        "pause_kind": pause_pkg.get("pause_kind") or "crisis_rule",
                        "crisis_id": pause_pkg["crisis_id"],
                        "title": pause_pkg["crisis_title"],
                        "detail": pause_pkg["crisis_detail"],
                        "risk_inventory": pause_pkg.get("risk_inventory"),
                        "plan_evaluations": pause_pkg.get("plan_evaluations"),
                        "tick": state.tick,
                        "header": header,
                        "metrics_history": state.metrics_history,
                        "narrative": state.narrative,
                        "detail_events": state.detail_events[-64:],
                        "horizon_forecast": state.horizon_forecast,
                        "finance_enabled": bool(scenario.operational_finance.enabled),
                        "exercise_type": scenario.exercise_type,
                        "decision_layer_active_summary": getattr(
                            state, "decision_layer_active_summary", ""
                        )
                        or "",
                        "decision_layer_history": list(
                            getattr(state, "decision_layer_history", []) or []
                        )[-24:],
                    }
                )
            else:
                q.put(
                    {
                        "phase": "complete",
                        "elapsed_seconds": round(elapsed, 2),
                        "header": header,
                        "final_metrics": state.snapshot_metrics(),
                        "metrics_history": state.metrics_history,
                        "narrative": state.narrative,
                        "detail_events": state.detail_events,
                        "horizon_forecast": state.horizon_forecast,
                        "finance_enabled": bool(scenario.operational_finance.enabled),
                        "exercise_type": scenario.exercise_type,
                        "decision_layer_active_summary": getattr(
                            state, "decision_layer_active_summary", ""
                        )
                        or "",
                        "decision_layer_history": list(
                            getattr(state, "decision_layer_history", []) or []
                        )[-24:],
                    }
                )
        except HTTPException as he:
            q.put(
                {
                    "phase": "error",
                    "error": he.detail if isinstance(he.detail, str) else str(he.detail),
                    "traceback": "",
                }
            )
        except Exception as e:
            q.put(
                {
                    "phase": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            q.put(None)

    return _sse_stream_from_worker(worker)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "people.webapp:app",
        host="127.0.0.1",
        port=8770,
        reload=False,
    )


if __name__ == "__main__":
    main()
