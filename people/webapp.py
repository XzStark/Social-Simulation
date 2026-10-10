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
import hmac
import hashlib
from html import escape as html_escape
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from people.config import WebSearchProvider
from people.scenario_overrides import merge_scenario_overrides, override_payload_from_model
from people.schema_versions import PEOPLE_API_SCHEMA_VERSION, PEOPLE_CHECKPOINT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent

_ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3, "auditor": 4}

_STREAM_LOCK = threading.Lock()
_STREAM_TOTAL_ACTIVE = 0
_STREAM_ACTIVE_BY_CLIENT: dict[str, int] = {}
_STREAM_TOTAL_LIMIT = int(os.environ.get("PEOPLE_STREAMS_TOTAL", "8") or "8")
_STREAM_PER_CLIENT_LIMIT = int(os.environ.get("PEOPLE_STREAMS_PER_CLIENT", "2") or "2")


def _check_access(request: Request, *, min_role: str = "operator") -> None:
    expected = (os.environ.get("PEOPLE_API_TOKEN") or "").strip()
    if expected:
        got = (request.headers.get("x-people-token") or "").strip()
        if got != expected:
            raise HTTPException(status_code=401, detail="认证失败：缺少或错误的 x-people-token")
    got_role = (request.headers.get("x-people-role") or "viewer").strip().lower()
    need = _ROLE_LEVEL.get(min_role, 2)
    cur = _ROLE_LEVEL.get(got_role, 1)
    if cur < need:
        raise HTTPException(status_code=403, detail=f"权限不足：需要 {min_role} 及以上角色")


def _client_id(request: Request) -> str:
    cid = (request.headers.get("x-people-client-id") or "").strip()
    if cid:
        return cid[:64]
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


@contextlib.contextmanager
def _acquire_stream_slot(request: Request) -> Any:
    cid = _client_id(request)
    global _STREAM_TOTAL_ACTIVE
    with _STREAM_LOCK:
        cur_total = _STREAM_TOTAL_ACTIVE
        cur_client = _STREAM_ACTIVE_BY_CLIENT.get(cid, 0)
        if _STREAM_TOTAL_LIMIT > 0 and cur_total >= _STREAM_TOTAL_LIMIT:
            raise HTTPException(status_code=429, detail="并发演练过多：请稍后再试或关闭其它窗口后重试。")
        if _STREAM_PER_CLIENT_LIMIT > 0 and cur_client >= _STREAM_PER_CLIENT_LIMIT:
            raise HTTPException(status_code=429, detail="该客户端并发演练过多：请先停止一条正在运行的推演。")
        _STREAM_TOTAL_ACTIVE = cur_total + 1
        _STREAM_ACTIVE_BY_CLIENT[cid] = cur_client + 1
    try:
        yield
    finally:
        with _STREAM_LOCK:
            _STREAM_TOTAL_ACTIVE = max(0, _STREAM_TOTAL_ACTIVE - 1)
            if cid in _STREAM_ACTIVE_BY_CLIENT:
                _STREAM_ACTIVE_BY_CLIENT[cid] = max(0, _STREAM_ACTIVE_BY_CLIENT[cid] - 1)


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
    internal_issues_enabled: bool | None = None
    internal_issue_trigger_pause: bool | None = None
    internal_issue_cooldown_ticks: int | None = Field(default=None, ge=0, le=20)
    internal_issue_probability_policy: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_enterprise: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_startup: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_shop: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_sampling_scope: Literal["auto", "policy", "enterprise", "startup", "shop"] | None = None
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
    internal_issues_enabled: bool | None = None
    internal_issue_trigger_pause: bool | None = None
    internal_issue_cooldown_ticks: int | None = Field(default=None, ge=0, le=20)
    internal_issue_probability_policy: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_enterprise: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_startup: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_probability_shop: float | None = Field(default=None, ge=0.0, le=1.0)
    internal_issue_sampling_scope: Literal["auto", "policy", "enterprise", "startup", "shop"] | None = None
    plan_search_enabled: bool = False
    plan_search_provider: WebSearchProvider | None = None
    plan_search_focus: Literal[
        "general",
        "new_product_launch",
        "acquisition_mna",
        "store_expansion",
        "policy_rollout",
        "competitor_patent_scan",
        "local_store_scan",
    ] = "general"
    # 决策类型：通用 / 商务合作 / 投资谈判 / 内部政治与组织调整
    decision_kind: Literal["general", "business_deal", "investment", "internal_politics"] = "general"


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


class InjectBlackSwanRequest(BaseModel):
    checkpoint_token: str | None = None
    run_control_token: str | None = None
    preset_id: str | None = None
    title: str | None = None
    detail: str | None = None
    severity: float = Field(default=1.0, ge=0.1, le=3.0)
    metric_deltas: dict[str, float] | None = None
    cash_delta_million: float | None = None
    fiscal_delta_billion: float | None = None
    debt_delta_million: float | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> "InjectBlackSwanRequest":
        if not (self.checkpoint_token or self.run_control_token):
            raise ValueError("请提供 checkpoint_token 或 run_control_token")
        if not (self.preset_id or (self.title and str(self.title).strip())):
            raise ValueError("请提供 preset_id，或至少填写自定义 title")
        return self


class PauseRuntimeRequest(BaseModel):
    run_control_token: str = Field(..., min_length=8)
    reason: str | None = None


class ReportExportRequest(BaseModel):
    header: dict[str, Any] = Field(default_factory=dict)
    final_metrics: dict[str, Any] = Field(default_factory=dict)
    narrative: list[str] = Field(default_factory=list)
    decision_layer_active_summary: str | None = None
    mode: Literal["professional", "public"] = "professional"
    title: str | None = None


class AuditVerifyRequest(BaseModel):
    audit_chain: list[dict[str, Any]] = Field(default_factory=list)


def _verify_audit_chain(chain: list[dict[str, Any]]) -> dict[str, Any]:
    secret = (os.environ.get("PEOPLE_AUDIT_SECRET") or "").strip()
    prev = ""
    errors: list[dict[str, Any]] = []
    for idx, e in enumerate(chain or []):
        tick = e.get("tick")
        h = e.get("hash")
        p = e.get("prev_hash")
        if idx == 0:
            # allow empty prev for first element
            if p not in ("", None):
                errors.append({"index": idx, "reason": "first_prev_hash_not_empty", "prev_hash": p})
        else:
            if str(p or "") != str(prev or ""):
                errors.append({"index": idx, "reason": "prev_hash_mismatch", "prev_hash": p, "expected": prev})
        if secret:
            if not e.get("sig"):
                errors.append({"index": idx, "reason": "missing_sig"})
            else:
                msg = f"{tick}|{h}|{p}".encode("utf-8")
                exp = hmac.new(secret.encode("utf-8"), msg, digestmod=hashlib.sha256).hexdigest()
                if str(e.get("sig")) != exp:
                    errors.append({"index": idx, "reason": "sig_mismatch", "got": e.get("sig"), "expected": exp})
        prev = h
    ok = len(errors) == 0 and len(chain or []) > 0
    return {"ok": ok, "errors": errors[:8], "n": len(chain or [])}


class PublicModeInferRequest(BaseModel):
    scenario_text: str = Field(..., min_length=4, max_length=4000)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    fast: str | None = None


class SpecialEvaluateRequest(BaseModel):
    focus: Literal["new_product_launch", "acquisition_mna"] = "new_product_launch"
    scenario_text: str = Field(..., min_length=4, max_length=6000)
    decision_context_hint: dict[str, Any] | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    fast: str | None = None
    plan_search_enabled: bool = True
    plan_search_provider: WebSearchProvider | None = None


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


def _build_black_swan_from_request(body: InjectBlackSwanRequest) -> dict[str, Any]:
    from people.black_swan import build_black_swan_event

    return build_black_swan_event(
        preset_id=(body.preset_id.strip() if body.preset_id else None),
        title=(body.title.strip() if body.title else None),
        detail=(body.detail.strip() if body.detail else None),
        severity=body.severity,
        metric_deltas=body.metric_deltas,
        cash_delta_million=body.cash_delta_million,
        fiscal_delta_billion=body.fiscal_delta_billion,
        debt_delta_million=body.debt_delta_million,
    )


def _build_plan_search_extra_queries(
    *,
    focus: str,
    scenario: Any,
    plain_solution: str,
) -> str:
    ex_type = str(getattr(scenario, "exercise_type", "product") or "product")
    region = (
        str(getattr(scenario.policy_context, "jurisdiction_name", "") or "")
        or str(getattr(scenario.regional_grounding, "region_label", "") or "")
    ).strip()
    issuer_name = str(getattr(scenario.issuer, "reputation_brief", "") or "").strip()
    sector = str(getattr(scenario.regional_grounding, "business_sector_brief", "") or "").strip()
    ms = getattr(scenario, "market_competitors", []) or []
    comp_names = []
    for m in ms[:6]:
        nm = str(getattr(m, "name", "") or "").strip()
        if nm:
            comp_names.append(nm)
    base: list[str] = []
    if ex_type == "policy":
        base.extend(
            [
                f"{region} 政策 执行 阻力 民生 反馈".strip(),
                f"{region} 基层 执行 难点 信访 舆情".strip(),
                f"{region} 行政 审批 合规 典型 问题".strip(),
            ]
        )
    else:
        base.extend(
            [
                f"{region} {sector or '行业'} 竞品 对手 公司 简介".strip(),
                f"{sector or '行业'} 专利 布局 纠纷 侵权 风险".strip(),
                f"{sector or '行业'} 新品 上线 失败 案例 渠道 反馈".strip(),
            ]
        )
        if comp_names:
            for cn in comp_names[:4]:
                base.append(f"{cn} 公司 简介 主要产品 市场份额")
                base.append(f"{cn} 专利 诉讼 合规 风险")
        if "收购" in plain_solution or "并购" in plain_solution or focus == "acquisition_mna":
            base.extend(
                [
                    f"{sector or '行业'} 并购 审批 反垄断 案例",
                    f"{sector or '行业'} 并购 整合 失败 原因",
                ]
            )
        if ("开店" in plain_solution) or ("门店" in plain_solution) or focus == "store_expansion":
            base.extend(
                [
                    f"{region} {sector or ''} 门店 数量 评价 客流".strip(),
                    f"{region} 同城 平台 门店 评分 差评 原因".strip(),
                ]
            )
    if issuer_name:
        base.append(f"{issuer_name[:40]} 负面 舆情 合规 争议")
    # attach the user decision itself as intent query
    ps = (plain_solution or "").strip().replace("\n", " ")
    if ps:
        base.append(ps[:160])
    # de-dup and keep compact
    seen: set[str] = set()
    out: list[str] = []
    for q in base:
        qq = " ".join(str(q).split()).strip()
        if not qq or qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
        if len(out) >= 14:
            break
    return "\n".join(out)


def _execute_simulation(
    body: RunRequest, progress_callback: Any, pop_black_swan_events: Any = None
) -> tuple[Any, Any, dict[str, Any], dict[str, Any] | None]:
    from people.config import Scenario
    from people.engine import RunConfig, run_single_simulation
    from people.providers.registry import get_client

    path = _safe_scenario_file(body.scenario_file)
    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        scenario = Scenario.load(path)
        # 全局 tick 上限保护：防止单场景设置极大 ticks 占满资源
        try:
            max_ticks = int(os.environ.get("PEOPLE_MAX_TICKS", "0") or "0")
        except ValueError:
            max_ticks = 0
        if max_ticks > 0 and scenario.ticks > max_ticks:
            scenario = scenario.model_copy(update={"ticks": max_ticks})
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
            pop_black_swan_events=pop_black_swan_events,
        )
        header = _header_from_scenario(scenario, seed=body.seed)
    return state, scenario, header, pause_pkg


def _execute_resume_simulation(
    body: "ResumeStreamRequest", progress_callback: Any, pop_black_swan_events: Any = None
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
                pop_black_swan_events=pop_black_swan_events,
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
                pop_black_swan_events=pop_black_swan_events,
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
            pop_black_swan_events=pop_black_swan_events,
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
    from people.checkpoint_migrations import migrate_checkpoint_package
    from people.config import Scenario
    from people.context import build_decision_context, enrich_decision_context_for_plan_evaluate
    from people.engine import RunConfig, _hydrate_finance, build_initial_state
    from people.pause_store import copy_checkpoint
    from people.providers.registry import get_client
    from people.rules import aggregate_from_cohorts, finalize_tick
    from people.state import WorldState
    from people.web_search_context import fetch_web_grounding_snippets

    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        if body.checkpoint_token and body.checkpoint_token.strip():
            pkg = migrate_checkpoint_package(copy_checkpoint(body.checkpoint_token.strip()))
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
        plain = body.plain_solution.strip()
        if body.plan_search_enabled:
            focus_query_map = {
                "general": "",
                "new_product_launch": "新品发布 上线节奏 渠道反馈 合规 竞品反应",
                "acquisition_mna": "并购 收购 反垄断 审查 整合 风险",
                "store_expansion": "开店 扩张 商圈 客流 房租 用工 合规",
                "policy_rollout": "政策落地 基层执行 民生反馈 稳定风险",
                "competitor_patent_scan": "竞品 多公司 简介 产品线 专利 布局 诉讼 风险",
                "local_store_scan": "当地 门店 数量 评分 评价 客流 同城平台",
            }
            focus_q = focus_query_map.get(str(body.plan_search_focus or "general"), "")
            rich_q = _build_plan_search_extra_queries(
                focus=str(body.plan_search_focus or "general"),
                scenario=scenario,
                plain_solution=plain,
            )
            rg = scenario.regional_grounding.model_copy(
                update={
                    "enabled": True,
                    "web_search_enabled": True,
                    "web_search_provider": body.plan_search_provider or scenario.regional_grounding.web_search_provider,
                    "region_label": (
                        scenario.regional_grounding.region_label
                        or scenario.policy_context.jurisdiction_name
                        or ""
                    ),
                    "business_sector_brief": scenario.regional_grounding.business_sector_brief
                    or scenario.issuer.supply_chain_position
                    or "",
                    "web_search_extra_queries": (focus_q + "\n" + rich_q).strip(),
                }
            )
            try:
                snippets, meta = fetch_web_grounding_snippets(scenario, rg)
            except Exception as e:
                snippets, meta = "", {"used": False, "error": f"{type(e).__name__}: {e}"}
            if snippets:
                dctx["plan_external_search_snippets"] = snippets
            dctx["plan_external_search_meta"] = meta

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

        digest = professionalize_plain_resolution(
            cfg.fast_client,
            scenario=scenario,
            state=state,
            plain_solution=plain,
            decision_context=dctx,
            temperature=0.25,
            decision_kind=str(body.decision_kind or "general"),
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
                "safety_score",
                "safety_summary",
                "risk_items",
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
            "safety_score",
            "safety_summary",
            "risk_items",
        )
        return {k: d.get(k) for k in keys if d.get(k) is not None}

    return {
        "agent_recommendations": (
            (desk or {}).get("recommended_internal_next_steps")
            or (digest or {}).get("intent_bullets")
            or []
        ),
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
            decision_kind=str(body.decision_kind or "general"),
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
                "decision_kind": str(body.decision_kind or "general"),
            }

    return out


app = FastAPI(title="people", docs_url=None, redoc_url=None)

# Shared, self-hosted presentation assets; simulation endpoints are unchanged.
app.mount("/people/assets", StaticFiles(directory=str(ROOT / "web" / "assets")), name="people-assets")
# Support existing reverse proxies that strip the /people prefix.
app.mount("/assets", StaticFiles(directory=str(ROOT / "web" / "assets")), name="people-assets-prefix-stripped")


@app.get("/")
def index_landing_page() -> FileResponse:
    """
    公网入口页：在有 landing.html 时展示「政企决策演练 / 婚恋演练」选择，
    否则回退到原来的 people 单页演练台。
    """
    landing = ROOT / "web" / "landing.html"
    if landing.is_file():
        return FileResponse(landing)
    # 回退：保持原有行为，直接进入 people 演练台
    html = ROOT / "web" / "index.html"
    if not html.is_file():
        raise HTTPException(status_code=500, detail="缺少 web/index.html")
    return FileResponse(html)


@app.get("/people")
@app.get("/people/")
def people_page() -> FileResponse:
    """显式 people 入口，便于从统一 landing 跳转。"""
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
def api_finance_adjust_checkpoint(body: FinanceCheckpointAdjustRequest, request: Request) -> dict[str, Any]:
    _check_access(request, min_role="operator")
    from people.pause_store import copy_checkpoint, put_checkpoint
    from people.checkpoint_migrations import migrate_checkpoint_package
    from people.state import WorldState

    pkg = migrate_checkpoint_package(copy_checkpoint(body.checkpoint_token))
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
    pkg["checkpoint_schema_version"] = PEOPLE_CHECKPOINT_SCHEMA_VERSION
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
def api_plan_confirm_execute(body: PlanConfirmExecuteRequest, request: Request) -> JSONResponse | dict[str, Any]:
    _check_access(request, min_role="operator")
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
            "api_schema_version": PEOPLE_API_SCHEMA_VERSION,
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
            "timeline_uncertainty": list(getattr(state, "timeline_uncertainty", []) or [])[-200:],
            "audit_chain_tail": list(getattr(state, "audit_chain", []) or [])[-200:],
        }
        if pause_pkg is not None:
            pause_pkg = {**pause_pkg, "checkpoint_schema_version": PEOPLE_CHECKPOINT_SCHEMA_VERSION}
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


@app.post("/api/report/html", response_model=None)
def api_report_html(body: ReportExportRequest) -> dict[str, Any]:
    title = (body.title or "people 决策报告").strip() or "people 决策报告"
    mode_cn = "大众场景模式" if body.mode == "public" else "专业模式"
    hdr_rows = "".join(
        f"<tr><th>{html_escape(str(k))}</th><td>{html_escape(str(v))}</td></tr>"
        for k, v in (body.header or {}).items()
    )
    metric_rows = "".join(
        f"<tr><th>{html_escape(str(k))}</th><td>{html_escape(str(v))}</td></tr>"
        for k, v in (body.final_metrics or {}).items()
    )
    narrative_text = "\n".join(str(x) for x in (body.narrative or []) if str(x).strip())
    decision = (body.decision_layer_active_summary or "").strip()
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html_escape(title)}</title>
  <style>
    body {{ font-family: "Segoe UI","Microsoft YaHei",sans-serif; margin: 22px; color: #1f2937; }}
    h1 {{ margin: 0 0 10px 0; }}
    .meta {{ color: #4b5563; margin-bottom: 14px; font-size: 13px; }}
    h2 {{ margin-top: 20px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ width: 36%; background: #f9fafb; }}
    pre {{ white-space: pre-wrap; line-height: 1.55; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{html_escape(title)}</h1>
  <div class="meta">生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")} · 模式: {mode_cn}</div>
  <h2>场景头信息</h2>
  <table>{hdr_rows or "<tr><td colspan='2'>无</td></tr>"}</table>
  <h2>终局指标</h2>
  <table>{metric_rows or "<tr><td colspan='2'>无</td></tr>"}</table>
  <h2>决策层摘要</h2>
  <pre>{html_escape(decision or "无")}</pre>
  <h2>叙事日志</h2>
  <pre>{html_escape(narrative_text or "无")}</pre>
</body>
</html>"""
    return {"ok": True, "html": html}


@app.post("/api/audit/verify", response_model=None)
def api_audit_verify(body: AuditVerifyRequest) -> dict[str, Any]:
    return _verify_audit_chain(body.audit_chain or [])


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
            "X-People-Api-Schema": PEOPLE_API_SCHEMA_VERSION,
        },
    )


@app.post("/api/run/stream")
def api_run_stream(body: RunRequest, request: Request) -> StreamingResponse:
    _check_access(request, min_role="viewer")
    """Server-Sent Events：逐 tick 推送；遇危机则 phase=crisis_pause（含 checkpoint_token）；否则 complete。"""

    from people.pause_store import put_checkpoint
    from people.runtime_control import (
        create_runtime_control_token,
        pop_black_swans,
        release_runtime_control_token,
    )

    def worker(q: queue.Queue[Any]) -> None:
        with _acquire_stream_slot(request):
            ctrl_token = create_runtime_control_token()
        try:

            def cb(ev: dict[str, Any]) -> None:
                if ev.get("phase") == "run_start":
                    ev = {**ev, "run_control_token": ctrl_token}
                q.put(ev)

            t0 = time.perf_counter()
            state, scenario, header, pause_pkg = _execute_simulation(
                body, progress_callback=cb, pop_black_swan_events=lambda: pop_black_swans(ctrl_token)
            )
            elapsed = time.perf_counter() - t0
            if pause_pkg is not None:
                pause_pkg = {**pause_pkg, "checkpoint_schema_version": PEOPLE_CHECKPOINT_SCHEMA_VERSION}
                tok = put_checkpoint(pause_pkg)
                q.put(
                    {
                        "phase": "crisis_pause",
                        "api_schema_version": PEOPLE_API_SCHEMA_VERSION,
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
                        "run_control_token": ctrl_token,
                    }
                )
            else:
                q.put(
                    {
                        "phase": "complete",
                        "api_schema_version": PEOPLE_API_SCHEMA_VERSION,
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
                        "run_control_token": ctrl_token,
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
            release_runtime_control_token(ctrl_token)
            q.put(None)

    return _sse_stream_from_worker(worker)


@app.post("/api/run/resume/stream")
def api_run_resume_stream(body: ResumeStreamRequest, request: Request) -> StreamingResponse:
    _check_access(request, min_role="viewer")
    from people.pause_store import put_checkpoint
    from people.runtime_control import (
        create_runtime_control_token,
        pop_black_swans,
        release_runtime_control_token,
    )

    def worker(q: queue.Queue[Any]) -> None:
        with _acquire_stream_slot(request):
            ctrl_token = create_runtime_control_token()
        try:

            def cb(ev: dict[str, Any]) -> None:
                if ev.get("phase") == "resume_start":
                    ev = {**ev, "run_control_token": ctrl_token}
                q.put(ev)

            t0 = time.perf_counter()
            state, scenario, header, pause_pkg = _execute_resume_simulation(
                body, progress_callback=cb, pop_black_swan_events=lambda: pop_black_swans(ctrl_token)
            )
            elapsed = time.perf_counter() - t0
            if pause_pkg is not None:
                pause_pkg = {**pause_pkg, "checkpoint_schema_version": PEOPLE_CHECKPOINT_SCHEMA_VERSION}
                tok = put_checkpoint(pause_pkg)
                q.put(
                    {
                        "phase": "crisis_pause",
                        "api_schema_version": PEOPLE_API_SCHEMA_VERSION,
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
                        "run_control_token": ctrl_token,
                    }
                )
            else:
                q.put(
                    {
                        "phase": "complete",
                        "api_schema_version": PEOPLE_API_SCHEMA_VERSION,
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
                        "run_control_token": ctrl_token,
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
            release_runtime_control_token(ctrl_token)
            q.put(None)

    return _sse_stream_from_worker(worker)


@app.get("/api/black-swan/presets", response_model=None)
def api_black_swan_presets() -> dict[str, Any]:
    from people.black_swan import list_black_swan_presets

    return {"ok": True, "presets": list_black_swan_presets()}


@app.post("/api/black-swan/inject", response_model=None)
def api_black_swan_inject(body: InjectBlackSwanRequest, request: Request) -> JSONResponse | dict[str, Any]:
    _check_access(request, min_role="operator")
    from people.black_swan import apply_black_swan_event
    from people.pause_store import put_checkpoint, take_checkpoint
    from people.runtime_control import enqueue_black_swan
    from people.state import WorldState

    try:
        ev = _build_black_swan_from_request(body)
        if body.run_control_token:
            ok = enqueue_black_swan(body.run_control_token.strip(), ev)
            if not ok:
                raise HTTPException(status_code=404, detail="run_control_token 无效或运行已结束")
            return {
                "ok": True,
                "queued": True,
                "target": "runtime",
                "title": ev.get("title"),
                "preset_id": ev.get("preset_id"),
            }

        tok = (body.checkpoint_token or "").strip()
        pkg = take_checkpoint(tok)
        if pkg is None:
            raise HTTPException(status_code=404, detail="checkpoint 不存在或已失效")
        state = WorldState.model_validate(pkg["state_dict"])
        deltas = apply_black_swan_event(state, ev, source="checkpoint")
        pkg["state_dict"] = state.model_dump()
        new_tok = put_checkpoint(pkg)
        return {
            "ok": True,
            "queued": False,
            "target": "checkpoint",
            "checkpoint_token": new_tok,
            "event": {
                "title": ev.get("title"),
                "preset_id": ev.get("preset_id"),
                "severity": ev.get("severity"),
                "deltas": deltas,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()},
        )


def _pause_runtime_request_handler(body: PauseRuntimeRequest) -> dict[str, Any]:
    from people.runtime_control import enqueue_pause_request

    ok = enqueue_pause_request(body.run_control_token.strip(), reason=body.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="run_control_token 无效或运行已结束")
    return {"ok": True, "queued": True, "action": "pause_requested"}


@app.post("/api/run-control/pause", response_model=None)
def api_run_control_pause(body: PauseRuntimeRequest, request: Request) -> dict[str, Any]:
    _check_access(request, min_role="operator")
    return _pause_runtime_request_handler(body)


@app.post("/api/run/pause-request", response_model=None)
def api_run_pause_request(body: PauseRuntimeRequest, request: Request) -> dict[str, Any]:
    _check_access(request, min_role="operator")
    """与 `/api/run-control/pause` 相同；与 `/api/run/stream` 同前缀，便于代理/排查。"""
    return _pause_runtime_request_handler(body)


@app.post("/api/public-mode/infer", response_model=None)
def api_public_mode_infer(body: PublicModeInferRequest) -> JSONResponse | dict[str, Any]:
    from people.providers.registry import get_client

    text = body.scenario_text.strip()
    low = text.lower()
    inferred = "product"
    confidence = 0.6
    why = "根据描述中的业务动作与组织目标进行推断。"
    if any(x in low for x in ("政策", "条例", "执法", "县", "市", "省", "监管口径")):
        inferred = "policy"
        confidence = 0.78
        why = "描述包含政策制定/执行与行政层级语义。"
    elif any(x in low for x in ("创业", "mvp", "融资", "现金流", "小团队")):
        inferred = "startup"
        confidence = 0.76
        why = "描述包含创业团队与融资/跑道等关键词。"
    elif any(x in low for x in ("开店", "门店", "商圈", "到店", "街边店")):
        inferred = "shop"
        confidence = 0.8
        why = "描述重点在门店经营与本地客流。"
    elif any(x in low for x in ("集团", "上市", "品牌", "渠道", "供应链")):
        inferred = "enterprise"
        confidence = 0.74
        why = "描述以品牌、渠道与供应链协同为核心。"

    dept_map: dict[str, list[str]] = {
        "policy": ["政策办/法制", "宣传与融媒体", "公安与信访", "财政与发改", "基层执行条线"],
        "enterprise": ["品牌公关", "法务合规", "供应链与交付", "客服与渠道", "财务与资金"],
        "startup": ["创始人/CEO", "产品技术小队", "增长运营", "外包法务", "兼职财务"],
        "shop": ["门店运营", "采购与供应", "社群与本地营销", "财务收支", "物业与街道协同"],
        "product": ["产品与研发", "品牌公关", "法务合规", "供应链", "客服与渠道"],
    }
    actions_map: dict[str, list[str]] = {
        "policy": ["先明确目标边界与受影响人群", "同步法务/合规审查口径", "设置基层反馈与舆情复盘节奏"],
        "enterprise": ["先冻结对外口径和客服问答", "并行梳理供应链替代方案", "按现金/资源上限分阶段推进"],
        "startup": ["先收敛范围做最小可行动作", "关键动作由创始人盯住节奏", "高风险环节优先外包并设验收点"],
        "shop": ["先保供与价格稳定", "按商圈客流做轻量投放", "建立店员反馈与负评处置机制"],
        "product": ["先明确灰度范围与回滚条件", "建立跨部门每日站会", "提前准备高风险场景应急预案"],
    }
    media_map: dict[str, list[str]] = {
        "policy": ["本地融媒体", "政务新媒体", "社区群/家长群/业主群", "短视频平台"],
        "enterprise": ["主流媒体", "财经媒体", "短视频平台", "论坛/社媒"],
        "startup": ["短视频平台", "社群与私域", "垂类社区", "KOL/KOC"],
        "shop": ["短视频平台", "同城生活平台", "本地社群", "到店评价平台"],
        "product": ["短视频平台", "科技媒体", "社媒平台", "用户社区"],
    }
    mode_hint_map: dict[str, str] = {
        "policy": "professional",
        "enterprise": "professional",
        "startup": "public",
        "shop": "public",
        "product": "professional",
    }

    # Optional LLM refinement (best-effort)
    overlay = _openai_env_overlay_from_body(body)
    with _env_overlay(overlay):
        try:
            fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
            fast = get_client(fast_spec, allow_dry=True)
            raw = fast.complete_json(
                system=(
                    "你是场景分诊助手。给定用户场景描述，输出 JSON: "
                    '{"mode":"policy|enterprise|startup|shop|product","confidence":0~1,'
                    '"reason":"...", "departments":["..."], "next_actions":["..."]}'
                ),
                user=text,
                temperature=0.2,
            )
            if isinstance(raw, dict):
                m = str(raw.get("mode") or inferred).strip()
                if m in ("policy", "enterprise", "startup", "shop", "product"):
                    inferred = m
                c = raw.get("confidence")
                if isinstance(c, (int, float)):
                    confidence = max(0.0, min(1.0, float(c)))
                rs = str(raw.get("reason") or "").strip()
                if rs:
                    why = rs[:240]
                d = raw.get("departments")
                if isinstance(d, list) and d:
                    dept_map[inferred] = [str(x).strip() for x in d[:8] if str(x).strip()]
                a = raw.get("next_actions")
                if isinstance(a, list) and a:
                    actions_map[inferred] = [str(x).strip() for x in a[:8] if str(x).strip()]
        except Exception:
            pass

    recommended_mode = mode_hint_map.get(inferred, "professional")
    return {
        "ok": True,
        "inferred_mode": inferred,
        "confidence": round(confidence, 3),
        "reason": why,
        "recommended_mode": recommended_mode,
        "audience_profile": (
            "大众场景优先：创业公司/门店；政策与大型企业建议切换专业模式。"
            if recommended_mode == "professional"
            else "适合大众场景模式，建议先做小步快跑与低成本试错。"
        ),
        "media_channels": media_map.get(inferred, media_map["product"]),
        "departments": dept_map.get(inferred, dept_map["product"]),
        "next_actions": actions_map.get(inferred, actions_map["product"]),
    }


@app.post("/api/public-mode/special-evaluate", response_model=None)
def api_public_mode_special_evaluate(body: SpecialEvaluateRequest) -> JSONResponse | dict[str, Any]:
    from people.config import Scenario
    from people.providers.registry import get_client
    from people.special_evaluator import run_special_evaluation
    from people.web_search_context import fetch_web_grounding_snippets

    try:
        overlay = _openai_env_overlay_from_body(body)
        with _env_overlay(overlay):
            fast_spec = body.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
            fast = get_client(fast_spec, allow_dry=True)
            dctx = body.decision_context_hint or {}
            snippets = ""
            if body.plan_search_enabled:
                # Use a tiny synthetic scenario to reuse existing search tooling.
                s = Scenario()
                s = s.model_copy(update={"player_brief": body.scenario_text[:800], "exercise_type": "product"})
                rg = s.regional_grounding.model_copy(
                    update={
                        "enabled": True,
                        "web_search_enabled": True,
                        "web_search_provider": body.plan_search_provider or s.regional_grounding.web_search_provider,
                        "web_search_extra_queries": (
                            "新品发布 上线节奏 渠道承载 客服峰值"
                            if body.focus == "new_product_launch"
                            else "并购 收购 审批 反垄断 整合 冲突"
                        ),
                    }
                )
                try:
                    snippets, _meta = fetch_web_grounding_snippets(s, rg)
                except Exception:
                    snippets = ""
            out = run_special_evaluation(
                fast,
                focus=body.focus,
                scenario_text=body.scenario_text,
                decision_context=dctx if isinstance(dctx, dict) else {},
                search_snippets=snippets,
                temperature=0.2,
            )
            return {"ok": True, "result": out}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": traceback.format_exc()},
        )


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
