from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from people.config import Scenario
from people.engine import RunConfig, run_ensemble_parallel, run_single_simulation, summarize_ensemble
from people.state import WorldState
from people.validation_tools import compare_tick_series, load_metrics_history_from_json, read_metrics_csv
from people.providers.registry import get_client
from people.roster import count_llm_agents_upper_bound


def scenario_from_args(args: argparse.Namespace) -> Scenario:
    s = Scenario.load(args.scenario)
    updates: dict = {}
    bf = getattr(args, "brief_file", None)
    if bf:
        updates["player_brief"] = Path(bf).read_text(encoding="utf-8")
    elif getattr(args, "brief", None):
        updates["player_brief"] = args.brief
    rf = getattr(args, "reference_file", None)
    if rf:
        updates["reference_cases_brief"] = Path(rf).read_text(encoding="utf-8")
    ex = getattr(args, "exercise", None)
    if ex is not None:
        updates["exercise_type"] = ex
    if updates:
        s = s.model_copy(update=updates)
    return s


def _print_scenario_header(con: Console, scenario: Scenario) -> None:
    pc = scenario.policy_context
    iss = scenario.issuer
    ref = (scenario.reference_cases_brief or "").strip()
    con.print(f"[bold]场景[/bold] {scenario.name}  ·  ticks={scenario.ticks}  ·  seed 见参数")
    if scenario.exercise_type == "policy":
        con.print(
            f"  [政策] 层级 [cyan]{pc.admin_level}[/cyan]  "
            f"{pc.jurisdiction_name or '（未填地名）'}",
        )
    else:
        con.print(
            f"  [产品] 发行方 [cyan]{iss.archetype}[/cyan]  "
            f"brand_equity={iss.brand_equity:.2f}",
        )
    if ref:
        con.print(f"  [参考案例] 已加载约 [cyan]{len(ref)}[/cyan] 字（脱敏摘要）")
    layers = [getattr(c, "class_layer", "mixed") for c in scenario.cohorts]
    if layers:
        con.print(f"  [人群阶层 class_layer] {', '.join(layers)}")


def _resolve_specs(args: argparse.Namespace) -> tuple[str, str]:
    primary = args.primary or os.environ.get("PEOPLE_MODEL_PRIMARY") or "openai:gpt-4o"
    fast = args.fast or os.environ.get("PEOPLE_MODEL_FAST") or "openai:gpt-4o-mini"
    return primary, fast


def _apply_regional_grounding_cli(scenario: Scenario, args: argparse.Namespace) -> Scenario:
    rg = scenario.regional_grounding
    u: dict = {}
    if getattr(args, "region", None):
        u["region_label"] = str(args.region).strip()
    if getattr(args, "grounding_mode", None):
        u["mode"] = args.grounding_mode
    if getattr(args, "business_scale", None):
        u["business_scale"] = args.business_scale
    if getattr(args, "business_sector", None):
        u["business_sector_brief"] = str(args.business_sector).strip()
    if getattr(args, "known_local_policy", None):
        u["user_known_local_policy_brief"] = str(args.known_local_policy).strip()
    if getattr(args, "known_central_policy", None):
        u["user_known_central_policy_brief"] = str(args.known_central_policy).strip()
    if getattr(args, "web_search", False):
        u["web_search_enabled"] = True
    if getattr(args, "web_search_provider", None):
        u["web_search_provider"] = args.web_search_provider
        u["web_search_enabled"] = True
    if getattr(args, "regional_grounding", False):
        u["enabled"] = True
    elif u:
        u["enabled"] = True
    else:
        return scenario
    return scenario.model_copy(update={"regional_grounding": rg.model_copy(update=u)})


def cmd_run(args: argparse.Namespace) -> None:
    con = Console()
    scenario = scenario_from_args(args)
    scenario = _apply_regional_grounding_cli(scenario, args)
    _print_scenario_header(con, scenario)
    primary_spec, fast_spec = _resolve_specs(args)
    primary = get_client(primary_spec, allow_dry=True)
    fast = get_client(fast_spec, allow_dry=True)
    cfg = RunConfig(
        primary_client=primary,
        fast_client=fast,
        primary_model_slot=primary_spec,
        fast_model_slot=fast_spec,
    )
    if getattr(args, "show_budget", False):
        lo, hi = count_llm_agents_upper_bound(scenario)
        con.print(f"[roster] 每 tick 约 [bold]{lo}–{hi}[/bold] 次 LLM 智能体（固定人设 + 池化抽样）")
    state, pause_pkg = run_single_simulation(scenario, cfg, seed=args.seed)
    if pause_pkg is not None:
        con.print(
            "[yellow]演练已暂停（YAML 危机规则或「关键推演满 N 次」内部风险盘点）。"
            "请在网页端提交方案并续跑；CLI 单次 run 仅展示暂停当刻快照。[/yellow]",
        )
    if scenario.exercise_type == "product":
        tip = "policy_support = 对发行方动作的支持/好感（含购买意愿）proxy"
    else:
        tip = "policy_support = 对政策/施政的接受度"
    con.print(f"[bold]演练类型[/bold] {scenario.exercise_type} — {tip}")
    con.print("[bold]终局指标[/bold]", state.snapshot_metrics())
    dump_m = getattr(args, "dump_metrics_json", None)
    if dump_m:
        Path(dump_m).write_text(
            json.dumps({"metrics_history": state.metrics_history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        con.print(f"[bold]已写入 metrics_history[/bold] {dump_m}")
    dump_full = getattr(args, "dump_full_state", None)
    if dump_full:
        from people.attribution_report import build_plain_report
        from people.causal_consistency import audit_scenario_causal_consistency
        from people.decision_support import build_decision_support_bundle
        from people.experiment_manifest import build_experiment_manifest

        payload = {
            "format": "people_run_dump_v1",
            "seed": int(args.seed),
            "scenario_dict": scenario.model_dump(),
            "state_dict": state.model_dump(),
            "用人话_结果怎么来的": build_plain_report(state),
            "决策辅助_粗检": build_decision_support_bundle(state, scenario),
            "experiment_manifest": build_experiment_manifest(
                scenario,
                seed=int(args.seed),
                primary_model_slot=primary_spec,
                fast_model_slot=fast_spec,
            ),
            "因果一致性审计_静态": audit_scenario_causal_consistency(scenario),
        }
        Path(dump_full).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        con.print(f"[bold]已写入完整存档[/bold] {dump_full}（供 explain 子命令解读）")
    con.print("[bold]叙事摘录[/bold]")
    for line in state.narrative[-20:]:
        con.print(line)


def cmd_ensemble(args: argparse.Namespace) -> None:
    con = Console()
    scenario = _apply_regional_grounding_cli(scenario_from_args(args), args)
    _print_scenario_header(con, scenario)
    primary_spec, fast_spec = _resolve_specs(args)
    if scenario.regional_grounding.enabled:
        from people.regional_grounding import prebake_regional_grounding_for_ensemble

        cfg0 = RunConfig(
            primary_client=get_client(primary_spec, allow_dry=True),
            fast_client=get_client(fast_spec, allow_dry=True),
            primary_model_slot=primary_spec,
            fast_model_slot=fast_spec,
        )
        scenario = prebake_regional_grounding_for_ensemble(scenario, cfg0)
    results = run_ensemble_parallel(
        scenario,
        n_runs=args.runs,
        base_seed=args.seed,
        primary_spec=primary_spec,
        fast_spec=fast_spec,
        max_workers=args.workers,
    )
    summary = summarize_ensemble(
        results,
        threshold_key=args.threshold_key,
        threshold=args.threshold,
        scenario=scenario,
        model_slots={"primary": primary_spec, "fast": fast_spec},
    )
    con.print(
        "[bold]数字汇总（JSON）[/bold]",
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("用人话说", "标准输出_决策简报")},
            ensure_ascii=False,
            indent=2,
        ),
    )
    plain = summary.get("用人话说")
    if plain:
        con.print("[bold]通俗解读（波动与极端情况）[/bold]")
        con.print(json.dumps(plain, ensure_ascii=False, indent=2))
    briefing = summary.get("标准输出_决策简报")
    if briefing:
        con.print("[bold]标准决策简报（风险分布 / 尾部 / 分歧 / 建议）[/bold]")
        con.print(json.dumps(briefing, ensure_ascii=False, indent=2))
    if args.dump_json:
        from people.causal_consistency import audit_scenario_causal_consistency
        from people.experiment_manifest import build_experiment_manifest, scenario_content_hash

        dump = {
            "summary": summary,
            "scenario": {
                "name": scenario.name,
                "exercise_type": scenario.exercise_type,
                "player_brief": scenario.player_brief,
                "reference_cases_brief": scenario.reference_cases_brief,
                "policy_context": scenario.policy_context.model_dump(),
                "issuer": scenario.issuer.model_dump(),
            },
            "runs": results,
            "reproducibility_bundle": {
                "scenario_content_sha256": scenario_content_hash(scenario),
                "experiment_manifest": build_experiment_manifest(
                    scenario,
                    seed=int(args.seed),
                    primary_model_slot=primary_spec,
                    fast_model_slot=fast_spec,
                ),
            },
            "因果一致性审计_静态": audit_scenario_causal_consistency(scenario),
        }
        Path(args.dump_json).write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        con.print(f"已写入 {args.dump_json}")

    t = Table(title="各次 run 终局")
    t.add_column("seed")
    for k in (
        "policy_support",
        "unrest",
        "issuer_trust_proxy",
        "supply_chain_stress",
        "sentiment",
        "rumor_level",
    ):
        t.add_column(k)
    for r in results:
        m = r["final_metrics"]
        t.add_row(
            str(r["seed"]),
            f'{m.get("policy_support", 0):.3f}',
            f'{m.get("unrest", 0):.3f}',
            f'{m.get("issuer_trust_proxy", 0):.3f}',
            f'{m.get("supply_chain_stress", 0):.3f}',
            f'{m.get("sentiment", 0):.3f}',
            f'{m.get("rumor_level", 0):.3f}',
        )
    con.print(t)


def cmd_validate(args: argparse.Namespace) -> None:
    con = Console()
    if bool(args.sim_json) == bool(args.sim_csv):
        con.print("[red]请二选一：--sim-csv 或 --sim-json[/red]")
        raise SystemExit(2)
    if args.sim_json:
        sim_rows = load_metrics_history_from_json(args.sim_json)
    else:
        sim_rows = read_metrics_csv(args.sim_csv)
    truth_rows = read_metrics_csv(args.truth_csv)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if getattr(args, "keys", None) else None
    report = compare_tick_series(sim_rows, truth_rows, keys=keys)
    con.print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_stability(args: argparse.Namespace) -> None:
    """与 ensemble 相同，默认多跑几次并突出通俗解读。"""
    cmd_ensemble(args)


def cmd_calibrate(args: argparse.Namespace) -> None:
    from people.calibration_loop import load_truth_rows, merge_realism_into_scenario_yaml, run_calibration

    con = Console()
    scenario = scenario_from_args(args)
    _print_scenario_header(con, scenario)
    truth_rows = load_truth_rows(args.truth_csv)
    primary_spec, fast_spec = _resolve_specs(args)
    primary = get_client(primary_spec, allow_dry=True)
    fast = get_client(fast_spec, allow_dry=True)
    cfg = RunConfig(
        primary_client=primary,
        fast_client=fast,
        primary_model_slot=primary_spec,
        fast_model_slot=fast_spec,
    )
    out = run_calibration(
        scenario,
        cfg,
        seed=int(args.seed),
        truth_rows=truth_rows,
        grid_spec=args.grid,
        refine_passes=int(args.refine_passes),
        refine_step=float(args.refine_step),
    )
    if out.get("error"):
        con.print(f"[red]{out['error']}[/red]")
        raise SystemExit(2)
    con.print("[bold]校准结果（通俗）[/bold]")
    con.print(json.dumps(out.get("用人话说"), ensure_ascii=False, indent=2))
    con.print("[bold]最好的一组写实参数（可贴回 YAML realism）[/bold]")
    con.print(json.dumps(out.get("最好的一组参数"), ensure_ascii=False, indent=2))
    closure = out.get("校准闭环")
    if closure:
        con.print("[bold]校准闭环（版本指纹）[/bold]")
        con.print(json.dumps(closure, ensure_ascii=False, indent=2))
    ws = getattr(args, "write_scenario", None)
    if ws and out.get("最好的一组参数"):
        merge_realism_into_scenario_yaml(args.scenario, out["最好的一组参数"], ws)
        con.print(f"[bold]已写入合并最优 realism 的场景[/bold] {ws}（PyYAML 会丢失注释，请 diff 审阅）")
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        con.print(f"完整过程已写入 {args.dump_json}")


def cmd_sensitivity(args: argparse.Namespace) -> None:
    from people.calibration_loop import load_truth_rows, run_sensitivity_realism

    con = Console()
    scenario = scenario_from_args(args)
    truth_rows = load_truth_rows(args.truth_csv)
    primary_spec, fast_spec = _resolve_specs(args)
    cfg = RunConfig(
        primary_client=get_client(primary_spec, allow_dry=True),
        fast_client=get_client(fast_spec, allow_dry=True),
        primary_model_slot=primary_spec,
        fast_model_slot=fast_spec,
    )
    vals = [float(x.strip()) for x in args.values.split(",") if x.strip()]
    out = run_sensitivity_realism(
        scenario,
        cfg,
        seed=int(args.seed),
        truth_rows=truth_rows,
        param=args.param.strip(),
        values=vals,
    )
    con.print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_scenario_audit(args: argparse.Namespace) -> None:
    from people.causal_consistency import audit_scenario_causal_consistency
    from people.experiment_manifest import scenario_content_hash

    con = Console()
    scenario = Scenario.load(args.scenario)
    rep = audit_scenario_causal_consistency(scenario)
    rep["scenario_content_sha256"] = scenario_content_hash(scenario)
    rep["scenario_name"] = scenario.name
    con.print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep.get("ok", False):
        raise SystemExit(2)


def cmd_explain(args: argparse.Namespace) -> None:
    from people.attribution_report import build_plain_report, explain_metric_at_tick
    from people.config import Scenario
    from people.decision_support import build_decision_support_bundle

    con = Console()
    raw = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    if raw.get("format") != "people_run_dump_v1" and "state_dict" not in raw:
        con.print("[red]JSON 需含 state_dict，或由 run --dump-full-state 生成。[/red]")
        raise SystemExit(2)
    state = WorldState.model_validate(raw["state_dict"])
    if args.metric and args.tick is not None:
        rep = explain_metric_at_tick(state, args.metric.strip(), int(args.tick))
    else:
        rep = build_plain_report(state)
        if isinstance(raw.get("scenario_dict"), dict):
            scen = Scenario.model_validate(raw["scenario_dict"])
            rep = {**rep, "决策辅助_粗检": build_decision_support_bundle(state, scen)}
        elif raw.get("决策辅助_粗检"):
            rep = {**rep, "决策辅助_粗检": raw["决策辅助_粗检"]}
    con.print(json.dumps(rep, ensure_ascii=False, indent=2))


def add_regional_grounding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--regional-grounding",
        action="store_true",
        help="模拟前多轮地区情境检索（维基可选+快模型），再叠加指挥台 brief",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="地区标签（写入 regional_grounding.region_label，可与 jurisdiction 并用）",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("llm_only", "wikipedia_then_llm"),
        default=None,
        help="地区检索模式：仅 LLM 或 维基摘要后再 LLM",
    )
    parser.add_argument(
        "--business-scale",
        choices=("unset", "street_shop", "sme_chain", "regional_group", "national_group"),
        default=None,
        help="商户/企业体量（product 演练）；可自动对齐 issuer 体量",
    )
    parser.add_argument("--business-sector", default=None, help="业态说明，如 社区火锅店、连锁餐饮")
    parser.add_argument("--known-local-policy", default=None, help="你已知的当地既有政策摘要（可选）")
    parser.add_argument("--known-central-policy", default=None, help="你已知的中央/部委层面摘要（可选）")
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="地区 grounding 前抓取开放网页检索摘要（DuckDuckGo 或 Brave，snippet 级；非法规库）",
    )
    parser.add_argument(
        "--web-search-provider",
        choices=("duckduckgo", "brave"),
        default=None,
        help="网页检索提供商；设此项时隐含开启 web_search（Brave 需环境变量 BRAVE_API_KEY）",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="people — 政策制定者 / 产品发行方 社会与市场反响演练（多模型 MVP）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="单次模拟")
    pr.add_argument("--scenario", default=str(ROOT / "scenarios" / "default.yaml"))
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--primary", default=None, help="如 openai:gpt-4o 或 anthropic:claude-sonnet-4-20250514")
    pr.add_argument("--fast", default=None, help="轻量模型，用于 cohort 批量等")
    pr.add_argument(
        "--exercise",
        choices=("policy", "product"),
        default=None,
        help="policy=政策沙盘；product=产品/市场反响沙盘（覆盖 YAML）",
    )
    pr.add_argument("--brief", default=None, help="用户下达的政策要点或产品/发行说明（短文本）")
    pr.add_argument("--brief-file", default=None, help="同上，从 UTF-8 文件读取；优先于 --brief")
    pr.add_argument("--reference-file", default=None, help="脱敏真实案例 UTF-8 文本，写入 reference_cases_brief")
    pr.add_argument("--show-budget", action="store_true", help="打印每 tick LLM 调用人数区间（人设名册 + 抽样）")
    pr.add_argument(
        "--dump-metrics-json",
        default=None,
        help="将 metrics_history 写入 JSON（供 validate --sim-json 使用）",
    )
    pr.add_argument(
        "--dump-full-state",
        default=None,
        help="写入完整 state+scenario+通俗归因 JSON（供 explain 子命令）",
    )
    add_regional_grounding_args(pr)
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("ensemble", help="并行多次（蒙特卡洛式频率估计）")
    pe.add_argument("--scenario", default=str(ROOT / "scenarios" / "default.yaml"))
    pe.add_argument("--seed", type=int, default=42, help="起始随机种子，每次 +1")
    pe.add_argument("--runs", type=int, default=8)
    pe.add_argument("--workers", type=int, default=4)
    pe.add_argument("--primary", default=None)
    pe.add_argument("--fast", default=None)
    pe.add_argument("--threshold-key", default="policy_support")
    pe.add_argument("--threshold", type=float, default=0.55)
    pe.add_argument("--dump-json", default=None, help="完整结果 JSON 路径")
    pe.add_argument("--exercise", choices=("policy", "product"), default=None)
    pe.add_argument("--brief", default=None)
    pe.add_argument("--brief-file", default=None)
    pe.add_argument("--reference-file", default=None)
    add_regional_grounding_args(pe)
    pe.set_defaults(func=cmd_ensemble)

    pst = sub.add_parser(
        "stability",
        help="稳定性：多次运行 + 通俗解读（最好/最差种子、哪项指标抖得最厉害）",
    )
    pst.add_argument("--scenario", default=str(ROOT / "scenarios" / "default.yaml"))
    pst.add_argument("--seed", type=int, default=42)
    pst.add_argument("--runs", type=int, default=12)
    pst.add_argument("--workers", type=int, default=4)
    pst.add_argument("--primary", default=None)
    pst.add_argument("--fast", default=None)
    pst.add_argument("--threshold-key", default="policy_support")
    pst.add_argument("--threshold", type=float, default=0.55)
    pst.add_argument("--dump-json", default=None)
    pst.add_argument("--exercise", choices=("policy", "product"), default=None)
    pst.add_argument("--brief", default=None)
    pst.add_argument("--brief-file", default=None)
    pst.add_argument("--reference-file", default=None)
    add_regional_grounding_args(pst)
    pst.set_defaults(func=cmd_stability)

    pcal = sub.add_parser(
        "calibrate",
        help="校准闭环：对照真值表自动试 realism 参数网格，可选微调，输出最好的一组",
    )
    pcal.add_argument("--scenario", default=str(ROOT / "scenarios" / "default.yaml"))
    pcal.add_argument("--truth-csv", required=True, help="含 tick 与指标列的真值表")
    pcal.add_argument("--seed", type=int, default=42)
    pcal.add_argument(
        "--grid",
        required=True,
        help="例如 llm_effect_multiplier:0.28,0.38|policy_from_sentiment_weight:0.02,0.028",
    )
    pcal.add_argument("--refine-passes", type=int, default=0, help=">0 时对最优解做小幅比例微调")
    pcal.add_argument("--refine-step", type=float, default=0.05)
    pcal.add_argument("--primary", default=None)
    pcal.add_argument("--fast", default=None)
    pcal.add_argument("--dump-json", default=None, help="写入全部尝试过程")
    pcal.add_argument(
        "--write-scenario",
        default=None,
        metavar="PATH",
        help="将最优 realism 合并写入该 YAML 路径（建议另存为新文件；会丢失注释）",
    )
    pcal.add_argument("--exercise", choices=("policy", "product"), default=None)
    pcal.add_argument("--brief", default=None)
    pcal.add_argument("--brief-file", default=None)
    pcal.add_argument("--reference-file", default=None)
    pcal.set_defaults(func=cmd_calibrate)

    paudit = sub.add_parser(
        "scenario-audit",
        help="静态审计：因果层/触发器/危机规则指标域与因果边有向环（可 CI 门禁）",
    )
    paudit.add_argument("--scenario", required=True, help="场景 YAML 路径")
    paudit.set_defaults(func=cmd_scenario_audit)

    psen = sub.add_parser(
        "sensitivity",
        help="敏感性：只动 realism 里某一个参数，看和真值的平均误差怎么变",
    )
    psen.add_argument("--scenario", required=True)
    psen.add_argument("--truth-csv", required=True)
    psen.add_argument("--param", required=True, help="RealismConfig 字段名，如 llm_effect_multiplier")
    psen.add_argument("--values", required=True, help="逗号分隔，如 0.28,0.38,0.48")
    psen.add_argument("--seed", type=int, default=42)
    psen.add_argument("--primary", default=None)
    psen.add_argument("--fast", default=None)
    psen.add_argument("--dump-json", default=None)
    psen.add_argument("--exercise", choices=("policy", "product"), default=None)
    psen.add_argument("--brief", default=None)
    psen.add_argument("--brief-file", default=None)
    psen.add_argument("--reference-file", default=None)
    psen.set_defaults(func=cmd_sensitivity)

    pex = sub.add_parser("explain", help="解读 run --dump-full-state 的存档：结果怎么来的")
    pex.add_argument("--from-json", required=True, help="dump-full-state 生成的 JSON")
    pex.add_argument("--metric", default=None, help="若指定则只解释该指标")
    pex.add_argument("--tick", type=int, default=None, help="配合 --metric 使用")
    pex.set_defaults(func=cmd_explain)

    pv = sub.add_parser("validate", help="校准/验证：对比仿真时间序列与外部真值 CSV（MAE 等）")
    pv.add_argument("--truth-csv", required=True, help="含 tick 列及指标列的真值表")
    pv.add_argument("--sim-csv", default=None, help="仿真输出 CSV（与 --sim-json 二选一）")
    pv.add_argument(
        "--sim-json",
        default=None,
        help="含 metrics_history 数组的 JSON，或纯 metrics_history 数组",
    )
    pv.add_argument(
        "--keys",
        default=None,
        help="逗号分隔指标列名；默认取两边共有数值列",
    )
    pv.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
