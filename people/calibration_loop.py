"""
用外部真值表自动试参数 → 再跑 → 选更贴近的一组（网格 + 简单微调）。
"""

from __future__ import annotations

import itertools
import statistics
from pathlib import Path
from typing import Any

import yaml

from people.config import Scenario
from people.engine import RunConfig, run_single_simulation
from people.experiment_manifest import scenario_content_hash
from people.validation_tools import average_mae, compare_tick_series, ensure_tick_column, read_metrics_csv


def parse_grid_spec(spec: str) -> dict[str, list[float]]:
    """
    形如：llm_effect_multiplier:0.28,0.38|policy_from_sentiment_weight:0.02,0.028
    键必须是 RealismConfig 上的字段名。
    """
    out: dict[str, list[float]] = {}
    for chunk in spec.split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, rest = chunk.split(":", 1)
        name = name.strip()
        vals = [float(x.strip()) for x in rest.split(",") if x.strip()]
        if name and vals:
            out[name] = vals
    return out


def _patch_realism(scenario: Scenario, updates: dict[str, float]) -> Scenario:
    rd = scenario.realism.model_dump()
    for k, v in updates.items():
        if k in rd:
            rd[k] = float(v)
    R = type(scenario.realism)
    return scenario.model_copy(update={"realism": R.model_validate(rd)})


def run_calibration(
    scenario: Scenario,
    cfg: RunConfig,
    *,
    seed: int,
    truth_rows: list[dict[str, Any]],
    grid_spec: str,
    refine_passes: int = 0,
    refine_step: float = 0.05,
) -> dict[str, Any]:
    """
    grid_spec：多参数笛卡尔积；refine_passes>0 时对最优解每个参数做 ±refine_step 比例微调。
    """
    grid = parse_grid_spec(grid_spec)
    if not grid:
        return {"error": "网格为空，请用 --grid 例如 llm_effect_multiplier:0.3,0.38,0.48"}

    keys = list(grid.keys())
    history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_params: dict[str, float] | None = None
    best_report: dict[str, Any] | None = None

    for combo in itertools.product(*[grid[k] for k in keys]):
        upd = {keys[i]: float(combo[i]) for i in range(len(keys))}
        sc2 = _patch_realism(scenario, upd)
        state, _pause = run_single_simulation(sc2, cfg, seed=seed)
        sim_rows = ensure_tick_column([dict(x) for x in state.metrics_history])
        report = compare_tick_series(sim_rows, truth_rows)
        score = average_mae(report)
        if score is None:
            score = 1e9
        history.append({"试的参数": upd, "平均误差": round(score, 6), "明细": report})
        if best_score is None or score < best_score:
            best_score = score
            best_params = upd
            best_report = report

    assert best_params is not None
    sha_before = scenario_content_hash(scenario)

    for _ in range(max(0, refine_passes)):
        improved = False
        base = dict(best_params)
        for k, v in list(base.items()):
            for mult in (1.0 - refine_step, 1.0 + refine_step):
                nv = max(0.001, float(v) * mult)
                trial = {**base, k: nv}
                sc2 = _patch_realism(scenario, trial)
                state, _ = run_single_simulation(sc2, cfg, seed=seed)
                sim_rows = ensure_tick_column([dict(x) for x in state.metrics_history])
                report = compare_tick_series(sim_rows, truth_rows)
                score = average_mae(report) or 1e9
                history.append({"试的参数": trial, "平均误差": round(score, 6), "明细": report, "阶段": "微调"})
                if best_score is not None and score < best_score:
                    best_score = score
                    best_params = trial
                    best_report = report
                    improved = True
                    base = dict(best_params)
        if not improved:
            break

    sc_best = _patch_realism(scenario, best_params)
    sha_after = scenario_content_hash(sc_best)
    closure = {
        "scenario_content_sha256_校准前": sha_before,
        "scenario_content_sha256_应用最优realism后": sha_after,
        "最优参数涉及的_realism_键": sorted(best_params.keys()),
        "说明": "闭环：试参 → 选优 → 可用 merge_realism_into_scenario_yaml 写回新 YAML → 再 validate/ensemble；"
        "PyYAML 写回会丢失注释与部分排版。",
    }
    return {
        "用人话说": {
            "做了什么": "按您给的真值表，自动试了几组「写实强度」参数，选平均误差更小的一组。",
            "最好的一组": best_params,
            "平均误差大约": best_score,
            "说明": "误差越小越贴近真值表，但不代表未来真实世界也一定如此。",
        },
        "最好的一组参数": best_params,
        "平均误差": best_score,
        "与真值对比摘要": best_report,
        "全部尝试记录": history,
        "校准闭环": closure,
    }


def _weighted_objective_from_report(report: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    """
    多指标加权目标：对 compare_tick_series 的输出做加权 MAE 汇总。
    - weights=None：退化为 average_mae
    - weights={"policy_support": 2.0, "rumor_level": 1.0}：按指标加权
    """
    if not isinstance(report, dict):
        return 1e9
    if not weights:
        return float(average_mae(report) or 1e9)
    wsum = 0.0
    acc = 0.0
    for k, w in weights.items():
        if not w:
            continue
        row = report.get(k)
        if not isinstance(row, dict):
            continue
        mae = row.get("mae")
        if not isinstance(mae, (int, float)):
            continue
        ww = float(w)
        wsum += abs(ww)
        acc += abs(ww) * float(mae)
    if wsum <= 1e-12:
        return float(average_mae(report) or 1e9)
    return float(acc / wsum)


def run_calibration_bayes(
    scenario: Scenario,
    cfg: RunConfig,
    *,
    seed: int,
    truth_rows: list[dict[str, Any]],
    # search space: {"llm_effect_multiplier": [0.25, 0.65], "policy_from_sentiment_weight": [0.0, 0.08]}
    space: dict[str, tuple[float, float]],
    n_trials: int = 40,
    metric_weights: dict[str, float] | None = None,
    # 联合优化：对“多 seed 运行”的方差做惩罚（越大越不稳定）
    ensemble_seeds: list[int] | None = None,
    ensemble_var_lambda: float = 0.0,
) -> dict[str, Any]:
    """
    以 optuna 做贝叶斯优化（TPE）：
    - 支持多指标加权目标函数
    - 可选：联合考虑 ensemble 的终局方差（稳定性惩罚）
    """
    try:
        import optuna  # type: ignore
    except Exception as e:
        return {"error": f"缺少 optuna 依赖：{e}. 请安装 requirements.txt 后重试。"}

    if not space:
        return {"error": "space 为空：请提供待校准 realism 参数范围"}
    keys = list(space.keys())
    for k in keys:
        lo, hi = space[k]
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or float(hi) <= float(lo):
            return {"error": f"space[{k}] 非法：需要 (low, high) 且 high>low"}

    ens_seeds = list(ensemble_seeds or [])
    if ens_seeds and seed not in ens_seeds:
        ens_seeds = [seed] + ens_seeds

    history: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        upd: dict[str, float] = {}
        for k in keys:
            lo, hi = space[k]
            upd[k] = float(trial.suggest_float(k, float(lo), float(hi)))
        sc2 = _patch_realism(scenario, upd)

        # 基准 seed 的拟合误差
        state, _pause = run_single_simulation(sc2, cfg, seed=seed)
        sim_rows = ensure_tick_column([dict(x) for x in state.metrics_history])
        report = compare_tick_series(sim_rows, truth_rows)
        base_score = _weighted_objective_from_report(report, metric_weights)

        # ensemble 方差惩罚（终局/关键指标的方差 proxy）
        var_pen = 0.0
        if ens_seeds and ensemble_var_lambda > 0:
            finals: list[dict[str, float]] = []
            for sd in ens_seeds[:8]:
                st2, _ = run_single_simulation(sc2, cfg, seed=int(sd))
                finals.append(st2.snapshot_metrics())
            # 仅对 weights 中出现的指标（或默认 policy_support）计算方差
            watch = list(metric_weights.keys()) if metric_weights else ["policy_support"]
            for mk in watch:
                vals = [float(r.get(mk)) for r in finals if isinstance(r.get(mk), (int, float))]
                if len(vals) >= 2:
                    var_pen += float(statistics.pvariance(vals))
            var_pen *= float(ensemble_var_lambda)

        score = float(base_score + var_pen)
        history.append(
            {
                "params": upd,
                "score": score,
                "base_score": base_score,
                "var_penalty": var_pen,
            }
        )
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(max(1, n_trials)))
    best = dict(study.best_params or {})
    best_score = float(study.best_value)
    sc_best = _patch_realism(scenario, {k: float(best[k]) for k in best})
    return {
        "ok": True,
        "method": "bayes_optuna_tpe",
        "best_params": best,
        "best_score": best_score,
        "trials": history[-min(300, len(history)) :],
    }


def run_sensitivity_realism(
    scenario: Scenario,
    cfg: RunConfig,
    *,
    seed: int,
    truth_rows: list[dict[str, Any]],
    param: str,
    values: list[float],
) -> dict[str, Any]:
    """单参数扫描：看调这一个旋钮时，和真值的差距怎么变。"""
    rows_out: list[dict[str, Any]] = []
    for v in values:
        sc2 = _patch_realism(scenario, {param: float(v)})
        state, _ = run_single_simulation(sc2, cfg, seed=seed)
        sim_rows = ensure_tick_column([dict(x) for x in state.metrics_history])
        report = compare_tick_series(sim_rows, truth_rows)
        score = average_mae(report)
        rows_out.append(
            {
                "参数": param,
                "取值": v,
                "平均误差": score,
                "明细": report,
            }
        )
    best = min(rows_out, key=lambda r: (r["平均误差"] is None, r["平均误差"] or 1e9))
    return {
        "用人话说": f"只动「{param}」时，误差最小的一次是取 {best['取值']}（平均误差 {best['平均误差']}）。",
        "表格": rows_out,
    }


def load_truth_rows(path: str) -> list[dict[str, Any]]:
    return read_metrics_csv(path)


def merge_realism_into_scenario_yaml(
    source: str | Path,
    best_params: dict[str, float],
    dest: str | Path,
) -> None:
    """
    将校准得到的最优 realism 键值合并进场景 YAML，写入 dest（可覆盖源文件，建议另存为新文件）。
    使用 PyYAML safe_dump：注释与锚点会丢失，请在版本管理中 diff 审阅。
    """
    src = Path(source)
    dst = Path(dest)
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("场景 YAML 根节点须为 mapping")
    r = raw.get("realism")
    if r is None:
        raw["realism"] = {}
        r = raw["realism"]
    if not isinstance(r, dict):
        raise ValueError("realism 须为 mapping")
    for k, v in best_params.items():
        r[k] = float(v)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, default_flow_style=False, width=120),
        encoding="utf-8",
    )
