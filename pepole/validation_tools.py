from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def _float_row(d: dict[str, Any], key: str) -> float | None:
    if key not in d or d[key] is None or str(d[key]).strip() == "":
        return None
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return None


def ensure_tick_column(rows: list[dict[str, Any]], tick_key: str = "tick") -> list[dict[str, Any]]:
    """若缺少 tick 列，则按行序从 1 递增（适配 metrics_history 导出）。"""
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        r = dict(row)
        raw = r.get(tick_key)
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            r[tick_key] = float(i + 1)
        out.append(r)
    return out


def compare_tick_series(
    sim_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    *,
    keys: list[str] | None = None,
    tick_key: str = "tick",
) -> dict[str, Any]:
    """
    将两条按 tick 对齐的时间序列做简单残差统计（同 tick 键必须一致）。
    keys 默认取两边共有的数值列（除 tick 外）。
    """
    sim_rows = ensure_tick_column(list(sim_rows), tick_key=tick_key)
    truth_rows = ensure_tick_column(list(truth_rows), tick_key=tick_key)
    if not sim_rows or not truth_rows:
        return {"error": "empty sim or truth rows"}

    def by_tick(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for row in rows:
            tv = _float_row(row, tick_key)
            if tv is None:
                continue
            out[int(tv)] = row
        return out

    sim_bt = by_tick(sim_rows)
    tr_bt = by_tick(truth_rows)
    common_ticks = sorted(set(sim_bt) & set(tr_bt))
    if not common_ticks:
        return {"error": "no overlapping tick values"}

    if keys is None:
        sample = {**sim_bt[common_ticks[0]], **tr_bt[common_ticks[0]]}
        keys = [k for k in sample if k != tick_key and _float_row(sample, k) is not None]

    per_key: dict[str, Any] = {}
    for k in keys:
        abs_errs: list[float] = []
        pairs: list[tuple[float, float]] = []
        for t in common_ticks:
            a = _float_row(sim_bt[t], k)
            b = _float_row(tr_bt[t], k)
            if a is None or b is None:
                continue
            abs_errs.append(abs(a - b))
            pairs.append((a, b))
        if not abs_errs:
            continue
        mae = sum(abs_errs) / len(abs_errs)
        per_key[k] = {
            "n": len(abs_errs),
            "mae": round(mae, 6),
            "max_abs": round(max(abs_errs), 6),
        }
    return {"ticks_compared": len(common_ticks), "tick_span": [common_ticks[0], common_ticks[-1]], "per_metric": per_key}


def average_mae(compare_report: dict[str, Any]) -> float | None:
    """把 validate 输出压成单个分数，供自动调参用；无法计算时返回 None。"""
    if compare_report.get("error"):
        return None
    per = compare_report.get("per_metric") or {}
    if not per:
        return None
    maes = [float(v["mae"]) for v in per.values() if isinstance(v, dict) and "mae" in v]
    if not maes:
        return None
    return sum(maes) / len(maes)


def load_metrics_history_from_json(path: str | Path) -> list[dict[str, Any]]:
    """支持 { \"metrics_history\": [...] } 或纯数组。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("metrics_history"), list):
        return [dict(x) for x in raw["metrics_history"]]
    if isinstance(raw, list):
        return [dict(x) for x in raw]
    raise ValueError("JSON 需为数组或含 metrics_history 数组的对象")
