#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_CSV = "src/etf_momentum/web/data/calendar_timing_param_search_results.csv"
DEFAULT_OUTPUT_DIR = "src/etf_momentum/web/data"
DEFAULT_OUTPUT_PREFIX = "calendar_timing_robust"

MAXIMIZE_METRICS = {
    "cumulative_return",
    "annualized_return",
    "sharpe_ratio",
    "ulcer_performance_index",
    "win_rate",
    "payoff_ratio",
    "kelly_fraction",
}
MINIMIZE_METRICS = {
    "annualized_volatility",
    "max_drawdown_recovery_days",
    "ulcer_index",
}

REQUIRED_COLUMNS = {
    "symbol",
    "decision_day",
    "hold_days",
    "sharpe_ratio",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "max_drawdown_recovery_days",
    "ulcer_index",
    "ulcer_performance_index",
    "win_rate",
    "payoff_ratio",
    "kelly_fraction",
}


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _infer_objective_mode(metric: str, objective_mode: str) -> str:
    mode = str(objective_mode).strip().lower()
    if mode in {"max", "min"}:
        return mode
    if metric in MAXIMIZE_METRICS:
        return "max"
    if metric in MINIMIZE_METRICS:
        return "min"
    return "max"


def _percentile_score(series: pd.Series, *, mode: str) -> pd.Series:
    if str(mode).lower() == "max":
        return series.rank(method="average", pct=True, ascending=True)
    return series.rank(method="average", pct=True, ascending=False)


def _local_robust_stats_for_symbol(
    sdf: pd.DataFrame,
    *,
    decision_radius: int,
    hold_radius: int,
) -> pd.DataFrame:
    out = sdf.copy()
    local_top_ratio: list[float] = []
    local_pct_mean: list[float] = []
    local_pct_std: list[float] = []
    local_stability: list[float] = []
    local_neighbor_count: list[int] = []

    for _, row in out.iterrows():
        d0 = int(row["decision_day"])
        h0 = int(row["hold_days"])
        nb = out[
            (out["decision_day"].sub(d0).abs() <= int(decision_radius))
            & (out["hold_days"].sub(h0).abs() <= int(hold_radius))
        ]
        if nb.empty:
            local_neighbor_count.append(0)
            local_top_ratio.append(float("nan"))
            local_pct_mean.append(float("nan"))
            local_pct_std.append(float("nan"))
            local_stability.append(float("nan"))
            continue
        p = nb["objective_percentile"].astype(float)
        top_ratio = float(nb["is_top_pct"].mean())
        p_mean = float(p.mean())
        p_std = float(p.std(ddof=0)) if len(p) > 1 else 0.0
        stability = float(max(0.0, min(1.0, 1.0 - p_std)))
        local_neighbor_count.append(int(len(nb)))
        local_top_ratio.append(top_ratio)
        local_pct_mean.append(p_mean)
        local_pct_std.append(p_std)
        local_stability.append(stability)

    out["local_neighbor_count"] = local_neighbor_count
    out["local_top_ratio"] = local_top_ratio
    out["local_pct_mean"] = local_pct_mean
    out["local_pct_std"] = local_pct_std
    out["local_stability"] = local_stability
    out["asset_robust_score"] = (
        out["objective_percentile"] * 0.50
        + out["local_top_ratio"] * 0.30
        + out["local_stability"] * 0.20
    )
    return out


def _build_symbol_pattern_summary(scored: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sym, sdf in scored.groupby("symbol", sort=True):
        top = sdf[sdf["is_top_pct"]].copy()
        if top.empty:
            out.append(
                {
                    "symbol": str(sym),
                    "top_count": 0,
                    "decision_day_min": None,
                    "decision_day_max": None,
                    "decision_day_median": None,
                    "hold_days_min": None,
                    "hold_days_max": None,
                    "hold_days_median": None,
                    "decision_day_mode": None,
                    "hold_days_mode": None,
                }
            )
            continue
        dec_mode = (
            int(top["decision_day"].mode().iloc[0])
            if not top["decision_day"].mode().empty
            else None
        )
        hold_mode = (
            int(top["hold_days"].mode().iloc[0])
            if not top["hold_days"].mode().empty
            else None
        )
        out.append(
            {
                "symbol": str(sym),
                "top_count": int(len(top)),
                "decision_day_min": int(top["decision_day"].min()),
                "decision_day_max": int(top["decision_day"].max()),
                "decision_day_median": float(top["decision_day"].median()),
                "hold_days_min": int(top["hold_days"].min()),
                "hold_days_max": int(top["hold_days"].max()),
                "hold_days_median": float(top["hold_days"].median()),
                "decision_day_mode": dec_mode,
                "hold_days_mode": hold_mode,
            }
        )
    return out


def _aggregate_combo_scores(scored: pd.DataFrame) -> pd.DataFrame:
    gb = scored.groupby(["decision_day", "hold_days"], sort=True)
    combo = gb.agg(
        symbol_count=("symbol", "count"),
        coverage_top_count=("is_top_pct", "sum"),
        mean_objective_percentile=("objective_percentile", "mean"),
        min_objective_percentile=("objective_percentile", "min"),
        max_objective_percentile=("objective_percentile", "max"),
        std_objective_percentile=("objective_percentile", "std"),
        mean_asset_robust_score=("asset_robust_score", "mean"),
        min_asset_robust_score=("asset_robust_score", "min"),
        mean_local_top_ratio=("local_top_ratio", "mean"),
        min_local_top_ratio=("local_top_ratio", "min"),
        mean_local_stability=("local_stability", "mean"),
        min_local_stability=("local_stability", "min"),
        mean_sharpe=("sharpe_ratio", "mean"),
        min_sharpe=("sharpe_ratio", "min"),
        mean_cum_return=("cumulative_return", "mean"),
        min_cum_return=("cumulative_return", "min"),
        mean_max_drawdown=("max_drawdown", "mean"),
        worst_max_drawdown=("max_drawdown", "min"),
    ).reset_index()
    combo["coverage_top_count"] = combo["coverage_top_count"].astype(int)
    combo["std_objective_percentile"] = combo["std_objective_percentile"].fillna(0.0)
    combo["universal_robust_score"] = (
        combo["min_objective_percentile"] * 0.40
        + combo["mean_objective_percentile"] * 0.25
        + combo["min_local_top_ratio"] * 0.20
        + combo["mean_local_stability"] * 0.10
        + combo["mean_asset_robust_score"] * 0.15
        - combo["std_objective_percentile"] * 0.10
    )
    combo = combo.sort_values(
        by=[
            "universal_robust_score",
            "min_objective_percentile",
            "mean_objective_percentile",
            "mean_sharpe",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return combo


def _pick_recommended_combo(
    combo: pd.DataFrame,
    *,
    symbol_count: int,
    min_top_coverage: int,
    allow_coverage_relax: bool,
) -> tuple[pd.Series, int, bool]:
    require = int(min_top_coverage)
    if require < 1:
        require = int(symbol_count)
    require = min(require, int(symbol_count))

    candidate = combo[combo["coverage_top_count"] >= require]
    relaxed = False
    if candidate.empty and allow_coverage_relax:
        for k in range(require - 1, 0, -1):
            candidate = combo[combo["coverage_top_count"] >= k]
            if not candidate.empty:
                require = int(k)
                relaxed = True
                break
    if candidate.empty:
        candidate = combo
        relaxed = True
        require = int(combo["coverage_top_count"].max()) if not combo.empty else 0

    best = candidate.iloc[0]
    return best, require, relaxed


def _build_report_markdown(
    *,
    objective: str,
    objective_mode: str,
    top_pct: float,
    symbol_count: int,
    used_min_top_coverage: int,
    relaxed_coverage: bool,
    best: pd.Series,
    symbol_pattern_summary: list[dict[str, Any]],
    top_candidates: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Calendar Timing Robust Parameter Research")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- objective: `{objective}` ({objective_mode})")
    lines.append(f"- top percentile: `{top_pct:.1%}`")
    lines.append(f"- symbol count: `{symbol_count}`")
    lines.append(
        f"- min top-coverage used: `{used_min_top_coverage}`"
        + (" (relaxed)" if relaxed_coverage else "")
    )
    lines.append("")
    lines.append("## Recommended Universal Parameters")
    lines.append("")
    lines.append(f"- decision_day: `{int(best['decision_day'])}`")
    lines.append(f"- hold_days: `{int(best['hold_days'])}`")
    lines.append(f"- universal_robust_score: `{float(best['universal_robust_score']):.6f}`")
    lines.append(f"- coverage_top_count: `{int(best['coverage_top_count'])}`")
    lines.append(
        "- percentile stats: "
        f"mean `{float(best['mean_objective_percentile']):.4f}`, "
        f"min `{float(best['min_objective_percentile']):.4f}`, "
        f"std `{float(best['std_objective_percentile']):.4f}`"
    )
    lines.append(
        "- local robustness: "
        f"mean local_top_ratio `{float(best['mean_local_top_ratio']):.4f}`, "
        f"min local_top_ratio `{float(best['min_local_top_ratio']):.4f}`, "
        f"mean local_stability `{float(best['mean_local_stability']):.4f}`"
    )
    lines.append("")
    lines.append("## Top10% Pattern Summary by Symbol")
    lines.append("")
    lines.append("| symbol | top_count | decision_day range | decision_day median | hold_days range | hold_days median |")
    lines.append("|---|---:|---|---:|---|---:|")
    for row in symbol_pattern_summary:
        if int(row["top_count"]) <= 0:
            lines.append(f"| {row['symbol']} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {row['symbol']} | {int(row['top_count'])} | "
            f"{int(row['decision_day_min'])}..{int(row['decision_day_max'])} | "
            f"{float(row['decision_day_median']):.2f} | "
            f"{int(row['hold_days_min'])}..{int(row['hold_days_max'])} | "
            f"{float(row['hold_days_median']):.2f} |"
        )
    lines.append("")
    lines.append("## Top Universal Candidates")
    lines.append("")
    lines.append("| rank | decision_day | hold_days | score | coverage_top | min_pct | mean_pct |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for i, (_, row) in enumerate(top_candidates.iterrows(), start=1):
        lines.append(
            f"| {i} | {int(row['decision_day'])} | {int(row['hold_days'])} | "
            f"{float(row['universal_robust_score']):.6f} | "
            f"{int(row['coverage_top_count'])} | "
            f"{float(row['min_objective_percentile']):.4f} | "
            f"{float(row['mean_objective_percentile']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    input_csv = Path(str(args.input_csv))
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError(f"missing required columns in csv: {missing_cols}")

    objective = str(args.objective).strip()
    if objective not in set(df.columns):
        raise ValueError(f"objective not found in csv columns: {objective}")
    objective_mode = _infer_objective_mode(objective, args.objective_mode)

    top_pct = float(args.top_pct)
    if not (0.0 < top_pct < 1.0):
        raise ValueError("top_pct must be in (0, 1)")

    decision_radius = int(args.decision_radius)
    hold_radius = int(args.hold_radius)
    if decision_radius < 0 or hold_radius < 0:
        raise ValueError("decision_radius and hold_radius must be >= 0")

    # Clean and normalize key fields.
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["decision_day"] = pd.to_numeric(df["decision_day"], errors="coerce").astype("Int64")
    df["hold_days"] = pd.to_numeric(df["hold_days"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["symbol", "decision_day", "hold_days"]).copy()
    df["decision_day"] = df["decision_day"].astype(int)
    df["hold_days"] = df["hold_days"].astype(int)

    numeric_cols = [
        objective,
        "sharpe_ratio",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "max_drawdown_recovery_days",
        "ulcer_index",
        "ulcer_performance_index",
        "win_rate",
        "payoff_ratio",
        "kelly_fraction",
    ]
    for c in numeric_cols:
        df[c] = df[c].map(_as_float)

    scored_parts: list[pd.DataFrame] = []
    quantile_thresholds: dict[str, float] = {}
    for sym, sdf in df.groupby("symbol", sort=True):
        x = sdf.copy()
        x["objective_raw"] = x[objective].astype(float)
        x["objective_percentile"] = _percentile_score(
            x["objective_raw"], mode=objective_mode
        )
        thr = float(x["objective_percentile"].quantile(1.0 - top_pct))
        quantile_thresholds[str(sym)] = thr
        x["is_top_pct"] = x["objective_percentile"] >= thr
        x = _local_robust_stats_for_symbol(
            x,
            decision_radius=decision_radius,
            hold_radius=hold_radius,
        )
        scored_parts.append(x)
    scored = pd.concat(scored_parts, ignore_index=True)

    symbol_pattern_summary = _build_symbol_pattern_summary(scored)
    combo = _aggregate_combo_scores(scored)
    symbols = sorted(scored["symbol"].unique().tolist())
    symbol_count = len(symbols)

    best, used_min_top_coverage, relaxed_coverage = _pick_recommended_combo(
        combo,
        symbol_count=symbol_count,
        min_top_coverage=int(args.min_top_coverage),
        allow_coverage_relax=bool(args.allow_coverage_relax),
    )

    # Add per-symbol diagnostics for best combo.
    best_diagnostics = scored[
        (scored["decision_day"] == int(best["decision_day"]))
        & (scored["hold_days"] == int(best["hold_days"]))
    ].copy()
    best_symbol_rows: list[dict[str, Any]] = []
    for _, row in best_diagnostics.sort_values("symbol").iterrows():
        best_symbol_rows.append(
            {
                "symbol": str(row["symbol"]),
                "objective_raw": _as_float(row["objective_raw"]),
                "objective_percentile": _as_float(row["objective_percentile"]),
                "is_top_pct": bool(row["is_top_pct"]),
                "asset_robust_score": _as_float(row["asset_robust_score"]),
                "local_top_ratio": _as_float(row["local_top_ratio"]),
                "local_stability": _as_float(row["local_stability"]),
                "sharpe_ratio": _as_float(row["sharpe_ratio"]),
                "annualized_return": _as_float(row["annualized_return"]),
                "max_drawdown": _as_float(row["max_drawdown"]),
            }
        )

    top_n = max(1, int(args.top_n))
    top_candidates = combo.head(top_n).copy()

    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip() or DEFAULT_OUTPUT_PREFIX

    scored_csv = output_dir / f"{prefix}_symbol_scored.csv"
    combo_csv = output_dir / f"{prefix}_combo_scored.csv"
    top_csv = output_dir / f"{prefix}_top_candidates.csv"
    rec_json = output_dir / f"{prefix}_recommendation.json"
    report_md = output_dir / f"{prefix}_report.md"
    pattern_csv = output_dir / f"{prefix}_top10_pattern_by_symbol.csv"

    scored.to_csv(scored_csv, index=False, encoding="utf-8")
    combo.to_csv(combo_csv, index=False, encoding="utf-8")
    top_candidates.to_csv(top_csv, index=False, encoding="utf-8")
    pd.DataFrame(symbol_pattern_summary).to_csv(pattern_csv, index=False, encoding="utf-8")

    recommendation = {
        "meta": {
            "input_csv": str(input_csv),
            "objective": objective,
            "objective_mode": objective_mode,
            "top_pct": float(top_pct),
            "decision_radius": int(decision_radius),
            "hold_radius": int(hold_radius),
            "symbol_count": int(symbol_count),
            "symbols": symbols,
            "min_top_coverage_requested": int(args.min_top_coverage),
            "min_top_coverage_used": int(used_min_top_coverage),
            "relaxed_coverage": bool(relaxed_coverage),
            "quantile_thresholds": quantile_thresholds,
        },
        "recommended": {
            "decision_day": int(best["decision_day"]),
            "hold_days": int(best["hold_days"]),
            "universal_robust_score": _as_float(best["universal_robust_score"]),
            "coverage_top_count": int(best["coverage_top_count"]),
            "mean_objective_percentile": _as_float(best["mean_objective_percentile"]),
            "min_objective_percentile": _as_float(best["min_objective_percentile"]),
            "std_objective_percentile": _as_float(best["std_objective_percentile"]),
            "mean_local_top_ratio": _as_float(best["mean_local_top_ratio"]),
            "min_local_top_ratio": _as_float(best["min_local_top_ratio"]),
            "mean_local_stability": _as_float(best["mean_local_stability"]),
            "mean_asset_robust_score": _as_float(best["mean_asset_robust_score"]),
            "mean_sharpe": _as_float(best["mean_sharpe"]),
            "min_sharpe": _as_float(best["min_sharpe"]),
            "mean_cum_return": _as_float(best["mean_cum_return"]),
            "worst_max_drawdown": _as_float(best["worst_max_drawdown"]),
        },
        "recommended_by_symbol": best_symbol_rows,
        "top_candidates": top_candidates.to_dict(orient="records"),
        "top10_pattern_by_symbol": symbol_pattern_summary,
        "artifacts": {
            "symbol_scored_csv": str(scored_csv),
            "combo_scored_csv": str(combo_csv),
            "top_candidates_csv": str(top_csv),
            "top10_pattern_csv": str(pattern_csv),
            "report_md": str(report_md),
        },
    }
    rec_json.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = _build_report_markdown(
        objective=objective,
        objective_mode=objective_mode,
        top_pct=top_pct,
        symbol_count=symbol_count,
        used_min_top_coverage=used_min_top_coverage,
        relaxed_coverage=relaxed_coverage,
        best=best,
        symbol_pattern_summary=symbol_pattern_summary,
        top_candidates=top_candidates,
    )
    report_md.write_text(report, encoding="utf-8")

    print(f"[INFO] objective={objective} ({objective_mode}), top_pct={top_pct:.1%}")
    print(
        "[INFO] recommended "
        f"decision_day={int(best['decision_day'])}, hold_days={int(best['hold_days'])}, "
        f"score={float(best['universal_robust_score']):.6f}, "
        f"coverage_top={int(best['coverage_top_count'])}/{symbol_count}"
    )
    print(f"[INFO] symbol scored: {scored_csv}")
    print(f"[INFO] combo scored: {combo_csv}")
    print(f"[INFO] top candidates: {top_csv}")
    print(f"[INFO] recommendation: {rec_json}")
    print(f"[INFO] report: {report_md}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Research robust calendar-timing parameters from grid-search results "
            "using top-percentile and local-neighborhood robustness analysis."
        )
    )
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--objective", default="sharpe_ratio")
    p.add_argument(
        "--objective-mode",
        default="auto",
        choices=["auto", "max", "min"],
        help="auto infers mode by metric type.",
    )
    p.add_argument(
        "--top-pct",
        type=float,
        default=0.10,
        help="Top percentile used for candidate zone, e.g. 0.10 means top 10%%.",
    )
    p.add_argument("--decision-radius", type=int, default=1)
    p.add_argument("--hold-radius", type=int, default=1)
    p.add_argument(
        "--min-top-coverage",
        type=int,
        default=-1,
        help=(
            "Minimum number of symbols where combo must be in top_pct. "
            "Use -1 to require all symbols."
        ),
    )
    p.add_argument(
        "--allow-coverage-relax",
        action="store_true",
        help="Allow fallback to lower coverage when no combo meets min-top-coverage.",
    )
    p.add_argument("--top-n", type=int, default=20)
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
