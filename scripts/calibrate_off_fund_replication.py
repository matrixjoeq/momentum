#!/usr/bin/env python3
"""Calibrate replication regularization on an independent passive-index set."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etf_momentum.analysis.off_fund_replication import (
    ReplicationFactorMeta,
    replicate_fund_by_constrained_weights,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT = "src/etf_momentum/data/off_fund_replication_calibration_v2.json"
DEVELOPMENT_CODES = [
    "000368",
    "110020",
    "160119",
    "011860",
    "019870",
]
ACCEPTANCE_CODES_EXCLUDED = [
    "001015",
    "003016",
    "003957",
    "004194",
    "004513",
    "005313",
    "006593",
    "007994",
    "014344",
    "015867",
    "016936",
    "017644",
    "017846",
    "019918",
    "019923",
    "100032",
]


def _post_json(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/analysis/off-fund/replicate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _asset_churn(series: list[dict[str, Any]]) -> float:
    distances: list[float] = []
    for previous, current in zip(series, series[1:]):
        left = dict(previous.get("asset_weights") or {})
        right = dict(current.get("asset_weights") or {})
        keys = set(left) | set(right)
        risky = sum(
            abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
        )
        cash = abs(
            float(previous.get("cash_weight") or 0.0)
            - float(current.get("cash_weight") or 0.0)
        )
        distances.append(0.5 * (risky + cash))
    return _mean(distances)


def _evaluate(
    base_url: str,
    *,
    lambda_temporal: float,
    start: str,
    end: str,
) -> dict[str, Any]:
    output = _post_json(
        base_url,
        {
            "codes": DEVELOPMENT_CODES,
            "start": start,
            "end": end,
            "template_id": "cn_equity_size",
            "template_version": 2,
            "rolling_window": 120,
            "min_samples": 120,
            "include_weight_series": True,
            "max_series_points": 2000,
            "advanced_mode": True,
            "lambda_substitution": 0.0,
            "lambda_temporal": lambda_temporal,
        },
    )
    successful = [
        item for item in output.get("items", []) if item.get("status") == "ok"
    ]
    if not output.get("ok") or len(successful) != len(DEVELOPMENT_CODES):
        raise RuntimeError(
            f"calibration candidate failed: {output.get('error') or output}"
        )
    tracking_errors = [float(item["tracking_error"]) for item in successful]
    group_churn = [
        float(item["stability"]["mean_group_weight_distance"]) for item in successful
    ]
    asset_churn = [
        _asset_churn(list(item.get("weight_series") or [])) for item in successful
    ]
    return {
        "lambda_temporal": lambda_temporal,
        "mean_oos_tracking_error": _mean(tracking_errors),
        "max_oos_tracking_error": max(tracking_errors),
        "mean_group_weight_distance": _mean(group_churn),
        "mean_asset_weight_distance": _mean(asset_churn),
        "fund_count": len(successful),
    }


def _synthetic_substitution_candidate(
    *,
    lambda_substitution: float,
    lambda_temporal: float,
) -> dict[str, Any]:
    tracking_errors: list[float] = []
    asset_churn: list[float] = []
    sensitivity_distances: list[float] = []
    sensitivity_te_changes: list[float] = []
    for seed in (104, 207, 311, 419, 523):
        rng = np.random.default_rng(seed)
        size = 420
        dates = pd.bdate_range("2022-01-03", periods=size)
        base_large = rng.normal(0.0002, 0.011, size)
        base_small = rng.normal(0.0001, 0.014, size)
        factors = {
            "LARGE_A": base_large + rng.normal(0.0, 0.00035, size),
            "LARGE_B": base_large + rng.normal(0.0, 0.00035, size),
            "SMALL_A": base_small + rng.normal(0.0, 0.00040, size),
            "SMALL_B": base_small + rng.normal(0.0, 0.00040, size),
        }
        phase = np.arange(size, dtype=float)
        large_weight = 0.55 + 0.08 * np.sin(phase / 80.0)
        small_weight = 0.30 - 0.05 * np.sin(phase / 80.0)
        target = (
            large_weight * base_large
            + small_weight * base_small
            + rng.normal(0.0, 0.0012, size)
        )

        def nav(returns: np.ndarray) -> pd.Series:
            return pd.Series(np.cumprod(1.0 + returns), index=dates)

        factor_close = pd.DataFrame(
            {key: nav(returns) for key, returns in factors.items()}
        )
        metadata = {
            key: ReplicationFactorMeta(
                key=key,
                label=key,
                reporting_group=("large" if key.startswith("LARGE") else "small"),
                substitution_group=(
                    "large_proxy" if key.startswith("LARGE") else "small_proxy"
                ),
                asset_class="equity",
            )
            for key in factors
        }
        output = replicate_fund_by_constrained_weights(
            fund_nav=nav(target),
            factor_close_df=factor_close,
            factor_meta=metadata,
            rolling_window=120,
            min_samples=120,
            include_series=True,
            max_series_points=2000,
            lambda_substitution=lambda_substitution,
            lambda_temporal=lambda_temporal,
            compute_latest_diagnostics=True,
            extra_solve_budget=30,
        )
        if output.get("status") != "ok":
            raise RuntimeError(f"synthetic calibration failed: {output}")
        tracking_errors.append(
            float(output["oos_metrics"]["tracking_error_annualized"])
        )
        asset_churn.append(_asset_churn(output["series"]))
        for item in output["factor_sensitivity"]:
            if item.get("scenario") not in {
                "within_group_addition",
                "within_group_replacement",
            }:
                continue
            if item.get("status") not in {"ok", "unstable_redundancy"}:
                continue
            sensitivity_distances.append(float(item["group_weight_distance"]))
            sensitivity_te_changes.append(
                abs(float(item["fit_tracking_error_improvement"]))
            )
    return {
        "lambda_substitution": lambda_substitution,
        "mean_oos_tracking_error": _mean(tracking_errors),
        "mean_asset_weight_distance": _mean(asset_churn),
        "p95_sensitivity_group_distance": float(
            np.quantile(sensitivity_distances, 0.95)
        ),
        "p95_sensitivity_te_change": float(np.quantile(sensitivity_te_changes, 0.95)),
        "seed_count": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    temporal_candidates = [
        _evaluate(
            args.base_url,
            lambda_temporal=lambda_temporal,
            start=args.start,
            end=args.end,
        )
        for lambda_temporal in (0.0, 0.005, 0.02, 0.05, 0.2, 0.8)
    ]
    best_te = min(row["mean_oos_tracking_error"] for row in temporal_candidates)
    baseline_temporal = next(
        row for row in temporal_candidates if row["lambda_temporal"] == 0.0
    )
    eligible_temporal = [
        row
        for row in temporal_candidates
        if row["mean_oos_tracking_error"] <= best_te * 1.01
        and row["mean_group_weight_distance"]
        <= 0.60 * baseline_temporal["mean_group_weight_distance"]
    ]
    selected_temporal = min(
        eligible_temporal,
        key=lambda row: (
            row["lambda_temporal"],
            row["mean_oos_tracking_error"],
        ),
    )
    substitution_candidates = [
        _synthetic_substitution_candidate(
            lambda_substitution=value,
            lambda_temporal=float(selected_temporal["lambda_temporal"]),
        )
        for value in (0.0, 0.0005, 0.002, 0.005, 0.02, 0.08)
    ]
    best_substitution_te = min(
        row["mean_oos_tracking_error"] for row in substitution_candidates
    )
    baseline_substitution = next(
        row for row in substitution_candidates if row["lambda_substitution"] == 0.0
    )
    eligible_substitution = [
        row
        for row in substitution_candidates
        if row["mean_oos_tracking_error"] <= best_substitution_te * 1.01
        and row["mean_asset_weight_distance"]
        <= 0.60 * baseline_substitution["mean_asset_weight_distance"]
    ]
    selected_substitution = min(
        eligible_substitution,
        key=lambda row: (
            row["lambda_substitution"],
            row["mean_oos_tracking_error"],
        ),
    )
    selected = {
        "lambda_substitution": selected_substitution["lambda_substitution"],
        "lambda_temporal": selected_temporal["lambda_temporal"],
        "stability_group_distance_threshold": round(
            max(
                0.02,
                1.5 * selected_substitution["p95_sensitivity_group_distance"],
            ),
            4,
        ),
        "stability_te_improvement_threshold": round(
            max(
                0.0005,
                1.5 * selected_substitution["p95_sensitivity_te_change"],
            ),
            4,
        ),
    }
    artifact = {
        "calibration_id": "off_fund_replication_v2_passive_index_2024_2025",
        "model_version": "off_fund_replication_v2",
        "development_period": {"start": args.start, "end": args.end},
        "development_codes": DEVELOPMENT_CODES,
        "acceptance_codes_explicitly_excluded": ACCEPTANCE_CODES_EXCLUDED,
        "temporal_factor_design": "cn_equity_size_v2",
        "substitution_factor_design": (
            "five deterministic correlated-proxy walk-forward seeds"
        ),
        "rolling_window": 120,
        "selection_rule": {
            "temporal": (
                "Choose the smallest lambda within 1% of minimum passive-fund "
                "mean OOS TE that reduces reporting-group churn by at least 40%."
            ),
            "substitution": (
                "Choose the smallest lambda within 1% of minimum synthetic-"
                "proxy mean OOS TE that reduces asset churn by at least 40%."
            ),
            "thresholds": ("max(practical floor, 1.5 × selected-candidate p95)."),
        },
        "selected": selected,
        "temporal_candidates": temporal_candidates,
        "substitution_candidates": substitution_candidates,
        "limitations": [
            "Temporal calibration uses a local historical snapshot.",
            "Substitution calibration uses deterministic proxy perturbations.",
            "This does not calibrate statistical confidence intervals.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
