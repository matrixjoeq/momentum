from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import etf_momentum.analysis.off_fund_replication as replication_module
from etf_momentum.analysis.off_fund_replication import (
    ReplicationConfigError,
    ReplicationFactorMeta,
    nav_to_returns,
    replicate_fund_by_constrained_weights,
    select_replication_factor_series,
)


def _nav(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.cumprod(1.0 + returns), index=dates, dtype=float)


def _meta(
    key: str,
    reporting_group: str,
    substitution_group: str | None = None,
) -> ReplicationFactorMeta:
    return ReplicationFactorMeta(
        key=key,
        label=key,
        reporting_group=reporting_group,
        substitution_group=substitution_group or key,
        asset_class="equity",
    )


def _fixture() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(20260903)
    dates = pd.bdate_range("2022-01-03", periods=420)
    f1 = rng.normal(0.00025, 0.011, len(dates))
    f2 = rng.normal(0.00015, 0.009, len(dates))
    noise = rng.normal(0.0, 0.0003, len(dates))
    target = 0.60 * f1 + 0.25 * f2 + 0.00008 + noise
    fund_nav = _nav(target, dates)
    factor_close = pd.DataFrame(
        {"F1": _nav(f1, dates), "F2": _nav(f2, dates)},
        index=dates,
    )
    return fund_nav, factor_close


def test_replication_recovers_long_only_weights_and_cash() -> None:
    fund_nav, factor_close = _fixture()
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=True,
        max_series_points=1000,
        lambda_temporal=0.0,
    )
    assert out["status"] == "ok"
    assert out["asset_weights"]["F1"] == pytest.approx(0.60, abs=0.03)
    assert out["asset_weights"]["F2"] == pytest.approx(0.25, abs=0.03)
    assert out["cash_weight"] == pytest.approx(0.15, abs=0.04)
    assert sum(out["asset_weights"].values()) + out["cash_weight"] == pytest.approx(
        1.0, abs=1e-8
    )
    assert out["oos_metrics"]["sample_days"] > 20
    assert out["oos_metrics"]["sample"] == "oos_lag1"
    assert out["oos_metrics"]["tracking_error_annualized"] < 0.02
    assert out["stability"]["sample"] == "rolling_group_weights"

    previous_nav = 1.0
    for row in out["series"]:
        if row["effective_date"] is None:
            continue
        expected = previous_nav * (1.0 + row["replication_return"])
        assert row["replication_nav"] == pytest.approx(expected, abs=1e-12)
        assert row["residual"] == pytest.approx(
            row["target_return"] - row["replication_return"], abs=1e-12
        )
        previous_nav = row["replication_nav"]


def test_factor_order_does_not_change_latest_result() -> None:
    fund_nav, factor_close = _fixture()
    meta = {"F1": _meta("F1", "large"), "F2": _meta("F2", "small")}
    kwargs = {
        "fund_nav": fund_nav,
        "factor_meta": meta,
        "rolling_window": 120,
        "min_samples": 80,
        "include_series": False,
        "max_series_points": 0,
    }
    left = replicate_fund_by_constrained_weights(
        factor_close_df=factor_close[["F1", "F2"]], **kwargs
    )
    right = replicate_fund_by_constrained_weights(
        factor_close_df=factor_close[["F2", "F1"]], **kwargs
    )
    assert left["asset_weights"] == pytest.approx(right["asset_weights"], abs=1e-8)
    assert left["cash_weight"] == pytest.approx(right["cash_weight"], abs=1e-8)
    assert left["oos_metrics"] == pytest.approx(right["oos_metrics"], abs=1e-10)


def test_near_zero_factor_is_dropped_instead_of_absorbing_cash() -> None:
    fund_nav, factor_close = _fixture()
    factor_close["ZERO"] = 1.0
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
            "ZERO": _meta("ZERO", "cash", "cash_like"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
    )
    assert out["status"] == "ok"
    assert out["asset_weights"]["ZERO"] == 0.0
    assert "ZERO" in out["dropped_near_zero_vol"]


def test_exact_duplicates_share_weight_inside_substitution_group() -> None:
    fund_nav, factor_close = _fixture()
    factor_close["F1_COPY"] = factor_close["F1"]
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large", "large_proxy"),
            "F1_COPY": _meta("F1_COPY", "large", "large_proxy"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
        lambda_temporal=0.0,
    )
    assert out["status"] == "ok"
    assert out["asset_weights"]["F1"] == pytest.approx(
        out["asset_weights"]["F1_COPY"], abs=1e-12
    )
    assert (
        out["asset_weights"]["F1"] + out["asset_weights"]["F1_COPY"]
    ) == pytest.approx(0.60, abs=0.03)


def test_exact_duplicates_in_different_groups_are_rejected() -> None:
    fund_nav, factor_close = _fixture()
    factor_close["F1_COPY"] = factor_close["F1"]
    with pytest.raises(ReplicationConfigError, match="substitution_group"):
        replicate_fund_by_constrained_weights(
            fund_nav=fund_nav,
            factor_close_df=factor_close,
            factor_meta={
                "F1": _meta("F1", "large", "proxy_a"),
                "F1_COPY": _meta("F1_COPY", "large", "proxy_b"),
                "F2": _meta("F2", "small"),
            },
            rolling_window=120,
            min_samples=80,
            include_series=False,
            max_series_points=0,
        )


def test_coverage_detects_mutually_masking_short_history_factors() -> None:
    fund_nav, factor_close = _fixture()
    short_start = factor_close.index[-100]
    factor_close["SHORT_A"] = factor_close["F1"].where(
        factor_close.index >= short_start
    )
    factor_close["SHORT_B"] = factor_close["F2"].where(
        factor_close.index >= short_start
    )
    meta = {
        "F1": _meta("F1", "large"),
        "F2": _meta("F2", "small"),
        "SHORT_A": _meta("SHORT_A", "short_a"),
        "SHORT_B": _meta("SHORT_B", "short_b"),
    }
    conflict = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta=meta,
        rolling_window=80,
        min_samples=60,
        include_series=False,
        max_series_points=0,
    )
    assert conflict["status"] == "coverage_conflict"
    assert set(conflict["coverage"]["culprit_factors"]) == {
        "SHORT_A",
        "SHORT_B",
    }

    dropped = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta=meta,
        rolling_window=80,
        min_samples=60,
        include_series=False,
        max_series_points=0,
        drop_short_history_factors=True,
    )
    assert dropped["status"] == "ok"
    assert set(dropped["dropped_short_history_factors"]) == {
        "SHORT_A",
        "SHORT_B",
    }


def test_requested_window_is_not_silently_shortened() -> None:
    fund_nav, factor_close = _fixture()
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav.iloc[:100],
        factor_close_df=factor_close.iloc[:100],
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=252,
        min_samples=40,
        include_series=False,
        max_series_points=0,
    )
    assert out["status"] == "insufficient_samples"
    assert "required days=252" in out["warnings"][0]


def test_include_series_does_not_change_latest_weights_or_metrics() -> None:
    fund_nav, factor_close = _fixture()
    kwargs = {
        "fund_nav": fund_nav,
        "factor_close_df": factor_close,
        "factor_meta": {
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        "rolling_window": 120,
        "min_samples": 80,
        "max_series_points": 500,
    }
    with_series = replicate_fund_by_constrained_weights(include_series=True, **kwargs)
    without_series = replicate_fund_by_constrained_weights(
        include_series=False, **kwargs
    )
    assert with_series["asset_weights"] == pytest.approx(
        without_series["asset_weights"], abs=1e-10
    )
    assert with_series["oos_metrics"] == pytest.approx(
        without_series["oos_metrics"], abs=1e-12
    )


def test_walk_forward_does_not_use_future_target_returns() -> None:
    fund_nav, factor_close = _fixture()
    changed = fund_nav.copy()
    cutoff = fund_nav.index[300]
    future_returns = changed.pct_change(fill_method=None)
    future_returns.loc[future_returns.index > cutoff] += 0.03
    changed = _nav(
        future_returns.fillna(0.0).to_numpy(dtype=float),
        fund_nav.index,
    )
    kwargs = {
        "factor_close_df": factor_close,
        "factor_meta": {
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        "rolling_window": 120,
        "min_samples": 80,
        "include_series": True,
        "max_series_points": 1000,
    }
    original = replicate_fund_by_constrained_weights(fund_nav=fund_nav, **kwargs)
    modified = replicate_fund_by_constrained_weights(fund_nav=changed, **kwargs)
    original_rows = {
        row["effective_date"]: row
        for row in original["series"]
        if row["effective_date"] and pd.Timestamp(row["effective_date"]) <= cutoff
    }
    modified_rows = {
        row["effective_date"]: row
        for row in modified["series"]
        if row["effective_date"] and pd.Timestamp(row["effective_date"]) <= cutoff
    }
    assert original_rows.keys() == modified_rows.keys()
    for date in original_rows:
        assert original_rows[date]["asset_weights"] == pytest.approx(
            modified_rows[date]["asset_weights"], abs=1e-10
        )


def test_latest_identifiability_intervals_contain_published_weights() -> None:
    fund_nav, factor_close = _fixture()
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
        compute_latest_diagnostics=True,
        extra_solve_budget=30,
    )
    intervals = out["identifiability"]["group_weight_intervals"]
    assert intervals
    assert out["extra_solves_used"] <= 30
    for bounds in intervals.values():
        assert bounds["lower"] <= bounds["published"] <= bounds["upper"]


def test_alias_selection_optimizes_common_coverage() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    base = pd.Series(np.linspace(1.0, 1.2, 100), index=dates)
    close = pd.DataFrame(
        {
            "A_LONG": base.where(base.index < dates[70]),
            "A_COMMON": base.where(
                (base.index >= dates[20]) & (base.index < dates[80])
            ),
            "B_LONG": base.where(base.index >= dates[30]),
            "B_COMMON": base.where(
                (base.index >= dates[20]) & (base.index < dates[80])
            ),
        }
    )
    selected, meta, _ = select_replication_factor_series(
        close_df=close,
        factor_rows=[
            {
                "key": "A",
                "label": "A",
                "aliases": ["A_LONG", "A_COMMON"],
            },
            {
                "key": "B",
                "label": "B",
                "aliases": ["B_LONG", "B_COMMON"],
            },
        ],
        target_nav=base,
    )
    selected_codes = {row["key"]: row["selected_code"] for row in meta}
    assert list(selected.columns) == ["A", "B"]
    assert selected_codes == {"A": "A_COMMON", "B": "B_COMMON"}


def test_adding_exact_duplicate_preserves_group_and_oos_returns() -> None:
    fund_nav, factor_close = _fixture()
    base = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large", "large_proxy"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=True,
        max_series_points=1000,
    )
    duplicated_close = factor_close.assign(F1_COPY=factor_close["F1"])
    duplicated = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=duplicated_close,
        factor_meta={
            "F1": _meta("F1", "large", "large_proxy"),
            "F1_COPY": _meta("F1_COPY", "large", "large_proxy"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=True,
        max_series_points=1000,
    )
    assert base["group_weights"] == pytest.approx(duplicated["group_weights"], abs=1e-8)
    base_returns = [
        row["replication_return"]
        for row in base["series"]
        if row["replication_return"] is not None
    ]
    duplicate_returns = [
        row["replication_return"]
        for row in duplicated["series"]
        if row["replication_return"] is not None
    ]
    assert base_returns == pytest.approx(duplicate_returns, abs=1e-10)


def test_reporting_group_does_not_imply_substitution_penalty() -> None:
    fund_nav, factor_close = _fixture()
    out = replicate_fund_by_constrained_weights(
        fund_nav=_nav(
            0.85
            * factor_close["F1"].pct_change(fill_method=None).fillna(0.0).to_numpy(),
            factor_close.index,
        ),
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "equity", "proxy_one"),
            "F2": _meta("F2", "equity", "proxy_two"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
        lambda_substitution=10.0,
        lambda_temporal=0.0,
    )
    assert out["asset_weights"]["F1"] > 0.75
    assert out["asset_weights"]["F2"] < 0.05


def test_failed_window_breaks_temporal_smoothing(monkeypatch) -> None:
    fund_nav, factor_close = _fixture()
    original_fit = replication_module._fit_window
    call_count = 0

    def fail_once(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            return None
        return original_fit(**kwargs)

    monkeypatch.setattr(replication_module, "_fit_window", fail_once)
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=True,
        max_series_points=1000,
    )
    assert "solver_failed_windows=1" in out["warnings"]
    assert any(row["smooth_break"] for row in out["series"])
    assert out["latest_solver"]["gradient_check_error"] < 2e-5


def test_solver_rejects_non_finite_window() -> None:
    x = np.ones((50, 2), dtype=float)
    x[10, 0] = np.nan
    result = replication_module._fit_window(
        x=x,
        y=np.ones(50, dtype=float),
        keys=["A", "B"],
        factor_meta={
            "A": _meta("A", "a"),
            "B": _meta("B", "b"),
        },
        previous_asset_weights=None,
        previous_group_weights=None,
        lambda_substitution=0.0,
        lambda_temporal=0.0,
    )
    assert result is None


def test_cash_only_target_publishes_one_hundred_percent_cash() -> None:
    _, factor_close = _fixture()
    flat_nav = pd.Series(1.0, index=factor_close.index)
    out = replicate_fund_by_constrained_weights(
        fund_nav=flat_nav,
        factor_close_df=factor_close,
        factor_meta={
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
    )
    assert out["status"] == "ok"
    assert out["cash_weight"] == pytest.approx(1.0, abs=1e-10)
    assert sum(out["asset_weights"].values()) == pytest.approx(0.0, abs=1e-10)


def test_default_regularization_limits_te_cost_and_reduces_group_churn() -> None:
    fund_nav, factor_close = _fixture()
    kwargs = {
        "fund_nav": fund_nav,
        "factor_close_df": factor_close,
        "factor_meta": {
            "F1": _meta("F1", "large"),
            "F2": _meta("F2", "small"),
        },
        "rolling_window": 120,
        "min_samples": 80,
        "include_series": False,
        "max_series_points": 0,
    }
    calibrated = replicate_fund_by_constrained_weights(**kwargs)
    unregularized = replicate_fund_by_constrained_weights(
        **kwargs,
        lambda_substitution=0.0,
        lambda_temporal=0.0,
    )
    assert (
        calibrated["oos_metrics"]["tracking_error_annualized"]
        <= 1.01 * unregularized["oos_metrics"]["tracking_error_annualized"]
    )
    assert (
        calibrated["stability"]["mean_group_weight_distance"]
        <= unregularized["stability"]["mean_group_weight_distance"]
    )
    assert (
        calibrated["model_parameters"]["calibration_status"]
        == "independent_walk_forward_v1"
    )


def test_model_defaults_match_versioned_calibration_artifact() -> None:
    artifact_path = (
        Path(__file__).parents[1]
        / "src/etf_momentum/data/off_fund_replication_calibration_v2.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    selected = artifact["selected"]
    assert replication_module.DEFAULT_LAMBDA_SUBSTITUTION == pytest.approx(
        selected["lambda_substitution"]
    )
    assert replication_module.DEFAULT_LAMBDA_TEMPORAL == pytest.approx(
        selected["lambda_temporal"]
    )
    assert replication_module.STABILITY_GROUP_DISTANCE_THRESHOLD == pytest.approx(
        selected["stability_group_distance_threshold"]
    )
    assert replication_module.STABILITY_TE_IMPROVEMENT_THRESHOLD == pytest.approx(
        selected["stability_te_improvement_threshold"]
    )
    assert replication_module.MODEL_CALIBRATION_DATASET == artifact["calibration_id"]


def test_variance_explained_is_not_clamped_at_zero() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-02", periods=180)
    factor_a = rng.normal(0.0, 0.01, len(dates))
    factor_b = rng.normal(0.0, 0.01, len(dates))
    target = rng.normal(0.0, 0.01, len(dates))
    out = replicate_fund_by_constrained_weights(
        fund_nav=_nav(target, dates),
        factor_close_df=pd.DataFrame(
            {
                "A": _nav(factor_a, dates),
                "B": _nav(factor_b, dates),
            }
        ),
        factor_meta={
            "A": _meta("A", "a"),
            "B": _meta("B", "b"),
        },
        rolling_window=40,
        min_samples=40,
        include_series=False,
        max_series_points=0,
    )
    assert out["oos_metrics"]["variance_explained"] < 0.0


def test_temporal_smoothing_resumes_after_group_set_change() -> None:
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2023-01-02", periods=320)
    factor_a = rng.normal(0.0, 0.01, len(dates))
    factor_b = np.zeros(len(dates))
    factor_b[180:] = rng.normal(0.0, 0.01, len(dates) - 180)
    target = 0.7 * factor_a + rng.normal(0.0, 0.0003, len(dates))
    out = replicate_fund_by_constrained_weights(
        fund_nav=_nav(target, dates),
        factor_close_df=pd.DataFrame(
            {
                "A": _nav(factor_a, dates),
                "B": _nav(factor_b, dates),
            }
        ),
        factor_meta={
            "A": _meta("A", "a"),
            "B": _meta("B", "b"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=True,
        max_series_points=1000,
    )
    break_indexes = [
        idx for idx, row in enumerate(out["series"]) if row["smooth_break"]
    ]
    assert break_indexes
    first_break = break_indexes[0]
    assert first_break + 1 < len(out["series"])
    assert out["series"][first_break + 1]["smooth_break"] is False


def test_missing_nav_day_does_not_create_bridged_daily_return() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    nav = pd.Series([1.0, np.nan, 1.2, 1.32], index=dates)
    returns = nav_to_returns(nav)
    assert dates[2] not in returns.index
    assert returns.index.tolist() == [dates[3]]
    assert returns.iloc[0] == pytest.approx(0.1)


def test_group_set_break_restarts_from_versioned_zero_initialization() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(0.0, 0.01, (120, 2))
    y = 0.5 * x[:, 0] + 0.2 * x[:, 1]
    kwargs = {
        "x": x,
        "y": y,
        "keys": ["A", "B"],
        "factor_meta": {
            "A": _meta("A", "a"),
            "B": _meta("B", "b"),
        },
        "lambda_substitution": 0.0,
        "lambda_temporal": 1.0,
    }
    first = replication_module._fit_window(
        **kwargs,
        previous_asset_weights=None,
        previous_group_weights=None,
    )
    broken = replication_module._fit_window(
        **kwargs,
        previous_asset_weights={"A": 0.0, "B": 1.0},
        previous_group_weights={"different_group": 1.0},
    )
    assert first is not None and broken is not None
    assert broken.temporal_smoothing_used is False
    assert broken.weights == pytest.approx(first.weights, abs=1e-10)


def test_alias_joint_search_escapes_two_switch_local_optimum(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    index_by_name = {
        "TARGET": dates,
        "A1": dates[[3, 4, 5]],
        "A2": dates[[1, 2, 3, 4]],
        "B1": dates[[3, 4, 6]],
        "B2": dates[[1, 2, 3, 4]],
    }

    def fake_returns(series: pd.Series) -> pd.Series:
        index = index_by_name[str(series.name)]
        return pd.Series(0.0, index=index)

    monkeypatch.setattr(replication_module, "nav_to_returns", fake_returns)
    close = pd.DataFrame(
        {
            name: pd.Series(np.arange(8, dtype=float), index=dates)
            for name in ("A1", "A2", "B1", "B2")
        }
    )
    target = pd.Series(np.arange(8, dtype=float), index=dates, name="TARGET")
    _, meta, _ = select_replication_factor_series(
        close_df=close,
        factor_rows=[
            {"key": "A", "aliases": ["A1", "A2"]},
            {"key": "B", "aliases": ["B1", "B2"]},
        ],
        target_nav=target,
    )
    assert {item["key"]: item["selected_code"] for item in meta} == {
        "A": "A2",
        "B": "B2",
    }


def test_sensitivity_reports_all_applicable_scenario_types() -> None:
    fund_nav, base_close = _fixture()
    rng = np.random.default_rng(33)
    base_returns = base_close["F1"].pct_change(fill_method=None).fillna(0.0).to_numpy()
    near_returns = base_returns + rng.normal(0.0, 0.0002, len(base_returns))
    factor_close = pd.DataFrame(
        {
            "A": base_close["F1"],
            "A_COPY": base_close["F1"],
            "B": _nav(near_returns, base_close.index),
            "C": base_close["F2"],
        }
    )
    out = replicate_fund_by_constrained_weights(
        fund_nav=fund_nav,
        factor_close_df=factor_close,
        factor_meta={
            "A": _meta("A", "large", "large_proxy"),
            "A_COPY": _meta("A_COPY", "large", "large_proxy"),
            "B": _meta("B", "large", "large_proxy"),
            "C": _meta("C", "small", "small_proxy"),
        },
        rolling_window=120,
        min_samples=80,
        include_series=False,
        max_series_points=0,
        compute_latest_diagnostics=True,
        extra_solve_budget=30,
    )
    scenarios = {item["scenario"] for item in out["factor_sensitivity"]}
    assert scenarios == {
        "exact_duplicate",
        "within_group_replacement",
        "within_group_addition",
        "cross_group_addition",
    }
