from __future__ import annotations
# pylint: disable=import-error

import math
import inspect

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, update

import etf_momentum.analysis.bt_trend as bt_trend_mod
from etf_momentum.analysis.bt_trend import (
    compute_trend_backtest_bt,
    compute_trend_portfolio_backtest_bt,
)
from etf_momentum.analysis.trend import (
    TrendInputs,
    TrendPortfolioInputs,
    compute_trend_backtest,
    compute_trend_portfolio_backtest,
)
from etf_momentum.db.session import make_session_factory
from etf_momentum.db.models import EtfPool, EtfPrice
from tests.helpers.rotation_case_data import seed_prices


def test_bt_trend_indicator_helpers_are_not_imported_from_legacy_trend_module():
    src = inspect.getsource(bt_trend_mod)
    banned = [
        "_ema as _ema_legacy",
        "_moving_average as _moving_average_legacy",
        "_macd_core as _macd_core_legacy",
        "_atr_from_hlc as _atr_from_hlc_legacy",
        "_efficiency_ratio as _efficiency_ratio_legacy",
        "_pos_from_donchian as _pos_from_donchian_legacy",
        "_pos_from_tsmom as _pos_from_tsmom_legacy",
        "_compute_impulse_state as _compute_impulse_state_legacy",
    ]
    for token in banned:
        assert token not in src


def test_bt_trend_indicator_pipeline_has_no_new_raw_rolling_or_ewm_usage():
    src = inspect.getsource(bt_trend_mod)
    # Indicator helper zone ends before input validation helpers; raw rolling/ewm
    # is only allowed in that helper zone as legacy fallback/comparison baseline.
    split_marker = "def _validate_bt_single_inputs("
    assert split_marker in src
    helper_zone, runtime_zone = src.split(split_marker, 1)
    assert ".rolling(" in helper_zone
    assert ".ewm(" in helper_zone
    assert ".rolling(" not in runtime_zone
    assert ".ewm(" not in runtime_zone


def test_bt_trend_imports_only_non_indicator_helpers_from_legacy_trend_module():
    src = inspect.getsource(bt_trend_mod)
    assert "from .trend import (" not in src
    assert "import etf_momentum.analysis.trend" not in src
    assert "import etf_momentum.analysis.trend as" not in src
    assert "from etf_momentum.analysis import trend" not in src


def test_bt_trend_core_indicators_keep_talib_first_path():
    src = inspect.getsource(bt_trend_mod)
    required_tokens = [
        "def _ema(",
        'return _talib_unary_series(s, "EMA"',
        "def _moving_average(",
        '"SMA", fallback=legacy',
        '"KAMA",',
        "def _macd_core(",
        "talib.MACD(",
        "def _atr_from_hlc(",
        "talib.ATR(",
        "def _rolling_linreg_slope(",
        "talib.LINEARREG_SLOPE(",
        "def _donchian_prev_high(",
        '"MAX", fallback=legacy',
        "def _donchian_prev_low(",
        '"MIN", fallback=legacy',
        "def _tsmom_rocp(",
        '"ROCP", fallback=legacy',
        "def _momentum_delta(",
        "talib.MOM(",
        "def _rolling_std(",
        "talib.STDDEV(",
        "def _rolling_sma(",
        "talib.SMA(",
        "def _efficiency_ratio(",
        "talib.SUM(",
    ]
    for token in required_tokens:
        assert token in src


def test_bt_trend_signal_builder_uses_indicator_wrappers_only():
    src = inspect.getsource(bt_trend_mod)
    start = src.find("def _build_signal_position(")
    assert start >= 0
    end = src.find("def _build_bt_frame(", start)
    assert end > start
    block = src[start:end]
    assert ".rolling(" not in block
    assert ".ewm(" not in block
    assert "talib." not in block


def test_bt_trend_raw_rolling_ewm_only_allowed_in_fallback_indicator_helpers():
    src = inspect.getsource(bt_trend_mod)
    allowed = {
        "_ema_fallback",
        "_kama_fallback",
        "_atr_from_hlc_fallback",
        "_rolling_linreg_slope",
        "_donchian_prev_high",
        "_donchian_prev_low",
        "_rolling_std",
        "_rolling_sma",
        "_efficiency_ratio",
    }
    cur_func = None
    offenders: list[str] = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("def ") and "(" in s:
            cur_func = s[4 : s.index("(")].strip()
        if ".rolling(" in ln or ".ewm(" in ln:
            if cur_func not in allowed:
                offenders.append(f"{cur_func}: {s}")
    assert not offenders


def test_bt_trend_indicator_paths_avoid_raw_diff_calls():
    src = inspect.getsource(bt_trend_mod)
    forbidden_tokens = [
        "ema.diff()",
        "hist.diff()",
        "p.diff().abs()",
        "(p - p.shift(",
    ]
    for token in forbidden_tokens:
        assert token not in src


def test_bt_trend_migrated_indicator_helpers_avoid_raw_shift_diff():
    # These helpers have been migrated to MOM/SUM wrapper paths and should
    # not regress to direct shift/diff arithmetic in their core logic.
    checks = {
        "_kama_fallback": ["p.diff(", "p - p.shift(", ".shift(er_w)"],
        "_efficiency_ratio": ["p.diff(", "p - p.shift("],
        "_compute_impulse_state": ["ema.diff(", "hist.diff("],
    }
    for fn, forbidden in checks.items():
        block = inspect.getsource(getattr(bt_trend_mod, fn))
        for token in forbidden:
            assert token not in block


def test_bt_trend_indicator_shift_diff_usage_is_whitelisted():
    checked = {
        "_kama_fallback",
        "_efficiency_ratio",
        "_compute_impulse_state",
    }
    allowed_tokens = {
        "_kama_fallback": {"_momentum_delta("},
        "_efficiency_ratio": {"_momentum_delta("},
        "_compute_impulse_state": {"_momentum_delta("},
    }
    for fn in checked:
        block = inspect.getsource(getattr(bt_trend_mod, fn))
        if ".diff(" in block or ".shift(" in block:
            found = {tok for tok in allowed_tokens[fn] if tok in block}
            assert found == allowed_tokens[fn]


def _flat_keys(obj: object, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.add(key)
            out |= _flat_keys(v, key)
    elif isinstance(obj, list) and obj:
        out |= _flat_keys(obj[0], f"{prefix}[]")
    return out


def _get(d: dict, path: str) -> object:
    cur: object = d
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _seed_case(engine) -> list:
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=220, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [100.0 + i * 0.35 + ((i % 17) - 8) * 0.1 for i, _ in enumerate(dates)],
            "B": [95.0 + i * 0.25 + ((i % 13) - 6) * 0.12 for i, _ in enumerate(dates)],
            "C": [
                105.0 + i * 0.15 + ((i % 11) - 5) * 0.08 for i, _ in enumerate(dates)
            ],
        },
        dates=dates,
    )
    return dates


@pytest.mark.parametrize("runner", [compute_trend_backtest, compute_trend_backtest_bt])
def test_single_backtest_prefix_is_invariant_to_future_extension(engine, runner):
    dates = _seed_case(engine)
    cutoff = dates[160]
    sf = make_session_factory(engine)
    common = dict(
        code="A",
        start=dates[0],
        strategy="tsmom",
        mom_lookback=20,
        exec_price="close",
        cost_bps=0.0,
        slippage_rate=0.0,
        quick_mode=True,
    )
    with sf() as db:
        short = runner(db, TrendInputs(end=cutoff, **common))
        extended = runner(db, TrendInputs(end=dates[-1], **common))

    short_dates = list(short["nav"]["dates"])
    extended_index = {day: index for index, day in enumerate(extended["nav"]["dates"])}
    # The cutoff row is a terminal projection for close execution; causal rows
    # strictly before it must not change when future observations are appended.
    causal_dates = short_dates[:-1]
    short_nav = short["nav"]["series"]["STRAT"]
    extended_nav = extended["nav"]["series"]["STRAT"]
    short_weight = short["signals"]["position_effective"]
    extended_weight = extended["signals"]["position_effective"]
    for index, day in enumerate(causal_dates):
        other = extended_index[day]
        assert float(short_nav[index]) == pytest.approx(
            float(extended_nav[other]), rel=0.0, abs=1e-12
        )
        assert float(short_weight[index]) == pytest.approx(
            float(extended_weight[other]), rel=0.0, abs=1e-12
        )


def test_bt_single_semantic_parity_keys_and_metrics(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="risk_budget",
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            ma_entry_filter_enabled=True,
            ma_entry_filter_type="sma",
            ma_entry_filter_fast=10,
            ma_entry_filter_slow=20,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="open",
            quick_mode=False,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    missing = sorted(_flat_keys(legacy) - _flat_keys(bt))
    assert not missing

    metric_paths = [
        "metrics.strategy.cumulative_return",
        "metrics.strategy.annualized_return",
        "metrics.strategy.max_drawdown",
        "metrics.strategy.sharpe_ratio",
        "metrics.strategy.calmar_ratio",
        "metrics.strategy.sortino_ratio",
        "metrics.benchmark.cumulative_return",
        "metrics.benchmark.calmar_ratio",
        "metrics.excess.cumulative_return",
        "metrics.excess.information_ratio",
    ]
    for p in metric_paths:
        lv = _get(legacy, p)
        bv = _get(bt, p)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
        if math.isfinite(float(lv)) and math.isfinite(float(bv)):
            assert abs(float(lv) - float(bv)) <= 1e-12


def test_bt_meta_params_execution_time_defaults_match_legacy(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            exec_price="close",
            atr_stop_execution_mode="next_day",
            r_take_profit_execution_mode="next_day",
            r_profit_scaleout_execution_mode="next_day",
            bias_v_take_profit_execution_mode="next_day",
            ma_trailing_stop_execution_mode="next_day",
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    l_params = ((legacy or {}).get("meta") or {}).get("params") or {}
    b_params = ((bt or {}).get("meta") or {}).get("params") or {}
    for key in [
        "atr_stop_execution_time",
        "r_take_profit_execution_time",
        "r_profit_scaleout_execution_time",
        "bias_v_take_profit_execution_time",
        "ma_trailing_stop_execution_time",
    ]:
        assert str(l_params.get(key) or "") == "close"
        assert str(b_params.get(key) or "") == str(l_params.get(key) or "")


def test_bt_portfolio_semantic_parity_keys_and_core_metrics(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="risk_budget",
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            ma_entry_filter_enabled=True,
            ma_entry_filter_type="sma",
            ma_entry_filter_fast=10,
            ma_entry_filter_slow=20,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="open",
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    missing = sorted(_flat_keys(legacy) - _flat_keys(bt))
    assert not missing

    # Portfolio path is now semantically aligned; keep tight tolerances on
    # key metrics while allowing tiny floating-point implementation drift.
    tolerances = {
        "metrics.strategy.cumulative_return": 1e-2,
        "metrics.strategy.annualized_return": 1e-2,
        "metrics.strategy.max_drawdown": 1e-2,
        "metrics.excess.cumulative_return": 1e-2,
        "metrics.excess.information_ratio": 0.2,
    }
    for p, tol in tolerances.items():
        lv = _get(legacy, p)
        bv = _get(bt, p)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
        if math.isfinite(float(lv)) and math.isfinite(float(bv)):
            assert abs(float(lv) - float(bv)) <= float(tol)

    for p in ("metrics.strategy.calmar_ratio", "metrics.benchmark.calmar_ratio"):
        lv = _get(legacy, p)
        bv = _get(bt, p)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))


@pytest.mark.parametrize(
    ("strategy", "exec_price", "position_sizing"),
    [
        ("ma_filter", "close", "equal"),
        ("ma_filter", "open", "risk_budget"),
        ("ma_cross", "close", "equal"),
        ("ma_cross", "open", "risk_budget"),
        ("donchian", "close", "equal"),
        ("donchian", "open", "risk_budget"),
        ("tsmom", "close", "equal"),
        ("tsmom", "open", "risk_budget"),
        ("linreg_slope", "close", "equal"),
        ("linreg_slope", "open", "risk_budget"),
        ("bias", "close", "equal"),
        ("bias", "open", "risk_budget"),
        ("macd_cross", "close", "equal"),
        ("macd_cross", "open", "risk_budget"),
        ("macd_zero_filter", "close", "equal"),
        ("macd_zero_filter", "open", "risk_budget"),
        ("macd_v", "close", "equal"),
        ("macd_v", "open", "risk_budget"),
        ("cci", "close", "equal"),
        ("cci", "open", "risk_budget"),
    ],
)
def test_bt_single_semantic_parity_matrix(
    engine, strategy: str, exec_price: str, position_sizing: str
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy=strategy,
            ma_type="kama" if strategy == "ma_filter" else "sma",
            sma_window=20,
            position_sizing=position_sizing,
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price=exec_price,
            quick_mode=False,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    # Extended migration patrol for single-asset path:
    # canonical combos must keep exact cumulative-return parity.
    l_cum = _get(legacy, "metrics.strategy.cumulative_return")
    b_cum = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(l_cum, (int, float)) and isinstance(b_cum, (int, float))
    if math.isfinite(float(l_cum)) and math.isfinite(float(b_cum)):
        assert abs(float(l_cum) - float(b_cum)) <= 1e-12

    # Matrix guard: broad scenarios should remain structurally aligned and
    # numerically stable (finite and not wildly divergent), while strict
    # near-equality is enforced in dedicated canonical tests above.
    for p, tol in {
        "metrics.strategy.annualized_return": 1.0,
        "metrics.strategy.max_drawdown": 1.0,
        "metrics.excess.cumulative_return": 1.0,
    }.items():
        lv = _get(legacy, p)
        bv = _get(bt, p)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
        if math.isfinite(float(lv)) and math.isfinite(float(bv)):
            assert abs(float(lv) - float(bv)) <= float(tol)


@pytest.mark.parametrize(
    ("strategy", "exec_price", "position_sizing"),
    [
        ("ma_filter", "close", "equal"),
        ("ma_filter", "open", "risk_budget"),
        ("ma_cross", "close", "equal"),
        ("ma_cross", "open", "risk_budget"),
        ("donchian", "close", "equal"),
        ("donchian", "open", "risk_budget"),
        ("tsmom", "close", "equal"),
        ("tsmom", "open", "risk_budget"),
        ("linreg_slope", "close", "equal"),
        ("linreg_slope", "open", "risk_budget"),
        ("bias", "close", "equal"),
        ("bias", "open", "risk_budget"),
        ("macd_cross", "close", "equal"),
        ("macd_cross", "open", "risk_budget"),
        ("macd_zero_filter", "close", "equal"),
        ("macd_zero_filter", "open", "risk_budget"),
        ("macd_v", "close", "equal"),
        ("macd_v", "open", "risk_budget"),
        ("cci", "close", "equal"),
        ("cci", "open", "risk_budget"),
    ],
)
def test_bt_portfolio_semantic_parity_matrix(
    engine, strategy: str, exec_price: str, position_sizing: str
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy=strategy,
            ma_type="kama" if strategy == "ma_filter" else "sma",
            sma_window=20,
            fast_window=12,
            slow_window=30,
            donchian_entry=20,
            donchian_exit=10,
            mom_lookback=60,
            position_sizing=position_sizing,
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price=exec_price,
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    # Extended migration patrol: all main strategy families should preserve
    # exact cumulative-return parity on canonical execution/sizing combos.
    l_cum = _get(legacy, "metrics.strategy.cumulative_return")
    b_cum = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(l_cum, (int, float)) and isinstance(b_cum, (int, float))
    if math.isfinite(float(l_cum)) and math.isfinite(float(b_cum)):
        assert abs(float(l_cum) - float(b_cum)) <= 1e-12

    for p, tol in {
        "metrics.strategy.annualized_return": 10.0,
        "metrics.strategy.max_drawdown": 1.0,
        "metrics.excess.cumulative_return": 10.0,
    }.items():
        lv = _get(legacy, p)
        bv = _get(bt, p)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
        if math.isfinite(float(lv)) and math.isfinite(float(bv)):
            assert abs(float(lv) - float(bv)) <= float(tol)


@pytest.mark.parametrize("strategy", ["macd_cross", "macd_v"])
def test_bt_single_macd_hist_threshold_semantic_parity(engine, strategy: str):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy=strategy,
            exec_price="close",
            cost_bps=5.0,
            slippage_rate=0.001,
            macd_hist_min=0.5 if strategy == "macd_cross" else 0.0,
            macd_v_hist_min=5.0 if strategy == "macd_v" else 0.0,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    l_cum = _get(legacy, "metrics.strategy.cumulative_return")
    b_cum = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(l_cum, (int, float)) and isinstance(b_cum, (int, float))
    if math.isfinite(float(l_cum)) and math.isfinite(float(b_cum)):
        assert abs(float(l_cum) - float(b_cum)) <= 1e-12


@pytest.mark.parametrize("strategy", ["macd_cross", "macd_v"])
def test_bt_portfolio_macd_hist_threshold_semantic_parity(engine, strategy: str):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy=strategy,
            exec_price="close",
            position_sizing="equal",
            cost_bps=5.0,
            slippage_rate=0.001,
            macd_hist_min=0.5 if strategy == "macd_cross" else 0.0,
            macd_v_hist_min=5.0 if strategy == "macd_v" else 0.0,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    l_cum = _get(legacy, "metrics.strategy.cumulative_return")
    b_cum = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(l_cum, (int, float)) and isinstance(b_cum, (int, float))
    if math.isfinite(float(l_cum)) and math.isfinite(float(b_cum)):
        assert abs(float(l_cum) - float(b_cum)) <= 3e-2


def test_bt_portfolio_effective_weights_semantic_parity(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="equal",
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    l_idx = pd.to_datetime(legacy["weights"]["dates"])
    b_idx = pd.to_datetime(bt["weights"]["dates"])
    l_w = pd.DataFrame(legacy["weights"]["series"], index=l_idx).sort_index()
    b_w = pd.DataFrame(bt["weights"]["series"], index=b_idx).sort_index()
    common = l_w.index.intersection(b_w.index)
    assert len(common) > 0
    l_aligned = l_w.reindex(common).fillna(0.0).astype(float)
    b_aligned = b_w.reindex(common).fillna(0.0).astype(float)
    max_abs = (b_aligned - l_aligned).abs().to_numpy().max()
    assert float(max_abs) <= 1e-12


def _series_frame(response: dict, key: str) -> pd.DataFrame:
    payload = response[key]
    return pd.DataFrame(
        payload["series"],
        index=pd.to_datetime(payload["dates"]),
    ).sort_index()


def _canonical_trigger_events(value: object, path: str = "") -> list[tuple]:
    rows: list[tuple] = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            child_path = f"{path}.{key}" if path else str(key)
            if key == "trigger_events" and isinstance(child, list):
                for event in child:
                    if not isinstance(event, dict):
                        continue
                    if not any(
                        event.get(field) is not None
                        for field in (
                            "date",
                            "trigger_date",
                            "execution_date",
                            "trigger_source",
                            "fill_price",
                        )
                    ):
                        continue
                    rows.append(
                        (
                            path,
                            event.get("date"),
                            event.get("trigger_date"),
                            event.get("execution_date"),
                            event.get("event_type"),
                            event.get("trigger_source"),
                            event.get("fill_price"),
                            event.get("requested_reduce_fraction"),
                        )
                    )
            else:
                rows.extend(_canonical_trigger_events(child, child_path))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_canonical_trigger_events(child, path))
    return sorted(rows, key=repr)


def _canonical_closed_trades(response: dict) -> list[tuple]:
    trades = (response.get("trade_statistics") or {}).get("closed_trades")
    if trades is None:
        trades = [
            row
            for row in (response.get("trade_statistics") or {}).get("trades", [])
            if bool(row.get("closed"))
        ]
    return sorted(
        (
            str(row.get("code")),
            str(row.get("entry_date")),
            str(row.get("exit_date")),
            float(row.get("return") or 0.0),
            float(row.get("holding_return") or 0.0),
            float(row.get("pnl_amount") or 0.0),
        )
        for row in trades
    )


def test_portfolio_daily_accounting_and_event_parity(engine) -> None:
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    inp = TrendPortfolioInputs(
        codes=["A", "B", "C"],
        start=dates[0],
        end=dates[-1],
        strategy="ma_filter",
        ma_type="kama",
        sma_window=20,
        position_sizing="equal",
        atr_stop_mode="trailing",
        atr_stop_n=2.0,
        r_take_profit_enabled=True,
        bias_v_take_profit_enabled=True,
        monthly_risk_budget_enabled=True,
        monthly_risk_budget_pct=0.06,
        cost_bps=5.0,
        slippage_rate=0.001,
        exec_price="close",
        quick_mode=False,
    )
    with sf() as db:
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    for key in ("weights", "weights_decision"):
        left = _series_frame(legacy, key)
        right = _series_frame(bt, key)
        pd.testing.assert_index_equal(left.index, right.index)
        pd.testing.assert_index_equal(left.columns, right.columns)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)

    left_nav = _series_frame(legacy, "nav")[["STRAT"]]
    right_nav = _series_frame(bt, "nav")[["STRAT"]]
    pd.testing.assert_frame_equal(
        left_nav,
        right_nav,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    left_cost = _series_frame(legacy, "return_decomposition")[["cost"]]
    right_cost = _series_frame(bt, "return_decomposition")[["cost"]]
    pd.testing.assert_frame_equal(
        left_cost,
        right_cost,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    assert _canonical_trigger_events(
        legacy.get("risk_controls") or {}
    ) == _canonical_trigger_events(bt.get("risk_controls") or {})
    left_trades = _canonical_closed_trades(legacy)
    right_trades = _canonical_closed_trades(bt)
    assert [row[:3] for row in left_trades] == [row[:3] for row in right_trades]
    np.testing.assert_allclose(
        [row[3:] for row in left_trades],
        [row[3:] for row in right_trades],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "runner",
    [compute_trend_portfolio_backtest, compute_trend_portfolio_backtest_bt],
)
def test_portfolio_availability_states_do_not_create_ghost_positions(
    engine, runner
) -> None:
    dates = _seed_case(engine)
    listing_date = dates[40]
    gap_date = dates[120]
    removal_date = dates[160]
    sf = make_session_factory(engine)
    with sf() as db:
        db.add_all(
            [
                EtfPool(
                    code="A",
                    name="A",
                    start_date=dates[0].strftime("%Y%m%d"),
                ),
                EtfPool(
                    code="B",
                    name="B",
                    start_date=listing_date.strftime("%Y%m%d"),
                ),
                EtfPool(
                    code="C",
                    name="C",
                    start_date=dates[0].strftime("%Y%m%d"),
                    end_date=removal_date.strftime("%Y%m%d"),
                ),
            ]
        )
        db.execute(
            delete(EtfPrice).where(
                EtfPrice.code == "B",
                EtfPrice.trade_date < listing_date,
            )
        )
        db.execute(
            delete(EtfPrice).where(
                EtfPrice.code == "C",
                (
                    (EtfPrice.trade_date == gap_date)
                    | (EtfPrice.trade_date > removal_date)
                ),
            )
        )
        db.commit()
        out = runner(
            db,
            TrendPortfolioInputs(
                codes=["A", "B", "C"],
                start=dates[0],
                end=dates[-1],
                strategy="tsmom",
                mom_lookback=20,
                tsmom_entry_threshold=0.01,
                tsmom_exit_threshold=0.0,
                position_sizing="equal",
                dynamic_universe=True,
                atr_stop_mode="static",
                cost_bps=0.0,
                slippage_rate=0.0,
                exec_price="close",
                quick_mode=False,
            ),
        )

    weights = _series_frame(out, "weights")
    assert (weights.loc[weights.index < pd.Timestamp(listing_date), "B"] == 0.0).all()
    gap_timestamp = pd.Timestamp(gap_date)
    assert weights.loc[gap_timestamp, "C"] == pytest.approx(
        weights.shift(1).loc[gap_timestamp, "C"]
    )
    decision = _series_frame(out, "weights_decision")
    post_removal = decision.index > pd.Timestamp(removal_date)
    assert (decision.loc[post_removal, "C"] == 0.0).all()
    post_dates = weights.index[weights.index > pd.Timestamp(removal_date)]
    assert (weights.loc[post_dates[1:], "C"] == 0.0).all()
    events = _canonical_trigger_events(out.get("risk_controls") or {})
    assert not any(
        gap_date.isoformat() in {str(event[1]), str(event[2]), str(event[3])}
        for event in events
    )


@pytest.mark.parametrize(
    "runner",
    [compute_trend_portfolio_backtest, compute_trend_portfolio_backtest_bt],
)
def test_portfolio_prefix_and_future_perturbation_are_causal(engine, runner) -> None:
    dates = _seed_case(engine)
    cutoff = dates[160]
    sf = make_session_factory(engine)
    common = dict(
        codes=["A", "B", "C"],
        start=dates[0],
        strategy="tsmom",
        mom_lookback=20,
        tsmom_entry_threshold=0.01,
        tsmom_exit_threshold=0.0,
        position_sizing="equal",
        dynamic_universe=True,
        atr_stop_mode="static",
        cost_bps=0.0,
        slippage_rate=0.0,
        exec_price="close",
        quick_mode=True,
    )
    with sf() as db:
        short = runner(db, TrendPortfolioInputs(end=cutoff, **common))
        extended = runner(db, TrendPortfolioInputs(end=dates[-1], **common))
        db.execute(
            update(EtfPrice)
            .where(EtfPrice.trade_date > cutoff)
            .values(
                close=EtfPrice.close * 7.0,
                open=EtfPrice.open * 7.0,
                high=EtfPrice.high * 7.0,
                low=EtfPrice.low * 7.0,
            )
        )
        db.commit()
        perturbed = runner(db, TrendPortfolioInputs(end=dates[-1], **common))

    causal_end = pd.Timestamp(cutoff)
    for key in ("weights", "weights_decision", "nav"):
        short_frame = _series_frame(short, key)
        extended_frame = _series_frame(extended, key).reindex(short_frame.index)
        perturbed_frame = _series_frame(perturbed, key).reindex(short_frame.index)
        columns = ["STRAT"] if key == "nav" else list(short_frame.columns)
        index = short_frame.index[short_frame.index < causal_end]
        pd.testing.assert_frame_equal(
            short_frame.loc[index, columns],
            extended_frame.loc[index, columns],
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            extended_frame.loc[index, columns],
            perturbed_frame.loc[index, columns],
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "runner",
    [compute_trend_portfolio_backtest, compute_trend_portfolio_backtest_bt],
)
def test_qfq_proportional_vintage_restatement_preserves_dimensionless_state(
    engine, runner
) -> None:
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    inp = TrendPortfolioInputs(
        codes=["A", "B", "C"],
        start=dates[0],
        end=dates[-1],
        strategy="tsmom",
        mom_lookback=20,
        tsmom_entry_threshold=0.01,
        tsmom_exit_threshold=0.0,
        position_sizing="equal",
        dynamic_universe=True,
        atr_stop_mode="static",
        cost_bps=0.0,
        slippage_rate=0.0,
        exec_price="close",
        quick_mode=True,
    )
    with sf() as db:
        original = runner(db, inp)
        db.execute(
            update(EtfPrice)
            .where(EtfPrice.adjust == "qfq")
            .values(
                close=EtfPrice.close * 13.0,
                open=EtfPrice.open * 13.0,
                high=EtfPrice.high * 13.0,
                low=EtfPrice.low * 13.0,
            )
        )
        db.commit()
        restated = runner(db, inp)

    for key in ("weights", "weights_decision"):
        pd.testing.assert_frame_equal(
            _series_frame(original, key),
            _series_frame(restated, key),
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("exec_price", "position_sizing"),
    [
        ("close", "equal"),
        ("close", "risk_budget"),
        ("close", "fixed_ratio"),
        ("close", "vol_target"),
        ("open", "equal"),
        ("open", "risk_budget"),
        ("open", "fixed_ratio"),
        ("open", "vol_target"),
    ],
)
def test_bt_single_sizing_modes_sentinel_parity(
    engine, exec_price: str, position_sizing: str
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            fast_window=12,
            slow_window=30,
            donchian_entry=20,
            donchian_exit=10,
            mom_lookback=60,
            position_sizing=position_sizing,
            fixed_pos_ratio=0.04,
            vol_target_ann=0.20,
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price=exec_price,
            quick_mode=False,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize(
    ("exec_price", "position_sizing"),
    [
        ("close", "equal"),
        ("close", "risk_budget"),
        ("close", "fixed_ratio"),
        ("close", "vol_target"),
        ("open", "equal"),
        ("open", "risk_budget"),
        ("open", "fixed_ratio"),
        ("open", "vol_target"),
    ],
)
def test_bt_portfolio_sizing_modes_sentinel_parity(
    engine, exec_price: str, position_sizing: str
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            fast_window=12,
            slow_window=30,
            donchian_entry=20,
            donchian_exit=10,
            mom_lookback=60,
            position_sizing=position_sizing,
            fixed_pos_ratio=0.04,
            vol_target_ann=0.20,
            risk_budget_pct=0.01,
            atr_stop_mode="trailing",
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price=exec_price,
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize(
    ("exec_price", "position_sizing"),
    [
        ("close", "equal"),
        ("close", "risk_budget"),
        ("close", "fixed_ratio"),
        ("close", "vol_target"),
        ("open", "equal"),
        ("open", "risk_budget"),
        ("open", "fixed_ratio"),
        ("open", "vol_target"),
    ],
)
def test_bt_portfolio_quick_mode_sentinel_parity(
    engine, exec_price: str, position_sizing: str
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            fast_window=12,
            slow_window=30,
            donchian_entry=20,
            donchian_exit=10,
            mom_lookback=60,
            position_sizing=position_sizing,
            fixed_pos_ratio=0.04,
            vol_target_ann=0.20,
            risk_budget_pct=0.01,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price=exec_price,
            quick_mode=True,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    # Quick mode should preserve the same API contract keys.
    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))

    # Also keep exact cumulative-return parity in quick mode.
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


def test_trend_single_rejects_oc2_exec_price_in_legacy_and_bt(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            exec_price="oc2",
        )
        with pytest.raises(ValueError, match="open\\|close"):
            compute_trend_backtest(db, inp)
        with pytest.raises(ValueError, match="open\\|close"):
            compute_trend_backtest_bt(db, inp)


def test_trend_portfolio_rejects_oc2_exec_price_in_legacy_and_bt(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            exec_price="oc2",
        )
        with pytest.raises(ValueError, match="open\\|close"):
            compute_trend_portfolio_backtest(db, inp)
        with pytest.raises(ValueError, match="open\\|close"):
            compute_trend_portfolio_backtest_bt(db, inp)


@pytest.mark.parametrize("quick_mode", [False, True])
def test_bt_single_random_entry_semantic_parity(engine, quick_mode: bool):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="random_entry",
            random_seed=42,
            random_hold_days=20,
            position_sizing="equal",
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=quick_mode,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("quick_mode", [False, True])
def test_bt_portfolio_random_entry_semantic_parity(engine, quick_mode: bool):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="random_entry",
            random_seed=42,
            random_hold_days=20,
            position_sizing="equal",
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=quick_mode,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("group_pick_policy", ["highest_sharpe", "earliest_entry"])
def test_bt_portfolio_group_filter_semantic_parity(engine, group_pick_policy: str):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="equal",
            group_enforce=True,
            group_pick_policy=group_pick_policy,
            group_max_holdings=1,
            asset_groups={"A": "G1", "B": "G1", "C": "G2"},
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


def test_bt_portfolio_dynamic_universe_semantic_parity(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="equal",
            dynamic_universe=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("atr_stop_mode", ["none", "static", "trailing"])
def test_bt_single_atr_stop_modes_semantic_parity(engine, atr_stop_mode: str):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="equal",
            atr_stop_mode=atr_stop_mode,
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=False,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("atr_stop_mode", ["none", "static", "trailing"])
def test_bt_portfolio_atr_stop_modes_semantic_parity(engine, atr_stop_mode: str):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="equal",
            atr_stop_mode=atr_stop_mode,
            atr_stop_n=2.0,
            r_take_profit_enabled=True,
            bias_v_take_profit_enabled=True,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=False,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("reentry_mode", ["reenter", "wait_next_entry"])
@pytest.mark.parametrize("quick_mode", [False, True])
def test_bt_single_tightening_reentry_semantic_parity(
    engine, reentry_mode: str, quick_mode: bool
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="risk_budget",
            risk_budget_pct=0.01,
            atr_stop_mode="tightening",
            atr_stop_n=2.0,
            atr_stop_m=0.5,
            atr_stop_reentry_mode=reentry_mode,
            r_take_profit_enabled=True,
            r_take_profit_reentry_mode=reentry_mode,
            bias_v_take_profit_enabled=True,
            bias_v_take_profit_reentry_mode=reentry_mode,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=quick_mode,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


@pytest.mark.parametrize("reentry_mode", ["reenter", "wait_next_entry"])
@pytest.mark.parametrize("quick_mode", [False, True])
def test_bt_portfolio_tightening_reentry_semantic_parity(
    engine, reentry_mode: str, quick_mode: bool
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            ma_type="kama",
            sma_window=20,
            position_sizing="risk_budget",
            risk_budget_pct=0.01,
            atr_stop_mode="tightening",
            atr_stop_n=2.0,
            atr_stop_m=0.5,
            atr_stop_reentry_mode=reentry_mode,
            r_take_profit_enabled=True,
            r_take_profit_reentry_mode=reentry_mode,
            bias_v_take_profit_enabled=True,
            bias_v_take_profit_reentry_mode=reentry_mode,
            monthly_risk_budget_enabled=True,
            monthly_risk_budget_pct=0.06,
            er_filter=True,
            impulse_entry_filter=True,
            er_exit_filter=True,
            cost_bps=5.0,
            slippage_rate=0.001,
            exec_price="close",
            quick_mode=quick_mode,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


def test_bt_single_ma_trailing_stop_semantic_parity(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendInputs(
            code="A",
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            sma_window=20,
            position_sizing="equal",
            exec_price="close",
            ma_trailing_stop_enabled=True,
            ma_trailing_stop_ma_type="ema",
            ma_trailing_stop_execution_mode="next_day",
            ma_trailing_stop_effective_delay_days=3,
            ma_trailing_stop_reduce_window=10,
            ma_trailing_stop_exit_window=20,
            ma_trailing_stop_reduce_fraction=0.33,
            r_take_profit_enabled=True,
            r_take_profit_tiers=[{"r_multiple": 2.0, "retrace_ratio": 0.3}],
            r_profit_scaleout_enabled=True,
            r_profit_scaleout_tiers=[{"r_multiple": 2.5, "reduce_fraction": 0.2}],
            quick_mode=True,
            cost_bps=0.0,
            slippage_rate=0.0,
        )
        legacy = compute_trend_backtest(db, inp)
        bt = compute_trend_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    assert (
        bool(_get(legacy, "risk_controls.ma_trailing_stop.enabled")) is True
        and bool(_get(bt, "risk_controls.ma_trailing_stop.enabled")) is True
    )
    assert str(
        _get(bt, "risk_controls.ma_trailing_stop.reduce_fraction_basis") or ""
    ) == ("initial_position")
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


def test_bt_portfolio_ma_trailing_stop_semantic_parity(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            sma_window=20,
            position_sizing="equal",
            exec_price="close",
            ma_trailing_stop_enabled=True,
            ma_trailing_stop_ma_type="sma",
            ma_trailing_stop_execution_mode="intraday",
            ma_trailing_stop_effective_delay_days=3,
            ma_trailing_stop_reduce_window=10,
            ma_trailing_stop_exit_window=20,
            ma_trailing_stop_reduce_fraction=0.33,
            r_take_profit_enabled=True,
            r_take_profit_tiers=[{"r_multiple": 2.0, "retrace_ratio": 0.3}],
            r_profit_scaleout_enabled=True,
            r_profit_scaleout_tiers=[{"r_multiple": 2.5, "reduce_fraction": 0.2}],
            quick_mode=True,
            cost_bps=0.0,
            slippage_rate=0.0,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert not sorted(_flat_keys(legacy) - _flat_keys(bt))
    assert (
        bool(_get(legacy, "risk_controls.ma_trailing_stop.enabled")) is True
        and bool(_get(bt, "risk_controls.ma_trailing_stop.enabled")) is True
    )
    lv = _get(legacy, "metrics.strategy.cumulative_return")
    bv = _get(bt, "metrics.strategy.cumulative_return")
    assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
    if math.isfinite(float(lv)) and math.isfinite(float(bv)):
        assert abs(float(lv) - float(bv)) <= 1e-12


def test_bt_portfolio_breakeven_addon_flags_and_addons_block_parity(engine):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        inp = TrendPortfolioInputs(
            codes=["A", "B", "C"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            sma_window=20,
            position_sizing="equal",
            exec_price="close",
            atr_stop_mode="none",
            r_profit_scaleout_enabled=True,
            r_profit_scaleout_execution_mode="intraday",
            r_profit_scaleout_breakeven_stop_enabled=False,
            r_profit_scaleout_tiers=[{"r_multiple": 2.5, "reduce_fraction": 0.2}],
            bias_v_take_profit_enabled=True,
            bias_v_take_profit_reentry_mode="reenter",
            bias_v_take_profit_execution_mode="intraday",
            bias_v_take_profit_breakeven_stop_enabled=False,
            bias_v_ma_window=20,
            bias_v_atr_window=20,
            bias_v_take_profit_tiers=[{"threshold": 3.0, "reduce_fraction": 0.2}],
            quick_mode=True,
            cost_bps=0.0,
            slippage_rate=0.0,
        )
        legacy = compute_trend_portfolio_backtest(db, inp)
        bt = compute_trend_portfolio_backtest_bt(db, inp)

    assert _get(bt, "risk_controls.addons.r_profit_scaleout_breakeven_stop") is not None
    assert (
        _get(bt, "risk_controls.addons.bias_v_take_profit_breakeven_stop") is not None
    )
    assert (
        bool(
            _get(
                legacy, "risk_controls.addons.r_profit_scaleout_breakeven_stop.enabled"
            )
        )
        is False
    )
    assert (
        bool(_get(bt, "risk_controls.addons.r_profit_scaleout_breakeven_stop.enabled"))
        is False
    )
    assert (
        bool(
            _get(
                legacy, "risk_controls.addons.bias_v_take_profit_breakeven_stop.enabled"
            )
        )
        is False
    )
    assert (
        bool(_get(bt, "risk_controls.addons.bias_v_take_profit_breakeven_stop.enabled"))
        is False
    )


def test_bt_single_ma_trailing_stop_override_remap_scales_and_matches_nav(
    engine, monkeypatch
):
    dates = _seed_case(engine)
    sf = make_session_factory(engine)

    def _fake_apply_intraday_or_arbitration_single(
        *, weights, event_sets, exec_price, open_sig, close_sig
    ):
        del event_sets, exec_price, open_sig, close_sig
        zeros = pd.Series(0.0, index=weights.index, dtype=float)
        ma_over = pd.Series(0.01, index=weights.index, dtype=float)
        return weights.astype(float).copy(), {
            "atr_stop": zeros.copy(),
            "bias_v_take_profit": zeros.copy(),
            "r_take_profit": zeros.copy(),
            "r_profit_scaleout": zeros.copy(),
            "ma_trailing_stop": ma_over,
        }

    def _fake_simulate_lot_account_weights(*, target_weights, **kwargs):
        del kwargs
        return target_weights.copy().astype(float), {}

    def _fake_remap_return_weights(*, w_eff_before, w_eff_after, w_ret_before):
        del w_eff_before, w_eff_after
        risk_scale = pd.Series(0.5, index=w_ret_before.index, dtype=float)
        return w_ret_before.copy().astype(float), risk_scale

    monkeypatch.setattr(
        bt_trend_mod,
        "_apply_intraday_or_arbitration_single",
        _fake_apply_intraday_or_arbitration_single,
    )

    with sf() as db:
        base_out = compute_trend_backtest_bt(
            db,
            TrendInputs(
                code="A",
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                exec_price="open",
                atr_stop_mode="none",
                ma_trailing_stop_enabled=True,
                ma_trailing_stop_ma_type="sma",
                ma_trailing_stop_execution_mode="intraday",
                ma_trailing_stop_effective_delay_days=1,
                ma_trailing_stop_reduce_window=3,
                ma_trailing_stop_exit_window=5,
                ma_trailing_stop_reduce_fraction=0.5,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
        base_ma = (
            ((base_out.get("return_decomposition") or {}).get("series") or {}).get(
                "ma_trailing_stop_override"
            )
        ) or []
        assert base_ma and any(abs(float(x)) > 1e-12 for x in base_ma)

        monkeypatch.setattr(
            bt_trend_mod,
            "simulate_lot_account_weights",
            _fake_simulate_lot_account_weights,
        )
        monkeypatch.setattr(
            bt_trend_mod,
            "remap_return_weights",
            _fake_remap_return_weights,
        )

        scaled_out = compute_trend_backtest_bt(
            db,
            TrendInputs(
                code="A",
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                exec_price="open",
                atr_stop_mode="none",
                ma_trailing_stop_enabled=True,
                ma_trailing_stop_ma_type="sma",
                ma_trailing_stop_execution_mode="intraday",
                ma_trailing_stop_effective_delay_days=1,
                ma_trailing_stop_reduce_window=3,
                ma_trailing_stop_exit_window=5,
                ma_trailing_stop_reduce_fraction=0.5,
                initial_account_amount=1_000_000.0,
                position_sizing="fixed_ratio",
                fixed_pos_ratio=0.8,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    scaled_decomp = ((scaled_out.get("return_decomposition") or {}).get("series")) or {}
    scaled_ma = [
        float(x) for x in (scaled_decomp.get("ma_trailing_stop_override") or [])
    ]
    scaled_risk = [float(x) for x in (scaled_decomp.get("risk_exit_override") or [])]
    scaled_net = [float(x) for x in (scaled_decomp.get("net") or [])]
    nav_series = [
        float(x)
        for x in (
            (((scaled_out.get("nav") or {}).get("series") or {}).get("STRAT")) or []
        )
    ]

    assert len(scaled_ma) == len(base_ma) == len(scaled_risk) == len(nav_series)
    for base_v, scaled_v in zip(base_ma, scaled_ma):
        assert float(scaled_v) == pytest.approx(float(base_v) * 0.5, abs=1e-12)
    for rv, mv in zip(scaled_risk, scaled_ma):
        assert float(rv) == pytest.approx(float(mv), abs=1e-12)

    assert len(scaled_net) == len(nav_series)
    nav_rebuild = (
        bt_trend_mod._as_nav(
            pd.Series(scaled_net, index=pd.RangeIndex(len(scaled_net)), dtype=float)
        )
        .astype(float)
        .tolist()
    )
    for got, rebuilt in zip(nav_series, nav_rebuild):
        assert float(got) == pytest.approx(float(rebuilt), abs=1e-12)
