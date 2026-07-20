from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from etf_momentum.analysis import futures_trend as fut_trend
from etf_momentum.db.futures_research_repo import FuturesGroupData


def _mk_exec_frame(idx: pd.DatetimeIndex, start_close: float) -> pd.DataFrame:
    close = pd.Series(
        [float(start_close + i) for i in range(len(idx))],
        index=idx,
        dtype=float,
    )
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": (close + 1.0).values,
            "Low": (close - 1.0).values,
            "Close": close.values,
            "Settle": close.values,
            "SignalClose": close.values,
        },
        index=idx,
    )


def test_futures_trend_backtest_respects_dynamic_universe_union(monkeypatch) -> None:
    idx_a = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
    idx_b = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    exec_map = {
        "A0": _mk_exec_frame(idx_a, 100.0),
        "B0": _mk_exec_frame(idx_b, 200.0),
    }

    def _fake_align(_db, *, pool_code: str, start: str, end: str):
        df = exec_map[str(pool_code)].copy()
        return df, df.copy(), {}

    def _fake_pool(_db, code: str):
        return SimpleNamespace(contract_multiplier=10.0, min_price_tick=1.0, code=code)

    def _fake_simulate(
        *,
        common_idx,
        exec_by_code,
        w_eff,
        cost_by_symbol,
        mults,
        margin_rate_frac,
        reserve_ratio,
        initial_equity_cny,
        exec_price,
        position_sizing,
        codes_sorted,
    ):
        return (
            pd.Series(1.0, index=pd.DatetimeIndex(common_idx), dtype=float),
            {"closed_trades": []},
        )

    monkeypatch.setattr(fut_trend, "_align_futures_trend_inputs", _fake_align)
    monkeypatch.setattr(fut_trend, "get_futures_pool_by_code", _fake_pool)
    monkeypatch.setattr(fut_trend, "simulate_discrete_lot_portfolio", _fake_simulate)

    group = FuturesGroupData(name="G", codes=["A0", "B0"], is_active=True)
    common_kwargs = {
        "db": None,
        "group": group,
        "start": "20240101",
        "end": "20240131",
        "exec_price": "close",
        "fast_ma": 2,
        "slow_ma": 3,
        "min_points": 2,
        "cost_bps": 0.0,
        "fee_side": "one_way",
        "slippage_type": "percent",
        "slippage_value": 0.0,
        "slippage_side": "one_way",
    }

    out_static = fut_trend.compute_futures_group_trend_backtest(
        dynamic_universe=False,
        **common_kwargs,
    )
    assert out_static["ok"] is True
    dates_static = [str(x["date"]) for x in out_static["series"]["strategy_nav"]]
    assert dates_static == ["2024-01-03", "2024-01-04"]
    assert out_static["meta"]["mode"] == "static_intersection"

    out_dynamic = fut_trend.compute_futures_group_trend_backtest(
        dynamic_universe=True,
        **common_kwargs,
    )
    assert out_dynamic["ok"] is True
    dates_dynamic = [str(x["date"]) for x in out_dynamic["series"]["strategy_nav"]]
    assert dates_dynamic == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert out_dynamic["meta"]["mode"] == "dynamic_union"


def test_futures_trend_executes_on_next_trading_day(monkeypatch) -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    exec_map = {"A0": _mk_exec_frame(idx, 100.0)}
    captured: dict[str, pd.DataFrame] = {}

    def _fake_align(_db, *, pool_code: str, start: str, end: str):
        df = exec_map[str(pool_code)].copy()
        return df, df.copy(), {}

    def _fake_pool(_db, code: str):
        return SimpleNamespace(contract_multiplier=10.0, min_price_tick=1.0, code=code)

    def _fake_build_panels(*args, **kwargs):
        common_idx = pd.DatetimeIndex(kwargs["common_idx"])
        score = pd.DataFrame({"A0": [0.0, 1.0, -1.0, -1.0]}, index=common_idx)
        sig = pd.DataFrame({"A0": [0.0, 1.0, -1.0, -1.0]}, index=common_idx)
        return score, sig

    def _fake_simulate(
        *,
        common_idx,
        exec_by_code,
        w_eff,
        cost_by_symbol,
        mults,
        margin_rate_frac,
        reserve_ratio,
        initial_equity_cny,
        exec_price,
        position_sizing,
        codes_sorted,
    ):
        captured["w_eff"] = w_eff.copy()
        return (
            pd.Series(1.0, index=pd.DatetimeIndex(common_idx), dtype=float),
            {"closed_trades": []},
        )

    monkeypatch.setattr(fut_trend, "_align_futures_trend_inputs", _fake_align)
    monkeypatch.setattr(fut_trend, "get_futures_pool_by_code", _fake_pool)
    monkeypatch.setattr(fut_trend, "build_ma_panels", _fake_build_panels)
    monkeypatch.setattr(fut_trend, "simulate_discrete_lot_portfolio", _fake_simulate)

    group = FuturesGroupData(name="G", codes=["A0"], is_active=True)
    out = fut_trend.compute_futures_group_trend_backtest(
        db=None,
        group=group,
        start="20240101",
        end="20240131",
        dynamic_universe=True,
        exec_price="close",
        trend_strategy="ma_cross",
        fast_ma=2,
        slow_ma=3,
        min_points=2,
        cost_bps=0.0,
        fee_side="one_way",
        slippage_type="percent",
        slippage_value=0.0,
        slippage_side="one_way",
    )
    assert out["ok"] is True
    assert out["meta"]["signal_execution_rule"] == "signal_t_execute_t_plus_1_close"
    assert out["meta"]["signal_lag_trading_days"] == 1
    w_eff = captured["w_eff"]
    assert [float(x) for x in w_eff["A0"].tolist()] == [0.0, 0.0, 1.0, -1.0]


def test_futures_trend_atr_stop_keeps_nav_on_lot_account_basis(monkeypatch) -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    exec_map = {"A0": _mk_exec_frame(idx, 100.0)}

    def _fake_align(_db, *, pool_code: str, start: str, end: str):
        df = exec_map[str(pool_code)].copy()
        return df, df.copy(), {}

    def _fake_pool(_db, code: str):
        return SimpleNamespace(contract_multiplier=10.0, min_price_tick=1.0, code=code)

    def _fake_build_panels(*args, **kwargs):
        common_idx = pd.DatetimeIndex(kwargs["common_idx"])
        score = pd.DataFrame({"A0": [0.0, 1.0, 1.0, 1.0]}, index=common_idx)
        sig = pd.DataFrame({"A0": [0.0, 1.0, 1.0, 1.0]}, index=common_idx)
        return score, sig

    def _fake_apply_atr_stop(
        base_pos,
        *,
        open_,
        close,
        high,
        low,
        mode,
        atr_basis,
        reentry_mode,
        atr_window,
        n_mult,
        m_step,
        same_day_stop,
    ):
        _ = (
            open_,
            close,
            high,
            low,
            mode,
            atr_basis,
            reentry_mode,
            atr_window,
            n_mult,
            m_step,
            same_day_stop,
        )
        return (
            base_pos,
            {
                "trigger_events": [
                    {
                        "date": "2024-01-03",
                        "execution_date": "2024-01-03",
                        "reduce_fraction": 1.0,
                        "fill_price": 90.0,
                    }
                ]
            },
        )

    def _fake_apply_intraday_stop_execution_portfolio(
        *,
        weights,
        atr_stop_by_asset,
        exec_price,
        open_sig_df,
        close_sig_df,
        stop_execution_mode=None,
    ):
        _ = (
            atr_stop_by_asset,
            exec_price,
            open_sig_df,
            close_sig_df,
            stop_execution_mode,
        )
        return (
            weights.copy(),
            pd.Series(0.5, index=weights.index, dtype=float),
        )

    def _should_not_call_lot_delta(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("post-hoc lot delta adjustment should not be called")

    def _fake_simulate(
        *,
        common_idx,
        exec_by_code,
        w_eff,
        cost_by_symbol,
        mults,
        margin_rate_frac,
        reserve_ratio,
        initial_equity_cny,
        exec_price,
        position_sizing,
        codes_sorted,
    ):
        _ = (
            exec_by_code,
            w_eff,
            cost_by_symbol,
            mults,
            margin_rate_frac,
            reserve_ratio,
            initial_equity_cny,
            exec_price,
            position_sizing,
            codes_sorted,
        )
        # Constant equity => NAV must stay flat at 1.0.
        return (
            pd.Series(1_000_000.0, index=pd.DatetimeIndex(common_idx), dtype=float),
            {"closed_trades": []},
        )

    monkeypatch.setattr(fut_trend, "_align_futures_trend_inputs", _fake_align)
    monkeypatch.setattr(fut_trend, "get_futures_pool_by_code", _fake_pool)
    monkeypatch.setattr(fut_trend, "build_ma_panels", _fake_build_panels)
    monkeypatch.setattr(fut_trend, "_apply_atr_stop", _fake_apply_atr_stop)
    monkeypatch.setattr(
        fut_trend,
        "_apply_intraday_stop_execution_portfolio",
        _fake_apply_intraday_stop_execution_portfolio,
    )
    monkeypatch.setattr(
        fut_trend, "_lot_engine_stop_override_delta", _should_not_call_lot_delta
    )
    monkeypatch.setattr(fut_trend, "simulate_discrete_lot_portfolio", _fake_simulate)

    group = FuturesGroupData(name="G", codes=["A0"], is_active=True)
    out = fut_trend.compute_futures_group_trend_backtest(
        db=None,
        group=group,
        start="20240101",
        end="20240131",
        dynamic_universe=True,
        exec_price="close",
        trend_strategy="ma_cross",
        fast_ma=2,
        slow_ma=3,
        min_points=2,
        cost_bps=0.0,
        fee_side="one_way",
        slippage_type="percent",
        slippage_value=0.0,
        slippage_side="one_way",
        atr_stop_mode="static",
    )
    assert out["ok"] is True
    nav_vals = [
        float(x["value"]) for x in (out.get("series") or {}).get("strategy_nav", [])
    ]
    assert nav_vals and all(abs(v - 1.0) < 1e-12 for v in nav_vals)
    port_meta = (out.get("meta") or {}).get("portfolio") or {}
    assert port_meta.get("intraday_stop_same_day_return_replacement") == "disabled"
