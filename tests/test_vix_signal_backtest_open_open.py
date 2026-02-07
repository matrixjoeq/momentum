import datetime as dt

import pandas as pd

from etf_momentum.strategy.vix_signal import backtest_vix_next_day_signal


def test_vix_signal_backtest_open_open_does_not_eat_gap_before_entry():
    """
    Guardrail against implicit look-ahead / optimistic PnL:

    If we start in cash and the strategy flips to long on CN date T (trade at open(T)),
    it must NOT earn the close(T-1)->open(T) gap.

    Under open-to-open accounting, PnL on date T is open(T)->open(T+1).
    """
    dates = [d.date() for d in pd.date_range("2024-01-02", periods=40, freq="B")]
    start = dates[0]
    end = dates[-1]

    # ETF open has a big jump on the entry day; then it stays flat.
    # If the backtest wrongly applies the position to a return that includes the gap,
    # it would show a large profit. Correct open_open should not.
    jump_i = 22  # > 20 so rolling threshold is available
    etf_open = pd.Series([100.0] * len(dates), index=dates, dtype=float)
    etf_open.iloc[jump_i:] = 200.0
    etf_close = etf_open.copy()

    # Build a US close series that keeps the strategy in CASH before the jump day,
    # then flips to LONG exactly on the jump day:
    # - before jump day: idx_log_ret > 0 => target cash
    # - at jump day: idx_log_ret < 0 => BUY at open(jump day)
    us_dates = [d - dt.timedelta(days=1) for d in dates]
    vals = [100.0 + 0.1 * i for i in range(len(us_dates))]  # mostly increasing => cash
    # Force a one-day drop that maps to the jump CN date.
    # Mapping: CN date = US date + 1 (next trading day).
    drop_us_date = dates[jump_i] - dt.timedelta(days=1)
    drop_idx = us_dates.index(drop_us_date)
    vals[drop_idx] = vals[drop_idx - 1] - 1.0  # drop vs prev => negative idx_log_ret on jump day
    idx_close_us = pd.Series(vals, index=us_dates, dtype=float)

    out = backtest_vix_next_day_signal(
        etf_close_cn=etf_close,
        etf_open_cn=etf_open,
        index_close_us=idx_close_us,
        start=start,
        end=end,
        index="VXN",
        index_align="cn_next_trading_day",
        calendar="XSHG",
        lookback_window=20,
        threshold_quantile=0.01,
        min_abs_ret=0.0,
        trade_cost_bps=0.0,
        initial_nav=1.0,
        initial_position="cash",
        exec_model="open_open",
    )
    assert out["ok"] is True
    meta = out.get("meta") or {}
    assert (meta.get("alignment_check") or {}).get("ok") is True
    assert meta.get("exec_model") == "open_open"

    s = out.get("series") or {}
    nav = s.get("nav_strategy") or []
    assert len(nav) == len(dates)
    # The only big move is the gap at entry open; open_open should not profit from it.
    assert abs(float(nav[-1]) - 1.0) < 1e-9

    trades = out.get("trades") or []
    assert any(str(t.get("action")).upper() == "BUY" for t in trades)

