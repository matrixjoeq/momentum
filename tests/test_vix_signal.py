import datetime as dt

import pandas as pd

from etf_momentum.strategy.vix_signal import VixSignalInputs, generate_next_action


def test_vix_signal_buy_sell_hold():
    # Build a tiny US close series with known returns:
    # US dates: 2024-01-02 close 10, 2024-01-03 close 11 (up), 2024-01-04 close 10 (down)
    us_dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    s = pd.Series([10.0, 11.0, 10.0], index=us_dates)

    # Use no alignment to keep dates simple; target is last date.
    out = generate_next_action(
        VixSignalInputs(
            index_close_us=s,
            index="VIX",
            index_align="none",
            current_position="cash",
            lookback_window=20,
            threshold_quantile=0.0,  # effectively allow any move
            min_abs_ret=0.0,
            target_cn_trade_date=dt.date(2024, 1, 4),
        )
    )
    assert out["ok"] is True
    # last return is down => risk-on => target long => BUY from cash
    assert out["action"] == "BUY"
    assert out["target_position"] == "long"

