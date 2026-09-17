import pandas as pd
import pytest

from etf_momentum.analysis import bt_trend, trend
from etf_momentum.analysis.baseline import hfq_close_daily_equal_weight_returns
from etf_momentum.analysis.r_multiple import enrich_trades_with_r_metrics


@pytest.mark.parametrize(
    "ledger_builder",
    [
        trend._trade_returns_from_weight_series,
        bt_trend._trade_returns_from_weight_series,
    ],
)
def test_close_trade_ledger_uses_return_weight_and_risk_exit_override(
    ledger_builder,
) -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    effective_weight = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates)
    close_return_weight = pd.Series([0.0, 0.0, 1.0, 1.0], index=dates)
    close_to_close_return = pd.Series([0.0, 0.5, 0.1, -0.2], index=dates)
    risk_exit_override = pd.Series([0.0, 0.0, 0.0, 0.03], index=dates)
    prices = pd.Series([10.0, 15.0, 16.5, 13.2], index=dates)

    ledger = ledger_builder(
        effective_weight,
        close_to_close_return,
        return_weight=close_return_weight,
        return_override=risk_exit_override,
        cost_bps=0.0,
        slippage_rate=0.0,
        exec_price=prices,
        dates=dates,
    )

    # Entry close return is excluded; the held +10% day and exit -17% day
    # (including the +3% risk-fill correction) belong to the closed episode.
    assert ledger["returns"] == pytest.approx([(1.1 * 0.83) - 1.0], abs=1e-12)
    assert ledger["closed_returns"] == pytest.approx([(1.1 * 0.83) - 1.0], abs=1e-12)
    assert ledger["open_mtm_returns"] == []
    assert ledger["trades"][0]["closed"] is True


@pytest.mark.parametrize(
    "ledger_builder",
    [
        trend._trade_returns_from_weight_series,
        bt_trend._trade_returns_from_weight_series,
    ],
)
def test_open_trade_ledger_includes_entry_day_and_excludes_exit_day(
    ledger_builder,
) -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    weights = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates)
    returns = pd.Series([0.0, 0.1, 0.2, 0.3], index=dates)

    ledger = ledger_builder(
        weights,
        returns,
        return_weight=weights,
        return_override=pd.Series(0.0, index=dates),
        cost_bps=0.0,
        slippage_rate=0.0,
        exec_price=pd.Series([10.0, 11.0, 13.2, 17.16], index=dates),
        dates=dates,
    )

    expected = (1.0 + 0.1) * (1.0 + 0.2) - 1.0
    assert ledger["closed_returns"] == pytest.approx([expected])
    assert ledger["trades"][0]["entry_date"] == dates[1].date().isoformat()
    assert ledger["trades"][0]["exit_date"] == dates[3].date().isoformat()


@pytest.mark.parametrize(
    "ledger_builder",
    [
        trend._trade_returns_from_weight_series,
        bt_trend._trade_returns_from_weight_series,
    ],
)
def test_trade_statistics_returns_exclude_open_mtm_episodes(ledger_builder) -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    weights = pd.Series([0.0, 1.0, 1.0], index=dates)
    returns = pd.Series([0.0, 0.1, 0.2], index=dates)
    prices = pd.Series([10.0, 11.0, 13.2], index=dates)

    ledger = ledger_builder(
        weights,
        returns,
        return_weight=weights,
        return_override=pd.Series(0.0, index=dates),
        cost_bps=0.0,
        slippage_rate=0.0,
        exec_price=prices,
        dates=dates,
    )

    assert ledger["returns"] == []
    assert ledger["closed_returns"] == []
    assert ledger["open_mtm_returns"] == pytest.approx([0.32], abs=1e-12)
    assert ledger["trades"][0]["closed"] is False


@pytest.mark.parametrize(
    "ledger_builder",
    [
        trend._trade_returns_from_weight_series,
        bt_trend._trade_returns_from_weight_series,
    ],
)
def test_slippage_rate_is_full_absolute_spread_split_across_sides(
    ledger_builder,
) -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    weights = pd.Series([0.0, 1.0, 0.0], index=dates)
    prices = pd.Series([2.0, 2.0, 2.0], index=dates)

    ledger = ledger_builder(
        weights,
        pd.Series(0.0, index=dates),
        return_weight=weights,
        return_override=pd.Series(0.0, index=dates),
        cost_bps=0.0,
        slippage_rate=0.01,
        exec_price=prices,
        dates=dates,
    )

    assert ledger["returns"] == pytest.approx([(1.0 - 0.0025) ** 2 - 1.0])
    trade = ledger["trades"][0]
    assert trade["entry_price"] == pytest.approx(2.005)
    assert trade["exit_price"] == pytest.approx(1.995)


def test_r_statistics_exclude_open_mtm_episode() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    trades = [
        {
            "code": "AAA",
            "entry_date": dates[0].date().isoformat(),
            "exit_date": dates[1].date().isoformat(),
            "return": 0.1,
            "closed": True,
        },
        {
            "code": "AAA",
            "entry_date": dates[1].date().isoformat(),
            "exit_date": dates[2].date().isoformat(),
            "return": -0.2,
            "closed": False,
        },
    ]
    nav = pd.Series([1.0, 1.1, 0.88], index=dates)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 8.8]}, index=dates)
    atr = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)

    result = enrich_trades_with_r_metrics(
        trades,
        nav=nav,
        weights=weights,
        exec_price=prices,
        atr=atr,
        atr_mult=2.0,
        risk_budget_pct=0.01,
        cost_bps=0.0,
        slippage_rate=0.0,
    )

    assert len(result["trades"]) == 2
    assert result["statistics"]["scope"] == "closed_trades_only"
    assert result["statistics"]["overall"]["trade_count"] == 1
    assert result["statistics"]["open_mtm_trade_count"] == 1


def test_portfolio_risk_exit_override_is_preserved_by_asset() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="B")
    weights = pd.DataFrame(
        {"AAA": [0.0, 1.0, 1.0], "BBB": [0.0, 1.0, 1.0]}, index=dates
    )
    prices = pd.DataFrame(100.0, index=dates, columns=weights.columns)
    event_date = dates[2].date().isoformat()
    event_sets = {
        "atr_stop": {
            "AAA": {
                "execution_mode": "intraday",
                "trigger_events": [
                    {
                        "date": event_date,
                        "execution_mode": "intraday",
                        "execution_time": "close",
                        "fill_price": 95.0,
                        "reduce_fraction": 1.0,
                        "trigger_kind": "exit",
                    }
                ],
            },
            "BBB": {"execution_mode": "intraday", "trigger_events": []},
        }
    }

    _, overrides = trend._apply_intraday_or_arbitration_portfolio(
        weights=weights,
        event_sets_by_asset=event_sets,
        exec_price="close",
        open_sig_df=prices,
        close_sig_df=prices,
    )

    by_asset = overrides["_by_asset"]["atr_stop"]
    assert float(by_asset["BBB"].abs().sum()) == 0.0
    assert float(by_asset["AAA"].abs().sum()) > 0.0
    pd.testing.assert_series_equal(
        by_asset.sum(axis=1),
        overrides["atr_stop"],
        check_names=False,
        atol=1e-12,
        rtol=0.0,
    )


def test_dynamic_benchmark_does_not_forward_fill_missing_observations() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    close = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 133.1],
            "BBB": [100.0, 120.0, float("nan"), 180.0],
        },
        index=dates,
    )

    returns = hfq_close_daily_equal_weight_returns(close, dynamic_universe=True)

    assert float(returns.loc[dates[1]]) == pytest.approx(0.15)
    assert float(returns.loc[dates[2]]) == pytest.approx(0.10)
    # BBB has no valid adjacent close pair after its gap, so only AAA participates.
    assert float(returns.loc[dates[3]]) == pytest.approx(0.10)


def test_missing_ohlc_does_not_fabricate_intraday_stop() -> None:
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    base = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=dates)
    close = pd.Series([100.0, 100.0, 110.0, float("nan"), 110.0], index=dates)
    high = pd.Series([101.0, 101.0, 111.0, float("nan"), 111.0], index=dates)
    low = pd.Series([99.0, 99.0, 109.0, float("nan"), 109.0], index=dates)

    out, stats = trend._apply_atr_stop(
        base,
        open_=close,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=1,
        n_mult=1.0,
        m_step=0.5,
    )

    assert stats["trigger_count"] == 0
    assert float(out.loc[dates[3]]) == pytest.approx(1.0)
