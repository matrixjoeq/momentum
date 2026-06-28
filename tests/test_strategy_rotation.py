import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import (
    TRADING_COST_PROXY_CODE,
    RotationInputs,
    backtest_rotation,
)
from tests.helpers.price_seed import add_price_all_adjustments


def test_backtest_rotation_basic_outputs(session_factory):
    sf = session_factory
    with sf() as db:
        # create minimal none/hfq/qfq prices for two codes over 40 days
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(50)]
        for i, d in enumerate(dates):
            # AAA trends up, BBB flat
            for adj in ("hfq", "qfq", "none"):
                db.add(
                    EtfPrice(
                        code="AAA",
                        trade_date=d,
                        close=100 + i,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
                db.add(
                    EtfPrice(
                        code="BBB",
                        trade_date=d,
                        close=100,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=10,
                skip_days=0,
                cost_bps=0.0,
            ),
        )

    assert out["nav"]["series"]["ROTATION"][0] == pytest.approx(1.0)
    assert "EW_REBAL" in out["nav"]["series"]
    assert "EXCESS" in out["nav"]["series"]
    assert out["price_basis"]["signal"] == "qfq"
    assert "none" in out["price_basis"]["strategy_nav"]
    assert out["price_basis"]["benchmark_nav"] == "hfq"
    assert out["win_payoff"]["rebalance"] == "monthly"
    assert "kelly_fraction" in out["win_payoff"]
    assert "abs_kelly_fraction" in out["win_payoff"]
    assert "strategy" in out["metrics"]
    assert "avg_daily_turnover" in out["metrics"]["strategy"]
    assert "avg_annual_turnover" in out["metrics"]["strategy"]
    assert "avg_daily_trade_count" in out["metrics"]["strategy"]
    assert "avg_annual_trade_count" in out["metrics"]["strategy"]
    assert "excess_vs_equal_weight" in out["metrics"]
    assert "period_returns" in out
    assert "weekly" in out["period_returns"]
    assert "current_holdings" in out
    assert isinstance(out.get("current_holdings"), list)
    assert "event_study" in out
    assert (out.get("market_regime") or {}).get("enabled") is True
    assert "strategy_by_dominant_state" in (out.get("market_regime") or {})
    assert set((out["event_study"] or {}).get("windows", {}).keys()) >= {
        "1d",
        "5d",
        "10d",
        "20d",
    }
    ev1 = ((out.get("event_study") or {}).get("windows") or {}).get("1d") or {}
    assert "profit_frequency" in (ev1.get("signal") or {})
    assert "bucket_probabilities" in (ev1.get("signal") or {})
    assert "bucket_profiles" in (ev1.get("signal") or {})
    assert "profit_frequency_mean" in (ev1.get("random_baseline") or {})
    assert "bucket_profiles_mean" in (ev1.get("random_baseline") or {})
    assert "delta_profit_frequency" in (ev1.get("comparison") or {})
    assert "delta_bucket_profiles" in (ev1.get("comparison") or {})
    assert "rolling" in out
    assert "returns" in out["rolling"]
    assert "corporate_actions" in out
    if out["period_details"]:
        assert "buys" in out["period_details"][0]
        assert "sells" in out["period_details"][0]


def test_rotation_close_exec_uses_forward_corp_action_fallback(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(8)]
        # none has a split-like cliff between day2->day3; hfq remains smooth.
        none_px = [100.0, 101.0, 102.0, 10.2, 10.3, 10.4, 10.5, 10.6]
        hfq_px = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        for d, p_none, p_hfq in zip(dates, none_px, hfq_px):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    close=float(p_none),
                    source="eastmoney",
                    adjust="none",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    close=float(p_hfq),
                    source="eastmoney",
                    adjust="hfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    close=float(p_hfq),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=[code],
                start=start,
                end=dates[-1],
                rebalance="daily",
                top_k=1,
                lookback_days=1,
                skip_days=0,
                exec_price="close",
                cost_bps=0.0,
            ),
        )

    nav = [float(x) for x in out["nav"]["series"]["ROTATION"]]
    assert nav
    # If fallback is one-day late, nav around split day collapses ~90%.
    assert min(nav) > 0.90


def test_rotation_trade_statistics_have_samples_user_case_like(session_factory):
    """
    User-case-like trade-stats regression (compressed vs prod-scale dates to keep CI fast):
    - multi assets with staggered drift so top-2 stay rankable
    - weekly rotation, Monday anchor, close execution
    - top2 adaptive, dynamic universe on, no entry/exit filters
    - cost 2bps + slippage 0.001

    Same synthetic idea as before: persistent leaders; needs end-of-backtest trade closure
    for non-empty closed-trade stats.
    """
    sf = session_factory
    start = dt.date(2022, 1, 4)
    end = dt.date(2023, 10, 31)
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    codes = [f"G{i:02d}" for i in range(1, 7)]
    with sf() as db:
        n = max(1, len(dates) - 1)
        for k, code in enumerate(codes):
            drift = 0.00010 + 0.00003 * float(
                k
            )  # higher-index codes have stronger trend
            for i, d in enumerate(dates):
                px = 100.0 * ((1.0 + drift) ** (float(i) / float(n) * float(n)))
                add_price_all_adjustments(
                    db,
                    code=code,
                    day=d,
                    close=float(px),
                    open_price=float(px),
                    high=float(px),
                    low=float(px),
                )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=end,
                dynamic_universe=True,
                rebalance="weekly",
                rebalance_anchor=1,
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                position_mode="adaptive",
                entry_backfill=False,
                score_method="raw_mom",
                lookback_days=20,
                skip_days=0,
                cost_bps=2.0,
                slippage_rate=0.001,
                atr_stop_mode="none",
                group_enforce=False,
                trend_filter=False,
                trend_exit_filter=False,
                bias_filter=False,
                bias_exit_filter=False,
                rsi_filter=False,
                chop_filter=False,
            ),
        )
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = ts.get("by_code") or {}
    assert int(overall.get("total_trades") or 0) > 0
    assert "holding_bias_v_ge_5_count" in overall
    assert "holding_bias_v_ge_5_max_per_holding" in overall
    assert "holding_bias_v_ge_5_per_holding_distribution" in overall
    assert any(int((v or {}).get("total_trades") or 0) > 0 for v in by_code.values())
    for one in by_code.values():
        assert "holding_bias_v_ge_5_count" in (one or {})
        assert "holding_bias_v_ge_5_max_per_holding" in (one or {})
        assert "holding_bias_v_ge_5_per_holding_distribution" in (one or {})
    rs = out.get("r_statistics") or {}
    assert int(((rs.get("overall") or {}).get("trade_count") or 0) > 0)
    assert "recent_100" in rs
    assert int(((rs.get("recent_100") or {}).get("effective_count") or 0) > 0)
    assert "sqn" in (rs.get("overall") or {})
    score_pack = rs.get("trade_system_score") or {}
    assert "overall" in score_pack
    assert "weights" in score_pack
    first_trade = (ts.get("trades") or [None])[0] or {}
    assert "initial_r_amount" in first_trade
    assert "r_multiple" in first_trade


def test_rotation_risk_budget_position_mode_scales_by_atr(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + float(i) * 0.8
            p2 = 90.0 + float(i) * 0.6
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(p1),
                open_price=float(p1),
                high=float(p1 * 1.01),
                low=float(p1 * 0.99),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(p2),
                open_price=float(p2),
                high=float(p2 * 1.01),
                low=float(p2 * 0.99),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                top_k=2,
                position_mode="risk_budget",
                risk_budget_atr_window=20,
                risk_budget_pct=0.01,
                lookback_days=10,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    assert str(out.get("position_mode") or "") == "risk_budget"
    w = pd.DataFrame((out.get("weights") or {}).get("series") or {})
    if not w.empty:
        expo = w.sum(axis=1)
        assert float(expo.max()) <= 1.0000001
        assert float(expo.max()) > 0.0


def test_rotation_cash_management_uses_511880_qfq(session_factory):
    sf = session_factory
    cash_code = "511880"
    start = dt.date(2024, 1, 2)
    dates = [d.date() for d in pd.date_range(start, periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + float(i) * 0.6
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(px),
                open_price=float(px),
                high=float(px * 1.10),
                low=float(px * 0.90),
            )
            cash_px = 100.0 + float(i) * 0.04
            add_price_all_adjustments(
                db,
                code=cash_code,
                day=d,
                close=float(cash_px),
                open_price=float(cash_px),
                high=float(cash_px),
                low=float(cash_px),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                top_k=1,
                position_mode="risk_budget",
                risk_budget_atr_window=20,
                risk_budget_pct=0.001,
                lookback_days=10,
                skip_days=0,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    m = ((out.get("metrics") or {}).get("strategy")) or {}
    assert str(m.get("cash_management_proxy_code") or "") == cash_code
    assert bool(m.get("cash_management_data_available")) is True
    assert float(m.get("cash_management_return_contribution") or 0.0) > 0.0

    decomp = out.get("return_decomposition") or {}
    series = decomp.get("series") or {}
    cash_series = [float(x) for x in (series.get("cash_management") or [])]
    overnight = [float(x) for x in (series.get("overnight") or [])]
    intraday = [float(x) for x in (series.get("intraday") or [])]
    interaction = [float(x) for x in (series.get("interaction") or [])]
    gross = [float(x) for x in (series.get("gross") or [])]
    net = [float(x) for x in (series.get("net") or [])]
    cost = [float(x) for x in (series.get("cost") or [])]
    assert any(abs(x) > 0.0 for x in cash_series)
    for i in range(len(gross)):
        assert gross[i] == pytest.approx(
            overnight[i] + intraday[i] + interaction[i] + cash_series[i], abs=1e-12
        )
        assert net[i] == pytest.approx(gross[i] - cost[i], abs=1e-12)

    attr = out.get("attribution") or {}
    ret_codes = {
        str(x.get("code") or "")
        for x in (((attr.get("return") or {}).get("by_code")) or [])
    }
    risk_codes = {
        str(x.get("code") or "")
        for x in (((attr.get("risk") or {}).get("by_code")) or [])
    }
    assert "AAA" in ret_codes
    assert "AAA" in risk_codes
    assert cash_code in ret_codes
    assert cash_code in risk_codes


def test_rotation_inverse_vol_position_mode_overweights_lower_vol_asset(
    session_factory,
):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + float(i) * 0.7 + (3.0 if (i % 2 == 0) else -3.0)
            p2 = 95.0 + float(i) * 0.7 + (0.8 if (i % 2 == 0) else -0.8)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(p1),
                open_price=float(p1),
                high=float(p1 * 1.01),
                low=float(p1 * 0.99),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(p2),
                open_price=float(p2),
                high=float(p2 * 1.01),
                low=float(p2 * 0.99),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                top_k=2,
                position_mode="inverse_vol",
                vol_window=20,
                lookback_days=10,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    assert str(out.get("position_mode") or "") == "inverse_vol"
    iv_holds = [
        h
        for h in (out.get("holdings") or [])
        if bool(
            (((h.get("risk_controls") or {}).get("inverse_vol") or {}).get("enabled"))
        )
    ]
    assert iv_holds
    iv_meta = (iv_holds[0].get("risk_controls") or {}).get("inverse_vol") or {}
    assert int(iv_meta.get("vol_window") or 0) == 20
    by_code = iv_meta.get("by_code") or {}
    if "AAA" in by_code and "BBB" in by_code:
        inv_a = float((by_code.get("AAA") or {}).get("inv_vol_raw") or 0.0)
        inv_b = float((by_code.get("BBB") or {}).get("inv_vol_raw") or 0.0)
        if inv_a > 0.0 and inv_b > 0.0:
            assert inv_b > inv_a


def test_rotation_daily_rebalance_increases_turnover_and_cost_drag(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=90, freq="B")]
    with sf() as db:
        px_a = 100.0
        px_b = 100.0
        for i, d in enumerate(dates):
            # Alternating relative moves create persistent drift between assets.
            px_a *= 1.01 if (i % 2 == 0) else 0.99
            px_b *= 0.99 if (i % 2 == 0) else 1.01
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(px_a),
                open_price=float(px_a),
                high=float(px_a),
                low=float(px_a),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(px_b),
                open_price=float(px_b),
                high=float(px_b),
                low=float(px_b),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="weekly",
            rebalance_anchor=1,
            top_k=2,
            position_mode="adaptive",
            lookback_days=5,
            skip_days=0,
            exec_price="close",
            cost_bps=100.0,
            slippage_rate=0.0,
        )
        out_base = backtest_rotation(db, RotationInputs(**base))
        out_daily = backtest_rotation(
            db, RotationInputs(**{**base, "daily_rebalance": True})
        )
    m_base = (out_base.get("metrics") or {}).get("strategy") or {}
    m_daily = (out_daily.get("metrics") or {}).get("strategy") or {}
    assert bool(out_base.get("daily_rebalance")) is False
    assert bool(out_daily.get("daily_rebalance")) is True
    assert float(m_daily.get("avg_daily_turnover") or 0.0) > float(
        m_base.get("avg_daily_turnover") or 0.0
    )
    assert float(m_daily.get("cumulative_return") or 0.0) < float(
        m_base.get("cumulative_return") or 0.0
    )


def test_rotation_daily_rebalance_respects_exec_price_mode(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            c1 = 100.0 + 0.7 * i
            c2 = 110.0 - 0.4 * i
            o1 = c1 * (0.99 if (i % 2 == 0) else 1.01)
            o2 = c2 * (1.01 if (i % 2 == 0) else 0.99)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(c1),
                open_price=float(o1),
                high=float(max(o1, c1)),
                low=float(min(o1, c1)),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(c2),
                open_price=float(o2),
                high=float(max(o2, c2)),
                low=float(min(o2, c2)),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="weekly",
            rebalance_anchor=1,
            top_k=2,
            position_mode="adaptive",
            daily_rebalance=True,
            lookback_days=5,
            skip_days=0,
            cost_bps=20.0,
            slippage_rate=0.001,
        )
        out_open = backtest_rotation(
            db, RotationInputs(**{**base, "exec_price": "open"})
        )
        out_close = backtest_rotation(
            db, RotationInputs(**{**base, "exec_price": "close"})
        )
    nav_open = float(
        ((out_open.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    nav_close = float(
        ((out_close.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    assert nav_open != pytest.approx(nav_close, rel=0.0, abs=1e-10)


@pytest.mark.parametrize("position_mode", ["inverse_vol", "risk_budget"])
def test_rotation_daily_rebalance_updates_dynamic_position_modes(
    session_factory, position_mode: str
):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=120, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            # Regime-shifting volatility profile to force dynamic sizing changes.
            amp_a = 5.0 if i < 60 else 1.0
            amp_b = 1.0 if i < 60 else 5.0
            p1 = 100.0 + 0.45 * i + (amp_a if (i % 2 == 0) else -amp_a)
            p2 = 105.0 + 0.20 * i + (amp_b if (i % 2 == 0) else -amp_b)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(p1),
                open_price=float(p1),
                high=float(p1 * 1.02),
                low=float(p1 * 0.98),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(p2),
                open_price=float(p2),
                high=float(p2 * 1.02),
                low=float(p2 * 0.98),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="monthly",
            rebalance_anchor=5,
            top_k=2,
            position_mode=position_mode,
            lookback_days=10,
            skip_days=0,
            exec_price="close",
            cost_bps=0.0,
            slippage_rate=0.0,
            risk_budget_atr_window=20,
            risk_budget_pct=0.01,
            vol_window=20,
        )
        out_base = backtest_rotation(db, RotationInputs(**base))
        out_daily = backtest_rotation(
            db, RotationInputs(**{**base, "daily_rebalance": True})
        )
    m_base = (out_base.get("metrics") or {}).get("strategy") or {}
    m_daily = (out_daily.get("metrics") or {}).get("strategy") or {}
    nav_base = float(
        ((out_base.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    nav_daily = float(
        ((out_daily.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    assert float(m_daily.get("avg_daily_turnover") or 0.0) > float(
        m_base.get("avg_daily_turnover") or 0.0
    )
    assert nav_daily != pytest.approx(nav_base, rel=0.0, abs=1e-10)


@pytest.mark.parametrize("position_mode", ["inverse_vol", "risk_budget"])
def test_rotation_daily_rebalance_recalculates_within_single_segment(
    session_factory, position_mode: str
):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=75, freq="B")]
    with sf() as db:
        px_a = 100.0
        px_b = 100.0
        for i, d in enumerate(dates):
            # One annual decision is anchored inside the sample window (day 40),
            # while the next yearly decision is out of range. Early days keep both
            # assets similar; later days force volatility divergence.
            if i < 25:
                r_a = 1.002 if (i % 2 == 0) else 0.998
                r_b = 1.002 if (i % 2 == 0) else 0.998
            else:
                r_a = 1.020 if (i % 2 == 0) else 0.980
                r_b = 1.004 if (i % 2 == 0) else 0.996
            px_a *= float(r_a)
            px_b *= float(r_b)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(px_a),
                open_price=float(px_a),
                high=float(px_a * 1.01),
                low=float(px_a * 0.99),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(px_b),
                open_price=float(px_b),
                high=float(px_b * 1.01),
                low=float(px_b * 0.99),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="yearly",
            rebalance_anchor=40,
            top_k=2,
            position_mode=position_mode,
            lookback_days=5,
            skip_days=0,
            exec_price="close",
            cost_bps=0.0,
            slippage_rate=0.0,
            vol_window=20,
            risk_budget_atr_window=20,
            risk_budget_pct=0.01,
        )
        out_base = backtest_rotation(db, RotationInputs(**base))
        out_daily = backtest_rotation(
            db, RotationInputs(**{**base, "daily_rebalance": True})
        )
    nav_base = float(
        ((out_base.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    nav_daily = float(
        ((out_daily.get("nav") or {}).get("series") or {}).get("ROTATION")[-1]
    )
    m_base = (out_base.get("metrics") or {}).get("strategy") or {}
    m_daily = (out_daily.get("metrics") or {}).get("strategy") or {}
    # With zero trading cost/slippage and a single rebalance decision in-range,
    # fixed decision-date targets would produce identical NAV. A difference here
    # confirms daily_rebalance recomputes dynamic position-mode targets each day.
    assert nav_daily != pytest.approx(nav_base, rel=0.0, abs=1e-10)
    assert float(m_daily.get("avg_daily_turnover") or 0.0) > float(
        m_base.get("avg_daily_turnover") or 0.0
    )


def test_rotation_attribution_net_consistency_with_cost_proxy(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    with sf() as db:
        px_a = 100.0
        px_b = 100.0
        for i, d in enumerate(dates):
            # Alternate leader to force frequent turnover and non-trivial cost drag.
            ra = 1.010 if (i % 2 == 0) else 0.992
            rb = 0.992 if (i % 2 == 0) else 1.010
            px_a *= float(ra)
            px_b *= float(rb)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(px_a),
                open_price=float(px_a),
                high=float(px_a * 1.01),
                low=float(px_a * 0.99),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(px_b),
                open_price=float(px_b),
                high=float(px_b * 1.01),
                low=float(px_b * 0.99),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="daily",
                top_k=1,
                lookback_days=1,
                skip_days=0,
                exec_price="close",
                cost_bps=20.0,
                slippage_rate=0.001,
            ),
        )

    m = ((out.get("metrics") or {}).get("strategy")) or {}
    attr_ret = ((out.get("attribution") or {}).get("return")) or {}
    rows = list(attr_ret.get("by_code") or [])
    by_code = {str(r.get("code") or ""): r for r in rows}
    total_contrib = sum(float((r.get("return_contribution") or 0.0)) for r in rows)

    assert float(attr_ret.get("total_return") or 0.0) == pytest.approx(
        float(m.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(total_contrib) == pytest.approx(
        float(m.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert TRADING_COST_PROXY_CODE in by_code
    assert (
        float(
            (by_code[TRADING_COST_PROXY_CODE] or {}).get("return_contribution") or 0.0
        )
        < 0.0
    )


def test_rotation_daily_rebalance_with_rr_and_dd_controls(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=110, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + 0.30 * i + (2.0 if (i % 3 == 0) else -1.0)
            p2 = 95.0 + 0.15 * i + (1.5 if (i % 3 == 0) else -0.8)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(p1),
                open_price=float(p1),
                high=float(p1 * 1.01),
                low=float(p1 * 0.99),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(p2),
                open_price=float(p2),
                high=float(p2 * 1.01),
                low=float(p2 * 0.99),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="weekly",
            rebalance_anchor=1,
            top_k=2,
            position_mode="adaptive",
            lookback_days=5,
            skip_days=0,
            exec_price="close",
            cost_bps=80.0,
            slippage_rate=0.0,
            rr_sizing=True,
            rr_years=1.0,
            rr_thresholds=[0.0, 0.05],
            rr_weights=[1.0, 0.7, 0.4],
            dd_control=True,
            dd_threshold=0.08,
            dd_reduce=0.5,
            dd_sleep_days=10,
        )
        out_base = backtest_rotation(db, RotationInputs(**base))
        out_daily = backtest_rotation(
            db, RotationInputs(**{**base, "daily_rebalance": True})
        )
    m_base = (out_base.get("metrics") or {}).get("strategy") or {}
    m_daily = (out_daily.get("metrics") or {}).get("strategy") or {}
    assert float(m_daily.get("avg_daily_turnover") or 0.0) > float(
        m_base.get("avg_daily_turnover") or 0.0
    )
    assert float(m_daily.get("cumulative_return") or 0.0) < float(
        m_base.get("cumulative_return") or 0.0
    )


def test_rotation_negative_top_k_selects_lower_momentum_names(session_factory):
    """Inverse TopK: hold the lowest-momentum names (e.g. BBB vs AAA in a two-asset uptrend)."""
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=60, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(100.0 + 0.5 * i),
                open_price=float(100.0 + 0.5 * i),
                high=float(100.0 + 0.5 * i),
                low=float(100.0 + 0.5 * i),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(90.0 + 0.2 * i),
                open_price=float(90.0 + 0.2 * i),
                high=float(90.0 + 0.2 * i),
                low=float(90.0 + 0.2 * i),
            )
        db.commit()
        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="daily",
            top_k=1,
            lookback_days=5,
            skip_days=0,
            exec_price="close",
            cost_bps=0.0,
            slippage_rate=0.0,
        )
        out_top = backtest_rotation(db, RotationInputs(**base))
        out_inv = backtest_rotation(db, RotationInputs(**{**base, "top_k": -1}))

    periods_top = out_top.get("holdings") or []
    periods_inv = out_inv.get("holdings") or []
    assert periods_top and periods_inv
    # After warmup, picks should diverge: top-1 favors AAA, bottom-1 favors BBB.
    last_p_top = sorted([str(x) for x in (periods_top[-1].get("picks") or [])])
    last_p_inv = sorted([str(x) for x in (periods_inv[-1].get("picks") or [])])
    assert last_p_top == ["AAA"]
    assert last_p_inv == ["BBB"]


def test_rotation_top_k_zero_raises(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [start]
    with sf() as db:
        add_price_all_adjustments(
            db,
            code="AAA",
            day=dates[0],
            close=100.0,
            open_price=100.0,
            high=100.0,
            low=100.0,
        )
        db.commit()
        with pytest.raises(ValueError, match="non-zero"):
            backtest_rotation(
                db,
                RotationInputs(
                    codes=["AAA"],
                    start=dates[0],
                    end=dates[0],
                    rebalance="daily",
                    top_k=0,
                    lookback_days=1,
                    skip_days=0,
                    cost_bps=0.0,
                    slippage_rate=0.0,
                ),
            )


def test_rotation_topk_larger_than_pool_still_runs(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=60, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(100.0 + 0.5 * i),
                open_price=float(100.0 + 0.5 * i),
                high=float(100.0 + 0.5 * i),
                low=float(100.0 + 0.5 * i),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(90.0 + 0.2 * i),
                open_price=float(90.0 + 0.2 * i),
                high=float(90.0 + 0.2 * i),
                low=float(90.0 + 0.2 * i),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                top_k=10,  # larger than pool size
                lookback_days=5,
                skip_days=0,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    nav = (out.get("nav") or {}).get("series", {}).get("ROTATION", [])
    assert nav and float(nav[-1]) > 0.0


def test_rotation_floating_topk_selects_positive_excess_assets(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            add_price_all_adjustments(
                db,
                code="BENCH",
                day=d,
                close=float(100.0),
                open_price=float(100.0),
                high=float(100.0),
                low=float(100.0),
            )
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(100.0 + 0.6 * i),
                open_price=float(100.0 + 0.6 * i),
                high=float(100.0 + 0.6 * i),
                low=float(100.0 + 0.6 * i),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(100.0 - 0.2 * i),
                open_price=float(100.0 - 0.2 * i),
                high=float(100.0 - 0.2 * i),
                low=float(100.0 - 0.2 * i),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["BENCH", "AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="daily",
                top_k_mode="floating",
                floating_benchmark_code="BENCH",
                lookback_days=10,
                skip_days=0,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    periods = out.get("holdings") or []
    assert periods
    last_picks = sorted([str(x) for x in (periods[-1].get("picks") or [])])
    assert last_picks == ["AAA"]


def test_rotation_floating_topk_fallback_to_benchmark_and_anchor_start(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=80, freq="B")]
    bench_start = dates[20]
    with sf() as db:
        for i, d in enumerate(dates):
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(100.0),
                open_price=float(100.0),
                high=float(100.0),
                low=float(100.0),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(100.0 - 0.1 * i),
                open_price=float(100.0 - 0.1 * i),
                high=float(100.0 - 0.1 * i),
                low=float(100.0 - 0.1 * i),
            )
            if d >= bench_start:
                add_price_all_adjustments(
                    db,
                    code="BENCH",
                    day=d,
                    close=float(100.0 + 0.3 * i),
                    open_price=float(100.0 + 0.3 * i),
                    high=float(100.0 + 0.3 * i),
                    low=float(100.0 + 0.3 * i),
                )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["BENCH", "AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="daily",
                top_k_mode="floating",
                floating_benchmark_code="BENCH",
                lookback_days=10,
                skip_days=0,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    assert ((out.get("date_range") or {}).get("start")) == bench_start.strftime(
        "%Y%m%d"
    )
    periods = out.get("holdings") or []
    assert periods
    last_picks = sorted([str(x) for x in (periods[-1].get("picks") or [])])
    assert last_picks == ["BENCH"]


def test_rotation_event_study_counts_membership_switches(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        dates = [d.date() for d in pd.date_range("2020-01-01", "2021-12-31", freq="B")]
        a = 100.0
        b = 100.0
        for i, d in enumerate(dates):
            regime = (i // 35) % 2
            if regime == 0:
                a *= 1.006
                b *= 0.994
            else:
                a *= 0.994
                b *= 1.006
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(a),
                open_price=float(a),
                high=float(a),
                low=float(a),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(b),
                open_price=float(b),
                high=float(b),
                low=float(b),
            )
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                top_k=1,
                lookback_days=10,
                skip_days=0,
                score_method="raw_mom",
                cost_bps=0.0,
            ),
        )
    ev = out.get("event_study", {})
    assert int(ev.get("entry_count", 0)) >= 4
