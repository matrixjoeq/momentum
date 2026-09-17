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


@pytest.mark.parametrize("dynamic_universe", [False, True])
def test_backtest_rotation_skips_untradable_candidates_regardless_dynamic_universe(
    session_factory, dynamic_universe: bool
):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB", "159985"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(60)]
        for i, d in enumerate(dates):
            for code, px in [("AAA", 100.0 + i), ("BBB", 90.0 + i * 0.8)]:
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
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=10,
                dynamic_universe=bool(dynamic_universe),
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    skipped = list(out.get("untradable_codes_skipped") or [])
    assert "159985" in skipped
    w_series = ((out.get("weights") or {}).get("series") or {}).get("159985") or []
    assert w_series
    assert all(float(x) == 0.0 for x in w_series)
    for one in out.get("period_details") or []:
        assert "159985" not in list((one or {}).get("picks") or [])


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
    assert rs.get("scope") == "closed_trades_only"
    assert int(rs.get("all_episode_count") or 0) > 0
    assert int((rs.get("overall") or {}).get("trade_count") or 0) == 0
    assert int(rs.get("open_mtm_trade_count") or 0) == int(
        rs.get("all_episode_count") or 0
    )
    assert "recent_100" in rs
    assert int((rs.get("recent_100") or {}).get("effective_count") or 0) == 0
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


def test_rotation_momentum_correction_reorders_ranking(session_factory):
    """Momentum correction multiplies base score by recent N-day return before ranking."""
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=40, freq="B")]
    aaa = [100.0 + 2.0 * i for i in range(len(dates))]
    bbb = [100.0 + 1.0 * i for i in range(len(dates))]
    # Keep medium-term momentum of AAA stronger, but force a short-term pullback
    # so correction term can flip the final ranking to BBB.
    aaa[-2] = 160.0
    aaa[-1] = 150.0
    with sf() as db:
        for d, pa, pb in zip(dates, aaa, bbb):
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa),
                low=float(pa),
            )
            add_price_all_adjustments(
                db,
                code="BBB",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb),
                low=float(pb),
            )
        db.commit()

        base = dict(
            codes=["AAA", "BBB"],
            start=dates[0],
            end=dates[-1],
            rebalance="daily",
            top_k=1,
            lookback_days=6,
            skip_days=4,
            exec_price="close",
            cost_bps=0.0,
            slippage_rate=0.0,
            score_method="raw_mom",
        )
        out_base = backtest_rotation(db, RotationInputs(**base))
        out_corr = backtest_rotation(
            db,
            RotationInputs(
                **{
                    **base,
                    "momentum_correction_enabled": True,
                    "momentum_correction_window": 3,
                }
            ),
        )

    periods_base = out_base.get("holdings") or []
    periods_corr = out_corr.get("holdings") or []
    assert periods_base and periods_corr
    last_base = sorted([str(x) for x in (periods_base[-1].get("picks") or [])])
    last_corr = sorted([str(x) for x in (periods_corr[-1].get("picks") or [])])
    assert last_base == ["AAA"]
    assert last_corr == ["BBB"]


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


def test_rotation_momentum_correction_window_range_validation(session_factory):
    sf = session_factory
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(6)]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + float(i)
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=px,
                open_price=px,
                high=px,
                low=px,
            )
        db.commit()
        with pytest.raises(ValueError, match="skip_days >= 3"):
            backtest_rotation(
                db,
                RotationInputs(
                    codes=["AAA"],
                    start=dates[0],
                    end=dates[-1],
                    rebalance="daily",
                    top_k=1,
                    lookback_days=2,
                    skip_days=2,
                    momentum_correction_enabled=True,
                    momentum_correction_window=3,
                    cost_bps=0.0,
                ),
            )
        with pytest.raises(ValueError, match="\\[3, skip_days\\]"):
            backtest_rotation(
                db,
                RotationInputs(
                    codes=["AAA"],
                    start=dates[0],
                    end=dates[-1],
                    rebalance="daily",
                    top_k=1,
                    lookback_days=2,
                    skip_days=4,
                    momentum_correction_enabled=True,
                    momentum_correction_window=5,
                    cost_bps=0.0,
                ),
            )


def test_rotation_bias_exit_rules_stack_partial_reduction(session_factory):
    sf = session_factory
    start = dt.date(2023, 1, 1)
    dates = [d.date() for d in pd.date_range(start, periods=140, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = float(100.0 * (1.02**i))
            add_price_all_adjustments(
                db,
                code="AAA",
                day=d,
                close=px,
                open_price=px,
                high=px,
                low=px,
            )
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=5,
                top_k=1,
                lookback_days=5,
                skip_days=3,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
                bias_exit_filter=True,
                asset_bias_rules=[
                    {
                        "code": "*",
                        "stage": "exit",
                        "bias_type": "bias",
                        "bias_ma_window": 20,
                        "threshold_type": "fixed",
                        "fixed_value": 10,
                        "op": ">",
                        "reduce_position_ratio": 0.5,
                    },
                    {
                        "code": "*",
                        "stage": "exit",
                        "bias_type": "bias",
                        "bias_ma_window": 20,
                        "threshold_type": "fixed",
                        "fixed_value": 15,
                        "op": ">",
                        "reduce_position_ratio": 0.5,
                    },
                    {
                        "code": "*",
                        "stage": "exit",
                        "bias_type": "bias",
                        "bias_ma_window": 60,
                        "threshold_type": "fixed",
                        "fixed_value": 20,
                        "op": ">",
                        "reduce_position_ratio": 0.5,
                    },
                ],
            ),
        )

    events = list(out.get("daily_exit_events") or [])
    assert events
    first = events[0]
    assert str(first.get("action") or "") == "partial_reduce"
    assert float(first.get("from_weight") or 0.0) == pytest.approx(1.0, abs=1e-12)
    assert float(first.get("to_weight") or 0.0) == pytest.approx(0.5, abs=1e-12)
    ratios = []
    for ev in events:
        fw = float(ev.get("from_weight") or 0.0)
        tw = float(ev.get("to_weight") or 0.0)
        if fw > 1e-12:
            ratios.append(tw / fw)
    assert any(r <= 0.125 + 1e-12 for r in ratios)
    hist = list(out.get("historical_trades") or [])
    partial_rows = [
        x for x in hist if str((x or {}).get("trade_action") or "") == "partial_reduce"
    ]
    assert partial_rows
    assert any(
        "乖离率退出" in str((x or {}).get("exit_reason") or "") for x in partial_rows
    )
    assert any(float((x or {}).get("reduce_ratio") or 0.0) > 0.0 for x in partial_rows)


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


def test_rotation_atr_scheme_requires_non_none_mode(session_factory):
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
        with pytest.raises(ValueError, match="stop_scheme=atr requires atr_stop_mode"):
            backtest_rotation(
                db,
                RotationInputs(
                    codes=["AAA"],
                    start=dates[0],
                    end=dates[0],
                    rebalance="daily",
                    top_k=1,
                    lookback_days=1,
                    skip_days=0,
                    cost_bps=0.0,
                    slippage_rate=0.0,
                    stop_scheme="atr",
                    atr_stop_mode="none",
                ),
            )


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


def test_rotation_atr_stop_exits_only_triggered_asset(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["STOPA", "STOPB"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            pa = 100.0 + i * 0.8
            if i >= 36:
                pa = 68.0 + (i - 36) * 0.05
            pb = 100.0 + i * 0.25
            add_price_all_adjustments(
                db,
                code="STOPA",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa),
                low=float(pa),
            )
            add_price_all_adjustments(
                db,
                code="STOPB",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb),
                low=float(pb),
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="atr",
                atr_stop_mode="static",
                atr_stop_execution_mode="intraday",
                atr_stop_execution_time="close",
                atr_stop_window=5,
                atr_stop_n=1.0,
                atr_stop_m=0.5,
            ),
        )
    wa = [
        float(x)
        for x in ((out.get("weights") or {}).get("series") or {}).get("STOPA", [])
    ]
    wb = [
        float(x)
        for x in ((out.get("weights") or {}).get("series") or {}).get("STOPB", [])
    ]
    assert wa and wb and len(wa) == len(wb)
    assert any((a <= 1e-12) and (b > 1e-12) for a, b in zip(wa, wb))
    atr_events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        atr_events.extend(list(atr_meta.get("events") or []))
    assert any(str((e or {}).get("code") or "") == "STOPA" for e in atr_events)


def test_rotation_equity_budget_stop_exits_only_triggered_asset(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["EQA", "EQB"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            pa = 100.0 + i * 0.7
            if i >= 34:
                pa = 70.0 + (i - 34) * 0.08
            pb = 100.0 + i * 0.22
            add_price_all_adjustments(
                db,
                code="EQA",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa),
                low=float(pa),
            )
            add_price_all_adjustments(
                db,
                code="EQB",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb),
                low=float(pb),
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="equity_budget",
                equity_stop_risk_pct=0.02,
                atr_stop_execution_mode="intraday",
                atr_stop_execution_time="close",
                atr_stop_mode="none",
            ),
        )
    wa = [
        float(x)
        for x in ((out.get("weights") or {}).get("series") or {}).get("EQA", [])
    ]
    wb = [
        float(x)
        for x in ((out.get("weights") or {}).get("series") or {}).get("EQB", [])
    ]
    assert wa and wb and len(wa) == len(wb)
    assert any((a <= 1e-12) and (b > 1e-12) for a, b in zip(wa, wb))
    eq_events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        eq_events.extend(list(atr_meta.get("events") or []))
    effective_eq_events = [
        e for e in eq_events if float((e or {}).get("reduce_fraction") or 0.0) > 0.0
    ]
    assert any(str((e or {}).get("code") or "") == "EQA" for e in eq_events)
    assert effective_eq_events
    for ev in effective_eq_events:
        loss_pct = (ev or {}).get("equity_loss_pct_at_trigger")
        risk_pct = (ev or {}).get("equity_stop_risk_pct")
        hold_ret = (ev or {}).get("holding_return_at_trigger")
        wt_now = (ev or {}).get("weight_at_trigger")
        assert loss_pct is not None
        assert risk_pct is not None
        assert hold_ret is not None
        assert wt_now is not None
        assert float(loss_pct) <= -float(risk_pct) + 1e-12
        assert float(loss_pct) == pytest.approx(
            float(wt_now) * float(hold_ret), abs=1e-12
        )


def test_rotation_equity_budget_next_day_close_exec_delays_exit(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["EQN_A", "EQN_B"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            pa = 100.0 + i * 0.7
            if i >= 34:
                pa = 69.0 + (i - 34) * 0.08
            pb = 100.0 + i * 0.20
            add_price_all_adjustments(
                db,
                code="EQN_A",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa),
                low=float(pa),
            )
            add_price_all_adjustments(
                db,
                code="EQN_B",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb),
                low=float(pb),
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="equity_budget",
                equity_stop_risk_pct=0.02,
                atr_stop_execution_mode="next_day",
                atr_stop_execution_time="close",
                atr_stop_mode="none",
            ),
        )
    events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        events.extend([e for e in (atr_meta.get("events") or []) if e])
    events_eqn_a = [e for e in events if str((e or {}).get("code") or "") == "EQN_A"]
    effective_events_eqn_a = [
        e for e in events_eqn_a if float((e or {}).get("reduce_fraction") or 0.0) > 0.0
    ]
    assert len(effective_events_eqn_a) == 1
    assert any(
        str((e or {}).get("ignored_reason") or "") == "deferred_to_future_segment"
        for e in events_eqn_a
    )
    target = effective_events_eqn_a[0]
    trigger_date = str(target.get("trigger_date") or "")
    execution_date = str(target.get("execution_date") or "")
    assert execution_date and execution_date > trigger_date
    assert str(target.get("execution_mode") or "") == "next_day"
    assert str(target.get("execution_time") or "") == "close"
    dates_out = list(((out.get("weights") or {}).get("dates") or []))
    wa = list((((out.get("weights") or {}).get("series") or {}).get("EQN_A") or []))
    assert execution_date in dates_out
    idx = dates_out.index(execution_date)
    assert idx > 0
    assert float(wa[idx - 1]) > 1e-12
    assert float(wa[idx]) <= 1e-12


def test_rotation_equity_budget_next_day_exec_still_works_during_dd_sleep(
    session_factory,
):
    sf = session_factory
    with sf() as db:
        codes = ["EQS_A", "EQS_B"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            pa = 100.0 + i * 0.7
            if i >= 34:
                pa = 69.0 + (i - 34) * 0.08
            pb = 100.0 + i * 0.20
            add_price_all_adjustments(
                db,
                code="EQS_A",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa),
                low=float(pa),
            )
            add_price_all_adjustments(
                db,
                code="EQS_B",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb),
                low=float(pb),
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="equity_budget",
                equity_stop_risk_pct=0.02,
                atr_stop_execution_mode="next_day",
                atr_stop_execution_time="close",
                atr_stop_mode="none",
                dd_control=True,
                dd_threshold=0.01,
                dd_reduce=0.5,
                dd_sleep_days=10,
            ),
        )
    events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        events.extend(
            [
                e
                for e in (atr_meta.get("events") or [])
                if e and str((e or {}).get("code") or "") == "EQS_A"
            ]
        )
    effective = [
        e for e in events if float((e or {}).get("reduce_fraction") or 0.0) > 0.0
    ]
    assert len(effective) == 1
    target = effective[0]
    trigger_date = str(target.get("trigger_date") or "")
    execution_date = str(target.get("execution_date") or "")
    assert execution_date and execution_date > trigger_date

    exec_period = next(
        (
            p
            for p in (out.get("holdings") or [])
            if str((p or {}).get("start_date") or "") <= execution_date
            and execution_date <= str((p or {}).get("end_date") or "")
        ),
        None,
    )
    assert exec_period is not None
    assert bool((((exec_period or {}).get("dd_control") or {}).get("in_sleep"))) is True

    dates_out = list(((out.get("weights") or {}).get("dates") or []))
    wa = list((((out.get("weights") or {}).get("series") or {}).get("EQS_A") or []))
    assert execution_date in dates_out
    idx = dates_out.index(execution_date)
    assert idx > 0
    assert float(wa[idx - 1]) > 1e-12
    assert float(wa[idx]) <= 1e-12


def test_rotation_equity_budget_stop_uses_current_weight_contribution(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["EWC_A", "EWC_B"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=100, freq="B")]
        close_a = 100.0
        close_b = 100.0
        for i, d in enumerate(dates):
            close_a = close_a + 0.2
            close_b = close_b + 0.2
            if i == 55:
                close_a = (
                    close_a * 0.98
                )  # mild drawdown while A weight should be reduced
            if i == 70:
                close_b = (
                    close_b * 0.90
                )  # force at least one effective equity-budget stop event
            if i < 50:
                ha, la = close_a * 1.002, close_a * 0.998
                hb, lb = close_b * 1.03, close_b * 0.97
            else:
                ha, la = close_a * 1.06, close_a * 0.94
                hb, lb = close_b * 1.01, close_b * 0.99
            add_price_all_adjustments(
                db,
                code="EWC_A",
                day=d,
                close=float(close_a),
                open_price=float(close_a),
                high=float(ha),
                low=float(la),
            )
            add_price_all_adjustments(
                db,
                code="EWC_B",
                day=d,
                close=float(close_b),
                open_price=float(close_b),
                high=float(hb),
                low=float(lb),
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                position_mode="risk_budget",
                daily_rebalance=True,
                risk_budget_atr_window=5,
                risk_budget_pct=0.01,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="equity_budget",
                equity_stop_risk_pct=0.01,
                atr_stop_execution_mode="intraday",
                atr_stop_execution_time="close",
                atr_stop_mode="none",
            ),
        )

    events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        events.extend([e for e in (atr_meta.get("events") or []) if e])
    effective = [
        e for e in events if float((e or {}).get("reduce_fraction") or 0.0) > 0.0
    ]
    for ev in effective:
        loss_pct = (ev or {}).get("equity_loss_pct_at_trigger")
        risk_pct = (ev or {}).get("equity_stop_risk_pct")
        hold_ret = (ev or {}).get("holding_return_at_trigger")
        wt_now = (ev or {}).get("weight_at_trigger")
        assert loss_pct is not None
        assert risk_pct is not None
        assert hold_ret is not None
        assert wt_now is not None
        assert float(loss_pct) <= -float(risk_pct) + 1e-12
        assert float(loss_pct) == pytest.approx(
            float(wt_now) * float(hold_ret), abs=1e-12
        )
    # EWC_A should not be stopped by the mild -2% move after its weight is reduced.
    assert not any(str((e or {}).get("code") or "") == "EWC_A" for e in effective)
    hist = list(out.get("historical_trades") or [])
    effective_codes = {str((e or {}).get("code") or "") for e in effective}
    if effective_codes:
        assert any(
            str((x or {}).get("code") or "") in effective_codes
            and "止损" in str((x or {}).get("exit_reason") or "")
            for x in hist
        )


def test_rotation_equity_budget_stop_keeps_true_entry_across_rebalance_segments(
    session_factory,
):
    sf = session_factory
    with sf() as db:
        code = "EQREF"
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=35, freq="B")]
        prices = []
        for i, d in enumerate(dates):
            if i < 20:
                px = 100.0 + float(i) * 0.6  # uptrend before second segment
            elif i == 20:
                px = 108.0  # small pullback at/after segment boundary
            else:
                px = 108.0 + float(i - 20) * 0.05
            prices.append(float(px))
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
                codes=[code],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                rebalance_shift="prev",
                exec_price="close",
                top_k=1,
                position_mode="inverse_vol",
                daily_rebalance=False,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="equity_budget",
                equity_stop_risk_pct=0.01,
                atr_stop_execution_mode="intraday",
                atr_stop_execution_time="close",
                atr_stop_mode="none",
            ),
        )
    events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        events.extend(
            [
                e
                for e in (atr_meta.get("events") or [])
                if e
                and str((e or {}).get("code") or "") == code
                and float((e or {}).get("reduce_fraction") or 0.0) > 0.0
            ]
        )
    # If entry reference were reset at each rebalance segment, the mild pullback
    # near the segment boundary could falsely trigger a 1% equity-budget stop.
    assert not events
    cur = {str(x.get("code") or ""): x for x in (out.get("current_holdings") or [])}
    one = cur.get(code)
    assert one is not None
    w_dates = list(((out.get("weights") or {}).get("dates") or []))
    w_vals = list((((out.get("weights") or {}).get("series") or {}).get(code) or []))
    first_pos = next((i for i, v in enumerate(w_vals) if float(v) > 1e-12), None)
    assert first_pos is not None
    assert str(one.get("entry_date") or "") == str(w_dates[first_pos])


def test_rotation_atr_stop_keeps_true_entry_across_rebalance_segments(
    session_factory,
):
    sf = session_factory
    with sf() as db:
        code = "ATRREF"
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=45, freq="B")]
        for i, d in enumerate(dates):
            if i < 22:
                px = 100.0 + float(i) * 0.35
            elif i == 22:
                px = 112.0
            elif i == 23:
                px = 107.5
            else:
                px = 107.5 + float(i - 23) * 0.05
            add_price_all_adjustments(
                db,
                code=code,
                day=d,
                close=float(px),
                open_price=float(px),
                high=float(px + 0.6),
                low=float(px - 0.6),
            )
        db.commit()
        out = backtest_rotation(
            db,
            RotationInputs(
                codes=[code],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                rebalance_shift="prev",
                exec_price="close",
                top_k=1,
                position_mode="inverse_vol",
                daily_rebalance=False,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="atr",
                atr_stop_mode="static",
                atr_stop_execution_mode="intraday",
                atr_stop_execution_time="close",
                atr_stop_window=5,
                atr_stop_n=0.5,
                atr_stop_m=0.25,
            ),
        )
    events = []
    for p in out.get("holdings") or []:
        atr_meta = (p or {}).get("atr_stop") or {}
        events.extend(
            [
                e
                for e in (atr_meta.get("events") or [])
                if e
                and str((e or {}).get("code") or "") == code
                and float((e or {}).get("reduce_fraction") or 0.0) > 0.0
            ]
        )
    # A segment-start reference-price reset would spuriously lift stop_price and
    # trigger on the mild pullback after the jump.
    assert not events
    cur = {str(x.get("code") or ""): x for x in (out.get("current_holdings") or [])}
    one = cur.get(code)
    assert one is not None
    w_dates = list(((out.get("weights") or {}).get("dates") or []))
    w_vals = list((((out.get("weights") or {}).get("series") or {}).get(code) or []))
    first_pos = next((i for i, v in enumerate(w_vals) if float(v) > 1e-12), None)
    assert first_pos is not None
    assert str(one.get("entry_date") or "") == str(w_dates[first_pos])


def test_rotation_current_holdings_return_uses_entry_day_execution_price(
    session_factory,
):
    sf = session_factory
    start = dt.date(2024, 1, 2)
    dates = [d.date() for d in pd.date_range(start, periods=45, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + float(i)
            add_price_all_adjustments(
                db,
                code="HOLD",
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
                codes=["HOLD"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=1,
                rebalance_shift="prev",
                exec_price="close",
                top_k=1,
                lookback_days=5,
                skip_days=0,
                position_mode="inverse_vol",
                daily_rebalance=False,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="none",
                atr_stop_mode="none",
            ),
        )

    rows = list(out.get("current_holdings") or [])
    assert rows
    one = rows[0]
    entry_date = dt.date.fromisoformat(str(one.get("entry_date")))
    assert entry_date in dates
    entry_idx = dates.index(entry_date)
    last_px = 100.0 + float(len(dates) - 1)
    entry_px = 100.0 + float(entry_idx)
    expected = float(last_px / entry_px - 1.0)
    assert float(one.get("holding_return") or 0.0) == pytest.approx(expected, abs=1e-12)
    assert float(one.get("entry_price") or 0.0) == pytest.approx(entry_px, abs=1e-12)
    assert float(one.get("latest_price") or 0.0) == pytest.approx(last_px, abs=1e-12)
    assert one.get("equity_return") is not None
    htr = [
        x for x in (out.get("historical_trades") or []) if str(x.get("code")) == "HOLD"
    ]
    assert htr
    last_open = htr[-1]
    assert bool(last_open.get("closed")) is False
    assert float(last_open.get("entry_price") or 0.0) == pytest.approx(
        entry_px, abs=1e-12
    )
    assert float(last_open.get("exit_price") or 0.0) == pytest.approx(
        last_px, abs=1e-12
    )
    assert float(last_open.get("holding_return") or 0.0) == pytest.approx(
        expected, abs=1e-12
    )
    assert float(one.get("equity_return") or 0.0) == pytest.approx(
        float(last_open.get("total_equity_return") or 0.0),
        abs=1e-12,
    )


def test_rotation_r_take_profit_triggers_with_stop_scheme_none(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["RTPA", "RTPB"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            if i < 28:
                pa = 100.0 + i * 1.2
            elif i < 46:
                pa = 133.6 - (i - 28) * 1.8
            else:
                pa = 101.2 + (i - 46) * 0.10
            pb = 100.0 + i * 0.18
            add_price_all_adjustments(
                db,
                code="RTPA",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa) * 1.002,
                low=float(pa) * 0.998,
            )
            add_price_all_adjustments(
                db,
                code="RTPB",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb) * 1.002,
                low=float(pb) * 0.998,
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="none",
                atr_stop_mode="none",
                atr_stop_window=5,
                atr_stop_n=1.0,
                r_take_profit_enabled=True,
                r_take_profit_reentry_mode="reenter",
                r_take_profit_execution_mode="intraday",
                r_take_profit_execution_time="close",
                r_take_profit_tiers=[{"r_multiple": 0.5, "retrace_ratio": 0.1}],
            ),
        )
    assert bool(out.get("r_take_profit_enabled")) is True
    events = [e for e in (out.get("r_take_profit_events") or []) if e]
    assert any(float((e or {}).get("reduce_fraction") or 0.0) > 0.0 for e in events)
    one_period = next(iter(out.get("holdings") or []), {}) or {}
    rtp_meta = (one_period.get("r_take_profit") or {}) if one_period else {}
    assert str(rtp_meta.get("execution_mode") or "") == "intraday"
    assert str(rtp_meta.get("execution_time") or "") == "close"
    assert isinstance(rtp_meta.get("tiers"), list)
    stats = ((out.get("trade_statistics") or {}).get("overall")) or {}
    assert int(stats.get("r_take_profit_trigger_count") or 0) >= 1
    by_code = ((out.get("trade_statistics") or {}).get("by_code")) or {}
    assert (
        sum(
            int(((v or {}).get("r_take_profit_trigger_count") or 0))
            for v in by_code.values()
        )
        >= 1
    )


def test_rotation_r_take_profit_next_day_execution_cross_segment(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["RTPX"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=50, freq="B")]
        for i, d in enumerate(dates):
            if i < 19:
                px = 100.0 + i * 1.5
            elif i == 19:
                px = 102.0
            else:
                px = 101.5 - (i - 20) * 0.05
            add_price_all_adjustments(
                db,
                code="RTPX",
                day=d,
                close=float(px),
                open_price=float(px),
                high=float(px) * 1.002,
                low=float(px) * 0.998,
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=1,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="none",
                atr_stop_mode="none",
                atr_stop_window=5,
                atr_stop_n=1.0,
                r_take_profit_enabled=True,
                r_take_profit_reentry_mode="reenter",
                r_take_profit_execution_mode="next_day",
                r_take_profit_execution_time="close",
                r_take_profit_tiers=[{"r_multiple": 0.5, "retrace_ratio": 0.1}],
            ),
        )
    events = [e for e in (out.get("r_take_profit_events") or []) if e]
    target = next(
        (
            e
            for e in events
            if str((e or {}).get("code") or "") == "RTPX"
            and str((e or {}).get("execution_mode") or "") == "next_day"
            and float((e or {}).get("reduce_fraction") or 0.0) > 0.0
        ),
        None,
    )
    assert target is not None
    trigger_date = str(target.get("trigger_date") or "")
    execution_date = str(target.get("execution_date") or "")
    assert execution_date and execution_date > trigger_date

    dates_out = list(((out.get("weights") or {}).get("dates") or []))
    w = list((((out.get("weights") or {}).get("series") or {}).get("RTPX") or []))
    assert execution_date in dates_out
    idx = dates_out.index(execution_date)
    assert idx > 0
    assert float(w[idx - 1]) > 1e-12
    assert float(w[idx]) <= 1e-12


def test_rotation_r_take_profit_next_day_exec_still_works_during_dd_sleep(
    session_factory,
):
    sf = session_factory
    with sf() as db:
        codes = ["RSD_A", "RSD_B"]
        dates = [d.date() for d in pd.date_range("2024-01-02", periods=80, freq="B")]
        for i, d in enumerate(dates):
            if i < 24:
                pa = 100.0 + i * 1.4
            elif i < 36:
                pa = 133.6 - (i - 24) * 2.0
            else:
                pa = 109.6 - (i - 36) * 0.03
            pb = 100.0 + i * 0.02
            add_price_all_adjustments(
                db,
                code="RSD_A",
                day=d,
                close=float(pa),
                open_price=float(pa),
                high=float(pa) * 1.002,
                low=float(pa) * 0.998,
            )
            add_price_all_adjustments(
                db,
                code="RSD_B",
                day=d,
                close=float(pb),
                open_price=float(pb),
                high=float(pb) * 1.002,
                low=float(pb) * 0.998,
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
                rebalance_shift="prev",
                exec_price="close",
                top_k=2,
                lookback_days=5,
                skip_days=0,
                cost_bps=0.0,
                slippage_rate=0.0,
                stop_scheme="none",
                atr_stop_mode="none",
                atr_stop_window=5,
                atr_stop_n=1.0,
                r_take_profit_enabled=True,
                r_take_profit_reentry_mode="reenter",
                r_take_profit_execution_mode="next_day",
                r_take_profit_execution_time="close",
                r_take_profit_tiers=[{"r_multiple": 0.5, "retrace_ratio": 0.1}],
                dd_control=True,
                dd_threshold=0.005,
                dd_reduce=0.5,
                dd_sleep_days=10,
            ),
        )
    events = [
        e
        for e in (out.get("r_take_profit_events") or [])
        if str((e or {}).get("code") or "") == "RSD_A"
        and float((e or {}).get("reduce_fraction") or 0.0) > 0.0
    ]
    assert len(events) == 1
    target = events[0]
    trigger_date = str(target.get("trigger_date") or "")
    execution_date = str(target.get("execution_date") or "")
    assert execution_date and execution_date > trigger_date

    exec_period = next(
        (
            p
            for p in (out.get("holdings") or [])
            if str((p or {}).get("start_date") or "") <= execution_date
            and execution_date <= str((p or {}).get("end_date") or "")
        ),
        None,
    )
    assert exec_period is not None
    assert bool((((exec_period or {}).get("dd_control") or {}).get("in_sleep"))) is True

    dates_out = list(((out.get("weights") or {}).get("dates") or []))
    wa = list((((out.get("weights") or {}).get("series") or {}).get("RSD_A") or []))
    assert execution_date in dates_out
    idx = dates_out.index(execution_date)
    assert idx > 0
    assert float(wa[idx - 1]) > 1e-12
    assert float(wa[idx]) <= 1e-12
