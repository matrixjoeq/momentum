import pandas as pd
import pytest

from etf_momentum.analysis.calendar_timing_strategy import (
    CalendarTimingStrategyInputs,
    compute_calendar_timing_strategy_backtest,
)
from etf_momentum.db.models import EtfPrice


def test_calendar_timing_uses_hfq_benchmark_and_none_with_fallback_for_strategy(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-02", periods=60, freq="B")]
    # none: split-like cliff around day 20; hfq: smooth total-return path.
    none_px = []
    hfq_px = []
    for i, _ in enumerate(dates):
        if i < 20:
            none_px.append(100.0 + float(i))
        else:
            none_px.append(12.0 + 0.1 * float(i - 20))
        hfq_px.append(100.0 + float(i))

    with sf() as db:
        for d, p_none, p_hfq in zip(dates, none_px, hfq_px):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(p_none),
                    high=float(p_none),
                    low=float(p_none),
                    close=float(p_none),
                    source="eastmoney",
                    adjust="none",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(p_hfq),
                    high=float(p_hfq),
                    low=float(p_hfq),
                    close=float(p_hfq),
                    source="eastmoney",
                    adjust="hfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(p_hfq),
                    high=float(p_hfq),
                    low=float(p_hfq),
                    close=float(p_hfq),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="single",
                code=code,
                codes=None,
                start=dates[0],
                end=dates[-1],
                decision_day=1,
                hold_days=40,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
                rebalance_shift="prev",
            ),
        )

    meta = out.get("meta") or {}
    assert "hfq" in str(meta.get("benchmark_price_basis") or "").lower()
    assert "none" in str(meta.get("strategy_price_basis") or "").lower()

    bench_nav = [float(x) for x in (out.get("nav") or {}).get("series", {}).get("BUY_HOLD", [])]
    assert len(bench_nav) == len(dates)
    expected_final_bench = float(hfq_px[-1] / hfq_px[0])
    assert bench_nav[-1] == pytest.approx(expected_final_bench, rel=1e-8)

    strat_nav = [float(x) for x in (out.get("nav") or {}).get("series", {}).get("STRAT", [])]
    assert strat_nav
    # Without none->hfq fallback on the split day, strategy NAV would collapse deeply when in position.
    assert min(strat_nav) > 0.80


def test_calendar_timing_dynamic_universe_uses_union_interval(session_factory):
    sf = session_factory
    dates_all = [d.date() for d in pd.date_range("2024-01-02", periods=60, freq="B")]
    dates_late = dates_all[25:]
    with sf() as db:
        for i, d in enumerate(dates_all):
            p = 100.0 + float(i)
            for adj in ("none", "hfq", "qfq"):
                db.add(
                    EtfPrice(
                        code="A1",
                        trade_date=d,
                        open=float(p),
                        high=float(p),
                        low=float(p),
                        close=float(p),
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        for i, d in enumerate(dates_late):
            p = 80.0 + float(i)
            for adj in ("none", "hfq", "qfq"):
                db.add(
                    EtfPrice(
                        code="B1",
                        trade_date=d,
                        open=float(p),
                        high=float(p),
                        low=float(p),
                        close=float(p),
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        db.commit()

        out_intersection = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="portfolio",
                code=None,
                codes=["A1", "B1"],
                start=dates_all[0],
                end=dates_all[-1],
                decision_day=1,
                hold_days=10,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
                rebalance_shift="prev",
                dynamic_universe=False,
            ),
        )
        out_union = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="portfolio",
                code=None,
                codes=["A1", "B1"],
                start=dates_all[0],
                end=dates_all[-1],
                decision_day=1,
                hold_days=10,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
                rebalance_shift="prev",
                dynamic_universe=True,
            ),
        )

    n_inter = int(((out_intersection.get("metrics") or {}).get("strategy") or {}).get("sample_days") or 0)
    n_union = int(((out_union.get("metrics") or {}).get("strategy") or {}).get("sample_days") or 0)
    assert n_union > n_inter
    assert bool((out_union.get("meta") or {}).get("dynamic_universe")) is True


def test_calendar_timing_next_plan_includes_current_month_decision(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2026-01-05", "2026-03-23", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p = 100.0 + float(i) * 0.2
            for adj in ("none", "hfq", "qfq"):
                db.add(
                    EtfPrice(
                        code="A1",
                        trade_date=d,
                        open=float(p),
                        high=float(p),
                        low=float(p),
                        close=float(p),
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        db.commit()

        out = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="single",
                code="A1",
                codes=None,
                start=dates[0],
                end=dates[-1],  # asof = 2026-03-23
                decision_day=-2,
                hold_days=10,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
                rebalance_shift="prev",
                dynamic_universe=False,
            ),
        )

    plan = ((out.get("next_execution_plan") or {}).get("plan") or {})
    assert str(plan.get("type") or "") == "entry"
    # 2026-03-31 is the next trading day after natural decision day 2026-03-30.
    assert str(plan.get("decision_date") or "") == "2026-03-30"
    assert str(plan.get("execution_date") or "") == "2026-03-31"
    # Current holding-day semantics are "exit after N trading days from entry execution day",
    # so hold_days=10 from 2026-03-31 lands on 2026-04-15.
    assert str(plan.get("planned_exit_date") or "") == "2026-04-15"


def test_calendar_timing_trade_stats_by_code_no_extreme_explosions(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2022-01-03", periods=300, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + 0.05 * float(i)
            p2 = 80.0 + 0.03 * float(i)
            for code, px in [("A1", p1), ("B1", p2)]:
                for adj in ("none", "hfq", "qfq"):
                    db.add(
                        EtfPrice(
                            code=code,
                            trade_date=d,
                            open=float(px),
                            high=float(px),
                            low=float(px),
                            close=float(px),
                            source="eastmoney",
                            adjust=adj,
                        )
                    )
        db.commit()

        out = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="portfolio",
                code=None,
                codes=["A1", "B1"],
                start=dates[0],
                end=dates[-1],
                decision_day=1,
                hold_days=10,
                exec_price="close",
                cost_bps=2.0,
                slippage_rate=0.001,
                rebalance_shift="prev",
            ),
        )

    by_code = (((out.get("trade_statistics") or {}).get("by_code")) or {})
    for c, st in by_code.items():
        all_stats = (st or {}).get("all_stats") or {}
        mn = all_stats.get("min")
        mx = all_stats.get("max")
        if mn is not None:
            assert float(mn) > -10.0, f"{c} min trade return exploded: {mn}"
        if mx is not None:
            assert float(mx) < 10.0, f"{c} max trade return exploded: {mx}"
