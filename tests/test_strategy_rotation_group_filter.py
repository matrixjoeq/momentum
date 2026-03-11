import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                close=float(close),
                low=float(close),
                high=float(close),
                source="eastmoney",
                adjust=adj,
            )
        )


def test_group_enforce_strongest_score_keeps_one_per_group(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-19", freq="B")]
        for i, d in enumerate(dates):
            # A1/A2 belong to the same group and both trend strongly.
            _add_price(db, code="A1", day=d, close=100.0 + i * 2.0)
            _add_price(db, code="A2", day=d, close=100.0 + i * 1.8)
            # B1 is weaker but in another group.
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.6)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=2,
                lookback_days=3,
                group_enforce=True,
                group_pick_policy="strongest_score",
                asset_groups={"A1": "CN_EQ", "A2": "CN_EQ", "B1": "US_EQ"},
            ),
        )

    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg is not None
    assert seg["group_filter"]["enabled"] is True
    assert "A1" in seg["picks"]
    assert "A2" not in seg["picks"]
    assert "B1" in seg["picks"]


def test_group_enforce_earliest_entry_prefers_existing_holding(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-19", freq="B")]
        for i, d in enumerate(dates):
            if d <= dt.date(2024, 1, 5):
                a1 = 100.0 + i * 2.5
                a2 = 100.0 + i * 0.8
            else:
                a1 = 112.0 + (i - 4) * 0.2
                a2 = 103.0 + (i - 4) * 2.2
            _add_price(db, code="A1", day=d, close=a1)
            _add_price(db, code="A2", day=d, close=a2)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=3,
                group_enforce=True,
                group_pick_policy="earliest_entry",
                asset_groups={"A1": "CN_EQ", "A2": "CN_EQ"},
            ),
        )

    seg1 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    seg2 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-15"), None)
    assert seg1 is not None and seg2 is not None
    assert seg1["picks"] == ["A1"]
    assert seg2["group_filter"]["enabled"] is True
    assert seg2["group_filter"]["policy"] == "earliest_entry"
    assert seg2["picks"] == ["A1"]


def test_rotation_holds_cash_when_candidates_not_greater_than_topk(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-31", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.0)
            _add_price(db, code="A2", day=d, close=100.0 + i * 0.8)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=2,
                lookback_days=3,
                risk_off=False,
            ),
        )

    assert out["holdings"]
    assert all((h.get("picks") or []) == [] for h in out["holdings"])


def test_rotation_entry_backfill_refills_filtered_topk(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-04-30", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.2)  # strongest, but will be blocked by RSI rule
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.9)
            _add_price(db, code="C1", day=d, close=100.0 + i * 0.6)
        db.commit()

        base = RotationInputs(
            codes=["A1", "B1", "C1"],
            start=dates[0],
            end=dates[-1],
            rebalance="monthly",
            top_k=2,
            lookback_days=20,
            rsi_filter=True,
            rsi_window=14,
            rsi_overbought=100.0,
            rsi_oversold=0.0,
            rsi_block_overbought=True,
            rsi_block_oversold=False,
            asset_rsi_rules=[
                {
                    "code": "*",
                    "rsi_window": 14,
                    "rsi_overbought": 100.0,
                    "rsi_oversold": 0.0,
                    "rsi_block_overbought": True,
                    "rsi_block_oversold": False,
                },
                {
                    "code": "A1",
                    "rsi_window": 14,
                    "rsi_overbought": 0.0,
                    "rsi_oversold": 0.0,
                    "rsi_block_overbought": True,
                    "rsi_block_oversold": False,
                },
            ],
        )
        out_no_fill = backtest_rotation(db, base)
        out_fill = backtest_rotation(db, RotationInputs(**{**base.__dict__, "entry_backfill": True}))

    h_no = next((x for x in out_no_fill["holdings"] if x.get("mode") == "risk_on"), None)
    h_yes = next((x for x in out_fill["holdings"] if x.get("mode") == "risk_on"), None)
    assert h_no is not None and h_yes is not None
    assert "A1" not in (h_no.get("picks") or [])
    assert (h_no.get("picks") or []) == ["B1"]
    assert bool(((h_no.get("backfill") or {}).get("used"))) is False
    assert (h_yes.get("picks") or []) == ["B1", "C1"]
    assert bool(((h_yes.get("backfill") or {}).get("used"))) is True


def test_rotation_position_mode_fixed_vs_adaptive_changes_exposure(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-04-30", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.2)  # blocked by RSI rule
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.9)
            _add_price(db, code="C1", day=d, close=100.0 + i * 0.6)
            _add_price(db, code="D1", day=d, close=100.0 + i * 0.4)
        db.commit()

        base = RotationInputs(
            codes=["A1", "B1", "C1", "D1"],
            start=dates[0],
            end=dates[-1],
            rebalance="monthly",
            top_k=3,
            lookback_days=20,
            entry_backfill=False,
            rsi_filter=True,
            rsi_window=14,
            rsi_overbought=100.0,
            rsi_oversold=0.0,
            rsi_block_overbought=True,
            rsi_block_oversold=False,
            asset_rsi_rules=[
                {"code": "*", "rsi_window": 14, "rsi_overbought": 100.0, "rsi_oversold": 0.0, "rsi_block_overbought": True, "rsi_block_oversold": False},
                {"code": "A1", "rsi_window": 14, "rsi_overbought": 0.0, "rsi_oversold": 0.0, "rsi_block_overbought": True, "rsi_block_oversold": False},
            ],
        )
        out_adapt = backtest_rotation(db, base)
        out_fixed = backtest_rotation(db, RotationInputs(**{**base.__dict__, "position_mode": "fixed"}))

    h_adapt = next((x for x in out_adapt["holdings"] if x.get("mode") == "risk_on"), None)
    h_fixed = next((x for x in out_fixed["holdings"] if x.get("mode") == "risk_on"), None)
    assert h_adapt is not None and h_fixed is not None
    assert len(h_adapt.get("picks") or []) == 2
    assert len(h_fixed.get("picks") or []) == 2
    assert float(h_adapt.get("exposure") or 0.0) == pytest.approx(1.0, rel=1e-6)
    assert float(h_fixed.get("exposure") or 0.0) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_rotation_entry_match_n_can_relax_from_and_to_nofm(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-04-30", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.4)  # highest momentum
            _add_price(db, code="C1", day=d, close=100.0 + i * 0.9)  # medium momentum
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.1)  # low momentum (near flat)
        db.commit()

        base = RotationInputs(
            codes=["A1", "B1", "C1"],
            start=dates[0],
            end=dates[-1],
            rebalance="monthly",
            top_k=1,
            lookback_days=20,
            trend_filter=True,
            trend_sma_window=10,
            rsi_filter=True,
            rsi_window=14,
            rsi_overbought=100.0,
            rsi_oversold=0.0,
            rsi_block_overbought=True,
            rsi_block_oversold=False,
            asset_rsi_rules=[
                {"code": "*", "rsi_window": 14, "rsi_overbought": 100.0, "rsi_oversold": 0.0, "rsi_block_overbought": True, "rsi_block_oversold": False},
                {"code": "A1", "rsi_window": 14, "rsi_overbought": 0.0, "rsi_oversold": 0.0, "rsi_block_overbought": True, "rsi_block_oversold": False},
            ],
        )
        out_and = backtest_rotation(db, base)
        out_nofm = backtest_rotation(db, RotationInputs(**{**base.__dict__, "entry_match_n": 1}))

    picks_and = [tuple(x.get("picks") or []) for x in out_and["holdings"]]
    picks_nofm = [tuple(x.get("picks") or []) for x in out_nofm["holdings"]]
    assert ("A1",) not in picks_and
    assert ("A1",) in picks_nofm


def test_rotation_momentum_exit_rule_generates_daily_exit_event(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
        for i, d in enumerate(dates):
            # A1 rises first, then weakens so momentum score can fall below threshold.
            a = 100.0 + i * 1.2 if i < 35 else 142.0 + (i - 35) * 0.02
            b = 100.0 + i * 0.2
            _add_price(db, code="A1", day=d, close=a)
            _add_price(db, code="B1", day=d, close=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "B1"],
                start=dates[0],
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                asset_momentum_floor_rules=[
                    {"code": "*", "stage": "entry", "op": ">", "threshold": 0.0, "threshold_unit": "raw"},
                    {"code": "A1", "stage": "exit", "op": "<", "threshold": 0.02, "threshold_unit": "raw"},
                ],
            ),
        )

    events = out.get("daily_exit_events") or []
    assert events
    assert any(str(e.get("type")) == "momentum_rule" for e in events)
