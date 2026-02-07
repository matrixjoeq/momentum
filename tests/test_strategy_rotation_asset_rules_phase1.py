import datetime as dt

import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import (
    RotationInputs,
    _effective_rules_for_code,
    _merge_rule,
    backtest_rotation,
)


def _seed_prices(db, *, code: str, dates: list[dt.date], closes: list[float]) -> None:
    assert len(dates) == len(closes)
    for d, px in zip(dates, closes):
        for adj in ("hfq", "qfq", "none"):
            c = float(px)
            hi = c * 1.01
            lo = c * 0.99
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=c,
                    high=hi,
                    low=lo,
                    close=c,
                    source="eastmoney",
                    adjust=adj,
                )
            )


def test_effective_rules_for_code_merges_default_and_specific():
    rules = [
        {"code": "*", "rsi_window": 14, "rsi_overbought": 70.0},
        {"code": "AAA", "rsi_overbought": 55.0},
    ]
    eff = _effective_rules_for_code("AAA", rules)
    assert len(eff) == 1
    assert eff[0]["code"] == "AAA"
    # inherited from default
    assert eff[0]["rsi_window"] == 14
    # overridden
    assert eff[0]["rsi_overbought"] == 55.0

    eff_b = _effective_rules_for_code("BBB", rules)
    assert len(eff_b) == 1
    assert eff_b[0]["code"] == "*"


def test_merge_rule_field_level():
    base = {"code": "*", "a": 1, "b": 2, "c": None}
    override = {"code": "AAA", "b": 3, "c": 4, "d": None}
    out = _merge_rule(base, override)
    assert out["code"] == "AAA"
    assert out["a"] == 1
    assert out["b"] == 3
    assert out["c"] == 4
    assert "d" not in out  # None should not override into output


def test_rsi_asset_rule_overrides_global_threshold_and_excludes(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(120)]
        # AAA strong up -> RSI high; BBB flat -> RSI ~ 50
        a = [100.0 + i * 1.0 for i in range(len(dates))]
        b = [100.0 for _ in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=2,
                lookback_days=20,
                cost_bps=0.0,
                rsi_filter=True,
                # global thresholds are permissive (should not block)
                rsi_window=14,
                rsi_overbought=100.0,
                rsi_block_overbought=True,
                # per-asset rules: AAA is strict and should be excluded
                asset_rsi_rules=[
                    {"code": "*", "rsi_window": 14, "rsi_overbought": 100.0, "rsi_block_overbought": True},
                    {"code": "AAA", "rsi_overbought": 55.0, "rsi_block_overbought": True},
                ],
            ),
        )

    picks = [p.get("picks") for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert picks, "expected some risk_on periods"
    # At least one risk-on period should exclude AAA but keep BBB.
    assert any(("AAA" not in (ps or [])) and ("BBB" in (ps or [])) for ps in picks)


def test_rsi_asset_rules_multiple_matches_use_worst_case(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(120)]
        a = [100.0 + i * 1.0 for i in range(len(dates))]
        b = [100.0 for _ in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=2,
                lookback_days=20,
                cost_bps=0.0,
                rsi_filter=True,
                rsi_window=14,
                rsi_overbought=100.0,
                rsi_block_overbought=True,
                asset_rsi_rules=[
                    {"code": "*", "rsi_window": 14, "rsi_overbought": 100.0, "rsi_block_overbought": True},
                    # Two matching rules for AAA: one blocks, one doesn't -> worst-case => block
                    {"code": "AAA", "rsi_overbought": 55.0, "rsi_block_overbought": True},
                    {"code": "AAA", "rsi_overbought": 100.0, "rsi_block_overbought": False},
                ],
            ),
        )

    picks = [p.get("picks") for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert picks, "expected some risk_on periods"
    assert any(("AAA" not in (ps or [])) and ("BBB" in (ps or [])) for ps in picks)


def test_momentum_floor_asset_rule_excludes_specific_code(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(90)]
        # AAA strong up; BBB flat.
        a = [100.0 + i * 2.0 for i in range(len(dates))]
        b = [100.0 for _ in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                cost_bps=0.0,
                risk_off=False,
                momentum_floor=0.0,
                asset_momentum_floor_rules=[
                    {"code": "*", "momentum_floor": -1.0},
                    # Exclude AAA by setting a very high floor.
                    {"code": "AAA", "momentum_floor": 9e9},
                ],
            ),
        )

    # At least one risk-on period should hold BBB instead of AAA.
    picks = [p.get("picks") for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert picks, "expected some risk_on periods"
    assert any((ps == ["BBB"] or ("BBB" in (ps or []) and "AAA" not in (ps or []))) for ps in picks)

