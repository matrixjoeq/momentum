import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_bar(db, *, code: str, day: dt.date, close: float) -> None:
    high = close * 1.01
    low = close * 0.99
    open_ = close
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                open=open_,
                high=high,
                low=low,
                close=float(close),
                volume=1.0,
                amount=1.0,
                source="eastmoney",
                adjust=adj,
            )
        )


def _seed_prices_regime_switch(db, *, codes: list[str], start: dt.date, end: dt.date) -> list[dt.date]:
    """
    Create a deterministic regime-switch dataset so TopK=1 has incentives to rotate:
    - first third: AAA strongest
    - second third: BBB strongest
    - last third: CCC strongest
    """
    dates = list(pd.date_range(start, end, freq="B").date)
    assert len(codes) >= 3
    n = len(dates)
    cut1 = n // 3
    cut2 = 2 * n // 3
    for i, d in enumerate(dates):
        for j, code in enumerate(codes):
            base = 100.0 + i * 0.05 + j * 0.01
            if code == "AAA":
                bump = i * (0.25 if i < cut1 else 0.02)
            elif code == "BBB":
                bump = i * (0.25 if (cut1 <= i < cut2) else 0.02)
            else:
                bump = i * (0.25 if (i >= cut2) else 0.02)
            _add_bar(db, code=code, day=d, close=base + bump)
    db.commit()
    return dates


def _seed_prices_high_corr_switch(db, *, start: dt.date, end: dt.date) -> tuple[list[str], list[dt.date]]:
    """
    Construct a dataset that:
    - holds AAA in the first holding segment
    - would switch to BBB in the next segment due to momentum
    - but AAA/BBB are highly positively correlated, so corr gate blocks switching
    """
    codes = ["AAA", "BBB"]
    dates = list(pd.date_range(start, end, freq="B").date)
    for i, d in enumerate(dates):
        base = 100.0 + (2.0 if (i % 2 == 0) else -2.0) + i * 0.3
        aaa = base
        # BBB follows closely (high corr), but on first decision day BBB is worse, so AAA is picked.
        # With lookback_days=2 and weekly rebalance, the first decision day is typically the first Friday in range.
        bump = 0.0
        if d == dt.date(2024, 1, 5):
            bump = -2.0
        if d == dt.date(2024, 1, 12):
            bump = +0.5
        bbb = base + bump
        _add_bar(db, code="AAA", day=d, close=aaa)
        _add_bar(db, code="BBB", day=d, close=bbb)
    db.commit()
    return codes, dates


def _seed_prices_drawdown_crash(db, *, start: dt.date, end: dt.date) -> tuple[list[str], list[dt.date]]:
    """
    Construct a dataset that produces a clear peak-to-trough drawdown on the selected holding.
    AAA rallies then crashes to trigger drawdown control; BBB/CCC are flat-ish.
    """
    codes = ["AAA", "BBB", "CCC"]
    dates = list(pd.date_range(start, end, freq="B").date)
    for i, d in enumerate(dates):
        # AAA: rise then crash then slow recovery
        if i < 8:
            aaa = 100.0 + i * 3.0
        elif i == 8:
            aaa = 124.0
        elif i == 9:
            aaa = 105.0  # ~15% drawdown from 124
        else:
            aaa = 105.0 + (i - 9) * 0.3
        bbb = 100.0 + (i % 3) * 0.1
        ccc = 100.0 + (i % 5) * 0.05
        _add_bar(db, code="AAA", day=d, close=aaa)
        _add_bar(db, code="BBB", day=d, close=bbb)
        _add_bar(db, code="CCC", day=d, close=ccc)
    db.commit()
    return codes, dates


def _assert_common_payload(out: dict) -> None:
    assert "nav" in out and "series" in out["nav"]
    assert len(out["nav"]["dates"]) == len(out["nav"]["series"]["ROTATION"])
    assert "holdings" in out and isinstance(out["holdings"], list)


def _first_period_with_picks(out: dict) -> dict:
    for p in out.get("holdings", []):
        if p.get("picks"):
            return p
    raise AssertionError("expected at least one holding period with picks")


def _assert_tp_sl(out: dict, mode: str) -> None:
    # tp_sl.mode should exist on all periods; additional keys exist only when we have picks (risk assets).
    for p in out.get("holdings", []):
        tp = p.get("tp_sl") or {}
        assert tp.get("mode") == mode

    if mode == "none":
        return

    # For feature modes, assert that when there is at least one period with picks, we also have the expected keys.
    picked_periods = [p for p in out.get("holdings", []) if p.get("picks")]
    assert picked_periods, "expected at least one holding period with picks for tp/sl assertions"
    tp = picked_periods[0].get("tp_sl") or {}

    if mode == "prev_week_low_stop":
        assert isinstance(tp.get("stop_loss_level_by_code"), dict)
        assert "triggered" in tp  # bool
        return
    if mode in {"atr_chandelier_fixed", "atr_chandelier_progressive"}:
        assert tp.get("atr_window_used") is not None
        assert isinstance(tp.get("initial_stop_by_code"), dict)
        assert "triggered" in tp
        assert isinstance(tp.get("final_stop_by_code"), dict)
        return
    raise AssertionError(f"unexpected tp_sl_mode={mode}")


def _assert_corr_gate(out: dict, *, enabled: bool, expect_block: bool) -> None:
    for p in out.get("holdings", []):
        cf = p.get("corr_filter")
        assert isinstance(cf, dict)
        assert cf.get("enabled") in {True, False}
        if enabled:
            assert cf.get("enabled") is True
            assert isinstance(cf.get("window"), int)
            assert isinstance(cf.get("threshold"), float)
            assert "blocked" in cf
    if enabled and expect_block:
        assert any(bool((p.get("corr_filter") or {}).get("blocked")) for p in out.get("holdings", [])), "expected at least one blocked period"


def _assert_rr_sizing(out: dict, *, enabled: bool, expect_scaled: bool) -> None:
    exposures = []
    for p in out.get("holdings", []):
        rr = p.get("rr_sizing")
        assert isinstance(rr, dict)
        assert rr.get("enabled") in {True, False}
        if enabled:
            assert rr.get("enabled") is True
            assert isinstance(rr.get("years"), float)
            assert isinstance(rr.get("window_days"), int)
            assert "exposure" in rr
            if rr.get("exposure") is not None:
                exposures.append(float(rr["exposure"]))
    if enabled and expect_scaled:
        assert any(x < 0.999 for x in exposures), "expected at least one period with exposure < 1"


def _assert_dd_control(out: dict, *, enabled: bool, expect_trigger: bool, expect_sleep: bool) -> None:
    for p in out.get("holdings", []):
        dd = p.get("dd_control")
        assert isinstance(dd, dict)
        assert dd.get("enabled") in {True, False}
        if enabled:
            assert dd.get("enabled") is True
            assert isinstance(dd.get("threshold"), float)
            assert isinstance(dd.get("reduce"), float)
            assert isinstance(dd.get("sleep_days"), int)
            assert "in_sleep" in dd
            assert "triggered" in dd
    if enabled and expect_trigger:
        assert any(bool((p.get("dd_control") or {}).get("triggered")) for p in out.get("holdings", [])), "expected at least one dd trigger"
    if enabled and expect_sleep:
        assert any(bool((p.get("dd_control") or {}).get("in_sleep")) for p in out.get("holdings", [])), "expected at least one in_sleep period"
    if enabled and expect_trigger:
        # This fixture constructs one crash; should not repeatedly re-trigger every period.
        n = sum(1 for p in out.get("holdings", []) if bool((p.get("dd_control") or {}).get("triggered")))
        assert n <= 2


def _assert_mirror_control(out: dict, *, enabled: bool, expect_scaled: bool) -> None:
    """
    Mirror control is reported under per-period risk_controls.mirror_control.
    When enabled, it may cap total risk-asset exposure (cash remainder).
    """
    metas = []
    exposures = []
    for p in out.get("holdings", []):
        rc = (p.get("risk_controls") or {}).get("mirror_control")
        if rc is None:
            continue
        assert isinstance(rc, dict)
        assert rc.get("enabled") in {True, False}
        metas.append(rc)
        try:
            exposures.append(float(p.get("exposure")))
        except (TypeError, ValueError):  # pragma: no cover
            pass

    if enabled:
        assert metas, "expected mirror_control metadata when enabled"
        assert any(bool(m.get("enabled")) for m in metas)
    if enabled and expect_scaled:
        assert any((x < 0.999) for x in exposures), "expected at least one capped exposure < 1"


CASES = [
    dict(
        name="base",
        seed="regime",
        cfg=dict(),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="prev_period_stop",
        seed="regime",
        cfg=dict(tp_sl_mode="prev_week_low_stop"),
        expect=dict(
            tp_sl_mode="prev_week_low_stop",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="atr_fixed",
        seed="regime",
        cfg=dict(tp_sl_mode="atr_chandelier_fixed", atr_window=10, atr_mult=2.0),
        expect=dict(
            tp_sl_mode="atr_chandelier_fixed",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="atr_progressive",
        seed="regime",
        cfg=dict(tp_sl_mode="atr_chandelier_progressive", atr_window=10, atr_mult=2.0, atr_step=0.5, atr_min_mult=0.5),
        expect=dict(
            tp_sl_mode="atr_chandelier_progressive",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="corr_gate_blocks",
        seed="high_corr_switch",
        cfg=dict(corr_filter=True, corr_window=10, corr_threshold=0.5, lookback_days=2),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=True,
            corr_block=True,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="rr_sizing_scales",
        seed="regime",
        cfg=dict(rr_sizing=True, rr_years=0.2, rr_thresholds=[0.0], rr_weights=[1.0, 0.6]),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=True,
            rr_scaled=True,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="mirror_caps_exposure",
        seed="regime",
        cfg=dict(mirror_control=True, mirror_quantiles=[0.1], mirror_exposures=[0.3]),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=True,
            mirror_scaled=True,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="dd_control_triggers_sleep",
        seed="drawdown",
        cfg=dict(dd_control=True, dd_threshold=0.10, dd_reduce=1.0, dd_sleep_days=10, lookback_days=2),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=True,
            dd_trigger=True,
            dd_sleep=True,
        ),
    ),
    dict(
        name="combo_tp_sl_corr_rr",
        seed="regime",
        cfg=dict(
            tp_sl_mode="atr_chandelier_fixed",
            atr_window=10,
            atr_mult=2.0,
            corr_filter=True,
            corr_window=20,
            corr_threshold=0.5,
            rr_sizing=True,
            rr_years=0.2,
            rr_thresholds=[0.0],
            rr_weights=[1.0, 0.6],
        ),
        expect=dict(
            tp_sl_mode="atr_chandelier_fixed",
            corr_enabled=True,
            corr_block=False,
            rr_enabled=True,
            rr_scaled=True,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="risk_controls_combo",
        seed="regime",
        cfg=dict(
            trend_filter=True,
            trend_mode="each",
            trend_sma_window=20,
            rsi_filter=True,
            rsi_window=14,
            vol_monitor=True,
            vol_window=20,
            vol_target_ann=0.2,
            vol_max_ann=0.6,
            chop_filter=True,
            chop_mode="adx",
            chop_adx_window=20,
            chop_adx_threshold=20.0,
        ),
        expect=dict(
            tp_sl_mode="none",
            corr_enabled=False,
            corr_block=False,
            rr_enabled=False,
            rr_scaled=False,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
    dict(
        name="all_on",
        seed="regime",
        cfg=dict(
            tp_sl_mode="atr_chandelier_progressive",
            atr_window=10,
            atr_mult=2.0,
            atr_step=0.5,
            atr_min_mult=0.5,
            corr_filter=True,
            corr_window=20,
            corr_threshold=0.5,
            rr_sizing=True,
            rr_years=0.2,
            rr_thresholds=[0.0, 0.5, 1.0],
            rr_weights=[1.0, 0.8, 0.6, 0.4],
            trend_filter=True,
            trend_mode="universe",
            trend_sma_window=20,
            rsi_filter=True,
            rsi_window=14,
            rsi_overbought=100.0,
            rsi_block_overbought=False,
            vol_monitor=True,
            vol_window=20,
            vol_target_ann=0.2,
            vol_max_ann=0.6,
            chop_filter=True,
            chop_mode="er",
            chop_window=20,
            chop_er_threshold=0.25,
        ),
        expect=dict(
            tp_sl_mode="atr_chandelier_progressive",
            corr_enabled=True,
            corr_block=False,
            rr_enabled=True,
            rr_scaled=True,
            mirror_enabled=False,
            mirror_scaled=False,
            dd_enabled=False,
            dd_trigger=False,
            dd_sleep=False,
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda x: x["name"])
def test_rotation_parameter_behavior_matrix(session_factory, case):
    """
    Behavior assertion matrix:
    - Ensures parameter combinations run
    - AND asserts expected behavioral invariants per feature family.
    """
    sf = session_factory
    seed = case["seed"]
    cfg = dict(case["cfg"])
    expect = case["expect"]
    rebalance = cfg.pop("rebalance", "weekly")
    lookback_days = int(cfg.pop("lookback_days", 10))
    skip_days = int(cfg.pop("skip_days", 0))
    cost_bps = float(cfg.pop("cost_bps", 0.0))

    if seed == "high_corr_switch":
        with sf() as db:
            codes, dates = _seed_prices_high_corr_switch(db, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19))
            out = backtest_rotation(
                db,
                RotationInputs(
                    codes=codes,
                    start=dates[0],
                    end=dates[-1],
                    rebalance=rebalance,
                    top_k=1,
                    lookback_days=lookback_days,
                    skip_days=skip_days,
                    cost_bps=cost_bps,
                    risk_off=False,
                    **cfg,
                ),
            )
    elif seed == "drawdown":
        with sf() as db:
            codes, dates = _seed_prices_drawdown_crash(db, start=dt.date(2024, 1, 1), end=dt.date(2024, 2, 29))
            out = backtest_rotation(
                db,
                RotationInputs(
                    codes=codes,
                    start=dates[0],
                    end=dates[-1],
                    rebalance=rebalance,
                    top_k=1,
                    lookback_days=lookback_days,
                    skip_days=skip_days,
                    cost_bps=cost_bps,
                    risk_off=False,
                    **cfg,
                ),
            )
    else:
        codes = ["AAA", "BBB", "CCC"]
        with sf() as db:
            dates = _seed_prices_regime_switch(db, codes=codes, start=dt.date(2024, 1, 1), end=dt.date(2024, 8, 30))
            out = backtest_rotation(
                db,
                RotationInputs(
                    codes=codes,
                    start=dates[0],
                    end=dates[-1],
                    rebalance=rebalance,
                    top_k=1,
                    lookback_days=lookback_days,
                    skip_days=skip_days,
                    cost_bps=cost_bps,
                    risk_off=False,
                    **cfg,
                ),
            )

    _assert_common_payload(out)
    _assert_tp_sl(out, expect["tp_sl_mode"])
    _assert_corr_gate(out, enabled=bool(expect["corr_enabled"]), expect_block=bool(expect["corr_block"]))
    _assert_rr_sizing(out, enabled=bool(expect["rr_enabled"]), expect_scaled=bool(expect["rr_scaled"]))
    _assert_mirror_control(out, enabled=bool(expect["mirror_enabled"]), expect_scaled=bool(expect["mirror_scaled"]))
    _assert_dd_control(out, enabled=bool(expect["dd_enabled"]), expect_trigger=bool(expect["dd_trigger"]), expect_sleep=bool(expect["dd_sleep"]))

