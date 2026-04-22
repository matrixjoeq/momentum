import datetime as dt

from etf_momentum.data.off_fund_ingestion import (
    FundEvent,
    _build_adjusted_by_accum,
    _build_adjusted_by_events,
)


def test_build_adjusted_by_accum_keeps_latest_equal_to_raw() -> None:
    unit = {
        dt.date(2024, 1, 1): 1.00,
        dt.date(2024, 1, 2): 0.90,
        dt.date(2024, 1, 3): 0.95,
    }
    accum = {
        dt.date(2024, 1, 1): 1.00,
        dt.date(2024, 1, 2): 1.00,
        dt.date(2024, 1, 3): 1.10,
    }
    qfq, hfq = _build_adjusted_by_accum(unit, accum)
    # qfq latest anchors to latest raw nav
    assert abs(qfq[dt.date(2024, 1, 3)] - unit[dt.date(2024, 1, 3)]) < 1e-12
    # hfq follows unit * (accum/unit) = accum
    assert abs(hfq[dt.date(2024, 1, 2)] - accum[dt.date(2024, 1, 2)]) < 1e-12


def test_build_adjusted_by_events_dividend_adjusts_past_only() -> None:
    unit = {
        dt.date(2024, 1, 1): 1.00,
        dt.date(2024, 1, 2): 1.00,
        dt.date(2024, 1, 3): 0.95,  # ex-div day (cash 0.05)
        dt.date(2024, 1, 4): 0.96,
    }
    events = [
        FundEvent(
            effective_date=dt.date(2024, 1, 3),
            event_type="dividend",
            event_key="div",
            cash_dividend=0.05,
        )
    ]
    qfq, hfq, note = _build_adjusted_by_events(unit, events)
    assert note.startswith("events_applied=")
    # latest should remain unchanged in qfq
    assert abs(qfq[dt.date(2024, 1, 4)] - unit[dt.date(2024, 1, 4)]) < 1e-12
    # before ex-day should be adjusted down
    assert qfq[dt.date(2024, 1, 2)] < unit[dt.date(2024, 1, 2)]
    # hfq is positive and available
    assert hfq[dt.date(2024, 1, 1)] > 0
