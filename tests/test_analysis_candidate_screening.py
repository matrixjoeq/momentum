import datetime as dt

import pandas as pd

from etf_momentum.analysis.candidate_screening import (
    RotationCandidateScreenInputs,
    screen_rotation_candidates,
)
from etf_momentum.db.models import EtfPool
from tests.helpers.price_seed import add_price_all_adjustments


def _seed_price(db, code: str, dates: list[dt.date], closes: list[float]) -> None:
    for d, c in zip(dates, closes):
        add_price_all_adjustments(
            db,
            code=code,
            day=d,
            close=float(c),
            open_price=float(c),
            high=float(c),
            low=float(c),
        )


def test_candidate_screen_returns_significance_report(session_factory):
    dates = [d.date() for d in pd.date_range("2023-01-01", "2024-12-31", freq="B")]
    with session_factory() as db:
        db.add(EtfPool(code="A", name="中证A", start_date=None, end_date=None))
        db.add(EtfPool(code="B", name="中证B", start_date=None, end_date=None))
        db.add(EtfPool(code="C", name="纳指ETF", start_date=None, end_date=None))
        db.add(EtfPool(code="D", name="国债ETF", start_date=None, end_date=None))
        db.flush()

        n = len(dates)
        a = [100.0 + i * 0.25 for i in range(n)]
        b = [100.0 + i * 0.10 + (0.8 if i % 7 < 3 else -0.8) for i in range(n)]
        c = [95.0 + i * 0.22 + (0.6 if i % 9 < 4 else -0.4) for i in range(n)]
        d = [100.0 + i * 0.03 for i in range(n)]
        _seed_price(db, "A", dates, a)
        _seed_price(db, "B", dates, b)
        _seed_price(db, "C", dates, c)
        _seed_price(db, "D", dates, d)
        db.commit()

        out = screen_rotation_candidates(
            db,
            RotationCandidateScreenInputs(
                codes=["A", "B", "C", "D"],
                start=dates[0],
                end=dates[-1],
                lookback_days=252,
                top_n=3,
                min_n=2,
                max_pair_corr=0.8,
            ),
        )
    assert "selected_codes" in out and len(out["selected_codes"]) >= 2
    assert "significance_report" in out
    assert "rows" in out["significance_report"]
    assert len(out["significance_report"]["rows"]) == 4


def test_candidate_screen_respects_category_quotas(session_factory):
    dates = [d.date() for d in pd.date_range("2023-01-01", "2024-12-31", freq="B")]
    with session_factory() as db:
        db.add(EtfPool(code="CN1", name="沪深300ETF", start_date=None, end_date=None))
        db.add(EtfPool(code="US1", name="纳指ETF", start_date=None, end_date=None))
        db.add(EtfPool(code="BD1", name="国债ETF", start_date=None, end_date=None))
        db.add(EtfPool(code="CM1", name="有色ETF", start_date=None, end_date=None))
        db.flush()
        n = len(dates)
        _seed_price(db, "CN1", dates, [100.0 + i * 0.20 for i in range(n)])
        _seed_price(db, "US1", dates, [90.0 + i * 0.24 for i in range(n)])
        _seed_price(db, "BD1", dates, [100.0 + i * 0.04 for i in range(n)])
        _seed_price(db, "CM1", dates, [80.0 + i * 0.18 for i in range(n)])
        db.commit()

        out = screen_rotation_candidates(
            db,
            RotationCandidateScreenInputs(
                codes=["CN1", "US1", "BD1", "CM1"],
                start=dates[0],
                end=dates[-1],
                lookback_days=252,
                top_n=3,
                min_n=3,
                max_pair_corr=0.1,
                category_quotas={"US_EQ": 1, "BOND": 1},
            ),
        )
    picked = set(out["selected_codes"])
    assert "US1" in picked
    assert "BD1" in picked


def test_candidate_screen_excludes_low_sample_assets(session_factory):
    dates = [d.date() for d in pd.date_range("2023-01-01", "2024-12-31", freq="B")]
    with session_factory() as db:
        db.add(EtfPool(code="A1", name="中证A1", start_date=None, end_date=None))
        db.add(EtfPool(code="B1", name="中证B1", start_date=None, end_date=None))
        db.add(EtfPool(code="L1", name="低样本标的", start_date=None, end_date=None))
        db.flush()
        n = len(dates)
        _seed_price(db, "A1", dates, [100.0 + i * 0.18 for i in range(n)])
        _seed_price(db, "B1", dates, [95.0 + i * 0.20 for i in range(n)])
        _seed_price(db, "L1", dates[-10:], [80.0 + i * 0.50 for i in range(10)])
        db.commit()

        out = screen_rotation_candidates(
            db,
            RotationCandidateScreenInputs(
                codes=["A1", "B1", "L1"],
                start=dates[0],
                end=dates[-1],
                lookback_days=252,
                top_n=3,
                min_n=2,
                max_pair_corr=0.8,
            ),
        )
    assert "L1" not in set(out["selected_codes"])
    row_l1 = next(r for r in out["details"] if str(r.get("code")) == "L1")
    assert row_l1["eligible"] is False
    assert row_l1["not_selected_reason"] == "insufficient_samples"
