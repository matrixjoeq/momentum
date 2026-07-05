from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi.testclient import TestClient

from etf_momentum.db.models import EtfPool
from tests.helpers.price_seed import add_price_all_adjustments


def _seed_price_series(
    db,
    *,
    code: str,
    dates: list[dt.date],
    closes: list[float],
) -> None:
    for d, c in zip(dates, closes):
        px = float(c)
        add_price_all_adjustments(
            db,
            code=code,
            day=d,
            close=px,
            open_price=px,
            high=px,
            low=px,
        )


def test_rotation_candidate_screen_contract_accepts_skip_days(
    api_client: TestClient, session_factory
) -> None:
    dates = [d.date() for d in pd.date_range("2022-01-03", "2024-12-31", freq="B")]
    with session_factory() as db:
        db.add(EtfPool(code="A", name="中证A", start_date=None, end_date=None))
        db.add(EtfPool(code="B", name="中证B", start_date=None, end_date=None))
        n = len(dates)
        _seed_price_series(
            db,
            code="A",
            dates=dates,
            closes=[100.0 + 0.15 * i for i in range(n)],
        )
        _seed_price_series(
            db,
            code="B",
            dates=dates,
            closes=[90.0 + 0.12 * i + (0.4 if i % 9 < 4 else -0.2) for i in range(n)],
        )
        db.commit()

    payload = {
        "codes": ["A", "B"],
        "start": dates[0].strftime("%Y%m%d"),
        "end": dates[-1].strftime("%Y%m%d"),
        "adjust": "qfq",
        "lookback_days": 20,
        "skip_days": 0,
        "top_n": 2,
        "min_n": 1,
        "max_pair_corr": 0.95,
        "signif_horizon_days": 5,
    }
    resp = api_client.post("/api/analysis/rotation/candidate-screen", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["lookback_days"] == 20
    assert body["meta"]["skip_days"] == 0
    assert body["meta"]["signif_horizon_days"] == 5
    assert body["meta"]["signif_return_basis"] == "close_to_close_simple"
    rows = {str(r["code"]): r for r in body["significance_report"]["rows"]}
    assert int(rows["A"]["n"]) >= 30
    assert int(rows["B"]["n"]) >= 30


def test_rotation_candidate_screen_contract_rejects_negative_skip_days(
    api_client: TestClient,
) -> None:
    payload = {
        "codes": ["A", "B"],
        "start": "20240101",
        "end": "20241231",
        "adjust": "qfq",
        "lookback_days": 20,
        "skip_days": -1,
        "top_n": 2,
        "min_n": 1,
        "max_pair_corr": 0.9,
        "signif_horizon_days": 5,
    }
    resp = api_client.post("/api/analysis/rotation/candidate-screen", json=payload)
    assert resp.status_code == 422


def test_rotation_candidate_screen_contract_filters_nonpositive_close_samples(
    api_client: TestClient, session_factory
) -> None:
    dates = [d.date() for d in pd.date_range("2022-01-03", "2024-12-31", freq="B")]
    with session_factory() as db:
        db.add(EtfPool(code="X", name="异常样本", start_date=None, end_date=None))
        db.add(EtfPool(code="Z", name="对照样本", start_date=None, end_date=None))
        n = len(dates)
        base = [120.0 + 0.18 * i for i in range(n)]
        x = list(base)
        x[90] = 0.0
        _seed_price_series(db, code="X", dates=dates, closes=x)
        _seed_price_series(db, code="Z", dates=dates, closes=base)
        db.commit()

    payload = {
        "codes": ["X", "Z"],
        "start": dates[0].strftime("%Y%m%d"),
        "end": dates[-1].strftime("%Y%m%d"),
        "adjust": "qfq",
        "lookback_days": 20,
        "skip_days": 0,
        "top_n": 2,
        "min_n": 1,
        "max_pair_corr": 0.95,
        "signif_horizon_days": 5,
    }
    resp = api_client.post("/api/analysis/rotation/candidate-screen", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    rows = {str(r["code"]): r for r in body["significance_report"]["rows"]}
    assert int(rows["X"]["n"]) < int(rows["Z"]["n"])
    assert int(rows["X"]["n"]) >= 30
