from __future__ import annotations

import datetime as dt

import numpy as np
from fastapi.testclient import TestClient

from etf_momentum.db.models import EtfPrice, OffFundNav, OffFundPool
from etf_momentum.db.session import make_session_factory


def _cum_nav_from_returns(rets: np.ndarray, base: float = 1.0) -> np.ndarray:
    out = np.empty_like(rets, dtype=float)
    nav = float(base)
    for i, r in enumerate(rets):
        nav = nav * (1.0 + float(r))
        out[i] = nav
    return out


def _seed_classification_fixture(client: TestClient) -> tuple[str, str]:
    engine = client.app.state.engine
    sf = make_session_factory(engine)
    n = 320
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    x = np.arange(n, dtype=float)
    rng = np.random.default_rng(7)

    r300 = 0.0004 + 0.0060 * np.sin(x / 19.0)
    r500 = 0.0002 + 0.0050 * np.cos(x / 17.0)
    r1000 = 0.0001 + 0.0070 * np.sin(x / 13.0 + 0.2)
    r2000 = -0.0001 + 0.0090 * np.cos(x / 11.0 + 0.1)

    f1 = 0.72 * r300 + 0.18 * r500 + 0.03 * r1000 + rng.normal(0.0, 0.0008, size=n)
    f2 = 0.68 * r2000 + 0.16 * r1000 + 0.06 * r500 + rng.normal(0.0, 0.0009, size=n)
    f1 = np.clip(f1, -0.2, 0.2)
    f2 = np.clip(f2, -0.2, 0.2)

    nav_f1 = _cum_nav_from_returns(f1, base=1.0)
    nav_f2 = _cum_nav_from_returns(f2, base=1.0)
    px300 = _cum_nav_from_returns(r300, base=1.0)
    px500 = _cum_nav_from_returns(r500, base=1.0)
    px1000 = _cum_nav_from_returns(r1000, base=1.0)
    px2000 = _cum_nav_from_returns(r2000, base=1.0)

    with sf() as db:
        db.add(
            OffFundPool(
                code="FUND1",
                name="样本基金1",
                start_date="20240101",
                end_date="20241231",
            )
        )
        db.add(
            OffFundPool(
                code="FUND2",
                name="样本基金2",
                start_date="20240101",
                end_date="20241231",
            )
        )
        for d, n1, n2 in zip(dates, nav_f1, nav_f2):
            db.add(
                OffFundNav(
                    code="FUND1",
                    trade_date=d,
                    nav=float(n1),
                    accum_nav=None,
                    source="unit_test",
                    adjust="hfq",
                )
            )
            db.add(
                OffFundNav(
                    code="FUND2",
                    trade_date=d,
                    nav=float(n2),
                    accum_nav=None,
                    source="unit_test",
                    adjust="hfq",
                )
            )
        for d, p300, p500, p1000, p2000 in zip(dates, px300, px500, px1000, px2000):
            for code, px in (
                ("000300", p300),
                ("000905", p500),
                ("000852", p1000),
                ("932000", p2000),
            ):
                db.add(
                    EtfPrice(
                        code=code,
                        trade_date=d,
                        open=float(px),
                        high=float(px),
                        low=float(px),
                        close=float(px),
                        volume=1000.0,
                        amount=float(px * 1000.0),
                        source="unit_test",
                        adjust="hfq",
                    )
                )
        db.commit()
    return dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")


def test_api_off_fund_classify_cn_stock_profile(api_client: TestClient) -> None:
    start, end = _seed_classification_fixture(api_client)
    resp = api_client.post(
        "/api/analysis/off-fund/classify",
        json={
            "codes": ["FUND1", "FUND2"],
            "start": start,
            "end": end,
            "fund_adjust": "hfq",
            "benchmark_adjust": "hfq",
            "benchmark_profile": "cn_stock_core",
            "rolling_window": 252,
            "min_samples": 120,
            "dominance_gap": 0.08,
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    items = out["items"]
    assert len(items) == 2
    by_code = {x["code"]: x for x in items}
    assert by_code["FUND1"]["status"] == "ok"
    assert by_code["FUND2"]["status"] == "ok"
    assert by_code["FUND1"]["primary_asset_class"] == "equity"
    assert by_code["FUND2"]["primary_asset_class"] == "equity"
    assert by_code["FUND1"]["avg_r2"] is not None
    assert by_code["FUND2"]["avg_r2"] is not None
    assert by_code["FUND1"]["avg_r2"] > 0.2
    assert by_code["FUND2"]["avg_r2"] > 0.2
    assert "A股" in str(by_code["FUND1"]["label"])
    assert "A股" in str(by_code["FUND2"]["label"])


def test_api_off_fund_classify_returns_error_when_benchmark_empty(
    api_client: TestClient,
) -> None:
    engine = api_client.app.state.engine
    sf = make_session_factory(engine)
    d0 = dt.date(2024, 1, 1)
    with sf() as db:
        db.add(OffFundPool(code="FUNDX", name="样本X", start_date="20240101", end_date="20240201"))
        for i in range(140):
            d = d0 + dt.timedelta(days=i)
            db.add(
                OffFundNav(
                    code="FUNDX",
                    trade_date=d,
                    nav=float(1.0 + 0.001 * i),
                    accum_nav=None,
                    source="unit_test",
                    adjust="hfq",
                )
            )
        db.commit()
    resp = api_client.post(
        "/api/analysis/off-fund/classify",
        json={
            "codes": ["FUNDX"],
            "start": "20240101",
            "end": "20240630",
            "fund_adjust": "hfq",
            "benchmark_adjust": "hfq",
            "benchmark_profile": "cn_stock_core",
            "rolling_window": 120,
            "min_samples": 80,
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is False
    assert out["error"] == "empty_benchmark_series"
