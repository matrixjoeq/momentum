from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from etf_momentum.app import create_app
from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_engine, make_session_factory, session_scope


def test_fetch_fails_if_adjust_ranges_mismatch(tmp_path: pathlib.Path):
    import etf_momentum.api.routes as routes

    sqlite_path = tmp_path / "test.sqlite3"
    engine = make_engine(str(sqlite_path))
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    class AkMismatch:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            adj = kwargs.get("adjust")
            # make qfq missing the last day
            dates = ["2024-01-02", "2024-01-03"] if adj != "qfq" else ["2024-01-02"]
            return pd.DataFrame(
                {
                    "日期": dates,
                    "开盘": [1.0] * len(dates),
                    "最高": [1.05] * len(dates),
                    "最低": [0.98] * len(dates),
                    "收盘": [1.02] * len(dates),
                    "成交量": [10.0] * len(dates),
                    "成交额": [100.0] * len(dates),
                }
            )

    app = create_app()

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return AkMismatch()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare

    with TestClient(app) as c:
        c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
        r = c.post("/api/etf/510300/fetch")
        assert r.status_code == 500
        # fetch status should be failed with range_check info
        etfs = c.get("/api/etf?adjust=hfq").json()
        assert etfs[0]["last_fetch_status"] == "failed"
        assert "range_check:failed" in (etfs[0]["last_fetch_message"] or "")

