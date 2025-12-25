from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from etf_momentum.app import create_app
from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_engine, make_session_factory, session_scope


def test_fetch_selected_partial_failure_sets_failed_status(tmp_path: pathlib.Path):
    import etf_momentum.api.routes as routes

    sqlite_path = tmp_path / "test.sqlite3"
    engine = make_engine(str(sqlite_path))
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    class FakeAkMixed:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            symbol = kwargs.get("symbol")
            if symbol == "BAD":
                return pd.DataFrame(
                    {
                        "日期": ["2024-01-02", "2024-01-03"],
                        "开盘": [1.0, 1.0],
                        "最高": [1.0, 1.0],
                        "最低": [1.1, 1.1],  # invalid: low > high
                        "收盘": [1.05, 1.05],
                        "成交量": [10.0, 20.0],
                        "成交额": [100.0, 200.0],
                    }
                )
            return pd.DataFrame(
                {
                    "日期": ["2024-01-02", "2024-01-03"],
                    "开盘": [1.0, 1.02],
                    "最高": [1.05, 1.06],
                    "最低": [0.98, 1.00],
                    "收盘": [1.02, 1.03],
                    "成交量": [10.0, 20.0],
                    "成交额": [100.0, 200.0],
                }
            )

    app = create_app()

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return FakeAkMixed()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare

    with TestClient(app) as c:
        c.post("/api/etf", json={"code": "GOOD", "name": "GOOD", "start_date": "20240102", "end_date": "20240103"})
        c.post("/api/etf", json={"code": "BAD", "name": "BAD", "start_date": "20240102", "end_date": "20240103"})
        resp = c.post("/api/fetch-selected", json={"codes": ["GOOD", "BAD"], "adjust": "hfq"})
        assert resp.status_code == 200
        out = resp.json()
        by_code = {x["code"]: x for x in out}
        assert by_code["GOOD"]["status"] == "success"
        assert by_code["BAD"]["status"] == "failed"

