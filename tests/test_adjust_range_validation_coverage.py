from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from etf_momentum.app import create_app
from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_session_factory, make_sqlite_engine, session_scope


def _make_client(tmp_path: pathlib.Path, ak_obj):
    import etf_momentum.api.routes as routes

    _ = tmp_path  # kept for call signature compatibility
    engine = make_sqlite_engine()
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    app = create_app()
    # Prevent app lifespan from creating a MySQL engine during tests.
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return ak_obj

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare
    return TestClient(app), routes


def test_rollback_failure_is_reported_in_fetch_one(tmp_path: pathlib.Path):
    class AkMismatch:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            adj = kwargs.get("adjust")
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

    c, routes = _make_client(tmp_path, AkMismatch())
    old = routes.logical_rollback_batch
    routes.logical_rollback_batch = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("rb"))  # type: ignore
    try:
        with c:
            c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
            r = c.post("/api/etf/510300/fetch")
            assert r.status_code == 500
            assert "rollback:failed" in r.text
    finally:
        routes.logical_rollback_batch = old  # type: ignore


def test_rollback_failure_is_reported_in_fetch_all(tmp_path: pathlib.Path):
    class AkMismatch:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            adj = kwargs.get("adjust")
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

    c, routes = _make_client(tmp_path, AkMismatch())
    old = routes.logical_rollback_batch
    routes.logical_rollback_batch = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("rb"))  # type: ignore
    try:
        with c:
            c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
            r = c.post("/api/fetch-all")
            assert r.status_code == 200
            msg = r.json()[0]["message"] or ""
            assert "rollback:failed" in msg
    finally:
        routes.logical_rollback_batch = old  # type: ignore


def test_rollback_failure_is_reported_in_fetch_selected(tmp_path: pathlib.Path):
    class AkMismatch:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            adj = kwargs.get("adjust")
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

    c, routes = _make_client(tmp_path, AkMismatch())
    old = routes.logical_rollback_batch
    routes.logical_rollback_batch = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("rb"))  # type: ignore
    try:
        with c:
            c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
            r = c.post("/api/fetch-selected", json={"codes": ["510300"]})
            assert r.status_code == 200
            msg = r.json()[0]["message"] or ""
            assert "rollback:failed" in msg
    finally:
        routes.logical_rollback_batch = old  # type: ignore

