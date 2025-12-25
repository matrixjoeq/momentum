from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from etf_momentum.db.init_db import init_db
from etf_momentum.db.seed import ensure_default_policies
from etf_momentum.db.session import make_engine, make_session_factory


@pytest.fixture()
def sqlite_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "test.sqlite3"


@pytest.fixture()
def session_factory(sqlite_path: pathlib.Path):
    engine = make_engine(str(sqlite_path))
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture()
def api_client(sqlite_path: pathlib.Path):
    """
    API client isolated from local filesystem DB:
    - uses tmp sqlite
    - overrides FastAPI dependencies for get_session/get_akshare
    """
    from etf_momentum.app import create_app
    from etf_momentum.db.session import session_scope
    import etf_momentum.api.routes as routes

    engine = make_engine(str(sqlite_path))
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    class FakeAk:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

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
        return FakeAk()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare

    return TestClient(app)

