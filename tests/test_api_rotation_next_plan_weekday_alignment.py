def test_api_rotation_next_plan_shows_on_execution_day_tab(tmp_path):
    """
    This test needs >=21 trading days of history for next-plan (lookback=20),
    so we use a FakeAk that returns a date range instead of the default 2-row stub.
    """
    from fastapi.testclient import TestClient

    from etf_momentum.app import create_app
    from etf_momentum.db.init_db import init_db
    from etf_momentum.db.seed import ensure_default_policies
    from etf_momentum.db.session import make_session_factory, make_sqlite_engine, session_scope
    import etf_momentum.api.routes as routes

    engine = make_sqlite_engine()
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        ensure_default_policies(db)
        db.commit()

    class FakeAkRange:
        def fund_etf_hist_em(self, **kwargs):
            import pandas as pd

            start_date = str(kwargs.get("start_date") or "20230101")
            end_date = str(kwargs.get("end_date") or "20240105")
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            # business days as a proxy for trading days in tests
            ds = pd.date_range(start=start, end=end, freq="B")
            n = len(ds)
            base = pd.Series(range(n), dtype=float)
            close = 1.0 + base * 0.001
            open_ = close.shift(1).fillna(close.iloc[0]).astype(float)
            high = (close * 1.01).astype(float)
            low = (close * 0.99).astype(float)
            vol = pd.Series([10.0] * n)
            amt = pd.Series([100.0] * n)
            return pd.DataFrame(
                {
                    "日期": ds.strftime("%Y-%m-%d"),
                    "开盘": open_.to_numpy(),
                    "最高": high.to_numpy(),
                    "最低": low.to_numpy(),
                    "收盘": close.to_numpy(),
                    "成交量": vol.to_numpy(),
                    "成交额": amt.to_numpy(),
                }
            )

    app = create_app()
    # Prevent app lifespan from creating a MySQL engine during tests.
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return FakeAkRange()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare

    c = TestClient(app)
    # create pool entries for the fixed codes and ingest enough history
    for code, name in [
        ("159915", "创业板ETF"),
        ("511010", "国债ETF"),
        ("513100", "纳指ETF"),
        ("518880", "黄金ETF"),
    ]:
        c.post("/api/etf", json={"code": code, "name": name, "start_date": "20231101", "end_date": "20240105"})
        assert c.post(f"/api/etf/{code}/fetch").status_code == 200

    # 2024-01-04 is Thu; next trading day is 2024-01-05 (Fri) in XSHG.
    # Thu tab should NOT show the plan; Fri tab SHOULD show it.
    resp_thu = c.post("/api/analysis/rotation/next-plan", json={"anchor_weekday": 3, "asof": "20240104"})
    assert resp_thu.status_code == 200
    data_thu = resp_thu.json()
    assert data_thu["next_trading_day"] == "2024-01-05"
    assert data_thu["rebalance_effective_next_day"] is False

    resp_fri = c.post("/api/analysis/rotation/next-plan", json={"anchor_weekday": 4, "asof": "20240104"})
    assert resp_fri.status_code == 200
    data_fri = resp_fri.json()
    assert data_fri["next_trading_day"] == "2024-01-05"
    assert data_fri["rebalance_effective_next_day"] is True
    assert data_fri["pick_code"] is not None

