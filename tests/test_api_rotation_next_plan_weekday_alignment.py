from tests.helpers.api_test_client import FIXED_MINIPROGRAM_POOL, upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import post_json, post_json_ok


def _build_test_client_with_fake_ak(*, fetch_end_date: str):
    from fastapi.testclient import TestClient

    from etf_momentum.app import create_app
    from etf_momentum.db.init_db import init_db
    from etf_momentum.db.seed import ensure_default_policies
    from etf_momentum.db.session import (
        make_session_factory,
        make_sqlite_engine,
        session_scope,
    )
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
            end_date = str(kwargs.get("end_date") or fetch_end_date)
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
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
    app.state.engine = engine
    app.state.session_factory = sf

    def override_get_session():
        yield from session_scope(sf)

    def override_get_akshare():
        return FakeAkRange()

    app.dependency_overrides[routes.get_session] = override_get_session
    app.dependency_overrides[routes.get_akshare] = override_get_akshare
    return TestClient(app)


def _assert_next_plan_payload(
    c, *, anchor_weekday: int, asof: str = "20240104"
) -> dict:
    return post_json_ok(
        c,
        "/api/analysis/rotation/next-plan",
        {"anchor_weekday": int(anchor_weekday), "asof": str(asof)},
    )


def test_api_rotation_next_plan_shows_on_execution_day_tab():
    """
    This test needs >=21 trading days of history for next-plan (lookback=20),
    so we use a FakeAk that returns a date range instead of the default 2-row stub.
    """
    c = _build_test_client_with_fake_ak(fetch_end_date="20240105")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240105",
    )

    # 2024-01-04 is Thu; next trading day is 2024-01-05 (Fri) in XSHG.
    # Thu tab should NOT show the plan; Fri tab SHOULD show it.
    data_thu = _assert_next_plan_payload(c, anchor_weekday=4)
    assert data_thu["next_trading_day"] == "2024-01-05"
    assert data_thu["rebalance_effective_next_day"] is False

    data_fri = _assert_next_plan_payload(c, anchor_weekday=5)
    assert data_fri["next_trading_day"] == "2024-01-05"
    assert data_fri["rebalance_effective_next_day"] is True
    assert data_fri["pick_code"] is not None


def test_api_rotation_next_plan_does_not_require_future_price_row():
    """
    Real-world behavior: when asking "tomorrow plan" intraday / after close,
    the DB typically does NOT yet have any rows for the next trading day.

    next-plan should still return a concrete pick (not empty) based on asof-close decision.
    """
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )

    # 2024-01-04 is Thu; next trading day is 2024-01-05 (Fri) in XSHG.
    data = _assert_next_plan_payload(c, anchor_weekday=5)
    assert data["next_trading_day"] == "2024-01-05"
    assert data["rebalance_effective_next_day"] is True
    assert data["pick_code"] is not None
    assert data["pick_name"] is not None
    assert data["pick_exposure"] is not None
    assert float(data["pick_exposure"]) >= 0.0


def test_api_rotation_next_plan_accepts_equity_budget_stop_params():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "close",
        },
    )
    assert data["next_trading_day"] == "2024-01-05"
    assert data["pick_code"] is not None


def test_api_rotation_next_plan_auto_accepts_equity_budget_stop_params():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "close",
        },
    )
    assert data["next_trading_day"] == "2024-01-05"
    assert data["pick_code"] is not None


def test_api_rotation_next_plan_rejects_equity_budget_non_close_execution():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)


def test_api_rotation_next_plan_rejects_atr_scheme_with_none_mode():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "none",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "stop_scheme=atr requires atr_stop_mode" in str(err)


def test_api_rotation_next_plan_rejects_equity_budget_non_close_even_when_not_effective_day():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    # 2024-01-04 is Thu and next trading day is Fri.
    # Use Thu anchor (4) so next-plan would short-circuit without execution.
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 4,
            "asof": "20240104",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)


def test_api_rotation_next_plan_auto_rejects_equity_budget_non_close_execution():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)


def test_api_rotation_next_plan_rejects_missing_anchor_weekday():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "asof": "20240104",
        },
        expected_status=400,
    )
    assert "anchor_weekday must be 1..5" in str(err)


def test_api_rotation_next_plan_rejects_invalid_asof_format():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "2024-01-04",
        },
        expected_status=400,
    )
    assert "asof must be YYYYMMDD" in str(err)


def test_api_rotation_next_plan_auto_rejects_invalid_asof_format():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "2024-01-04",
        },
        expected_status=400,
    )
    assert "asof must be YYYYMMDD" in str(err)


def test_api_rotation_next_plan_auto_rejects_atr_scheme_with_none_mode():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "none",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "stop_scheme=atr requires atr_stop_mode" in str(err)


def test_api_rotation_next_plan_rejects_tightening_invalid_nm():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "tightening",
            "atr_stop_n": 0.5,
            "atr_stop_m": 1.0,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_n must be > atr_stop_m" in str(err)


def test_api_rotation_next_plan_auto_rejects_tightening_invalid_nm():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "tightening",
            "atr_stop_n": 0.5,
            "atr_stop_m": 1.0,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_n must be > atr_stop_m" in str(err)


def test_api_rotation_next_plan_rejects_invalid_atr_basis():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "static",
            "atr_stop_atr_basis": "oops",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_atr_basis must be one of: entry|latest" in str(err)


def test_api_rotation_next_plan_auto_rejects_invalid_atr_basis():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "static",
            "atr_stop_atr_basis": "oops",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_atr_basis must be one of: entry|latest" in str(err)


def test_api_rotation_next_plan_rejects_invalid_atr_reentry_mode():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "anchor_weekday": 5,
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "static",
            "atr_stop_reentry_mode": "hold",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_reentry_mode must be one of: reenter|wait_next_entry" in str(err)


def test_api_rotation_next_plan_auto_rejects_invalid_atr_reentry_mode():
    c = _build_test_client_with_fake_ak(fetch_end_date="20240104")
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20231101",
        end_date="20240104",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/next-plan-auto",
        {
            "asof": "20240104",
            "stop_scheme": "atr",
            "atr_stop_mode": "static",
            "atr_stop_reentry_mode": "hold",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
        },
        expected_status=422,
    )
    assert "atr_stop_reentry_mode must be one of: reenter|wait_next_entry" in str(err)
