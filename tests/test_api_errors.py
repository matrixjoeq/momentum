from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import delete_json, get_json, post_json


def test_upsert_date_validation_errors(api_client: TestClient) -> None:
    # bad start_date length
    post_json(
        api_client,
        "/api/etf",
        {
            "code": "510300",
            "name": "沪深300ETF",
            "start_date": "202401",
            "end_date": None,
        },
        expected_status=400,
    )

    # bad end_date length
    post_json(
        api_client,
        "/api/etf",
        {
            "code": "510300",
            "name": "沪深300ETF",
            "start_date": None,
            "end_date": "2024",
        },
        expected_status=400,
    )

    # start > end
    post_json(
        api_client,
        "/api/etf",
        {
            "code": "510300",
            "name": "沪深300ETF",
            "start_date": "20240201",
            "end_date": "20240101",
        },
        expected_status=400,
    )


def test_delete_missing_etf_404(api_client: TestClient) -> None:
    delete_json(api_client, "/api/etf/NOPE", expected_status=404)


def test_fetch_missing_etf_404(api_client: TestClient) -> None:
    post_json(api_client, "/api/etf/NOPE/fetch", {}, expected_status=404)


def test_fetch_failure_sets_status_failed(api_client: TestClient) -> None:
    # create ETF first
    post_json(api_client, "/api/etf", {"code": "510300", "name": "沪深300ETF"})

    class BadAk:
        def fund_etf_hist_em(self, **kwargs):
            raise RuntimeError("boom")

    # override akshare dependency to force failure
    import etf_momentum.api.routes as routes

    api_client.app.dependency_overrides[routes.get_akshare] = lambda: BadAk()

    post_json(api_client, "/api/etf/510300/fetch", {}, expected_status=500)

    # status should be marked failed
    item = get_json(api_client, "/api/etf")[0]
    assert item["last_fetch_status"] == "failed"
    # Fetchers may swallow provider exceptions and report an aggregated fetch failure message.
    assert (item["last_fetch_message"] or "") != ""


def test_fetch_all_has_failed_and_success(api_client: TestClient) -> None:
    post_json(api_client, "/api/etf", {"code": "510300", "name": "沪深300ETF"})
    post_json(api_client, "/api/etf", {"code": "588000", "name": "科创50ETF"})

    class MixedAk:
        def fund_etf_hist_em(self, **kwargs):
            if kwargs.get("symbol") == "588000":
                raise RuntimeError("bad-symbol")
            return pd.DataFrame(
                {
                    "日期": ["2024-01-02"],
                    "开盘": [1.0],
                    "最高": [1.15],
                    "最低": [0.95],
                    "收盘": [1.1],
                }
            )

    import etf_momentum.api.routes as routes

    api_client.app.dependency_overrides[routes.get_akshare] = lambda: MixedAk()
    data = post_json(api_client, "/api/fetch-all", {})
    assert any(x["code"] == "510300" and x["status"] == "success" for x in data)
    assert any(x["code"] == "588000" and x["status"] == "failed" for x in data)
