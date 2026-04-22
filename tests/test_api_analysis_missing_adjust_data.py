from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import delete_json, post_json


def test_api_analysis_errors_if_any_code_missing_adjust_data(api_client):
    c = api_client

    upsert_and_fetch_etfs(
        c,
        codes=["A", "B"],
        names={"A": "A", "B": "B"},
        start_date="20240101",
        end_date="20240103",
    )

    # delete B's hfq data to simulate missing adjust
    assert delete_json(c, "/api/etf/B/prices", params={"adjust": "hfq"})["deleted"] > 0

    # analysis on hfq should fail because B lacks hfq
    out = post_json(
        c,
        "/api/analysis/baseline",
        {
            "codes": ["A", "B"],
            "start": "20240102",
            "end": "20240103",
            "adjust": "hfq",
            "rebalance": "yearly",
        },
        expected_status=400,
    )
    assert "missing data" in out.get("detail", "")
