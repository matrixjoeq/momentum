from tests.helpers.api_test_client import FIXED_MINIPROGRAM_POOL, upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import post_json_ok


def test_api_rotation_weekly5_open_sim(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/rotation/weekly5-open",
        {"start": "20240102", "end": "20240103", "anchor_weekday": 5},
    )
    assert data["meta"]["type"] == "rotation_weekly5_open"
    assert data["meta"]["exec_price"] == "open"
    assert data["meta"]["rebalance_shift"] == "prev"
    by = data["by_anchor"]
    assert set(by.keys()) == {"5"}
    # spot-check one result payload shape
    one = by["5"]
    assert "nav" in one and "series" in one["nav"]
    assert "ROTATION" in one["nav"]["series"]
    assert "EW_REBAL" in one["nav"]["series"]
    assert "EXCESS" in one["nav"]["series"]
