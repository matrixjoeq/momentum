from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import get_json_ok


def test_etf_list_range_changes_with_adjust(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["510300"],
        names={"510300": "沪深300"},
        start_date="20240102",
        end_date="20240103",
    )

    # hfq should have range
    it_h = get_json_ok(c, "/api/etf?adjust=hfq")[0]
    assert it_h["last_data_start_date"] == "20240102"
    assert it_h["last_data_end_date"] == "20240103"

    # qfq should also have identical range
    it_q = get_json_ok(c, "/api/etf?adjust=qfq")[0]
    assert it_q["last_data_start_date"] == "20240102"
    assert it_q["last_data_end_date"] == "20240103"

    # none should also have identical range
    it_n = get_json_ok(c, "/api/etf?adjust=none")[0]
    assert it_n["last_data_start_date"] == "20240102"
    assert it_n["last_data_end_date"] == "20240103"

