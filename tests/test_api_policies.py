from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import get_json


def test_list_validation_policies(api_client: TestClient) -> None:
    data = get_json(api_client, "/api/validation-policies")
    assert isinstance(data, list)
    assert any(p["name"] == "cn_stock_etf_10" for p in data)

