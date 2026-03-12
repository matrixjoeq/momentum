from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import get_json, get_json_ok, post_json


def test_batches_list_get_and_rollback(api_client: TestClient) -> None:
    # create ETF and ingest to generate a batch
    upsert_and_fetch_etfs(
        api_client,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )

    batches = get_json_ok(api_client, "/api/batches")
    assert isinstance(batches, list)
    assert len(batches) >= 1
    batch_id = batches[0]["id"]
    assert batches[0]["val_max_abs_return"] is not None

    assert get_json_ok(api_client, f"/api/batches/{batch_id}")["id"] == batch_id

    # rollback should succeed (insert-only batch -> restore to empty, valid)
    out = post_json(api_client, f"/api/batches/{batch_id}/rollback", {})
    assert out["status"] in ("success", "snapshot_restored")


def test_get_batch_404(api_client: TestClient) -> None:
    get_json(api_client, "/api/batches/99999999", expected_status=404)


def test_rollback_batch_missing_returns_error(api_client: TestClient) -> None:
    post_json(api_client, "/api/batches/99999999/rollback", {}, expected_status=500)

