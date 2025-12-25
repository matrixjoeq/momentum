from __future__ import annotations

from fastapi.testclient import TestClient


def test_batches_list_get_and_rollback(api_client: TestClient) -> None:
    # create ETF and ingest to generate a batch
    resp = api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF", "start_date": "20240101", "end_date": "20240131"})
    assert resp.status_code == 200
    resp = api_client.post("/api/etf/510300/fetch", json={})
    assert resp.status_code == 200

    resp = api_client.get("/api/batches")
    assert resp.status_code == 200
    batches = resp.json()
    assert isinstance(batches, list)
    assert len(batches) >= 1
    batch_id = batches[0]["id"]
    assert batches[0]["val_max_abs_return"] is not None

    resp = api_client.get(f"/api/batches/{batch_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == batch_id

    # rollback should succeed (insert-only batch -> restore to empty, valid)
    resp = api_client.post(f"/api/batches/{batch_id}/rollback", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] in ("success", "snapshot_restored")


def test_get_batch_404(api_client: TestClient) -> None:
    resp = api_client.get("/api/batches/99999999")
    assert resp.status_code == 404


def test_rollback_batch_missing_returns_error(api_client: TestClient) -> None:
    resp = api_client.post("/api/batches/99999999/rollback", json={})
    assert resp.status_code == 500

