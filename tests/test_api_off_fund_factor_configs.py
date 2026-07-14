from __future__ import annotations

from fastapi.testclient import TestClient

from etf_momentum.analysis.off_fund_regression import DEFAULT_CN_STOCK_FACTORS


def test_off_fund_factor_configs_default_and_upsert(api_client: TestClient) -> None:
    c = api_client
    r0 = c.get("/api/off-fund/regression/factor-configs")
    assert r0.status_code == 200
    rows0 = r0.json()
    assert isinstance(rows0, list)
    assert len(rows0) >= 1
    assert any(bool(x.get("is_active")) for x in rows0)
    active0 = next(x for x in rows0 if bool(x.get("is_active")))
    assert isinstance(active0.get("effective_benchmark_factors"), list)
    assert len(active0["effective_benchmark_factors"]) == len(DEFAULT_CN_STOCK_FACTORS)

    r1 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "测试模板A",
            "set_active": True,
            "benchmark_profile": "cn_stock_core",
            "benchmark_factors": [
                {"key": "F300", "label": "沪深300", "aliases": ["000300", "510300"]},
                {"key": "F500", "label": "中证500", "aliases": ["000905", "510500"]},
            ],
        },
    )
    assert r1.status_code == 200
    out1 = r1.json()
    assert out1["name"] == "测试模板A"
    assert out1["is_active"] is True
    assert len(out1["benchmark_factors"]) == 2
    assert len(out1["effective_benchmark_factors"]) == 2

    r2 = c.get("/api/off-fund/regression/factor-configs")
    assert r2.status_code == 200
    rows2 = r2.json()
    by_name = {x["name"]: x for x in rows2}
    assert by_name["测试模板A"]["is_active"] is True
    assert len(by_name["测试模板A"]["benchmark_factors"]) == 2


def test_off_fund_factor_configs_activate_and_delete(api_client: TestClient) -> None:
    c = api_client
    r1 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "模板1",
            "set_active": False,
            "benchmark_profile": "cn_stock_core",
        },
    )
    assert r1.status_code == 200
    r2 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "模板2",
            "set_active": False,
            "benchmark_profile": "cn_stock_core",
        },
    )
    assert r2.status_code == 200

    ra = c.post("/api/off-fund/regression/factor-configs/模板2/activate")
    assert ra.status_code == 200
    rows = c.get("/api/off-fund/regression/factor-configs").json()
    by = {x["name"]: x for x in rows}
    assert by["模板2"]["is_active"] is True
    assert by["模板1"]["is_active"] is False

    rd = c.delete("/api/off-fund/regression/factor-configs/模板2")
    assert rd.status_code == 200
    rows2 = c.get("/api/off-fund/regression/factor-configs").json()
    names = [x["name"] for x in rows2]
    assert "模板2" not in names
    # deleting active should keep some config active
    assert any(bool(x.get("is_active")) for x in rows2)
