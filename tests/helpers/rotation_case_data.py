from __future__ import annotations

import datetime as dt

import pandas as pd
from httpx import Response

from etf_momentum.db.models import EtfPrice
from etf_momentum.db.session import make_session_factory


FIXED_MINIPROGRAM_CODES = ["159915", "511010", "513100", "518880"]


def seed_prices(engine, *, code_to_series: dict[str, list[float]], dates: list[dt.date]) -> None:
    """Seed test prices for none/hfq/qfq adjustments."""
    sf = make_session_factory(engine)
    with sf() as db:
        for code, series in code_to_series.items():
            for d, px in zip(dates, series):
                for adj in ("none", "hfq", "qfq"):
                    db.add(
                        EtfPrice(
                            code=str(code),
                            trade_date=d,
                            close=float(px),
                            low=float(px),
                            high=float(px),
                            source="eastmoney",
                            adjust=adj,
                        )
                    )
        db.commit()


def build_rotation_case_series() -> tuple[list[dt.date], dict[str, list[float]]]:
    """
    Synthetic case used by rotation API E2E tests.
    - A/C/D: high momentum with late spikes (higher BIAS risk)
    - B/E: slower trend (lower BIAS risk)
    """
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-07-31", freq="B")]

    def _spike(base: float, slope: float, spike_start: int, spike_slope: float) -> list[float]:
        out: list[float] = []
        for i, _ in enumerate(dates):
            v = base + i * slope
            if i >= spike_start:
                v += (i - spike_start) * spike_slope
            out.append(float(v))
        return out

    series = {
        "A": _spike(100.0, 0.12, 115, 1.8),
        "C": _spike(100.0, 0.11, 116, 1.7),
        "D": _spike(100.0, 0.10, 117, 1.6),
        "B": [100.0 + i * 0.04 for i, _ in enumerate(dates)],
        "E": [100.0 + i * 0.03 for i, _ in enumerate(dates)],
    }
    return dates, series


def map_case_series_to_miniprogram_codes(src: dict[str, list[float]]) -> dict[str, list[float]]:
    """Map synthetic A/B/C/D series to fixed mini-program ETF codes."""
    return {
        "159915": src["A"],
        "511010": src["B"],
        "513100": src["C"],
        "518880": src["D"],
    }


def fmt_ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def make_rotation_base_payload(
    *,
    codes: list[str],
    dates: list[dt.date],
    rebalance: str = "weekly",
    top_k: int = 3,
    lookback_days: int = 20,
    skip_days: int = 0,
    cost_bps: float = 0.0,
    position_mode: str = "adaptive",
    entry_backfill: bool = False,
) -> dict[str, object]:
    return {
        "codes": list(codes),
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "rebalance": str(rebalance),
        "top_k": int(top_k),
        "position_mode": str(position_mode),
        "entry_backfill": bool(entry_backfill),
        "lookback_days": int(lookback_days),
        "skip_days": int(skip_days),
        "cost_bps": float(cost_bps),
    }


def make_trend_rule(*, stage: str, op: str = ">=", trend_sma_window: int = 5, trend_ma_type: str = "ema") -> dict[str, object]:
    return {
        "code": "*",
        "stage": str(stage),
        "op": str(op),
        "trend_sma_window": int(trend_sma_window),
        "trend_ma_type": str(trend_ma_type),
    }


def make_bias_rule(
    *,
    stage: str,
    op: str,
    bias_ma_window: int = 5,
    threshold_type: str = "fixed",
    fixed_value: float = 1.5,
    min_periods: int = 5,
) -> dict[str, object]:
    return {
        "code": "*",
        "stage": str(stage),
        "op": str(op),
        "bias_ma_window": int(bias_ma_window),
        "threshold_type": str(threshold_type),
        "fixed_value": float(fixed_value),
        "min_periods": int(min_periods),
    }


def mc_metric_mean(data: dict, metric: str = "annualized_return") -> float:
    raw = (((data.get("mc") or {}).get("strategy") or {}).get("metrics", {})).get(metric, 0.0)
    if isinstance(raw, dict):
        return float(raw.get("mean", 0.0))
    return float(raw or 0.0)


def first_grid_metric(data: dict, metric: str = "annualized_return") -> float:
    grid = data.get("grid") or []
    if not grid:
        return 0.0
    first = grid[0] or {}
    metrics = first.get("metrics") or {}
    return float(metrics.get(metric) or 0.0)


def request_json(
    api_client,
    *,
    method: str,
    path: str,
    expected_status: int = 200,
    payload: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> dict | list:
    m = str(method).lower()
    if m == "post":
        if payload is None:
            resp = api_client.post(path, params=params)
        else:
            resp = api_client.post(path, json=payload, params=params)
    elif m == "get":
        resp = api_client.get(path, params=params)
    elif m == "delete":
        resp = api_client.delete(path, params=params)
    else:
        raise ValueError(f"unsupported method: {method}")
    assert resp.status_code == int(expected_status)
    return resp.json()


def request_response(
    api_client,
    *,
    method: str,
    path: str,
    expected_status: int | None = None,
    payload: dict[str, object] | None = None,
    params: dict[str, object] | None = None,
) -> Response:
    m = str(method).lower()
    if m == "post":
        if payload is None:
            resp = api_client.post(path, params=params)
        else:
            resp = api_client.post(path, json=payload, params=params)
    elif m == "get":
        resp = api_client.get(path, params=params)
    elif m == "delete":
        resp = api_client.delete(path, params=params)
    else:
        raise ValueError(f"unsupported method: {method}")
    if expected_status is not None:
        assert resp.status_code == int(expected_status)
    return resp


def post_response(
    api_client,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    expected_status: int | None = None,
    params: dict[str, object] | None = None,
) -> Response:
    return request_response(
        api_client,
        method="post",
        path=path,
        expected_status=expected_status,
        payload=payload,
        params=params,
    )


def get_response(
    api_client,
    path: str,
    *,
    expected_status: int | None = None,
    params: dict[str, object] | None = None,
) -> Response:
    return request_response(
        api_client,
        method="get",
        path=path,
        expected_status=expected_status,
        params=params,
    )


def delete_response(
    api_client,
    path: str,
    *,
    expected_status: int | None = None,
    params: dict[str, object] | None = None,
) -> Response:
    return request_response(
        api_client,
        method="delete",
        path=path,
        expected_status=expected_status,
        params=params,
    )


def post_json(api_client, path: str, payload: dict[str, object], *, expected_status: int = 200) -> dict | list:
    return request_json(
        api_client,
        method="post",
        path=path,
        expected_status=expected_status,
        payload=payload,
    )


def get_json(api_client, path: str, *, expected_status: int = 200, params: dict[str, object] | None = None) -> dict | list:
    return request_json(
        api_client,
        method="get",
        path=path,
        expected_status=expected_status,
        params=params,
    )


def delete_json(api_client, path: str, *, expected_status: int = 200, params: dict[str, object] | None = None) -> dict | list:
    return request_json(
        api_client,
        method="delete",
        path=path,
        expected_status=expected_status,
        params=params,
    )


def post_json_ok(api_client, path: str, payload: dict[str, object]) -> dict | list:
    return post_json(api_client, path, payload, expected_status=200)


def get_json_ok(api_client, path: str) -> dict | list:
    return get_json(api_client, path, expected_status=200)


def make_entry_filters_payload(*, bias_fixed_value: float = 1.5) -> dict[str, object]:
    return {
        "trend_filter": True,
        "bias_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry")],
        "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=bias_fixed_value)],
    }


def make_entry_exit_filters_payload(
    *,
    entry_bias_fixed_value: float = 1.5,
    exit_bias_fixed_value: float = 99.0,
) -> dict[str, object]:
    return {
        "trend_filter": True,
        "bias_filter": True,
        "trend_exit_filter": True,
        "bias_exit_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry"), make_trend_rule(stage="exit")],
        "asset_bias_rules": [
            make_bias_rule(stage="entry", op="<=", fixed_value=entry_bias_fixed_value),
            make_bias_rule(stage="exit", op=">=", fixed_value=exit_bias_fixed_value),
        ],
    }
