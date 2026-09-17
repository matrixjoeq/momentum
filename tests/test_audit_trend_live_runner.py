from __future__ import annotations

import datetime as dt
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_trend_live.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("audit_trend_live", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_counts_falls_back_to_quiet_progress_rows(tmp_path: Path) -> None:
    runner = _load_runner()
    log = tmp_path / "pytest.log"
    log.write_text(
        ("." * 72) + " [ 50%]\n" + ("." * 70) + "s. [100%]\n" + "warnings summary\n",
        encoding="utf-8",
    )

    assert runner._pytest_counts(log) == {  # pylint: disable=protected-access
        "passed": 143,
        "failed": 0,
        "skipped": 1,
        "errors": 0,
        "collected": 144,
    }


def test_build_stopped_bridge_is_additive_and_does_not_cross_gate() -> None:
    runner = _load_runner()

    bridge = runner._build_stopped_bridge(  # pylint: disable=protected-access
        metric="portfolio_return",
        model_value=Decimal("0.12"),
        live_value=Decimal("-0.03"),
        unavailable_layer="data_availability_and_price",
        reason="historical_price_vintage_unavailable",
        absolute_tolerance=Decimal("0.00000001"),
    )

    assert bridge["status"] == "stopped_unidentifiable"
    assert bridge["layers"][0]["status"] == "unidentifiable_stop"
    assert all(row["increment"] is None for row in bridge["layers"])
    assert Decimal(bridge["total_difference"]) == Decimal("-0.15")
    assert Decimal(bridge["unexplained_residual"]) == Decimal("-0.15")
    assert Decimal(bridge["rebuild_error"]) == Decimal("0.00")
    assert bridge["within_tolerance"] is True


def test_decimal_nav_identity_finds_first_divergence_and_float_bound() -> None:
    runner = _load_runner()
    rows = [
        {
            "nav_date": "2026-01-01",
            "equity": "100.00",
            "cash": "40.00",
            "market_value": "60.00",
        },
        {
            "nav_date": "2026-01-02",
            "equity": "101.00",
            "cash": "40.00",
            "market_value": "60.00",
        },
    ]

    result = runner._decimal_nav_identity(  # pylint: disable=protected-access
        rows,
        physical_types={
            "equity": "double",
            "cash": "double",
            "market_value": "double",
        },
    )

    assert result["first_divergence_date"] == "2026-01-02"
    assert Decimal(result["max_abs_decimal_residual"]) == Decimal("1.00")
    assert Decimal(result["float_error_upper_bound"]) < Decimal("0.000000000001")
    assert result["explained_by_float_error"] is False
    assert result["source_precision"] == "source_precision_unknown"


def test_decimal_nav_chain_reports_first_divergence() -> None:
    runner = _load_runner()
    rows = [
        {
            "nav_date": "2026-01-01",
            "nav_twr": "1.0",
            "daily_return_twr": "0",
        },
        {
            "nav_date": "2026-01-02",
            "nav_twr": "1.1",
            "daily_return_twr": "0.1",
        },
        {
            "nav_date": "2026-01-03",
            "nav_twr": "1.3",
            "daily_return_twr": "0.1",
        },
    ]

    result = runner._decimal_nav_chain(rows)  # pylint: disable=protected-access

    assert result["first_divergence_date"] == "2026-01-03"
    assert Decimal(result["max_abs_decimal_residual"]) == Decimal("0.09")
    assert result["explained_by_float_error"] is False


def test_artifact_envelope_binds_upstream_ids_and_hashes() -> None:
    runner = _load_runner()

    artifact = runner._artifact_envelope(  # pylint: disable=protected-access
        artifact_id="child-v1",
        artifact_type="test",
        payload={"value": 3},
        upstream=[{"artifact_id": "parent-v1", "sha256": "a" * 64}],
        generated_at=dt.datetime(2026, 8, 27, tzinfo=dt.timezone.utc),
    )

    assert artifact["artifact_id"] == "child-v1"
    assert artifact["artifact_type"] == "test"
    assert artifact["upstream"] == [{"artifact_id": "parent-v1", "sha256": "a" * 64}]
    assert artifact["payload"] == {"value": 3}
    assert artifact["generated_at"] == "2026-08-27T00:00:00+00:00"


def test_artifact_chain_validation_requires_resolvable_upstream(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    parent_path = tmp_path / "parent.json"
    child_path = tmp_path / "child.json"
    parent_path.write_text(json.dumps({"artifact_id": "parent-v1"}), encoding="utf-8")
    parent_hash = runner._sha256(parent_path)  # pylint: disable=protected-access
    child_path.write_text(
        json.dumps(
            {
                "artifact_id": "child-v1",
                "upstream": [{"artifact_id": "parent-v1", "sha256": parent_hash}],
            }
        ),
        encoding="utf-8",
    )
    nodes = [
        {
            "artifact_id": "parent-v1",
            "path": "parent.json",
            "sha256": parent_hash,
        },
        {
            "artifact_id": "child-v1",
            "path": "child.json",
            "sha256": runner._sha256(  # pylint: disable=protected-access
                child_path
            ),
        },
    ]

    result = runner._validate_chain_nodes(  # pylint: disable=protected-access
        tmp_path,
        nodes,
    )
    assert result["node_count"] == 2
    assert result["upstream_reference_count"] == 1
    assert result["upstream_hashes_valid"] is True

    with pytest.raises(ValueError, match="unchained upstream"):
        runner._validate_chain_nodes(  # pylint: disable=protected-access
            tmp_path,
            nodes[1:],
        )


def test_cashflow_classification_keeps_scope_and_income_separate() -> None:
    runner = _load_runner()
    rows = [
        {"flow_type": "transfer_in", "amount": "100"},
        {"flow_type": "dividend", "amount": "2.5"},
        {"flow_type": "manual", "amount": "-1"},
    ]

    classified = runner._classify_strategy_cashflows(  # pylint: disable=protected-access
        rows
    )

    assert classified["external_flow"] == pytest.approx(99.0)
    assert classified["investment_income"] == pytest.approx(2.5)
    assert classified["unknown"] == pytest.approx(0.0)
    scoped = runner._classify_cashflows_by_scope(  # pylint: disable=protected-access
        account_rows=[
            {"flow_type": "deposit", "amount": "100"},
            {"flow_type": "transfer_to_strategy", "amount": "-60"},
        ],
        strategy_rows=[
            {"flow_type": "transfer_in", "amount": "60"},
            {"flow_type": "dividend", "amount": "2.5"},
        ],
    )
    assert scoped["account"]["external_flow"] == pytest.approx(100.0)
    assert scoped["account"]["internal_transfer"] == pytest.approx(-60.0)
    assert scoped["strategy"]["external_flow"] == pytest.approx(60.0)
    assert scoped["strategy"]["investment_income"] == pytest.approx(2.5)


def test_independent_tsmom_trace_is_causal_and_does_not_bridge_price_gap() -> None:
    runner = _load_runner()
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 103.0, None, 99.0, 105.0], index=index)

    trace = runner._independent_tsmom_trace(  # pylint: disable=protected-access
        prices,
        lookback=1,
        entry_threshold=0.02,
        exit_threshold=0.0,
    )

    assert trace.loc[index[1], "momentum"] == pytest.approx(0.03)
    assert trace.loc[index[1], "signal"] == 1.0
    assert pd.isna(trace.loc[index[2], "momentum"])
    assert trace.loc[index[2], "signal"] == 1.0
    assert pd.isna(trace.loc[index[3], "momentum"])
    assert trace.loc[index[3], "signal"] == 1.0
    assert trace.loc[index[4], "momentum"] == pytest.approx(105.0 / 99.0 - 1.0)


@pytest.mark.parametrize(
    "mode", ["no_forced_liquidation", "reporting_only_liquidation"]
)
def test_terminal_liquidation_modes_never_rewrite_causal_episodes(mode: str) -> None:
    runner = _load_runner()
    episodes = [
        {"code": "A", "entry_date": "2026-01-01", "closed": False, "return": 0.1},
        {"code": "B", "entry_date": "2026-01-01", "closed": True, "return": -0.1},
    ]

    projection = runner._terminal_episode_projection(  # pylint: disable=protected-access
        episodes,
        mode=mode,
        as_of="2026-01-31",
    )

    assert projection["causal_episodes"] == episodes
    assert projection["closed_stat_returns"] == [-0.1]
    if mode == "no_forced_liquidation":
        assert projection["reporting_only_liquidations"] == []
    else:
        assert projection["reporting_only_liquidations"] == [
            {
                "code": "A",
                "entry_date": "2026-01-01",
                "reporting_date": "2026-01-31",
                "return": 0.1,
                "closed": False,
                "projection_only": True,
            }
        ]


def test_window_drawdowns_reports_reset_and_inherited_high_water() -> None:
    runner = _load_runner()
    response = {
        "nav": {
            "dates": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "series": {"STRAT": [1.0, 2.0, 1.5, 1.8]},
        }
    }

    result = runner._window_drawdowns(  # pylint: disable=protected-access
        response,
        start="2026-01-03",
        end="2026-01-04",
    )

    assert result["window_reset_high_water_max_drawdown"] == pytest.approx(0.0)
    assert result["inherited_high_water_max_drawdown"] == pytest.approx(-0.25)


def test_independent_daily_oracle_conserves_cash_market_value_and_nav() -> None:
    runner = _load_runner()

    rows = runner._independent_daily_oracle(  # pylint: disable=protected-access
        asset_returns=["0", "0.5", "0.1", "-0.2"],
        effective_weights=["0", "1", "1", "0"],
        return_weights=["0", "0", "1", "1"],
        fee_rate="0",
        slippage_spread="0",
        execution_prices=["10", "15", "16.5", "13.2"],
        initial_nav="1",
    )

    assert [Decimal(row["net_return"]) for row in rows] == [
        Decimal("0"),
        Decimal("0"),
        Decimal("0.1"),
        Decimal("-0.2"),
    ]
    assert [Decimal(row["nav"]) for row in rows] == [
        Decimal("1"),
        Decimal("1"),
        Decimal("1.1"),
        Decimal("0.88"),
    ]
    for row in rows:
        assert Decimal(row["nav"]) == (
            Decimal(row["cash"]) + Decimal(row["market_value"])
        )


def test_independent_risk_budget_sizing_records_every_constraint_stage() -> None:
    runner = _load_runner()

    result = runner._independent_risk_budget_sizing(  # pylint: disable=protected-access
        capital="100000",
        risk_budget_pct="0.0025",
        price="50",
        atr="2",
        stop_multiple="2",
        max_weight="0.03",
        lot_size=10,
        available_cash="2000",
    )

    assert Decimal(result["risk_amount"]) == Decimal("250")
    assert Decimal(result["ideal_quantity"]) == Decimal("62.5")
    assert Decimal(result["overcap_quantity"]) == Decimal("60")
    assert Decimal(result["round_lot_quantity"]) == Decimal("60")
    assert Decimal(result["cash_constrained_quantity"]) == Decimal("40")
