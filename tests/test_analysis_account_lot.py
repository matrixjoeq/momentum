import pandas as pd
import pytest

from etf_momentum.analysis.account_lot import simulate_lot_account_weights


def test_simulate_lot_periodic_threshold_uses_discrete_lot_shares() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    target_weights = pd.DataFrame(
        {"A": [0.50, 0.525, 0.53, 0.531]},
        index=idx,
        dtype=float,
    )
    exec_price = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=idx)

    _, meta = simulate_lot_account_weights(
        target_weights=target_weights,
        exec_price=exec_price,
        initial_account_amount=1_000_000.0,
        max_leverage_multiple=1.0,
        lot_size_shares=100,
        periodic_rebalance_enabled=True,
        periodic_rebalance_threshold_pct=0.05,
    )

    shares = (meta.get("shares_by_code") or {}).get("A") or []
    assert shares == [5000, 5000, 5300, 5300]
    stats = (meta.get("periodic_rebalance_stats") or {}).get("overall") or {}
    assert int(stats.get("periodic_rebalance_evaluated_count") or 0) == 3
    assert int(stats.get("periodic_rebalance_trigger_count") or 0) == 1
    assert int(stats.get("periodic_rebalance_skip_count") or 0) == 2
    assert float(meta.get("periodic_rebalance_threshold_pct") or 0.0) == pytest.approx(
        0.05, rel=0.0, abs=1e-12
    )
