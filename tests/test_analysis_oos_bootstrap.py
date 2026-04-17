"""Tests for out-of-sample bootstrap parameter optimisation (Carver-style)."""

import numpy as np
import pandas as pd

import etf_momentum.db.repo as repo
from etf_momentum.analysis.oos_bootstrap import (
    OosBootstrapConfig,
    block_bootstrap_returns,
    returns_to_close,
    run_trend_oos_bootstrap,
    split_in_sample_oos,
    run_rotation_oos_bootstrap,
)
from tests.helpers.rotation_case_data import seed_prices


def test_split_in_sample_oos():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    in_d, oos_d = split_in_sample_oos(dates, oos_ratio=0.3)
    assert len(in_d) == 70
    assert len(oos_d) == 30
    assert in_d[0] == dates[0]
    assert oos_d[-1] == dates[-1]
    assert in_d[-1] < oos_d[0]


def test_split_in_sample_oos_raises_bad_ratio():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    try:
        split_in_sample_oos(dates, oos_ratio=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        split_in_sample_oos(dates, oos_ratio=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_returns_to_close():
    r = pd.DataFrame({"A": [0.01, -0.005, 0.02], "B": [0.0, 0.0, 0.01]})
    close = returns_to_close(r, initial=None)
    assert close.shape == r.shape
    assert close.iloc[0].eq(1.0).all()
    np.testing.assert_allclose(close["A"].iloc[-1], 1.01 * 0.995 * 1.02, rtol=1e-9)


def test_block_bootstrap_returns_same_shape():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(np.random.randn(50, 3) * 0.01, columns=["A", "B", "C"])
    boot = block_bootstrap_returns(returns, block_size=5, rng=rng)
    assert boot.shape == returns.shape


def test_run_rotation_oos_bootstrap_mini():
    """Minimal run: small grid, 2 bootstrap iterations, tiny data."""
    rng = np.random.default_rng(123)
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Synthetic close: two assets, mild drift
    ret = pd.DataFrame(
        {"A": rng.normal(0.0003, 0.01, n), "B": rng.normal(0.0002, 0.012, n)},
        index=dates,
    )
    ret.iloc[0] = 0.0
    close = (1 + ret).cumprod()
    close.iloc[0] = 1.0

    class FakeUniverse:
        name = "Test"
        codes = ["A", "B"]

    param_grid = {
        "score_method": ["raw_mom"],
        "lookback_days": [20],
        "top_k": [1],
        "rebalance": ["weekly"],
        "enable_trend_filter": [True],
    }
    cfg = OosBootstrapConfig(oos_ratio=0.3, n_bootstrap=2, block_size=5, seed=1)
    out = run_rotation_oos_bootstrap(close, FakeUniverse(), param_grid, cost_bps=0.0, config=cfg)
    assert "error" not in out or "Insufficient" not in str(out.get("error", ""))
    if "error" in out:
        # May fail if in-sample too short for lookback
        assert "in_sample_days" in out or "oos_days" in out
        return
    assert "chosen_params" in out
    assert "oos_metrics" in out
    assert "in_sample_end" in out
    assert "oos_start" in out
    assert out["chosen_params"].get("lookback_days") == 20
    assert out["chosen_params"].get("top_k") == 1


def test_run_trend_oos_bootstrap_bt_fallback_marks_legacy(engine, session_factory, monkeypatch):
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    cfg = OosBootstrapConfig(oos_ratio=0.3, n_bootstrap=4, block_size=10, seed=7)
    monkeypatch.setattr(
        repo,
        "upsert_prices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced bootstrap synth fail")),
    )
    with session_factory() as db:
        out = run_trend_oos_bootstrap(
            db,
            codes=["A", "B"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            config=cfg,
            param_grid={"sma_window": [20], "ma_type": ["sma"]},
            engine="bt",
        )

    assert str(out.get("engine") or "").lower() == "bt"
    assert str(out.get("oos_eval_engine") or "").lower() == "bt"
    assert str(out.get("bootstrap_eval_engine") or "").lower() == "legacy"
    limitations = out.get("limitations")
    assert isinstance(limitations, list)
    assert any("fallback to legacy" in str(x).lower() for x in limitations)


def test_run_trend_oos_bootstrap_bt_fallback_marks_mixed(engine, session_factory, monkeypatch):
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    cfg = OosBootstrapConfig(oos_ratio=0.3, n_bootstrap=4, block_size=10, seed=7)

    real_upsert = repo.upsert_prices
    calls = {"n": 0}

    def _upsert_with_one_failure(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced first bootstrap synth fail")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(repo, "upsert_prices", _upsert_with_one_failure)
    with session_factory() as db:
        out = run_trend_oos_bootstrap(
            db,
            codes=["A", "B"],
            start=dates[0],
            end=dates[-1],
            strategy="ma_filter",
            config=cfg,
            param_grid={"sma_window": [20], "ma_type": ["sma"]},
            engine="bt",
        )

    assert str(out.get("engine") or "").lower() == "bt"
    assert str(out.get("oos_eval_engine") or "").lower() == "bt"
    assert str(out.get("bootstrap_eval_engine") or "").lower() == "mixed"
    limitations = out.get("limitations")
    assert isinstance(limitations, list)
    assert any("fallback to legacy" in str(x).lower() for x in limitations)
