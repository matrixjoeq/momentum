from __future__ import annotations

import numpy as np
import pandas as pd

from etf_momentum.analysis.vol_proxy import compute_gjr_garch_volatility


def _make_close_series(n: int = 420, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    ret = rng.normal(0.0004, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.Series(close, index=dates, dtype=float)


class _FakeFit:
    def __init__(self, n: int) -> None:
        self.convergence_flag = 0
        self.params = pd.Series(
            {
                "mu": 0.01,
                "omega": 0.02,
                "alpha[1]": 0.06,
                "gamma[1]": 0.04,
                "beta[1]": 0.88,
                "nu": 8.5,
            }
        )
        self.conditional_volatility = np.full(n, 1.5, dtype=float)
        self.std_resid = np.linspace(-0.8, 0.9, n, dtype=float)
        self.loglikelihood = -123.4
        self.aic = 260.1
        self.bic = 278.6


class _FakeModel:
    def __init__(self, ret: pd.Series) -> None:
        self._n = int(len(ret))

    def fit(self, disp: str = "off", show_warning: bool = False) -> _FakeFit:
        _ = (disp, show_warning)
        return _FakeFit(self._n)


def _fake_arch_model(ret: pd.Series, **kwargs) -> _FakeModel:
    _ = kwargs
    return _FakeModel(ret)


def test_compute_gjr_garch_volatility_success_with_factory_stub() -> None:
    close = _make_close_series()
    out = compute_gjr_garch_volatility(
        close,
        max_points=300,
        min_samples=120,
        arch_model_factory=_fake_arch_model,
    )
    assert out["ok"] is True
    params = out["params"] or {}
    assert params.get("persistence") is not None
    diag = out["diagnostics"] or {}
    assert diag.get("converged") is True
    assert int(diag.get("n_obs_returns") or 0) == 300
    interp = out["interpretation"] or {}
    assert interp.get("model_value") in {"high", "medium", "low"}
    series = out["series"] or {}
    assert len(series.get("vol_dates") or []) == len(
        series.get("cond_vol_annualized") or []
    )


def test_compute_gjr_garch_volatility_insufficient_samples() -> None:
    close = _make_close_series(n=40)
    out = compute_gjr_garch_volatility(
        close,
        min_samples=80,
        arch_model_factory=_fake_arch_model,
    )
    assert out["ok"] is False
    assert out["error"] == "insufficient_samples"


def test_compute_gjr_garch_volatility_dependency_unavailable() -> None:
    close = _make_close_series(n=160)
    out = compute_gjr_garch_volatility(
        close,
        min_samples=60,
        arch_model_factory="not_callable",
    )
    assert out["ok"] is False
    assert out["error"] == "dependency_unavailable"


def test_compute_gjr_garch_volatility_fit_failed() -> None:
    close = _make_close_series(n=240)

    class _BadModel:
        def fit(self, disp: str = "off", show_warning: bool = False):
            _ = (disp, show_warning)
            raise RuntimeError("solver failed")

    def _bad_arch_model(ret: pd.Series, **kwargs):
        _ = (ret, kwargs)
        return _BadModel()

    out = compute_gjr_garch_volatility(
        close,
        min_samples=120,
        arch_model_factory=_bad_arch_model,
    )
    assert out["ok"] is False
    assert out["error"] == "fit_failed"
