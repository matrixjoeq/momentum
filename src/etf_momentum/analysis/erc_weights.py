"""
Equal Risk Contribution (ERC) portfolio weights from a covariance matrix.

ERC weights satisfy (approximately) equal normalized marginal risk contributions:
    RC_i = w_i (Σw)_i / (w^T Σ w) ≈ 1/n

Uses scipy.optimize (project dependency). Falls back to inverse-volatility on failure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def solve_erc_weights(
    cov: np.ndarray,
    *,
    max_iter: int = 800,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Solve for long-only ERC weights (sum = 1) given a PSD covariance matrix.

    Parameters
    ----------
    cov
        (n, n) covariance matrix.
    max_iter
        Passed to SLSQP.
    tol
        Convergence tolerance on the sum of squared RC deviations from 1/n.
    """
    cov = np.asarray(cov, dtype=float)
    n = int(cov.shape[0])
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.ones(1, dtype=float)

    # Ensure numerical PSD
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    sigma = evecs @ np.diag(evals) @ evecs.T

    diag = np.diag(sigma)
    inv_vol = np.where(diag > 1e-18, 1.0 / np.sqrt(np.maximum(diag, 1e-18)), 0.0)
    s_iv = float(inv_vol.sum())
    x0 = (inv_vol / s_iv) if s_iv > 0 else np.full(n, 1.0 / n, dtype=float)

    def portfolio_var(wv: np.ndarray) -> float:
        return float(wv @ sigma @ wv)

    def risk_contrib_norm(wv: np.ndarray) -> np.ndarray:
        wv = np.maximum(wv, 1e-14)
        wv = wv / wv.sum()
        m = sigma @ wv
        pv = float(wv @ m)
        if pv <= 1e-20:
            return np.full(n, 1.0 / n, dtype=float)
        return (wv * m) / pv

    def objective(wv: np.ndarray) -> float:
        rc = risk_contrib_norm(wv)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    bounds = [(1e-9, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda wv: float(np.sum(wv)) - 1.0},)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": int(max_iter), "ftol": 1e-14},
    )
    w = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    sw = float(w.sum())
    if sw <= 0 or not np.isfinite(sw):
        w = np.maximum(x0, 0.0)
        sw = float(w.sum())
    w = w / sw

    # Refine / verify: if still poor, fall back to inverse-vol from diagonal
    rc_final = risk_contrib_norm(w)
    err = float(np.sum((rc_final - 1.0 / n) ** 2))
    if err > 1e-4 or not np.all(np.isfinite(w)):
        iv = 1.0 / np.sqrt(np.maximum(np.diag(sigma), 1e-18))
        w = np.maximum(iv, 0.0)
        w = w / float(np.sum(w)) if float(np.sum(w)) > 0 else np.full(n, 1.0 / n, dtype=float)
    _ = portfolio_var(w)
    return w.astype(float)


def erc_weights_from_return_history(
    hist: np.ndarray,
    *,
    min_obs: int = 2,
) -> np.ndarray:
    """
    ERC weights from a (T, n) return matrix (rows = time). Drops rows with any NaN.
    """
    r = np.asarray(hist, dtype=float)
    if r.ndim != 2 or r.shape[1] <= 0:
        return np.array([])
    t, n = r.shape
    if n == 1:
        return np.ones(1, dtype=float)
    mask = np.all(np.isfinite(r), axis=1)
    clean = r[mask]
    if clean.shape[0] < int(min_obs):
        return np.full(n, 1.0 / n, dtype=float)
    cov = np.cov(clean, rowvar=False, ddof=1)
    if cov.ndim == 0:
        return np.ones(1, dtype=float)
    return solve_erc_weights(cov)
