"""Tests for equal risk contribution (ERC) weight solver."""

from __future__ import annotations

import numpy as np

from etf_momentum.analysis.erc_weights import (
    erc_weights_from_return_history,
    solve_erc_weights,
)


def test_solve_erc_diagonal_covariance_matches_inverse_volatility_weights() -> None:
    vols = np.array([0.10, 0.20, 0.40])
    cov = np.diag(vols**2)
    w = solve_erc_weights(cov)
    inv_v = 1.0 / vols
    expected = inv_v / inv_v.sum()
    np.testing.assert_allclose(w, expected, rtol=1e-4, atol=1e-5)


def test_erc_weights_from_return_history_single_asset() -> None:
    r = np.random.default_rng(0).standard_normal((50, 1))
    w = erc_weights_from_return_history(r)
    assert w.shape == (1,)
    assert w[0] == 1.0


def test_erc_weights_from_return_history_insufficient_obs_falls_back_equal() -> None:
    r = np.array([[0.01, -0.02], [0.0, 0.01]])
    w = erc_weights_from_return_history(r, min_obs=10)
    np.testing.assert_allclose(w, np.array([0.5, 0.5]))
