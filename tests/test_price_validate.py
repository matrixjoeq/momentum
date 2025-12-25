from __future__ import annotations

import datetime as dt

import pytest

from etf_momentum.validation.price_validate import (  # pylint: disable=import-error
    PricePoint,
    ValidationError,
    ValidationPolicyParams,
    validate_price_series,
)


def test_validate_ok_series() -> None:
    pol = ValidationPolicyParams(max_abs_return=0.3, max_hl_spread=0.5, max_gap_days=10)
    pts = [
        PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0, 10, 100),
        PricePoint(dt.date(2024, 1, 3), 1.0, 1.2, 0.8, 1.1, 10, 100),
    ]
    validate_price_series(pts, policy=pol)


def test_validate_rejects_empty() -> None:
    pol = ValidationPolicyParams(max_abs_return=0.3, max_hl_spread=0.5, max_gap_days=10)
    with pytest.raises(ValidationError):
        validate_price_series([], policy=pol)


def test_validate_rejects_non_increasing_dates() -> None:
    pol = ValidationPolicyParams(max_abs_return=0.3, max_hl_spread=0.5, max_gap_days=10)
    pts = [
        PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0),
        PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0),
    ]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)


def test_validate_rejects_return_jump() -> None:
    pol = ValidationPolicyParams(max_abs_return=0.1, max_hl_spread=0.5, max_gap_days=10)
    pts = [
        PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0),
        PricePoint(dt.date(2024, 1, 3), 1.0, 1.1, 0.9, 1.3),
    ]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)


def test_validate_rejects_gap_too_large() -> None:
    pol = ValidationPolicyParams(max_abs_return=0.5, max_hl_spread=0.5, max_gap_days=3)
    pts = [
        PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0),
        PricePoint(dt.date(2024, 1, 10), 1.0, 1.1, 0.9, 1.0),
    ]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)


def test_validate_rejects_nonfinite_price() -> None:
    pol = ValidationPolicyParams(max_abs_return=1.0, max_hl_spread=1.0, max_gap_days=10)
    pts = [PricePoint(dt.date(2024, 1, 2), 1.0, float("nan"), 0.9, 1.0)]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)


def test_validate_rejects_negative_volume() -> None:
    pol = ValidationPolicyParams(max_abs_return=1.0, max_hl_spread=1.0, max_gap_days=10)
    pts = [PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.9, 1.0, volume=-1)]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)


def test_validate_rejects_low_nonpositive() -> None:
    pol = ValidationPolicyParams(max_abs_return=1.0, max_hl_spread=10.0, max_gap_days=10)
    pts = [PricePoint(dt.date(2024, 1, 2), 1.0, 1.1, 0.0, 1.0)]
    with pytest.raises(ValidationError):
        validate_price_series(pts, policy=pol)

