from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PricePoint:
    trade_date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None = None
    amount: float | None = None


@dataclass(frozen=True)
class ValidationPolicyParams:
    max_abs_return: float
    max_hl_spread: float
    max_gap_days: int


@dataclass(frozen=True)
class ValidationIssue:
    rule: str
    trade_date: str | None = None  # ISO date
    field: str | None = None
    value: float | int | str | None = None
    threshold: float | int | str | None = None
    detail: str | None = None


class ValidationError(ValueError):
    def __init__(self, message: str, *, issues: list[ValidationIssue] | None = None):
        super().__init__(message)
        self.issues = issues or []

    def to_dict(self) -> dict:
        return {
            "error_type": "validation",
            "message": str(self),
            "issues": [
                {
                    "rule": i.rule,
                    "trade_date": i.trade_date,
                    "field": i.field,
                    "value": i.value,
                    "threshold": i.threshold,
                    "detail": i.detail,
                }
                for i in self.issues
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _raise(rule: str, msg: str, *, trade_date: dt.date | None = None, field: str | None = None, value=None, threshold=None, detail: str | None = None) -> None:
    issue = ValidationIssue(
        rule=rule,
        trade_date=trade_date.isoformat() if trade_date else None,
        field=field,
        value=value,
        threshold=threshold,
        detail=detail,
    )
    raise ValidationError(msg, issues=[issue])


def _is_finite_positive(x: float | None) -> bool:
    if x is None:
        return True
    return math.isfinite(x) and x > 0


def _is_finite_nonneg(x: float | None) -> bool:
    if x is None:
        return True
    return math.isfinite(x) and x >= 0


def validate_price_series(
    points: list[PricePoint],
    *,
    policy: ValidationPolicyParams,
) -> None:
    """
    Validate a chronological price series for a single instrument.
    Raises ValidationError on failure.
    """
    if not points:
        _raise("empty_series", "empty price series")

    points = sorted(points, key=lambda p: p.trade_date)
    if any(points[i].trade_date >= points[i + 1].trade_date for i in range(len(points) - 1)):
        _raise("non_increasing_dates", "trade_date must be strictly increasing")

    prev_close: float | None = None
    prev_date: dt.date | None = None
    for p in points:
        # basic numeric checks
        for name, v in [("open", p.open), ("high", p.high), ("low", p.low), ("close", p.close)]:
            if v is not None and not math.isfinite(v):
                _raise("non_finite", f"{name} not finite", trade_date=p.trade_date, field=name, value=v)
        if not _is_finite_positive(p.open) or not _is_finite_positive(p.high) or not _is_finite_positive(p.low) or not _is_finite_positive(p.close):
            _raise("non_positive_price", "price must be >0", trade_date=p.trade_date)
        if not _is_finite_nonneg(p.volume) or not _is_finite_nonneg(p.amount):
            _raise("negative_volume_amount", "volume/amount must be >=0", trade_date=p.trade_date)

        # OHLC relation when all present
        if p.open is not None and p.high is not None and p.low is not None and p.close is not None:
            if not (p.low <= min(p.open, p.close) <= max(p.open, p.close) <= p.high):
                _raise("ohlc_invalid", "OHLC invalid", trade_date=p.trade_date, detail="low <= min(open,close) <= max(open,close) <= high")
            if p.low <= 0:
                _raise("low_nonpositive", "low must be >0", trade_date=p.trade_date, field="low", value=p.low)
            hl_spread = p.high / p.low - 1.0
            if hl_spread > policy.max_hl_spread:
                _raise("hl_spread_exceeded", "high/low spread too large", trade_date=p.trade_date, value=hl_spread, threshold=policy.max_hl_spread)

        # gap check (natural days)
        if prev_date is not None:
            gap = (p.trade_date - prev_date).days
            if gap > policy.max_gap_days:
                _raise("gap_exceeded", "date gap too large", trade_date=p.trade_date, value=gap, threshold=policy.max_gap_days, detail=f"{prev_date.isoformat()} -> {p.trade_date.isoformat()}")

        # return jump check
        if prev_close is not None and p.close is not None:
            r = p.close / prev_close - 1.0
            if abs(r) > policy.max_abs_return:
                _raise("abs_return_exceeded", "abs return too large", trade_date=p.trade_date, value=r, threshold=policy.max_abs_return)

        prev_close = p.close if p.close is not None else prev_close
        prev_date = p.trade_date

