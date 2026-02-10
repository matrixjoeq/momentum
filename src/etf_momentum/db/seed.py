from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ValidationPolicy


DEFAULT_POLICIES: list[dict] = [
    {
        "name": "cn_stock_etf_10",
        "description": "A股大多数跟踪指数ETF，10%涨跌幅，异常检测阈值略放宽。",
        # Some broad-market ETFs can occasionally exceed 12% in adjusted series due to corporate actions / data quirks.
        # Keep it only slightly relaxed to still catch true anomalies.
        "max_abs_return": 0.13,
        "max_hl_spread": 0.30,
        "max_gap_days": 15,
    },
    {
        "name": "chinext_related_20",
        "description": "创业板相关ETF，20%涨跌幅，异常检测阈值略放宽。",
        "max_abs_return": 0.22,
        "max_hl_spread": 0.40,
        "max_gap_days": 15,
    },
    {
        "name": "star_related_30",
        "description": "科创板相关ETF，30%涨跌幅，异常检测阈值略放宽。",
        "max_abs_return": 0.33,
        "max_hl_spread": 0.50,
        "max_gap_days": 15,
    },
    {
        "name": "bond_10_strict",
        "description": "债券ETF，价格更平滑，收益跳变阈值更严格。",
        "max_abs_return": 0.08,
        "max_hl_spread": 0.15,
        "max_gap_days": 15,
    },
    {
        "name": "qdii_commod_fx",
        "description": "QDII/商品/跨市资产，可能受汇率与跨市影响，阈值更宽。",
        "max_abs_return": 0.35,
        "max_hl_spread": 0.60,
        "max_gap_days": 15,
    },
]


def ensure_default_policies(db: Session) -> None:
    existing = {p.name for p in db.execute(select(ValidationPolicy.name)).all()}
    to_add = [p for p in DEFAULT_POLICIES if p["name"] not in existing]
    for p in to_add:
        db.add(ValidationPolicy(**p))

    # Update existing policies to match current defaults (safe, deterministic).
    for p in DEFAULT_POLICIES:
        obj = db.execute(select(ValidationPolicy).where(ValidationPolicy.name == p["name"])).scalar_one_or_none()
        if obj is None:
            continue
        obj.description = p.get("description")
        obj.max_abs_return = p["max_abs_return"]
        obj.max_hl_spread = p["max_hl_spread"]
        obj.max_gap_days = p["max_gap_days"]

    db.flush()

