from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferResult:
    policy_name: str
    source: str  # "name" | "lookup" | "default"


def _norm_text(x: str | None) -> str:
    return (x or "").strip().lower()


def infer_policy_name(*, code: str, name: str | None) -> InferResult:
    """
    Infer validation policy name from ETF name (primary signal).

    Note: With akshare==1.16.72 the available ETF metadata APIs don't reliably
    expose a robust 'asset_class' field. We therefore prioritize explicit
    keywords in the fund name, and allow UI/manual override as the final say.
    """
    n = _norm_text(name)
    if n:
        # STAR / 科创板
        if "科创" in n:
            return InferResult(policy_name="star_related_30", source="name")
        # ChiNext / 创业板
        if "创业" in n:
            return InferResult(policy_name="chinext_related_20", source="name")
        # Bonds
        if any(k in n for k in ["债", "国债", "信用", "可转债", "转债"]):
            return InferResult(policy_name="bond_10_strict", source="name")
        # QDII / cross-market / commodities
        if any(
            k in n
            for k in [
                "qdii",
                "纳指",
                "纳斯达克",
                "标普",
                "sp500",
                "恒生",
                "h股",
                "日经",
                "德国",
                "印度",
                "越南",
                "原油",
                "油",
                "黄金",
                "白银",
                "商品",
            ]
        ):
            return InferResult(policy_name="qdii_commod_fx", source="name")

    # Default: most A-share index ETFs (10% band)
    _ = code  # reserved for future code-based heuristics
    return InferResult(policy_name="cn_stock_etf_10", source="default")

