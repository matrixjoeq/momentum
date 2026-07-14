from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from etf_momentum.db.base import Base

# pylint: disable=not-callable


class ValidationPolicy(Base):
    __tablename__ = "validation_policy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Example: 0.12 for 12% max abs daily return used in anomaly detection.
    max_abs_return: Mapped[float] = mapped_column(Float, nullable=False)

    # Example: 0.30 for (high/low - 1) threshold.
    max_hl_spread: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)

    # Max allowed "long gap" in natural days inside a requested range.
    max_gap_days: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EtfPool(Base):
    __tablename__ = "etf_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD

    validation_policy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("validation_policy.id"), nullable=True
    )
    max_abs_return_override: Mapped[float | None] = mapped_column(Float, nullable=True)

    last_fetch_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Latest available data range in DB for this code (YYYYMMDD).
    last_data_start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EtfPrice(Base):
    __tablename__ = "etf_prices"
    __table_args__ = (
        UniqueConstraint(
            "code", "trade_date", "adjust", name="uq_etf_prices_code_trade_date_adjust"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="eastmoney")
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="qfq")

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EtfResearchGroup(Base):
    """
    ETF research groups persisted in DB.
    """

    __tablename__ = "etf_research_group"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EtfResearchGroupItem(Base):
    """
    Symbols bound to an ETF research group.
    """

    __tablename__ = "etf_research_group_item"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "code", name="uq_etf_research_group_item_group_code"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("etf_research_group.id"), index=True, nullable=False
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class IngestionBatch(Base):
    __tablename__ = "ingestion_batch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="eastmoney")
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="qfq")

    start_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    end_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pre_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    post_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Validation params used for this batch (effective, after override)
    val_max_abs_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_max_hl_spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_max_gap_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class IngestionItem(Base):
    __tablename__ = "ingestion_item"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "trade_date", name="uq_ingestion_item_batch_trade_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ingestion_batch.id"), index=True, nullable=False
    )

    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    action: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # "insert" | "update"

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EtfPriceAudit(Base):
    __tablename__ = "etf_price_audit"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "code", "trade_date", name="uq_audit_batch_code_trade_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ingestion_batch.id"), index=True, nullable=False
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    adjust: Mapped[str] = mapped_column(String(8), nullable=False)

    audited_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OffFundPool(Base):
    """
    Off-exchange mutual fund candidate pool.
    Isolated from ETF/macros to avoid cross-impact.
    """

    __tablename__ = "off_fund_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD

    last_fetch_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    last_data_start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class OffFundNav(Base):
    """
    Daily NAV series for off-exchange mutual funds.
    adjust:
      - none: raw unit nav
      - qfq: forward-adjusted
      - hfq: backward-adjusted
    """

    __tablename__ = "off_fund_navs"
    __table_args__ = (
        UniqueConstraint(
            "code",
            "trade_date",
            "adjust",
            name="uq_off_fund_navs_code_trade_date_adjust",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    nav: Mapped[float | None] = mapped_column(Float, nullable=True)
    accum_nav: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="eastmoney")
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="none")

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OffFundEvent(Base):
    """
    Dividend/split records used for adjusted NAV reconstruction fallback.
    """

    __tablename__ = "off_fund_events"
    __table_args__ = (
        UniqueConstraint(
            "code",
            "effective_date",
            "event_type",
            "event_key",
            name="uq_off_fund_events_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    effective_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # dividend|split
    event_key: Mapped[str] = mapped_column(String(128), nullable=False, default="")

    cash_dividend: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # cash per share
    split_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # e.g. 1.2 means 1 -> 1.2
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="eastmoney")

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OffFundRegressionFactorConfig(Base):
    """
    Saved factor configuration profiles for off-fund regression classification.
    """

    __tablename__ = "off_fund_regression_factor_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    benchmark_profile: Mapped[str] = mapped_column(
        String(64), nullable=False, default="cn_stock_core"
    )
    benchmark_factors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class OffFundResearchState(Base):
    """
    Shared global state for off-fund research parameter panel.
    Single row (id=1): date range, pricing basis, sizing/rebalance controls.
    """

    __tablename__ = "off_fund_research_state"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False, default=1
    )
    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="hfq")
    risk_free_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.025)
    inner_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="risk_parity_cov"
    )
    rp_window: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    rebalance_cycle: Mapped[str] = mapped_column(
        String(16), nullable=False, default="daily"
    )
    drift_rebalance_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    drift_abs_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.05
    )
    drift_rel_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.25
    )
    pair_chart_prefs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class GlobalBenchmarkPool(Base):
    """
    Global benchmark index candidate pool.
    Independent from ETF/off-fund/futures pools.
    """

    __tablename__ = "global_benchmark_pool"
    __table_args__ = (
        UniqueConstraint(
            "code", "series_kind", name="uq_global_benchmark_pool_code_kind"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    series_kind: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False, default="price"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    code_format: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provider_hint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provider_symbol: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD

    last_fetch_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    last_data_start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class GlobalBenchmarkPrice(Base):
    """
    Global benchmark index daily prices.
    Price basis is fixed to raw/unadjusted (`none`).
    """

    __tablename__ = "global_benchmark_prices"
    __table_args__ = (
        UniqueConstraint(
            "code",
            "series_kind",
            "trade_date",
            "adjust",
            name="uq_global_benchmark_prices_code_kind_trade_date_adjust",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    series_kind: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False, default="price"
    )
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="none")

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FuturesPool(Base):
    """
    Futures candidate pool.
    Isolated from ETF/off-fund/macro tables.
    """

    __tablename__ = "futures_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    min_margin_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    contract_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    min_price_tick: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    last_fetch_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    last_data_start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)

    contract_extend_calendar_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=366
    )
    contract_parallel: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_contract_fetch_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_contract_fetch_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    last_contract_fetch_message: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FuturesPrice(Base):
    """
    Futures daily prices from Sina (via AkShare).
    Futures have no dividend adjustment here; only raw (none) is stored.
    """

    __tablename__ = "futures_prices"
    __table_args__ = (
        UniqueConstraint(
            "code",
            "trade_date",
            "adjust",
            name="uq_futures_prices_code_trade_date_adjust",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pool_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("futures_pool.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    settle: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    hold: Mapped[float | None] = mapped_column(Float, nullable=True)

    dominant_contract_suffix: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    roll_from_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    roll_to_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    roll_from_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    roll_from_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    roll_to_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    roll_to_close: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="sina")
    adjust: Mapped[str] = mapped_column(String(8), nullable=False, default="none")

    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FuturesContractFetchStatus(Base):
    """
    Per-deliverable-month contract ingestion status for a futures pool row (main symbol).
    """

    __tablename__ = "futures_contract_fetch_status"
    __table_args__ = (
        UniqueConstraint(
            "pool_id", "contract_code", name="uq_futures_contract_fetch_pool_contract"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pool_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("futures_pool.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    contract_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    rows_upserted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FuturesResearchGroup(Base):
    """
    Futures research groups (isolated from ETF/off-fund research groups).
    """

    __tablename__ = "futures_research_group"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FuturesResearchGroupItem(Base):
    """
    Symbols bound to a futures research group.
    """

    __tablename__ = "futures_research_group_item"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "code", name="uq_futures_research_group_item_group_code"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("futures_research_group.id"), index=True, nullable=False
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class FuturesResearchState(Base):
    """
    Shared global state for futures research page.
    Single row (id=1): start/end/dynamic_universe/current quick range selection.
    """

    __tablename__ = "futures_research_state"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False, default=1
    )
    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    dynamic_universe: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    quick_range_key: Mapped[str | None] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SimPortfolio(Base):
    __tablename__ = "sim_portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="默认账户")
    base_ccy: Mapped[str] = mapped_column(String(16), nullable=False, default="CNY")
    initial_cash: Mapped[float] = mapped_column(
        Float, nullable=False, default=1_000_000.0
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SimStrategyConfig(Base):
    __tablename__ = "sim_strategy_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_portfolio.id"), index=True, nullable=False
    )

    codes_json: Mapped[str] = mapped_column(String(512), nullable=False)  # json list
    rebalance: Mapped[str] = mapped_column(String(16), nullable=False, default="weekly")
    lookback_days: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    exec_price: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    rebalance_shift: Mapped[str] = mapped_column(
        String(8), nullable=False, default="prev"
    )
    risk_controls_json: Mapped[str] = mapped_column(
        String(1024), nullable=False, default="{}"
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SimVariant(Base):
    __tablename__ = "sim_variant"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "anchor_weekday", name="uq_sim_variant_portfolio_anchor"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_portfolio.id"), index=True, nullable=False
    )
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_strategy_config.id"), index=True, nullable=False
    )

    anchor_weekday: Mapped[int] = mapped_column(Integer, nullable=False)  # 1..5
    label: Mapped[str] = mapped_column(String(8), nullable=False)  # MON..FRI
    is_active: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # 0/1 for sqlite simplicity

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SimDecision(Base):
    __tablename__ = "sim_decision"
    __table_args__ = (
        UniqueConstraint(
            "variant_id", "decision_date", name="uq_sim_decision_variant_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_variant.id"), index=True, nullable=False
    )

    decision_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    effective_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    picked_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scores_json: Mapped[str] = mapped_column(String(4096), nullable=False, default="{}")
    prev_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason_json: Mapped[str] = mapped_column(String(2048), nullable=False, default="{}")

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SimTrade(Base):
    __tablename__ = "sim_trade"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_variant.id"), index=True, nullable=False
    )
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY/SELL
    price: Mapped[float] = mapped_column(Float, nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    decision_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sim_decision.id"), index=True, nullable=True
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SimPositionDaily(Base):
    __tablename__ = "sim_position_daily"
    __table_args__ = (
        UniqueConstraint(
            "variant_id", "trade_date", name="uq_sim_position_variant_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sim_variant.id"), index=True, nullable=False
    )
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    positions_json: Mapped[str] = mapped_column(
        String(4096), nullable=False, default="{}"
    )
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    nav: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    mdd: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SyncJob(Base):
    """
    Long-running admin jobs (e.g. market data sync) tracked in DB so callers can poll status.
    """

    __tablename__ = "sync_job"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_sync_job_dedupe_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False, default="sync_fixed_pool"
    )
    dedupe_key: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), index=True, nullable=False, default="queued"
    )  # queued|running|success|failed

    run_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    full_refresh: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    adjusts: Mapped[str] = mapped_column(
        String(64), nullable=False, default="qfq,hfq,none"
    )  # comma-separated

    progress_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class MacroSeriesMeta(Base):
    """
    Macro series metadata registry.

    Examples:
    - series_id=DGS10 provider=fred provider_symbol=DGS10 unit=%
    - series_id=DINIW provider=sina provider_symbol=DINIW unit=index
    - series_id=XAUUSD provider=stooq provider_symbol=XAUUSD unit=USD/oz
    - series_id=GC_FUT provider=yahoo provider_symbol=GC=F unit=USD/oz
    """

    __tablename__ = "macro_series_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # rates|fx|gold_spot|gold_fut|...

    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)

    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    calendar: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class MacroPrice(Base):
    """
    Unified daily macro series prices (OHLCV optional).
    For rates/fx, typically only close is populated.
    """

    __tablename__ = "macro_prices"
    __table_args__ = (
        UniqueConstraint(
            "series_id", "trade_date", name="uq_macro_prices_series_id_trade_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)

    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MacroIngestionBatch(Base):
    __tablename__ = "macro_ingestion_batch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    end_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="running"
    )  # running|success|failed
    message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class LiveAccount(Base):
    __tablename__ = "live_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    base_ccy: Mapped[str] = mapped_column(String(16), nullable=False, default="CNY")
    initial_cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LiveShareholderAccount(Base):
    __tablename__ = "live_shareholder_account"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "shareholder_account",
            name="uq_live_shareholder_account_account_holder",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    shareholder_account: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveStrategy(Base):
    __tablename__ = "live_strategy"
    __table_args__ = (
        UniqueConstraint("account_id", "name", name="uq_live_strategy_account_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LiveStrategyProfile(Base):
    __tablename__ = "live_strategy_profile"
    __table_args__ = (
        UniqueConstraint("strategy_id", name="uq_live_strategy_profile_strategy"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=False
    )
    strategy_type: Mapped[str] = mapped_column(
        String(24), nullable=False, default="etf_spot"
    )  # etf_spot|bond_repo
    capital_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="segregated"
    )  # segregated|shared_account_cash
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LiveAccountCashflow(Base):
    __tablename__ = "live_account_cashflow"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    flow_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    flow_type: Mapped[str] = mapped_column(
        String(24), nullable=False, default="deposit"
    )  # deposit|withdraw|transfer_to_strategy|transfer_from_strategy|dividend|manual
    transfer_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveStrategyCashflow(Base):
    __tablename__ = "live_strategy_cashflow"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=False
    )
    flow_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    flow_type: Mapped[str] = mapped_column(
        String(24), nullable=False, default="transfer_in"
    )  # transfer_in|transfer_out|manual|dividend
    transfer_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveTrade(Base):
    __tablename__ = "live_trade"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_live_trade_idempotency_key"),
        UniqueConstraint("broker_trade_no", name="uq_live_trade_broker_trade_no"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    strategy_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=False
    )
    shareholder_account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_shareholder_account.id"), index=True, nullable=False
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    trade_time: Mapped[str] = mapped_column(
        String(8), nullable=False, default="09:30:00"
    )
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY|SELL
    price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    broker_trade_no: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveRepoTradeDetail(Base):
    __tablename__ = "live_repo_trade_detail"
    __table_args__ = (
        UniqueConstraint("trade_id", name="uq_live_repo_trade_detail_trade"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_trade.id"), index=True, nullable=False
    )
    repo_action: Mapped[str] = mapped_column(String(8), nullable=False)  # OPEN|CLOSE
    principal_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    annual_rate_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    interest_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    day_count_basis: Mapped[int] = mapped_column(Integer, nullable=False, default=365)
    open_trade_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_trade.id"), index=True, nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LiveTradeAuditLog(Base):
    __tablename__ = "live_trade_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    account_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    strategy_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # update|delete
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveCorporateActionEvent(Base):
    __tablename__ = "live_corporate_action_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=True
    )
    strategy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=True
    )
    event_type: Mapped[str] = mapped_column(
        String(24), nullable=False
    )  # cash_dividend|split|share_conversion|code_change
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    new_code: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    event_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    effective_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    ratio_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveSymbolAlias(Base):
    __tablename__ = "live_symbol_alias"
    __table_args__ = (
        UniqueConstraint(
            "old_code", "effective_date", name="uq_live_symbol_alias_old_code_effective"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    old_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    new_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    effective_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveClosedRound(Base):
    __tablename__ = "live_closed_round"
    __table_args__ = (
        UniqueConstraint(
            "scope_type",
            "scope_id",
            "code",
            "round_no",
            name="uq_live_closed_round_scope_code_round_no",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(
        String(16), index=True, nullable=False
    )  # account|strategy
    scope_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    strategy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=True
    )
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    open_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    close_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    holding_duration_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buy_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sell_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    buy_qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sell_qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_buy_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_sell_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    return_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveClosedRoundLeg(Base):
    __tablename__ = "live_closed_round_leg"
    __table_args__ = (
        UniqueConstraint(
            "round_id", "sort_order", name="uq_live_closed_round_leg_round_sort"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_closed_round.id"), index=True, nullable=False
    )
    trade_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_trade.id"), index=True, nullable=False
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    trade_time: Mapped[str] = mapped_column(
        String(8), nullable=False, default="09:30:00"
    )


class LiveHoldingSnapshot(Base):
    __tablename__ = "live_holding_snapshot"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_date",
            "scope_type",
            "scope_id",
            "code",
            name="uq_live_holding_snapshot_scope_code_day",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    scope_type: Mapped[str] = mapped_column(
        String(16), index=True, nullable=False
    )  # account|strategy
    scope_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    strategy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=True
    )
    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    stale_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LiveNavDaily(Base):
    __tablename__ = "live_nav_daily"
    __table_args__ = (
        UniqueConstraint(
            "nav_date", "scope_type", "scope_id", name="uq_live_nav_daily_scope_day"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nav_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    scope_type: Mapped[str] = mapped_column(
        String(16), index=True, nullable=False
    )  # account|strategy
    scope_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("live_account.id"), index=True, nullable=False
    )
    strategy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("live_strategy.id"), index=True, nullable=True
    )
    equity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    external_flow: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trading_fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_return_twr: Mapped[float | None] = mapped_column(Float, nullable=True)
    daily_return_dietz: Mapped[float | None] = mapped_column(Float, nullable=True)
    nav_twr: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    nav_dietz: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    selection_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    timing_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_drag_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_drag_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    repo_carry_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    repo_fee_drag_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
