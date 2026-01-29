from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from etf_momentum.db.base import Base

# pylint: disable=not-callable


class ValidationPolicy(Base):
    __tablename__ = "validation_policy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
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
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class EtfPool(Base):
    __tablename__ = "etf_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)  # YYYYMMDD

    validation_policy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("validation_policy.id"), nullable=True
    )
    max_abs_return_override: Mapped[float | None] = mapped_column(Float, nullable=True)

    last_fetch_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_fetch_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_fetch_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Latest available data range in DB for this code (YYYYMMDD).
    last_data_start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_data_end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class EtfPrice(Base):
    __tablename__ = "etf_prices"
    __table_args__ = (UniqueConstraint("code", "trade_date", "adjust", name="uq_etf_prices_code_trade_date_adjust"),)

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
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class IngestionItem(Base):
    __tablename__ = "ingestion_item"
    __table_args__ = (UniqueConstraint("batch_id", "trade_date", name="uq_ingestion_item_batch_trade_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingestion_batch.id"), index=True, nullable=False)

    code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # "insert" | "update"

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EtfPriceAudit(Base):
    __tablename__ = "etf_price_audit"
    __table_args__ = (UniqueConstraint("batch_id", "code", "trade_date", name="uq_audit_batch_code_trade_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingestion_batch.id"), index=True, nullable=False)
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


class SimPortfolio(Base):
    __tablename__ = "sim_portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="默认账户")
    base_ccy: Mapped[str] = mapped_column(String(16), nullable=False, default="CNY")
    initial_cash: Mapped[float] = mapped_column(Float, nullable=False, default=1_000_000.0)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SimStrategyConfig(Base):
    __tablename__ = "sim_strategy_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_portfolio.id"), index=True, nullable=False)

    codes_json: Mapped[str] = mapped_column(String(512), nullable=False)  # json list
    rebalance: Mapped[str] = mapped_column(String(16), nullable=False, default="weekly")
    lookback_days: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    exec_price: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    rebalance_shift: Mapped[str] = mapped_column(String(8), nullable=False, default="prev")
    risk_controls_json: Mapped[str] = mapped_column(String(1024), nullable=False, default="{}")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SimVariant(Base):
    __tablename__ = "sim_variant"
    __table_args__ = (UniqueConstraint("portfolio_id", "anchor_weekday", name="uq_sim_variant_portfolio_anchor"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_portfolio.id"), index=True, nullable=False)
    config_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_strategy_config.id"), index=True, nullable=False)

    anchor_weekday: Mapped[int] = mapped_column(Integer, nullable=False)  # 0..4
    label: Mapped[str] = mapped_column(String(8), nullable=False)  # MON..FRI
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0/1 for sqlite simplicity

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SimDecision(Base):
    __tablename__ = "sim_decision"
    __table_args__ = (UniqueConstraint("variant_id", "decision_date", name="uq_sim_decision_variant_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_variant.id"), index=True, nullable=False)

    decision_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    effective_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    picked_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scores_json: Mapped[str] = mapped_column(String(4096), nullable=False, default="{}")
    prev_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason_json: Mapped[str] = mapped_column(String(2048), nullable=False, default="{}")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SimTrade(Base):
    __tablename__ = "sim_trade"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_variant.id"), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # BUY/SELL
    price: Mapped[float] = mapped_column(Float, nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    decision_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("sim_decision.id"), index=True, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SimPositionDaily(Base):
    __tablename__ = "sim_position_daily"
    __table_args__ = (UniqueConstraint("variant_id", "trade_date", name="uq_sim_position_variant_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_variant.id"), index=True, nullable=False)
    trade_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    positions_json: Mapped[str] = mapped_column(String(4096), nullable=False, default="{}")
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    nav: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    mdd: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SyncJob(Base):
    """
    Long-running admin jobs (e.g. market data sync) tracked in DB so callers can poll status.
    """

    __tablename__ = "sync_job"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_sync_job_dedupe_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="sync_fixed_pool")
    dedupe_key: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    status: Mapped[str] = mapped_column(String(16), index=True, nullable=False, default="queued")  # queued|running|success|failed

    run_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    full_refresh: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    adjusts: Mapped[str] = mapped_column(String(64), nullable=False, default="qfq,hfq,none")  # comma-separated

    progress_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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
    series_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)  # rates|fx|gold_spot|gold_fut|...

    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)

    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    calendar: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MacroPrice(Base):
    """
    Unified daily macro series prices (OHLCV optional).
    For rates/fx, typically only close is populated.
    """

    __tablename__ = "macro_prices"
    __table_args__ = (UniqueConstraint("series_id", "trade_date", name="uq_macro_prices_series_id_trade_date"),)

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
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MacroIngestionBatch(Base):
    __tablename__ = "macro_ingestion_batch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    end_date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")  # running|success|failed
    message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
