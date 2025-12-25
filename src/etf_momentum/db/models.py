from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
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

