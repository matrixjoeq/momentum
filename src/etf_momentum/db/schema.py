from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _sqlite_has_column(engine: Engine, table: str, column: str) -> bool:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(r[1] == column for r in rows)  # (cid, name, type, notnull, dflt_value, pk)


def _sqlite_add_column(engine: Engine, table: str, ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def _sqlite_table_exists(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table},
        ).fetchone()
    return row is not None


def _sqlite_unique_index_columns(engine: Engine, table: str) -> list[list[str]]:
    """
    Return list of unique index column lists for the table.
    """
    with engine.connect() as conn:
        idxs = conn.execute(text(f"PRAGMA index_list({table})")).fetchall()
        out: list[list[str]] = []
        for r in idxs:
            # (seq, name, unique, origin, partial)
            name = r[1]
            unique = int(r[2])
            if unique != 1:
                continue
            cols = conn.execute(text(f"PRAGMA index_info({name})")).fetchall()
            out.append([c[2] for c in cols])  # (seqno, cid, name)
        return out


def _ensure_etf_prices_unique_on_adjust(engine: Engine) -> None:
    """
    Legacy DBs had unique(code, trade_date). We need unique(code, trade_date, adjust)
    to store multiple adjustment types in the same DB.
    """
    if engine.dialect.name != "sqlite":
        return
    if not _sqlite_table_exists(engine, "etf_prices"):
        return
    uniqs = _sqlite_unique_index_columns(engine, "etf_prices")
    # already ok if any unique index includes (code, trade_date, adjust) in order
    if any(u[:3] == ["code", "trade_date", "adjust"] for u in uniqs):
        return
    # migrate only if legacy unique on (code, trade_date) exists
    if not any(u[:2] == ["code", "trade_date"] for u in uniqs):
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE etf_prices RENAME TO etf_prices_old"))
        # capture legacy columns (some old DBs may miss OHLCV fields)
        old_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(etf_prices_old)")).fetchall()]

        conn.execute(
            text(
                """
                CREATE TABLE etf_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(32) NOT NULL,
                    trade_date DATE NOT NULL,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    amount FLOAT,
                    source VARCHAR(32) NOT NULL DEFAULT 'eastmoney',
                    adjust VARCHAR(8) NOT NULL DEFAULT 'qfq',
                    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT uq_etf_prices_code_trade_date_adjust UNIQUE (code, trade_date, adjust)
                )
                """
            )
        )
        new_cols = [
            "id",
            "code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "source",
            "adjust",
            "ingested_at",
        ]
        select_exprs = []
        for c in new_cols:
            if c in old_cols:
                select_exprs.append(c)
            elif c == "ingested_at":
                select_exprs.append("CURRENT_TIMESTAMP AS ingested_at")
            else:
                select_exprs.append(f"NULL AS {c}")
        conn.execute(
            text(
                f"""
                INSERT INTO etf_prices ({", ".join(new_cols)})
                SELECT {", ".join(select_exprs)} FROM etf_prices_old
                """
            )
        )
        conn.execute(text("DROP TABLE etf_prices_old"))
        # indexes (best-effort; matches SQLAlchemy index=True fields)
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_etf_prices_code ON etf_prices(code)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_etf_prices_trade_date ON etf_prices(trade_date)"))


def ensure_sqlite_schema(engine: Engine) -> None:
    """
    Minimal SQLite schema evolution helper.

    This keeps existing local DB files usable while we haven't introduced Alembic yet.
    Only adds new nullable columns; it does not drop/rename columns.
    """
    if engine.dialect.name != "sqlite":
        return

    if not _sqlite_has_column(engine, "etf_pool", "validation_policy_id"):
        _sqlite_add_column(engine, "etf_pool", "validation_policy_id INTEGER")
    if not _sqlite_has_column(engine, "etf_pool", "max_abs_return_override"):
        _sqlite_add_column(engine, "etf_pool", "max_abs_return_override FLOAT")
    if not _sqlite_has_column(engine, "etf_pool", "last_data_start_date"):
        _sqlite_add_column(engine, "etf_pool", "last_data_start_date VARCHAR(8)")
    if not _sqlite_has_column(engine, "etf_pool", "last_data_end_date"):
        _sqlite_add_column(engine, "etf_pool", "last_data_end_date VARCHAR(8)")

    # ingestion_batch validation params (added later)
    if not _sqlite_has_column(engine, "ingestion_batch", "val_max_abs_return"):
        _sqlite_add_column(engine, "ingestion_batch", "val_max_abs_return FLOAT")
    if not _sqlite_has_column(engine, "ingestion_batch", "val_max_hl_spread"):
        _sqlite_add_column(engine, "ingestion_batch", "val_max_hl_spread FLOAT")
    if not _sqlite_has_column(engine, "ingestion_batch", "val_max_gap_days"):
        _sqlite_add_column(engine, "ingestion_batch", "val_max_gap_days INTEGER")

    _ensure_etf_prices_unique_on_adjust(engine)

