from __future__ import annotations

from sqlalchemy import text

from etf_momentum.db.init_db import init_db
from etf_momentum.db.session import make_engine


def test_init_db_adds_missing_columns_for_etf_pool(tmp_path) -> None:
    sqlite_path = tmp_path / "legacy.sqlite3"
    engine = make_engine(str(sqlite_path))
    # Create a legacy etf_pool table missing the new columns.
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE etf_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(32) NOT NULL UNIQUE,
                    name VARCHAR(128) NOT NULL,
                    start_date VARCHAR(8),
                    end_date VARCHAR(8),
                    last_fetch_at DATETIME,
                    last_fetch_status VARCHAR(32),
                    last_fetch_message VARCHAR(512),
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
    init_db(engine)
    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(etf_pool)")).fetchall()]
    assert "validation_policy_id" in cols
    assert "max_abs_return_override" in cols


def test_init_db_migrates_etf_prices_unique_to_include_adjust(tmp_path) -> None:
    sqlite_path = tmp_path / "legacy_prices.sqlite3"
    engine = make_engine(str(sqlite_path))
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE etf_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(32) NOT NULL,
                    trade_date DATE NOT NULL,
                    close FLOAT,
                    source VARCHAR(32) NOT NULL DEFAULT 'eastmoney',
                    adjust VARCHAR(8) NOT NULL DEFAULT 'qfq',
                    ingested_at DATETIME,
                    CONSTRAINT uq_etf_prices_code_trade_date UNIQUE (code, trade_date)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO etf_prices (code, trade_date, close, source, adjust)
                VALUES ('510300', '2024-01-02', 1.0, 'eastmoney', 'qfq')
                """
            )
        )
    init_db(engine)
    # should allow same code/date with different adjust after migration
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO etf_prices (code, trade_date, close, source, adjust)
                VALUES ('510300', '2024-01-02', 1.0, 'eastmoney', 'hfq')
                """
            )
        )

