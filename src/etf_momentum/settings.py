from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOMENTUM_",
        extra="ignore",
        # Local dev: use data/.env.local (gitignored). Cloud: environment variables.
        env_file=("data/.env.local", ".env"),
        env_file_encoding="utf-8",
    )

    data_dir: Path = Path("data")
    # Cloud deployment: if provided, use this SQLAlchemy database URL directly.
    # Example (MySQL 8): mysql+pymysql://user:pass@host:3306/dbname?charset=utf8mb4
    db_url: str | None = None

    # Local MySQL (preferred for local dev). Used only when db_url is not provided.
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "momentum"
    mysql_password: str = "momentum"
    mysql_db: str = "momentum"
    # Optional unix socket path (if you run mysqld with a socket in ./data/mysql).
    mysql_socket: str | None = None

    default_start_date: str = "20100101"  # YYYYMMDD
    default_end_date: str = "20991231"  # YYYYMMDD

    log_level: str = "INFO"

    # --- External data providers ---
    # FRED API key for US rates (e.g. DGS2/DGS5/DGS10/DGS30).
    # Set via environment variable: MOMENTUM_FRED_API_KEY
    fred_api_key: str | None = None

    # --- Market data sync (Cloud trigger preferred) ---
    # Recommended: use WeChat Cloud scheduled trigger to call /api/admin/sync/fixed-pool.
    # If you still want in-process scheduler, you can enable it explicitly.
    auto_sync_enabled: bool = False
    auto_sync_tz: str = "Asia/Shanghai"
    auto_sync_calendar: str = "XSHG"
    auto_sync_hour: int = 15
    auto_sync_minute: int = 15  # after close (15:00) with some buffer
    # For data consistency/completeness, refresh full history every run (qfq/hfq/none).
    # This is heavier than incremental mode but guarantees no stale history.
    auto_sync_full_refresh: bool = True

    # If set, /api/admin/sync/fixed-pool requires this token via header X-Momentum-Token or body.token
    sync_token: str | None = None


def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    if not s.db_url:
        # Local dev: construct DSN from mysql_* fields.
        # Use socket if provided, otherwise host/port.
        if s.mysql_socket:
            s.db_url = (
                f"mysql+pymysql://{s.mysql_user}:{s.mysql_password}"
                f"@localhost/{s.mysql_db}?charset=utf8mb4&unix_socket={s.mysql_socket}"
            )
        else:
            s.db_url = (
                f"mysql+pymysql://{s.mysql_user}:{s.mysql_password}"
                f"@{s.mysql_host}:{int(s.mysql_port)}/{s.mysql_db}?charset=utf8mb4"
            )
    return s

