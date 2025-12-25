from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOMENTUM_", extra="ignore")

    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/etf_momentum.sqlite3")

    default_start_date: str = "20100101"  # YYYYMMDD
    default_end_date: str = "20991231"  # YYYYMMDD

    log_level: str = "INFO"


def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    # ensure parent exists
    s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return s

