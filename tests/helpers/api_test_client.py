from __future__ import annotations

from typing import Iterable


FIXED_MINIPROGRAM_POOL: list[tuple[str, str]] = [
    ("159915", "创业板ETF"),
    ("511010", "国债ETF"),
    ("513100", "纳指ETF"),
    ("518880", "黄金ETF"),
]


def upsert_and_fetch_etfs(
    client,
    *,
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    names: dict[str, str] | None = None,
) -> None:
    """Create ETF pool rows then fetch prices for all codes."""
    name_map = names or {}
    for code in [str(x) for x in codes]:
        name = str(name_map.get(code) or f"ETF-{code}")
        client.post(
            "/api/etf",
            json={
                "code": code,
                "name": name,
                "start_date": str(start_date),
                "end_date": str(end_date),
            },
        )
        assert client.post(f"/api/etf/{code}/fetch").status_code == 200
