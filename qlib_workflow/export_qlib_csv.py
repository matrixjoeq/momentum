from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def _load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fetch_prices(db_path: str, codes: list[str], adjust: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        q = """
        SELECT code, trade_date, open, high, low, close, volume
        FROM etf_prices
        WHERE code IN ({codes}) AND adjust = ?
        ORDER BY trade_date ASC
        """.format(codes=",".join(["?"] * len(codes)))
        params = list(codes) + [adjust]
        df = pd.read_sql_query(q, con, params=params, parse_dates=["trade_date"])
        if df.empty:
            raise ValueError("no rows found in etf_prices for given codes/adjust")
        return df
    finally:
        con.close()


def _format_date(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series).dt.strftime("%Y-%m-%d")


def export_csv(db_path: str, adjust: str, symbol_map: dict[str, str], out_dir: Path) -> None:
    _ensure_dir(out_dir)
    codes = list(symbol_map.keys())
    df = _fetch_prices(db_path, codes=codes, adjust=adjust)

    for code, sym in symbol_map.items():
        sub = df[df["code"] == code].copy()
        if sub.empty:
            print(f"[WARN] no data for code={code}")
            continue
        sub["date"] = _format_date(sub["trade_date"])
        sub["factor"] = 1.0
        sub = sub[["date", "open", "high", "low", "close", "volume", "factor"]]
        sub = sub.dropna(subset=["close"])
        out_path = out_dir / f"{sym}.csv"
        sub.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    args = parser.parse_args()

    cfg = _load_toml(Path(args.config))
    ds = cfg.get("data_source", {})
    db_path = str(ds.get("sqlite_path", "")).strip()
    adjust = str(ds.get("adjust", "hfq")).strip()
    out_dir = Path(str(ds.get("csv_dir", "qlib_workflow/csv")))
    symbol_map = dict(ds.get("symbol_map", {}))
    if not db_path:
        raise ValueError("data_source.sqlite_path is required")
    if not symbol_map:
        raise ValueError("data_source.symbol_map is required")

    export_csv(db_path=db_path, adjust=adjust, symbol_map=symbol_map, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
