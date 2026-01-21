from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import qlib
from qlib.data import D


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    rebalance_weekday: int
    lookback_days: int
    topk: int
    hold_days: int
    risk_free: float


@dataclasses.dataclass(frozen=True)
class QlibConfig:
    provider_uri: str
    region: str


@dataclasses.dataclass(frozen=True)
class OptimizeConfig:
    metric: str
    lookback_days: list[int]
    topk: list[int]
    rebalance_weekday: list[int]


def _load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def _parse_dates(start: str, end: str) -> tuple[str, str]:
    # normalize to YYYY-MM-DD
    s = str(start)
    e = str(end)
    return s, e


def init_qlib(cfg: QlibConfig) -> None:
    qlib.init(provider_uri=cfg.provider_uri, region=cfg.region)


def load_close_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    # qlib returns MultiIndex (instrument, datetime) by default
    df = D.features(symbols, fields=["$close"], start_time=start, end_time=end, freq="day")
    if df is None or df.empty:
        raise ValueError("qlib returned empty price data; check provider_uri and symbols")
    df = df.reset_index()
    if "instrument" not in df.columns or "datetime" not in df.columns:
        raise ValueError("unexpected qlib features format")
    df = df.pivot(index="datetime", columns="instrument", values="$close").sort_index()
    return df.astype(float)


def _trading_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return [pd.Timestamp(x).normalize() for x in index]


def _calc_metrics(nav: pd.Series, rf: float) -> dict:
    ret = nav.pct_change().fillna(0.0)
    ann_factor = 252.0
    ann_ret = float((1.0 + ret.mean()) ** ann_factor - 1.0) if len(ret) else float("nan")
    ann_vol = float(ret.std(ddof=1) * np.sqrt(ann_factor)) if len(ret) else float("nan")
    sharpe = float((ret.mean() - rf / ann_factor) / (ret.std(ddof=1) or np.nan) * np.sqrt(ann_factor)) if len(ret) else float("nan")
    cum = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) else float("nan")
    return {
        "cumulative_return": cum,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
    }


def _score_momentum(close: pd.DataFrame, decision_idx: int, lookback: int) -> pd.Series:
    if decision_idx - lookback < 0:
        return pd.Series(dtype=float)
    last = close.iloc[decision_idx]
    prev = close.iloc[decision_idx - lookback]
    score = last / prev - 1.0
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    return score


def run_rotation(
    close: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.Series, list[dict]]:
    idx = _trading_days(close.index)
    if not idx:
        raise ValueError("no trading days")

    # execution days: match weekday on trading calendar
    exec_days = [i for i, d in enumerate(idx) if int(d.weekday()) == int(cfg.rebalance_weekday)]
    if not exec_days:
        raise ValueError("no execution days for given weekday")

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    trades: list[dict] = []

    for j, exec_i in enumerate(exec_days):
        # decision day = previous trading day
        decision_i = exec_i - 1
        if decision_i < 0:
            continue
        score = _score_momentum(close, decision_i, cfg.lookback_days)
        if score.empty:
            continue
        pick = score.sort_values(ascending=False).head(cfg.topk).index.tolist()
        w = 1.0 / float(len(pick)) if pick else 0.0

        start_i = exec_i
        end_i = (exec_days[j + 1] - 1) if j + 1 < len(exec_days) else min(len(idx) - 1, exec_i + cfg.hold_days - 1)
        weights.loc[close.index[start_i] : close.index[end_i], pick] = w

        trades.append(
            {
                "exec_date": str(idx[exec_i].date()),
                "decision_date": str(idx[decision_i].date()),
                "picks": list(pick),
                "scores": {k: float(score.get(k)) for k in pick},
            }
        )

    ret = close.pct_change().fillna(0.0)
    port_ret = (weights * ret).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()
    nav.iloc[0] = 1.0
    return nav, trades


def _grid(iterables: Iterable[Iterable[int]]) -> list[tuple[int, ...]]:
    from itertools import product

    return list(product(*iterables))


def optimize(
    close: pd.DataFrame,
    base: BacktestConfig,
    opt: OptimizeConfig,
) -> list[dict]:
    results: list[dict] = []
    for lookback, topk, wd in _grid([opt.lookback_days, opt.topk, opt.rebalance_weekday]):
        cfg = dataclasses.replace(base, lookback_days=int(lookback), topk=int(topk), rebalance_weekday=int(wd))
        nav, _ = run_rotation(close, cfg)
        metrics = _calc_metrics(nav, cfg.risk_free)
        results.append(
            {
                "rebalance_weekday": cfg.rebalance_weekday,
                "lookback_days": cfg.lookback_days,
                "topk": cfg.topk,
                **metrics,
            }
        )
    metric = opt.metric
    results.sort(key=lambda r: (r.get(metric) if np.isfinite(r.get(metric, float("nan"))) else -1e18), reverse=True)
    return results


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    parser.add_argument("--mode", choices=["backtest", "optimize"], default="optimize")
    args = parser.parse_args()

    cfg = _load_toml(Path(args.config))
    qcfg = QlibConfig(**cfg["qlib"])
    init_qlib(qcfg)

    symbols = list(cfg["universe"]["symbols"])
    start, end = _parse_dates(cfg["backtest"]["start"], cfg["backtest"]["end"])
    close = load_close_prices(symbols, start, end)

    bt_cfg = BacktestConfig(
        start=start,
        end=end,
        rebalance_weekday=int(cfg["backtest"]["rebalance_weekday"]),
        lookback_days=int(cfg["backtest"]["lookback_days"]),
        topk=int(cfg["backtest"]["topk"]),
        hold_days=int(cfg["backtest"]["hold_days"]),
        risk_free=float(cfg["backtest"]["risk_free"]),
    )

    out_dir = Path(__file__).with_name("outputs") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(out_dir)

    if args.mode == "backtest":
        nav, trades = run_rotation(close, bt_cfg)
        metrics = _calc_metrics(nav, bt_cfg.risk_free)
        nav.to_csv(out_dir / "nav.csv", header=["nav"])
        _write_csv(trades, out_dir / "trades.csv")
        _write_csv([metrics], out_dir / "metrics.csv")
        print("saved to", out_dir)
        return 0

    opt_cfg = OptimizeConfig(**cfg["optimize"])
    results = optimize(close, bt_cfg, opt_cfg)
    _write_csv(results, out_dir / "grid_results.csv")
    top = results[0] if results else {}
    _write_csv([top], out_dir / "best.csv")
    print("saved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
