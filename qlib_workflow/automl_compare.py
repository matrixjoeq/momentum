from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.workflow import R


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
    return str(start), str(end)


def init_qlib(cfg: QlibConfig) -> None:
    qlib.init(provider_uri=cfg.provider_uri, region=cfg.region)


def load_close_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    df = D.features(symbols, fields=["$close"], start_time=start, end_time=end, freq="day")
    if df is None or df.empty:
        raise ValueError("qlib returned empty price data; check provider_uri and symbols")
    df = df.reset_index()
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


def run_rotation(close: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.Series, list[dict]]:
    idx = _trading_days(close.index)
    exec_days = [i for i, d in enumerate(idx) if int(d.weekday()) == int(cfg.rebalance_weekday)]
    if not exec_days:
        raise ValueError("no execution days for given weekday")

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    trades: list[dict] = []
    for j, exec_i in enumerate(exec_days):
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
            }
        )

    ret = close.pct_change().fillna(0.0)
    nav = (1.0 + (weights * ret).sum(axis=1)).cumprod()
    nav.iloc[0] = 1.0
    return nav, trades


def _grid(iterables: Iterable[Iterable[int]]) -> list[tuple[int, ...]]:
    from itertools import product

    return list(product(*iterables))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_baseline_api(start: str, end: str, exec_weekday: int, db_url: str) -> dict:
    os.environ["MOMENTUM_DB_URL"] = db_url
    from fastapi.testclient import TestClient
    from etf_momentum.app import app

    payload = {"start": start.replace("-", ""), "end": end.replace("-", ""), "anchor_weekday": int(exec_weekday)}
    c = TestClient(app)
    r = c.post("/api/analysis/rotation/weekly5-open", json=payload)
    r.raise_for_status()
    data = r.json()
    by_anchor = data.get("by_anchor", {}).get(str(exec_weekday), {})
    metrics = (by_anchor.get("metrics") or {}).get("strategy") or {}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    parser.add_argument("--experiment", default="qlib_rotation_automl")
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

    opt_cfg = OptimizeConfig(**cfg["optimize"])
    out_dir = Path(__file__).with_name("outputs") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(out_dir)

    results: list[dict] = []
    for lookback, topk, wd in _grid([opt_cfg.lookback_days, opt_cfg.topk, opt_cfg.rebalance_weekday]):
        cfg_i = dataclasses.replace(bt_cfg, lookback_days=int(lookback), topk=int(topk), rebalance_weekday=int(wd))
        with R.start(experiment_name=args.experiment, recorder_name=f"wd{wd}_lb{lookback}_k{topk}"):
            R.log_params(
                start=cfg_i.start,
                end=cfg_i.end,
                rebalance_weekday=cfg_i.rebalance_weekday,
                lookback_days=cfg_i.lookback_days,
                topk=cfg_i.topk,
                hold_days=cfg_i.hold_days,
            )
            nav, _ = run_rotation(close, cfg_i)
            metrics = _calc_metrics(nav, cfg_i.risk_free)
            R.log_metrics(**metrics)
        results.append(
            {
                "rebalance_weekday": cfg_i.rebalance_weekday,
                "lookback_days": cfg_i.lookback_days,
                "topk": cfg_i.topk,
                **metrics,
            }
        )

    metric = opt_cfg.metric
    results.sort(key=lambda r: (r.get(metric) if np.isfinite(r.get(metric, float("nan"))) else -1e18), reverse=True)
    _write_csv(results, out_dir / "grid_results.csv")

    best = results[0] if results else {}
    _write_csv([best], out_dir / "best.csv")

    ds = cfg.get("data_source", {})
    db_path = str(ds.get("sqlite_path", "")).strip()
    if db_path:
        baseline = run_baseline_api(start=start, end=end, exec_weekday=int(bt_cfg.rebalance_weekday), db_url=f"sqlite:///{db_path}")
        _write_csv([baseline], out_dir / "baseline.csv")
        if best:
            compare = {
                "baseline_sharpe": float(baseline.get("sharpe_ratio", float("nan"))),
                "best_sharpe": float(best.get("sharpe", float("nan"))),
                "baseline_cum": float(baseline.get("cumulative_return", float("nan"))),
                "best_cum": float(best.get("cumulative_return", float("nan"))),
            }
            _write_csv([compare], out_dir / "compare.csv")

    print("saved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
