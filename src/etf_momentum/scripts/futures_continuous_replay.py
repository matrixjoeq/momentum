#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_COLS = ["open", "high", "low", "close", "volume", "hold", "settle", "amount"]
PRICE_COLS = ["open", "high", "low", "close", "settle"]
ERROR_FIELDS = ["open", "high", "low", "close", "volume", "amount", "hold", "settle"]


@dataclass(frozen=True)
class ReplayConfig:
    underlying: str
    main_symbol: str
    start_date: str
    end_date: str
    contract_start: str
    contract_end: str
    switch_threshold: float
    usable_rel_mean_max: float
    usable_rel_p95_max: float
    usable_min_fields: int
    output_dir: Path


def _month_iter(start_yymm: str, end_yymm: str) -> list[str]:
    sy = int(start_yymm[:2])
    sm = int(start_yymm[2:])
    ey = int(end_yymm[:2])
    em = int(end_yymm[2:])
    out: list[str] = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(f"{y:02d}{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _yymm_from_yyyymmdd(s: str) -> str:
    d = pd.to_datetime(s, format="%Y%m%d")
    return d.strftime("%y%m")


def _shift_yymm(yymm: str, months: int) -> str:
    base = pd.to_datetime(f"20{yymm}", format="%Y%m")
    shifted = base + pd.DateOffset(months=months)
    return shifted.strftime("%y%m")


def _normalize_contract_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower() for c in df.columns]
    mapping: dict[str, str] = {}
    for i, c in enumerate(cols):
        raw = str(df.columns[i])
        if c in {"date", "trade_date", "日期", "交易日期"}:
            mapping[raw] = "date"
        elif c in {"open", "开盘"}:
            mapping[raw] = "open"
        elif c in {"high", "最高"}:
            mapping[raw] = "high"
        elif c in {"low", "最低"}:
            mapping[raw] = "low"
        elif c in {"close", "收盘"}:
            mapping[raw] = "close"
        elif c in {"volume", "成交量"}:
            mapping[raw] = "volume"
        elif c in {"hold", "持仓", "持仓量", "open_interest", "oi"}:
            mapping[raw] = "hold"
        elif c in {"settle", "结算", "结算价"}:
            mapping[raw] = "settle"
        elif c in {"amount", "成交额", "turnover", "total_turnover"}:
            mapping[raw] = "amount"
    out = df.rename(columns=mapping).copy()
    if "date" not in out.columns:
        return pd.DataFrame(columns=["date", *NUM_COLS])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in NUM_COLS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[["date", *NUM_COLS]].dropna(subset=["date"]).sort_values("date")
    return out


def _fetch_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = ak.futures_zh_daily_sina(symbol=symbol)
    df = _normalize_contract_df(raw)
    if df.empty:
        return df
    s = pd.to_datetime(start_date, format="%Y%m%d")
    e = pd.to_datetime(end_date, format="%Y%m%d")
    return df[(df["date"] >= s) & (df["date"] <= e)].copy()


def _load_contract_universe(cfg: ReplayConfig) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    data: dict[str, pd.DataFrame] = {}
    coverage: list[dict[str, object]] = []
    for yymm in _month_iter(cfg.contract_start, cfg.contract_end):
        symbol = f"{cfg.underlying}{yymm}"
        try:
            df = _fetch_symbol(symbol, cfg.start_date, cfg.end_date)
            if df.empty:
                coverage.append({"symbol": symbol, "ok": False, "rows": 0, "start": None, "end": None, "error": "empty"})
                continue
            data[symbol] = df
            coverage.append(
                {
                    "symbol": symbol,
                    "ok": True,
                    "rows": int(len(df)),
                    "start": str(df["date"].min().date()),
                    "end": str(df["date"].max().date()),
                    "error": None,
                }
            )
        except Exception as e:  # noqa: BLE001
            coverage.append({"symbol": symbol, "ok": False, "rows": 0, "start": None, "end": None, "error": str(e)})
    return data, coverage


def _build_hold_table(contract_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for sym, df in contract_data.items():
        x = df[["date", "hold"]].copy()
        x["symbol"] = sym
        pieces.append(x)
    panel = pd.concat(pieces, ignore_index=True)
    tbl = panel.pivot_table(index="date", columns="symbol", values="hold", aggfunc="last").sort_index()
    return tbl


def _build_quote_panel(contract_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, df in contract_data.items():
        x = df.copy().set_index("date").sort_index()
        out[sym] = x
    return out


def _argmax_hold(holds: pd.Series) -> str | None:
    v = holds.dropna()
    if v.empty:
        return None
    return str(v.idxmax())


def _replay_dominant(cfg: ReplayConfig, contract_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    holds_tbl = _build_hold_table(contract_data)
    panel = _build_quote_panel(contract_data)
    if holds_tbl.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = list(holds_tbl.index)
    chosen: list[str] = []
    prev_dom: str | None = None

    for i, d in enumerate(dates):
        today_holds = holds_tbl.loc[d]
        if i == 0:
            cur = _argmax_hold(today_holds)
            if cur is None:
                continue
            chosen.append(cur)
            prev_dom = cur
            continue

        prev_d = dates[i - 1]
        prev_holds = holds_tbl.loc[prev_d]
        cur = prev_dom
        if cur is None:
            cur = _argmax_hold(today_holds)
        else:
            cur_hold = prev_holds.get(cur, np.nan)
            best = _argmax_hold(prev_holds)
            if best is not None and pd.notna(cur_hold):
                best_hold = prev_holds.get(best, np.nan)
                if best != cur and pd.notna(best_hold) and float(best_hold) > cfg.switch_threshold * float(cur_hold):
                    cur = best
        if cur is None or cur not in panel or d not in panel[cur].index:
            fallback = _argmax_hold(today_holds)
            if fallback is None:
                continue
            cur = fallback
        chosen.append(cur)
        prev_dom = cur

    replay_rows: list[dict[str, object]] = []
    for d, sym in zip(dates[: len(chosen)], chosen, strict=True):
        row = panel[sym].loc[d]
        replay_rows.append(
            {
                "date": d,
                "dominant_symbol": sym,
                "open": row.get("open", np.nan),
                "high": row.get("high", np.nan),
                "low": row.get("low", np.nan),
                "close": row.get("close", np.nan),
                "volume": row.get("volume", np.nan),
                "hold": row.get("hold", np.nan),
                "settle": row.get("settle", np.nan),
            }
        )
    replay_df = pd.DataFrame(replay_rows).sort_values("date")

    switches: list[dict[str, object]] = []
    if not replay_df.empty:
        prev = None
        for r in replay_df.itertuples(index=False):
            if prev is not None and r.dominant_symbol != prev:
                switches.append({"date": str(r.date.date()), "from_symbol": prev, "to_symbol": r.dominant_symbol})
            prev = r.dominant_symbol
    switch_df = pd.DataFrame(switches)
    return replay_df, switch_df


def _calc_price_diff(
    panel: dict[str, pd.DataFrame], old_sym: str, new_sym: str, date: pd.Timestamp, field: str
) -> float | None:
    old_df = panel.get(old_sym)
    new_df = panel.get(new_sym)
    if old_df is None or new_df is None:
        return None
    if date not in old_df.index or date not in new_df.index:
        return None
    a = old_df.loc[date].get(field, np.nan)
    b = new_df.loc[date].get(field, np.nan)
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def _build_adjusted_continuous(
    replay88_df: pd.DataFrame, switch_df: pd.DataFrame, panel: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replay88_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = replay88_df.copy().sort_values("date").reset_index(drop=True)
    base_idx = base.set_index("date")
    idx = base_idx.index
    loc_by_date = {d: i for i, d in enumerate(idx)}

    pre_adj = pd.Series(0.0, index=idx)
    post_adj = pd.Series(0.0, index=idx)
    if not switch_df.empty:
        for r in switch_df.itertuples(index=False):
            d = pd.to_datetime(r.date)
            if d not in loc_by_date:
                continue
            i = loc_by_date[d]
            if i <= 0:
                continue
            prev_d = idx[i - 1]
            pre_delta = _calc_price_diff(panel, r.from_symbol, r.to_symbol, prev_d, "close")
            if pre_delta is not None:
                pre_adj.loc[idx <= prev_d] += pre_delta
            post_delta = _calc_price_diff(panel, r.from_symbol, r.to_symbol, d, "open")
            if post_delta is not None:
                post_adj.loc[idx >= d] += post_delta

    def apply_adj(adj: pd.Series) -> pd.DataFrame:
        out = base_idx.copy()
        for c in PRICE_COLS:
            if c not in out.columns:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce") + adj
        # Follow the referenced rule: amount is not adjusted and set to 0.
        out["amount"] = 0.0
        return out.reset_index()

    replay888 = apply_adj(pre_adj)
    replay889 = apply_adj(post_adj)
    return replay888, replay889


def _build_joined_for_error(replay88_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    if replay88_df.empty or main_df.empty:
        return pd.DataFrame()

    left = replay88_df.copy().set_index("date")
    right = main_df.copy().set_index("date")
    both = left.join(right, how="inner", lsuffix="_replay88", rsuffix="_main0")
    return both


def _calc_error_stats(joined_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for f in ERROR_FIELDS:
        col_a = f"{f}_replay88"
        col_b = f"{f}_main0"
        a_raw = joined_df[col_a] if col_a in joined_df.columns else pd.Series(np.nan, index=joined_df.index)
        b_raw = joined_df[col_b] if col_b in joined_df.columns else pd.Series(np.nan, index=joined_df.index)
        a = pd.to_numeric(a_raw, errors="coerce")
        b = pd.to_numeric(b_raw, errors="coerce")
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        if n == 0:
            rows.append(
                {
                    "field": f,
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "max_abs": np.nan,
                    "mape": np.nan,
                    "p95_ape": np.nan,
                }
            )
            continue
        diff = (a[valid] - b[valid]).astype(float)
        abs_diff = diff.abs()
        denom = b[valid].replace(0, np.nan).abs()
        ape = abs_diff / denom
        mape = ape.mean(skipna=True)
        p95_ape = ape.quantile(0.95)
        rows.append(
            {
                "field": f,
                "n": n,
                "mae": float(abs_diff.mean()),
                "rmse": float(np.sqrt((diff**2).mean())),
                "max_abs": float(abs_diff.max()),
                "mape": float(mape) if pd.notna(mape) else np.nan,
                "p95_ape": float(p95_ape) if pd.notna(p95_ape) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_usability(cfg: ReplayConfig, err_df: pd.DataFrame, compare_summary: dict[str, object]) -> dict[str, object]:
    if not compare_summary.get("ok"):
        return {
            "usable": False,
            "reason": "comparison not available",
            "rule": {
                "rel_mean_max": cfg.usable_rel_mean_max,
                "rel_p95_max": cfg.usable_rel_p95_max,
                "min_fields": cfg.usable_min_fields,
            },
            "covered_fields": [],
            "failed_fields": [],
        }
    key_fields = ["open", "high", "low", "close", "settle"]
    covered: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for f in key_fields:
        row = err_df[err_df["field"] == f]
        if row.empty:
            continue
        r = row.iloc[0]
        n = int(r.get("n", 0) or 0)
        if n <= 0:
            continue
        mape = float(r.get("mape", np.nan))
        p95 = float(r.get("p95_ape", np.nan))
        item = {"field": f, "n": n, "mape": mape, "p95_ape": p95}
        covered.append(item)
        if (
            (not np.isnan(mape) and mape > cfg.usable_rel_mean_max)
            or (not np.isnan(p95) and p95 > cfg.usable_rel_p95_max)
        ):
            failed.append(item)
    usable = len(covered) >= cfg.usable_min_fields and len(failed) == 0
    reason = "pass" if usable else (
        "insufficient covered fields" if len(covered) < cfg.usable_min_fields else "relative error threshold exceeded"
    )
    return {
        "usable": usable,
        "reason": reason,
        "rule": {
            "rel_mean_max": cfg.usable_rel_mean_max,
            "rel_p95_max": cfg.usable_rel_p95_max,
            "min_fields": cfg.usable_min_fields,
        },
        "covered_fields": covered,
        "failed_fields": failed,
    }


def _compare_with_main(replay88_df: pd.DataFrame, main_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    if replay88_df.empty or main_df.empty:
        return pd.DataFrame(), {"ok": False, "reason": "empty replay or main"}, pd.DataFrame()

    both = _build_joined_for_error(replay88_df, main_df)
    if both.empty:
        return both.reset_index(), {"ok": False, "reason": "no overlap"}, pd.DataFrame()

    both["close_abs_diff"] = (both["close_replay88"] - both["close_main0"]).abs()
    both["close_rel_diff"] = both["close_abs_diff"] / both["close_main0"].replace(0, np.nan).abs()
    err_df = _calc_error_stats(both)
    summary = {
        "ok": True,
        "overlap_days": int(len(both)),
        "date_start": str(both.index.min().date()),
        "date_end": str(both.index.max().date()),
        "close_abs_diff_mean": float(both["close_abs_diff"].mean(skipna=True)),
        "close_abs_diff_median": float(both["close_abs_diff"].median(skipna=True)),
        "close_rel_diff_mean": float(both["close_rel_diff"].mean(skipna=True)),
        "close_rel_diff_median": float(both["close_rel_diff"].median(skipna=True)),
        "close_rel_diff_p95": float(both["close_rel_diff"].quantile(0.95)),
        "nonzero_close_diff_days": int((both["close_abs_diff"] > 0).sum()),
    }
    return both.reset_index(), summary, err_df


def _plot_kline(df: pd.DataFrame, title: str, out_png: Path) -> None:
    if df.empty:
        return
    x = pd.to_datetime(df["date"])
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")  # noqa: E741
    c = pd.to_numeric(df["close"], errors="coerce")
    valid = x.notna() & o.notna() & h.notna() & l.notna() & c.notna()
    x = x[valid]
    o = o[valid]
    h = h[valid]
    l = l[valid]  # noqa: E741
    c = c[valid]
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4.5))
    xs = mdates.date2num(x.dt.to_pydatetime())
    width = max(0.2, min(0.8, 180.0 / max(len(xs), 1)))
    for xi, oi, hi, li, ci in zip(xs, o, h, l, c, strict=True):
        color = "#d62728" if ci >= oi else "#1f77b4"
        ax.vlines(xi, li, hi, color=color, linewidth=0.8)
        bottom = min(oi, ci)
        height = abs(ci - oi)
        if height == 0:
            ax.hlines(ci, xi - width / 2, xi + width / 2, color=color, linewidth=1.0)
        else:
            rect = plt.Rectangle((xi - width / 2, bottom), width, height, facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
    ax.set_title(title)
    ax.xaxis_date()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _plot_ratio(compare_df: pd.DataFrame, out_png: Path, title: str) -> None:
    if compare_df.empty:
        return
    x = pd.to_datetime(compare_df["date"])
    y = pd.to_numeric(compare_df["close_replay88"], errors="coerce") / pd.to_numeric(compare_df["close_main0"], errors="coerce").replace(0, np.nan)
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(x, y, color="#4c78a8", linewidth=1.1)
    ax.axhline(1.0, color="#999999", linewidth=0.9, linestyle="--")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _fmt_num(v: object, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "-"
    if pct:
        return f"{float(v) * 100:.4f}%"
    return f"{float(v):,.6f}"


def _write_html_report(
    cfg: ReplayConfig,
    compare_summary: dict[str, object],
    err_df: pd.DataFrame,
    usability: dict[str, object],
    image_map: dict[str, str],
    out_html: Path,
) -> None:
    err_rows = []
    if not err_df.empty:
        for r in err_df.itertuples(index=False):
            err_rows.append(
                "<tr>"
                f"<td>{r.field}</td>"
                f"<td>{int(r.n)}</td>"
                f"<td>{_fmt_num(r.mae)}</td>"
                f"<td>{_fmt_num(r.rmse)}</td>"
                f"<td>{_fmt_num(r.max_abs)}</td>"
                f"<td>{_fmt_num(r.mape, pct=True)}</td>"
                f"<td>{_fmt_num(r.p95_ape, pct=True)}</td>"
                "</tr>"
            )
    failed_lines = []
    for it in usability.get("failed_fields", []):
        failed_lines.append(
            f"{it.get('field')}: mape={_fmt_num(it.get('mape'), pct=True)}, p95={_fmt_num(it.get('p95_ape'), pct=True)}"
        )
    failed_text = "<br/>".join(failed_lines) if failed_lines else "-"
    usable_flag = "可用" if usability.get("usable") else "不可用"
    usable_color = "#1a7f37" if usability.get("usable") else "#d1242f"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{cfg.underlying} 连续合约回放对比报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    h1, h2 {{ margin: 12px 0; }}
    .muted {{ color: #666; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e5e5e5; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>{cfg.underlying} 连续合约回放对比报告</h1>
  <div class="muted">主连标的：{cfg.main_symbol}；区间：{cfg.start_date}~{cfg.end_date}；月合约范围：{cfg.contract_start}~{cfg.contract_end}；切换阈值：{cfg.switch_threshold}</div>
  <div class="card">
    <h2>对比摘要</h2>
    <table>
      <tr><th>指标</th><th>值</th></tr>
      <tr><td>overlap_days</td><td>{compare_summary.get("overlap_days", "-")}</td></tr>
      <tr><td>close_abs_diff_mean</td><td>{_fmt_num(compare_summary.get("close_abs_diff_mean"))}</td></tr>
      <tr><td>close_abs_diff_median</td><td>{_fmt_num(compare_summary.get("close_abs_diff_median"))}</td></tr>
      <tr><td>close_rel_diff_mean</td><td>{_fmt_num(compare_summary.get("close_rel_diff_mean"), pct=True)}</td></tr>
      <tr><td>close_rel_diff_median</td><td>{_fmt_num(compare_summary.get("close_rel_diff_median"), pct=True)}</td></tr>
      <tr><td>close_rel_diff_p95</td><td>{_fmt_num(compare_summary.get("close_rel_diff_p95"), pct=True)}</td></tr>
      <tr><td>nonzero_close_diff_days</td><td>{compare_summary.get("nonzero_close_diff_days", "-")}</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>可用性判定</h2>
    <table>
      <tr><th>判定结果</th><td style="color:{usable_color}; font-weight:700;">{usable_flag}</td></tr>
      <tr><th>原因</th><td>{usability.get("reason", "-")}</td></tr>
      <tr><th>规则</th><td>MAPE ≤ {_fmt_num(cfg.usable_rel_mean_max, pct=True)} 且 P95 APE ≤ {_fmt_num(cfg.usable_rel_p95_max, pct=True)}，覆盖字段数 ≥ {cfg.usable_min_fields}</td></tr>
      <tr><th>失败字段</th><td>{failed_text}</td></tr>
    </table>
  </div>
  <div class="card"><h2>1) 原始 *0 主连 K 线</h2><img src="{image_map.get('k0','')}" alt="k0"/></div>
  <div class="card"><h2>2) 合成 *88 简单主连 K 线</h2><img src="{image_map.get('k88','')}" alt="k88"/></div>
  <div class="card"><h2>3) 合成 *888 前复权主连 K 线</h2><img src="{image_map.get('k888','')}" alt="k888"/></div>
  <div class="card"><h2>4) 合成 *889 后复权主连 K 线</h2><img src="{image_map.get('k889','')}" alt="k889"/></div>
  <div class="card"><h2>5) *88 / *0 收盘价比值图</h2><img src="{image_map.get('ratio','')}" alt="ratio"/></div>
  <div class="card">
    <h2>6) *88 相对 *0 误差统计</h2>
    <table>
      <tr><th>字段</th><th>有效样本数</th><th>MAE</th><th>RMSE</th><th>MaxAbs</th><th>MAPE</th><th>P95 APE</th></tr>
      {''.join(err_rows)}
    </table>
  </div>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


def parse_args() -> ReplayConfig:
    p = argparse.ArgumentParser(description="Replay dominant continuous futures from monthly contracts and compare with main continuous.")
    p.add_argument("--underlying", default="LC", help="Underlying symbol root, e.g. LC")
    p.add_argument("--main-symbol", default=None, help="Main continuous symbol, default: <underlying>0")
    p.add_argument("--start-date", default="20230721", help="YYYYMMDD")
    p.add_argument("--end-date", default="20260417", help="YYYYMMDD")
    p.add_argument("--contract-start", default=None, help="YYMM; default derives from start-date")
    p.add_argument("--contract-end", default=None, help="YYMM; default derives from end-date + 12 months")
    p.add_argument("--switch-threshold", type=float, default=1.1, help="Switch if best_hold > threshold * current_hold")
    p.add_argument("--usable-rel-mean-max", type=float, default=0.005, help="Usability threshold: max mean relative error (MAPE)")
    p.add_argument("--usable-rel-p95-max", type=float, default=0.02, help="Usability threshold: max P95 absolute percentage error")
    p.add_argument("--usable-min-fields", type=int, default=4, help="Minimum covered key fields among open/high/low/close/settle")
    p.add_argument("--output-dir", default="data/futures_replay", help="Output directory")
    a = p.parse_args()
    main_symbol = a.main_symbol or f"{a.underlying}0"
    inferred_contract_start = _yymm_from_yyyymmdd(str(a.start_date))
    inferred_contract_end = _shift_yymm(_yymm_from_yyyymmdd(str(a.end_date)), months=12)
    contract_start = str(a.contract_start) if a.contract_start else inferred_contract_start
    contract_end = str(a.contract_end) if a.contract_end else inferred_contract_end
    return ReplayConfig(
        underlying=str(a.underlying).upper(),
        main_symbol=str(main_symbol).upper(),
        start_date=str(a.start_date),
        end_date=str(a.end_date),
        contract_start=contract_start,
        contract_end=contract_end,
        switch_threshold=float(a.switch_threshold),
        usable_rel_mean_max=float(a.usable_rel_mean_max),
        usable_rel_p95_max=float(a.usable_rel_p95_max),
        usable_min_fields=int(a.usable_min_fields),
        output_dir=Path(a.output_dir),
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    contract_data, coverage = _load_contract_universe(cfg)
    coverage_df = pd.DataFrame(coverage).sort_values(["ok", "symbol"], ascending=[False, True])
    coverage_df.to_csv(cfg.output_dir / f"{cfg.underlying}_contract_coverage.csv", index=False, encoding="utf-8-sig")

    replay88_df, switch_df = _replay_dominant(cfg, contract_data)
    if not replay88_df.empty:
        replay88_df.to_csv(cfg.output_dir / f"{cfg.underlying}_replay_88.csv", index=False, encoding="utf-8-sig")
    if not switch_df.empty:
        switch_df.to_csv(cfg.output_dir / f"{cfg.underlying}_switches.csv", index=False, encoding="utf-8-sig")

    panel = _build_quote_panel(contract_data)
    replay888_df, replay889_df = _build_adjusted_continuous(replay88_df, switch_df, panel)
    if not replay888_df.empty:
        replay888_df.to_csv(cfg.output_dir / f"{cfg.underlying}_replay_888.csv", index=False, encoding="utf-8-sig")
    if not replay889_df.empty:
        replay889_df.to_csv(cfg.output_dir / f"{cfg.underlying}_replay_889.csv", index=False, encoding="utf-8-sig")

    try:
        main_df = _fetch_symbol(cfg.main_symbol, cfg.start_date, cfg.end_date)
    except Exception as e:  # noqa: BLE001
        main_df = pd.DataFrame()
        print(f"[WARN] failed to fetch main symbol {cfg.main_symbol}: {e}")
    if not main_df.empty:
        main_df.to_csv(cfg.output_dir / f"{cfg.main_symbol}_raw.csv", index=False, encoding="utf-8-sig")

    compare_df, compare_summary, err_df = _compare_with_main(replay88_df, main_df)
    if not compare_df.empty:
        compare_df.to_csv(cfg.output_dir / f"{cfg.underlying}_compare_replay88_vs_{cfg.main_symbol}.csv", index=False, encoding="utf-8-sig")
    if not err_df.empty:
        err_df.to_csv(cfg.output_dir / f"{cfg.underlying}_error_stats_replay88_vs_{cfg.main_symbol}.csv", index=False, encoding="utf-8-sig")
    usability = _evaluate_usability(cfg, err_df, compare_summary)

    # Charts + HTML report
    k0_png = cfg.output_dir / f"{cfg.main_symbol}_kline.png"
    k88_png = cfg.output_dir / f"{cfg.underlying}_88_kline.png"
    k888_png = cfg.output_dir / f"{cfg.underlying}_888_kline.png"
    k889_png = cfg.output_dir / f"{cfg.underlying}_889_kline.png"
    ratio_png = cfg.output_dir / f"{cfg.underlying}_ratio_88_over_{cfg.main_symbol}.png"
    report_html = cfg.output_dir / f"{cfg.underlying}_replay_report.html"

    _plot_kline(main_df, f"{cfg.main_symbol} 原始主连 K线", k0_png)
    _plot_kline(replay88_df, f"{cfg.underlying}88 合成简单主连 K线", k88_png)
    _plot_kline(replay888_df, f"{cfg.underlying}888 合成前复权主连 K线", k888_png)
    _plot_kline(replay889_df, f"{cfg.underlying}889 合成后复权主连 K线", k889_png)
    _plot_ratio(compare_df, ratio_png, f"{cfg.underlying}88 / {cfg.main_symbol} 收盘价比值")
    _write_html_report(
        cfg=cfg,
        compare_summary=compare_summary,
        err_df=err_df,
        usability=usability,
        image_map={
            "k0": k0_png.name,
            "k88": k88_png.name,
            "k888": k888_png.name,
            "k889": k889_png.name,
            "ratio": ratio_png.name,
        },
        out_html=report_html,
    )

    summary = {
        "underlying": cfg.underlying,
        "main_symbol": cfg.main_symbol,
        "date_range": {"start": cfg.start_date, "end": cfg.end_date},
        "contract_range": {"start": cfg.contract_start, "end": cfg.contract_end},
        "switch_threshold": cfg.switch_threshold,
        "contracts_ok": int((coverage_df["ok"] == True).sum()) if not coverage_df.empty else 0,  # noqa: E712
        "contracts_bad": int((coverage_df["ok"] == False).sum()) if not coverage_df.empty else 0,  # noqa: E712
        "replay_rows": int(len(replay88_df)),
        "replay_888_rows": int(len(replay888_df)),
        "replay_889_rows": int(len(replay889_df)),
        "switch_count": int(len(switch_df)),
        "compare": compare_summary,
        "usability": usability,
        "outputs": {
            "coverage_csv": str(cfg.output_dir / f"{cfg.underlying}_contract_coverage.csv"),
            "replay_88_csv": str(cfg.output_dir / f"{cfg.underlying}_replay_88.csv"),
            "replay_888_csv": str(cfg.output_dir / f"{cfg.underlying}_replay_888.csv"),
            "replay_889_csv": str(cfg.output_dir / f"{cfg.underlying}_replay_889.csv"),
            "switches_csv": str(cfg.output_dir / f"{cfg.underlying}_switches.csv"),
            "main_raw_csv": str(cfg.output_dir / f"{cfg.main_symbol}_raw.csv"),
            "compare_csv": str(cfg.output_dir / f"{cfg.underlying}_compare_replay88_vs_{cfg.main_symbol}.csv"),
            "error_stats_csv": str(cfg.output_dir / f"{cfg.underlying}_error_stats_replay88_vs_{cfg.main_symbol}.csv"),
            "kline_main_png": str(k0_png),
            "kline_88_png": str(k88_png),
            "kline_888_png": str(k888_png),
            "kline_889_png": str(k889_png),
            "ratio_png": str(ratio_png),
            "report_html": str(report_html),
            "summary_json": str(cfg.output_dir / f"{cfg.underlying}_replay_summary.json"),
        },
    }
    summary_path = cfg.output_dir / f"{cfg.underlying}_replay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
