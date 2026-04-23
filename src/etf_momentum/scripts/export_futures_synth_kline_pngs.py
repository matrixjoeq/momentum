#!/usr/bin/env python3
"""
Export full synthetic futures OHLC (88 / 888 / 889) as candlestick PNGs for QA.

Usage (from repo root, with DB configured in env / settings):
  python3 -m etf_momentum.scripts.export_futures_synth_kline_pngs --out var/futures_synth_kline_audit

Requires matplotlib (not in core pyproject; install if missing: python3 -m pip install matplotlib).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

from etf_momentum.data.futures_synthesize import _symbol_root_from_main
from etf_momentum.db.futures_repo import list_futures_pool, list_futures_prices
from etf_momentum.db.session import make_engine, make_session_factory
from etf_momentum.settings import get_settings


def _load_series(
    db,
    *,
    code: str,
    adjust: str,
    max_rows: int = 2_000_000,
):
    return list_futures_prices(
        db,
        code=code,
        adjust=adjust,
        start_date=None,
        end_date=None,
        limit=max_rows,
    )


def _plot_candles_png(
    *,
    dates: list[dt.date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    title: str,
    out_path: Path,
    max_draw_bars: int = 8000,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: python3 -m pip install matplotlib"
        ) from e

    n = len(dates)
    if n == 0:
        return
    stride = max(1, math.ceil(n / max_draw_bars))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    idx = sorted(set(idx))

    fig_w = min(48.0, max(12.0, len(idx) / 80.0))
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))
    x = list(range(len(idx)))
    for j, i in enumerate(idx):
        o, h, lo, c = opens[i], highs[i], lows[i], closes[i]
        col = "#e53935" if c >= o else "#26a69a"
        ax.plot([j, j], [lo, h], color=col, linewidth=0.6, solid_capstyle="round")
        bottom = min(o, c)
        height = abs(c - o) if abs(c - o) > 1e-12 else 1e-9 * (h - lo or 1.0)
        ax.add_patch(
            Rectangle(
                (j - 0.35, bottom),
                0.7,
                height,
                facecolor=col,
                edgecolor=col,
                linewidth=0.4,
            )
        )

    tick_step = max(1, len(idx) // 25)
    xt = x[::tick_step]
    xlabs = [dates[idx[j]].isoformat() for j in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=7)
    sub = f" (stride={stride}, plotted={len(idx)}/{n} bars)" if stride > 1 else ""
    ax.set_title(title + sub, fontsize=10)
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_close_line_png(
    *,
    dates: list[dt.date],
    closes: list[float],
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: python3 -m pip install matplotlib"
        ) from e

    if not dates:
        return
    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    xd = list(range(len(dates)))
    ax.plot(xd, closes, color="#1565c0", linewidth=0.8)
    tick_step = max(1, len(dates) // 30)
    ax.set_xticks(xd[::tick_step])
    ax.set_xticklabels(
        [dates[i].isoformat() for i in xd[::tick_step]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_title(title + " (close, full series)", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("var/futures_synth_kline_audit"),
        help="Output directory",
    )
    p.add_argument(
        "--skip-close-line",
        action="store_true",
        help="Do not write companion close-only line charts",
    )
    args = p.parse_args()
    out_root: Path = args.out.resolve()

    settings = get_settings()
    eng = make_engine(db_url=settings.db_url)
    sf = make_session_factory(eng)
    db = sf()

    pool = list_futures_pool(db)
    db.close()

    written = 0
    skipped = 0
    summary: list[str] = []

    synth_specs = (
        ("88", "none", "synth88_none"),
        ("888", "qfq", "synth888_qfq"),
        ("889", "hfq", "synth889_hfq"),
    )

    for item in pool:
        pool_code = str(item.code).strip().upper()
        root = _symbol_root_from_main(pool_code)
        subdir = out_root / pool_code.replace("/", "_")
        for suffix, adjust, label in synth_specs:
            code = f"{root}{suffix}"
            db = sf()
            rows = _load_series(db, code=code, adjust=adjust)
            db.close()
            if not rows:
                skipped += 1
                summary.append(f"{pool_code}\t{code}\t{adjust}\t0 rows")
                continue

            dates: list[dt.date] = []
            o_: list[float] = []
            h_: list[float] = []
            l_: list[float] = []
            c_: list[float] = []
            for r in rows:
                if r.open is None or r.high is None or r.low is None or r.close is None:
                    continue
                dates.append(r.trade_date)
                o_.append(float(r.open))
                h_.append(float(r.high))
                l_.append(float(r.low))
                c_.append(float(r.close))

            if len(dates) < 1:
                skipped += 1
                summary.append(f"{pool_code}\t{code}\t{adjust}\tno valid OHLC")
                continue

            t0, t1 = dates[0].isoformat(), dates[-1].isoformat()
            title = f"{pool_code} → {code} ({adjust})  {t0} … {t1}  n={len(dates)}"
            png = subdir / f"{label}_{code}.png"
            _plot_candles_png(
                dates=dates,
                opens=o_,
                highs=h_,
                lows=l_,
                closes=c_,
                title=title,
                out_path=png,
            )
            written += 1
            summary.append(f"{pool_code}\t{code}\t{adjust}\t{len(dates)}\t{png}")

            if not args.skip_close_line:
                clp = subdir / f"{label}_{code}_close_line.png"
                _plot_close_line_png(
                    dates=dates,
                    closes=c_,
                    title=f"{pool_code} → {code} ({adjust})",
                    out_path=clp,
                )
                written += 1

    sum_path = out_root / "index.tsv"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(
        "pool_code\tseries_code\tadjust\tbars\tpath_or_note\n"
        + "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Done. pool={len(pool)} written_png_steps={written} skipped_empty={skipped} out={out_root}"
    )
    print(f"Index: {sum_path}")


if __name__ == "__main__":
    main()
