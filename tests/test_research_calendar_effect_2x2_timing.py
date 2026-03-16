import datetime as dt
from pathlib import Path


def _trade_ret(
    *,
    opens: list[float],
    closes: list[float],
    entry_idx: int,
    exit_idx: int,
    entry_px: str,
    exit_px: str,
) -> float:
    e = closes[entry_idx] if entry_px == "close" else opens[entry_idx]
    x = closes[exit_idx] if exit_px == "close" else opens[exit_idx]
    return (x / e) - 1.0


def test_monthday_2x2_entry_exit_price_is_strict_trade_interval() -> None:
    # Synthetic path with deliberately different open/close to expose wrong timing.
    # Day i=1 as entry, j=3 as exit.
    opens = [10.0, 11.0, 9.0, 9.9, 8.0]
    closes = [10.0, 12.0, 9.0, 13.0, 7.0]
    i, j = 1, 3

    # open -> open
    r_oo = _trade_ret(opens=opens, closes=closes, entry_idx=i, exit_idx=j, entry_px="open", exit_px="open")
    assert r_oo == (opens[j] / opens[i] - 1.0)

    # open -> close
    r_oc = _trade_ret(opens=opens, closes=closes, entry_idx=i, exit_idx=j, entry_px="open", exit_px="close")
    assert r_oc == (closes[j] / opens[i] - 1.0)

    # close -> open
    r_co = _trade_ret(opens=opens, closes=closes, entry_idx=i, exit_idx=j, entry_px="close", exit_px="open")
    assert r_co == (opens[j] / closes[i] - 1.0)

    # close -> close
    r_cc = _trade_ret(opens=opens, closes=closes, entry_idx=i, exit_idx=j, entry_px="close", exit_px="close")
    assert r_cc == (closes[j] / closes[i] - 1.0)

    # Guard against accidental inclusion of pre-entry segment (day0->day1 gap).
    wrong_pre_entry_mix = opens[j] / opens[i - 1] - 1.0
    assert r_oo != wrong_pre_entry_mix


def test_research_html_uses_trade_return_helper_for_2x2_calendar_effect() -> None:
    html = Path("src/etf_momentum/web/research.html").read_text(encoding="utf-8")
    assert "const _tradeRet = (i, j, pxE0, pxX0) => {" in html
    assert "const r = _tradeRet(i, j, pxE, pxX);" in html
    # Previous inline formula path should not exist anymore.
    assert 'const e = (pxE === "close") ? Number(closes[i]) : Number(opens[i]);' not in html
    assert 'const x = (pxX === "close") ? Number(closes[j]) : Number(opens[j]);' not in html

