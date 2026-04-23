"""
Discrete-lot futures account simulator (RMB).

Uses settlement price for MTM and initial margin requirement; execution at Open or
Close per ``exec_price``. Sequential margin allocation follows sorted contract codes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _settle_price(row: pd.Series) -> float:
    if "Settle" in row.index:
        s = pd.to_numeric(row["Settle"], errors="coerce")
        if np.isfinite(float(s)) and float(s) > 0:
            return float(s)
    c = pd.to_numeric(row["Close"], errors="coerce")
    return float(c) if np.isfinite(float(c)) and float(c) > 0 else float("nan")


def _exec_price_row(row: pd.Series, exec_price: str) -> float:
    ep = str(exec_price or "close").strip().lower()
    if ep == "open":
        v = pd.to_numeric(row["Open"], errors="coerce")
    else:
        v = pd.to_numeric(row["Close"], errors="coerce")
    return float(v) if np.isfinite(float(v)) and float(v) > 0 else float("nan")


def _suffix_cell(row: pd.Series) -> str | None:
    if "dominant_contract_suffix" not in row.index:
        return None
    raw = row["dominant_contract_suffix"]
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    t = str(raw).strip()
    if t in {"", "nan", "None"}:
        return None
    return t


def _roll_exec_prices(row: pd.Series, exec_price: str) -> tuple[float, float]:
    """
    Return (old_contract_exit_px, new_contract_entry_px) on roll day.
    Falls back to current continuous execution price when explicit legs are unavailable.
    """
    ep = str(exec_price or "close").strip().lower()
    cpx = _exec_price_row(row, exec_price)
    if ep == "open":
        fcol = "roll_from_open"
        tcol = "roll_to_open"
    else:
        fcol = "roll_from_close"
        tcol = "roll_to_close"
    fpx = pd.to_numeric(row.get(fcol), errors="coerce")
    tpx = pd.to_numeric(row.get(tcol), errors="coerce")
    old_px = float(fpx) if np.isfinite(float(fpx)) and float(fpx) > 0 else float(cpx)
    new_px = float(tpx) if np.isfinite(float(tpx)) and float(tpx) > 0 else float(cpx)
    return old_px, new_px


def _margin_per_lot(settle_px: float, mult: float, margin_rate_frac: float) -> float:
    if not np.isfinite(settle_px) or settle_px <= 0:
        return float("nan")
    if not np.isfinite(mult) or mult <= 0:
        return float("nan")
    return float(settle_px) * float(mult) * float(margin_rate_frac)


def _fee_on_notional(notional: float, cost: object) -> float:
    r = float(max(0.0, getattr(cost, "commission_per_fill", 0.0))) + float(
        max(0.0, getattr(cost, "spread_per_fill", 0.0))
    )
    if not np.isfinite(notional) or notional <= 0:
        return 0.0
    return float(notional) * r


def _empty_trade_book() -> dict[str, Any]:
    return {
        "side": 0,
        "open_lots": 0,
        "avg_entry_px": float("nan"),
        "basis_notional_cny": 0.0,
        "realized_pnl_cny": 0.0,
        "entry_date": None,
    }


def _safe_ratio(num: float, den: float) -> float | None:
    if (not np.isfinite(float(num))) or (not np.isfinite(float(den))) or den == 0.0:
        return None
    return float(num / den)


def simulate_discrete_lot_portfolio(
    *,
    common_idx: pd.DatetimeIndex,
    exec_by_code: dict[str, pd.DataFrame],
    w_eff: pd.DataFrame,
    cost_by_symbol: dict[str, object],
    mults: dict[str, float],
    margin_rate_frac: float,
    reserve_ratio: float,
    initial_equity_cny: float,
    exec_price: str,
    position_sizing: str,
    codes_sorted: list[str],
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Daily loop: MTM → roll fees → sequential target lots from w_eff row → trade
    (with reverse same-|lots| rule) → record equity.

    ``w_eff`` must already include signal lag (e.g. shift(1)) matching the engine.
    """
    codes = [c for c in codes_sorted if c in exec_by_code and c in mults]
    if not codes:
        return pd.Series(dtype=float), {"error": "no_codes"}

    ps = str(position_sizing or "equal").strip().lower()
    res = float(reserve_ratio)
    if not np.isfinite(res) or res < 0.0 or res >= 1.0:
        res = 0.0
    mrate = float(margin_rate_frac)
    if not np.isfinite(mrate) or mrate <= 0.0:
        mrate = 0.15

    equity = float(initial_equity_cny)
    if not np.isfinite(equity) or equity <= 0.0:
        equity = 1.0

    lots: dict[str, int] = {c: 0 for c in codes}
    prev_settle: dict[str, float] = {c: float("nan") for c in codes}
    prev_suf: dict[str, str | None] = {c: None for c in codes}
    books: dict[str, dict[str, Any]] = {c: _empty_trade_book() for c in codes}

    eq_list: list[float] = []

    meta_roll_fees = 0.0
    meta_roll_events = 0
    meta_trade_fees = 0.0
    reserve_attempted_entries = 0
    reserve_blocked_entries = 0
    reserve_attempted_by_code: dict[str, int] = {c: 0 for c in codes}
    reserve_blocked_by_code: dict[str, int] = {c: 0 for c in codes}
    reserve_attempted_by_code_direction: dict[str, dict[str, int]] = {
        c: {"long": 0, "short": 0} for c in codes
    }
    reserve_blocked_by_code_direction: dict[str, dict[str, int]] = {
        c: {"long": 0, "short": 0} for c in codes
    }
    reserve_attempted_by_direction: dict[str, int] = {"long": 0, "short": 0}
    reserve_blocked_by_direction: dict[str, int] = {"long": 0, "short": 0}
    closed_trades: list[dict[str, Any]] = []

    def apply_lot_change(
        *,
        code: str,
        new_lots: int,
        px: float,
        mult: float,
        cost: object,
    ) -> float:
        """Apply fee drag; return fee subtotal for meta. Enforces reverse = two fills."""
        old = int(lots[code])
        tgt = int(new_lots)
        fee_sub = 0.0
        if old == tgt:
            return 0.0
        if old != 0 and tgt != 0 and (old > 0) != (tgt > 0):
            n1 = abs(old) * px * mult
            fee_sub += _fee_on_notional(n1, cost)
            n2 = abs(tgt) * px * mult
            fee_sub += _fee_on_notional(n2, cost)
            lots[code] = tgt
            return fee_sub
        delta = tgt - old
        n = abs(delta) * px * mult
        fee_sub = _fee_on_notional(n, cost)
        lots[code] = tgt
        return fee_sub

    def _close_trade(
        *,
        code: str,
        qty: int,
        px: float,
        dt: pd.Timestamp,
    ) -> None:
        if qty <= 0:
            return
        b = books[code]
        side = int(b.get("side", 0))
        if side == 0:
            return
        avg_px = float(b.get("avg_entry_px", float("nan")))
        if (not np.isfinite(avg_px)) or avg_px <= 0.0:
            return
        mult = float(mults[code])
        pnl = float(side) * float(qty) * mult * (float(px) - avg_px)
        b["realized_pnl_cny"] = float(b.get("realized_pnl_cny", 0.0)) + pnl
        b["open_lots"] = int(max(0, int(b.get("open_lots", 0)) - int(qty)))
        if int(b["open_lots"]) == 0:
            basis = float(max(0.0, float(b.get("basis_notional_cny", 0.0))))
            tr = _safe_ratio(float(b.get("realized_pnl_cny", 0.0)), basis)
            closed_trades.append(
                {
                    "code": str(code),
                    "direction": ("long" if side > 0 else "short"),
                    "entry_date": b.get("entry_date"),
                    "exit_date": pd.Timestamp(dt).date().isoformat(),
                    "open_lots": int(qty),
                    "return": tr,
                    "pnl_cny": float(b.get("realized_pnl_cny", 0.0)),
                    "basis_notional_cny": basis,
                }
            )
            books[code] = _empty_trade_book()

    def _open_trade(
        *,
        code: str,
        side: int,
        qty: int,
        px: float,
        dt: pd.Timestamp,
    ) -> None:
        if qty <= 0:
            return
        b = books[code]
        mult = float(mults[code])
        if int(b.get("side", 0)) == 0 or int(b.get("open_lots", 0)) == 0:
            b["side"] = int(side)
            b["open_lots"] = int(qty)
            b["avg_entry_px"] = float(px)
            b["basis_notional_cny"] = float(qty) * float(px) * mult
            b["realized_pnl_cny"] = 0.0
            b["entry_date"] = pd.Timestamp(dt).date().isoformat()
            return
        prev_lots = int(b.get("open_lots", 0))
        prev_avg = float(b.get("avg_entry_px", float("nan")))
        new_lots = prev_lots + int(qty)
        if new_lots <= 0:
            return
        if np.isfinite(prev_avg) and prev_avg > 0.0:
            b["avg_entry_px"] = float(
                (prev_avg * prev_lots + float(px) * int(qty)) / float(new_lots)
            )
        else:
            b["avg_entry_px"] = float(px)
        b["open_lots"] = int(new_lots)
        b["basis_notional_cny"] = (
            float(b.get("basis_notional_cny", 0.0)) + float(qty) * float(px) * mult
        )

    for i, d in enumerate(common_idx):
        settle_m: dict[str, float] = {}
        exec_m: dict[str, float] = {}
        suf_m: dict[str, str | None] = {}
        for c in codes:
            df = exec_by_code[str(c)]
            if d not in df.index:
                settle_m[c] = float("nan")
                exec_m[c] = float("nan")
                suf_m[c] = None
                continue
            row = df.loc[d]
            settle_m[c] = _settle_price(row)
            exec_m[c] = _exec_price_row(row, exec_price)
            suf_m[c] = _suffix_cell(row)

        # MTM
        if i > 0:
            for c in codes:
                lc = int(lots[c])
                if lc == 0:
                    continue
                ps0 = prev_settle[c]
                ps1 = settle_m[c]
                if (
                    np.isfinite(ps0)
                    and np.isfinite(ps1)
                    and np.isfinite(float(mults[c]))
                ):
                    equity += float(lc) * float(mults[c]) * (ps1 - ps0)

        # Roll round-turn fees (cash from unified pool); P&L already in stitched series via MTM
        if i > 0:
            for c in codes:
                lc = int(lots[c])
                if lc == 0:
                    continue
                s0 = prev_suf[c]
                s1 = suf_m[c]
                same = (s0 or "") == (s1 or "")
                if same:
                    continue
                row_today = (
                    exec_by_code[str(c)].loc[d]
                    if d in exec_by_code[str(c)].index
                    else None
                )
                if row_today is None:
                    continue
                old_px, new_px = _roll_exec_prices(row_today, exec_price)
                if (
                    (not np.isfinite(old_px))
                    or old_px <= 0
                    or (not np.isfinite(new_px))
                    or new_px <= 0
                ):
                    continue
                cost = cost_by_symbol[c]
                n_exit = abs(lc) * old_px * float(mults[c])
                n_entry = abs(lc) * new_px * float(mults[c])
                # Two fills (exit old + entry new) same |lots|
                rt = _fee_on_notional(n_exit, cost) + _fee_on_notional(n_entry, cost)
                equity -= rt
                meta_roll_fees += rt
                meta_roll_events += 1

        # Targets from w_eff
        try:
            wrow = w_eff.loc[d].reindex(codes).astype(float)
        except (KeyError, TypeError):
            wrow = pd.Series(0.0, index=codes, dtype=float)

        targets: dict[str, int] = {c: 0 for c in codes}

        # Explicit flat per signal → target 0 lots (full exit)
        for c in sorted(codes):
            wt = float(wrow.get(c, 0.0) or 0.0)
            if abs(wt) <= 1e-12 and int(lots[c]) != 0:
                targets[c] = 0

        active = sorted(
            [c for c in codes if abs(float(wrow.get(c, 0.0) or 0.0)) > 1e-12]
        )
        rev_set: set[str] = set()
        for c in sorted(codes):
            o = int(lots[c])
            wt = float(wrow.get(c, 0.0) or 0.0)
            if o != 0 and abs(wt) > 1e-12 and (o > 0) != (wt > 0):
                targets[c] = -o
                rev_set.add(c)

        m_max = max(0.0, equity * (1.0 - res))
        rem = float(m_max)

        alloc_list = sorted([c for c in active if c not in rev_set])
        n_act = len(alloc_list)
        for j, c in enumerate(alloc_list):
            wt = float(wrow.get(c, 0.0) or 0.0)
            side_name = "long" if wt > 0 else "short"
            is_new_entry_try = int(lots[c]) == 0
            if is_new_entry_try:
                reserve_attempted_entries += 1
                reserve_attempted_by_code[c] = int(
                    reserve_attempted_by_code.get(c, 0) + 1
                )
                reserve_attempted_by_code_direction[c][side_name] = int(
                    reserve_attempted_by_code_direction[c].get(side_name, 0) + 1
                )
                reserve_attempted_by_direction[side_name] = int(
                    reserve_attempted_by_direction.get(side_name, 0) + 1
                )
            mpl = _margin_per_lot(settle_m[c], float(mults[c]), mrate)
            if not np.isfinite(mpl) or mpl <= 0:
                if is_new_entry_try:
                    reserve_blocked_entries += 1
                    reserve_blocked_by_code[c] = int(
                        reserve_blocked_by_code.get(c, 0) + 1
                    )
                    reserve_blocked_by_code_direction[c][side_name] = int(
                        reserve_blocked_by_code_direction[c].get(side_name, 0) + 1
                    )
                    reserve_blocked_by_direction[side_name] = int(
                        reserve_blocked_by_direction.get(side_name, 0) + 1
                    )
                continue
            left = n_act - j
            if left <= 0:
                break
            if ps == "equal":
                budget = rem / float(left)
            else:
                wsum = sum(abs(float(wrow.get(a, 0.0) or 0.0)) for a in alloc_list[j:])
                wc = abs(float(wrow.get(c, 0.0) or 0.0))
                budget = rem * (wc / wsum) if wsum > 1e-18 else 0.0
            tl = int(np.floor(budget / mpl)) if budget > 0 else 0
            if tl <= 0:
                if is_new_entry_try:
                    reserve_blocked_entries += 1
                    reserve_blocked_by_code[c] = int(
                        reserve_blocked_by_code.get(c, 0) + 1
                    )
                    reserve_blocked_by_code_direction[c][side_name] = int(
                        reserve_blocked_by_code_direction[c].get(side_name, 0) + 1
                    )
                    reserve_blocked_by_direction[side_name] = int(
                        reserve_blocked_by_direction.get(side_name, 0) + 1
                    )
                continue
            sgn = 1 if float(wrow[c]) > 0 else -1
            targets[c] = int(sgn) * tl
            rem -= float(abs(targets[c])) * mpl

        # Execute in code order
        for c in sorted(codes):
            px = exec_m[c]
            if not np.isfinite(px) or px <= 0:
                continue
            old = int(lots[c])
            tgt = int(targets[c])
            fee = apply_lot_change(
                code=c,
                new_lots=tgt,
                px=float(px),
                mult=float(mults[c]),
                cost=cost_by_symbol[c],
            )
            equity -= fee
            meta_trade_fees += fee
            if old == tgt:
                continue
            if old != 0 and tgt != 0 and (old > 0) != (tgt > 0):
                _close_trade(code=c, qty=abs(old), px=float(px), dt=pd.Timestamp(d))
                _open_trade(
                    code=c,
                    side=(1 if tgt > 0 else -1),
                    qty=abs(tgt),
                    px=float(px),
                    dt=pd.Timestamp(d),
                )
            else:
                delta = int(tgt - old)
                if old == 0 and tgt != 0:
                    _open_trade(
                        code=c,
                        side=(1 if tgt > 0 else -1),
                        qty=abs(tgt),
                        px=float(px),
                        dt=pd.Timestamp(d),
                    )
                elif old != 0 and abs(tgt) < abs(old):
                    _close_trade(
                        code=c, qty=abs(delta), px=float(px), dt=pd.Timestamp(d)
                    )
                elif old != 0 and abs(tgt) > abs(old):
                    _open_trade(
                        code=c,
                        side=(1 if old > 0 else -1),
                        qty=abs(delta),
                        px=float(px),
                        dt=pd.Timestamp(d),
                    )
                elif old != 0 and tgt == 0:
                    _close_trade(code=c, qty=abs(old), px=float(px), dt=pd.Timestamp(d))

        eq_list.append(equity)
        for c in codes:
            prev_settle[c] = settle_m[c]
            prev_suf[c] = suf_m[c]

    eq_s = pd.Series(eq_list, index=common_idx, dtype=float)
    if len(common_idx) > 0:
        d_last = pd.Timestamp(common_idx[-1])
        for c in codes:
            b = books[c]
            ol = int(b.get("open_lots", 0))
            if ol <= 0:
                continue
            px_end = prev_settle.get(c, float("nan"))
            if (not np.isfinite(px_end)) or px_end <= 0.0:
                continue
            _close_trade(code=c, qty=ol, px=float(px_end), dt=d_last)
    meta: dict[str, Any] = {
        "initial_equity_cny": float(initial_equity_cny),
        "margin_rate_frac": float(mrate),
        "reserve_margin_ratio": float(res),
        "roll_fees_cny": float(meta_roll_fees),
        "roll_events": int(meta_roll_events),
        "trade_fees_cny": float(meta_trade_fees),
        "reserve_margin_attempted_entry_count": int(reserve_attempted_entries),
        "reserve_margin_blocked_entry_count": int(reserve_blocked_entries),
        "reserve_margin_attempted_entry_count_by_code": {
            str(k): int(v) for k, v in reserve_attempted_by_code.items()
        },
        "reserve_margin_blocked_entry_count_by_code": {
            str(k): int(v) for k, v in reserve_blocked_by_code.items()
        },
        "reserve_margin_attempted_entry_count_by_code_direction": {
            str(k): {"long": int(v.get("long", 0)), "short": int(v.get("short", 0))}
            for k, v in reserve_attempted_by_code_direction.items()
        },
        "reserve_margin_blocked_entry_count_by_code_direction": {
            str(k): {"long": int(v.get("long", 0)), "short": int(v.get("short", 0))}
            for k, v in reserve_blocked_by_code_direction.items()
        },
        "reserve_margin_attempted_entry_count_by_direction": {
            str(k): int(v) for k, v in reserve_attempted_by_direction.items()
        },
        "reserve_margin_blocked_entry_count_by_direction": {
            str(k): int(v) for k, v in reserve_blocked_by_direction.items()
        },
        "closed_trades": closed_trades,
        "engine": "lot_account",
    }
    return eq_s, meta
