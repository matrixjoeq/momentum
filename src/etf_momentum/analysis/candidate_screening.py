from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .baseline import load_close_prices, load_volume_amount
from ..db.models import EtfPool


@dataclass(frozen=True)
class RotationCandidateScreenInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    adjust: str = "hfq"
    lookback_days: int = 252
    top_n: int = 12
    min_n: int = 4
    max_pair_corr: float = 0.75
    factor_weights: dict[str, float] | None = None
    category_quotas: dict[str, int] | None = None
    signif_horizon_days: int = 20


def _max_drawdown_from_nav(nav: pd.Series) -> float:
    s = (
        pd.to_numeric(nav, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if s.empty:
        return float("nan")
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _rank01(s: pd.Series, *, ascending: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    r = x.rank(method="average", ascending=ascending, pct=True)
    return r.astype(float).fillna(r.mean() if np.isfinite(float(r.mean())) else 0.5)


def _pval_norm_2sided_from_t(t: float) -> float:
    if not np.isfinite(float(t)):
        return float("nan")
    z = abs(float(t))
    return float(math.erfc(z / math.sqrt(2.0)))


def _infer_category(code: str, name: str) -> str:
    txt = f"{str(code or '').upper()}|{str(name or '').upper()}"
    if (
        ("国债ETF" in txt)
        or ("30年国债" in txt)
        or ("债" in txt and "可转债" not in txt)
    ):
        return "BOND"
    if "可转债" in txt:
        return "CONVERTIBLE_BOND"
    if ("黄金" in txt) or ("GOLD" in txt):
        return "GOLD"
    if ("豆粕" in txt) or ("有色" in txt) or ("能源化工" in txt) or ("商品" in txt):
        return "COMMODITY"
    if ("恒生" in txt) or ("港股" in txt):
        return "HK_EQ"
    if (
        ("纳指" in txt)
        or ("标普" in txt)
        or ("道琼斯" in txt)
        or ("US" in txt and "ETF" in txt)
    ):
        return "US_EQ"
    if ("日经" in txt) or ("东证" in txt) or ("日本" in txt):
        return "JP_EQ"
    if ("德国" in txt) or ("法国" in txt) or ("欧洲" in txt) or ("EURO" in txt):
        return "EU_EQ"
    if (
        ("东南亚" in txt)
        or ("亚太" in txt)
        or ("新兴亚洲" in txt)
        or ("沙特" in txt)
        or ("巴西" in txt)
    ):
        return "EM_EQ"
    return "CN_EQ"


def _normalize_factor_weights(raw: dict[str, float] | None) -> dict[str, float]:
    keys = ["mom_63", "mom_126", "sharpe", "win_rate", "liquidity", "mdd"]
    base = {
        "mom_63": 0.25,
        "mom_126": 0.25,
        "sharpe": 0.20,
        "win_rate": 0.10,
        "liquidity": 0.10,
        "mdd": 0.10,
    }
    if not raw:
        return base
    out = {}
    s = 0.0
    for k in keys:
        v = float(raw.get(k, 0.0))
        if np.isfinite(v) and v > 0:
            out[k] = v
            s += v
        else:
            out[k] = 0.0
    if s <= 1e-12:
        return base
    for k in keys:
        out[k] = out[k] / s
    return out


def _is_corr_ok(
    c: str, selected: list[str], corr: pd.DataFrame, max_corr: float
) -> tuple[bool, float, str]:
    if not selected:
        return True, 0.0, ""
    max_abs = -1.0
    max_with = ""
    for s in selected:
        v = float(corr.loc[c, s]) if (c in corr.index and s in corr.columns) else 0.0
        av = abs(v)
        if av > max_abs:
            max_abs = av
            max_with = s
    return (max_abs <= max_corr), max_abs, max_with


def screen_rotation_candidates(
    db: Session, inp: RotationCandidateScreenInputs
) -> dict[str, Any]:
    codes = list(
        dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()])
    )
    if len(codes) < 2:
        raise ValueError("codes must have at least 2 assets")
    lb = int(inp.lookback_days)
    if lb < 20:
        raise ValueError("lookback_days must be >= 20")
    top_n = int(inp.top_n)
    min_n = int(inp.min_n)
    if top_n < 2:
        raise ValueError("top_n must be >= 2")
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if min_n > top_n:
        raise ValueError("min_n must be <= top_n")
    max_corr = float(inp.max_pair_corr)
    if (not np.isfinite(max_corr)) or max_corr < 0.0 or max_corr >= 1.0:
        raise ValueError("max_pair_corr must be in [0,1)")
    horizon = int(inp.signif_horizon_days)
    if horizon < 5:
        raise ValueError("signif_horizon_days must be >= 5")
    factor_w = _normalize_factor_weights(inp.factor_weights)
    quotas = {
        str(k).strip().upper(): int(v)
        for k, v in (inp.category_quotas or {}).items()
        if str(k).strip()
    }
    for k, v in quotas.items():
        if v < 0:
            raise ValueError(f"category quota must be >=0: {k}")

    close = load_close_prices(
        db, codes=codes, start=inp.start, end=inp.end, adjust=str(inp.adjust or "hfq")
    )
    if close.empty:
        raise ValueError("no price data in selected range")
    close = close.sort_index().ffill().replace([np.inf, -np.inf], np.nan)
    close = close[codes]
    # Research correlation uses log returns across modules.
    ret = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    if len(ret) > lb:
        ret = ret.iloc[-lb:]
    ret = ret.dropna(how="all")
    if ret.empty:
        raise ValueError("insufficient returns after cleaning")
    obs_count = ret.count(axis=0).reindex(codes).fillna(0).astype(int)

    px = close.reindex(ret.index).ffill()
    mom_63 = (px / px.shift(63) - 1.0).iloc[-1]
    mom_126 = (px / px.shift(126) - 1.0).iloc[-1]
    mean_ret = ret.mean()
    vol_ret = ret.std(ddof=1)
    sharpe = (mean_ret / vol_ret).replace([np.inf, -np.inf], np.nan) * np.sqrt(252.0)
    win_rate = (ret > 0).sum(axis=0) / ret.count(axis=0).replace(0, np.nan)

    mdd = {}
    for c in codes:
        rs = pd.to_numeric(ret[c], errors="coerce").astype(float).fillna(0.0)
        nav = (1.0 + rs).cumprod()
        mdd[c] = _max_drawdown_from_nav(nav)
    mdd_s = pd.Series(mdd, dtype=float)

    vol, amount = load_volume_amount(
        db, codes=codes, start=inp.start, end=inp.end, adjust=str(inp.adjust or "hfq")
    )
    liq = pd.Series(index=codes, dtype=float)
    if amount is not None and not amount.empty:
        amt = amount.reindex(px.index).ffill()
        liq = amt.tail(lb).median(axis=0)
    elif vol is not None and not vol.empty:
        vv = vol.reindex(px.index).ffill()
        liq = vv.tail(lb).median(axis=0)
    liq = (
        pd.to_numeric(liq, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    # Name/category metadata.
    name_by_code: dict[str, str] = {c: "" for c in codes}
    rows_pool = (
        db.query(EtfPool.code, EtfPool.name).filter(EtfPool.code.in_(codes)).all()
    )
    for c0, n0 in rows_pool:
        c = str(c0 or "").strip()
        if c:
            name_by_code[c] = str(n0 or "").strip()
    category_by_code = {c: _infer_category(c, name_by_code.get(c, "")) for c in codes}

    # Composite score (higher is better).
    # Weighted by user controls; default emphasizes trend persistence and risk-adjusted return.
    score_parts = pd.DataFrame(
        {
            "mom_63_rank": _rank01(mom_63, ascending=True),
            "mom_126_rank": _rank01(mom_126, ascending=True),
            "sharpe_rank": _rank01(sharpe, ascending=True),
            "win_rate_rank": _rank01(win_rate, ascending=True),
            "liq_rank": _rank01(np.log1p(liq.clip(lower=0.0)), ascending=True),
            "mdd_rank": _rank01(
                mdd_s, ascending=False
            ),  # less negative drawdown is better
        }
    ).reindex(codes)
    score_raw = (
        float(factor_w["mom_63"]) * score_parts["mom_63_rank"]
        + float(factor_w["mom_126"]) * score_parts["mom_126_rank"]
        + float(factor_w["sharpe"]) * score_parts["sharpe_rank"]
        + float(factor_w["win_rate"]) * score_parts["win_rate_rank"]
        + float(factor_w["liquidity"]) * score_parts["liq_rank"]
        + float(factor_w["mdd"]) * score_parts["mdd_rank"]
    )
    score_raw = pd.to_numeric(score_raw, errors="coerce").astype(float).fillna(0.0)

    # Advanced momentum significance: does momentum state predict near-future return?
    mom_sig = (px / px.shift(63) - 1.0).astype(float)
    fwd_ret = (
        np.log(px.shift(-horizon) / px).replace([np.inf, -np.inf], np.nan).astype(float)
    )
    signif_by_code: dict[str, dict[str, float | int | None | bool]] = {}
    for c in codes:
        m = pd.to_numeric(mom_sig[c], errors="coerce").astype(float)
        y = pd.to_numeric(fwd_ret[c], errors="coerce").astype(float)
        good = m.notna() & y.notna()
        m = m[good]
        y = y[good]
        if len(m) < 30:
            signif_by_code[c] = {
                "n": int(len(m)),
                "mom_positive_ratio": None,
                "mean_fwd_when_mom_pos": None,
                "mean_fwd_when_mom_nonpos": None,
                "effect_spread": None,
                "t_stat": None,
                "p_value": None,
                "ic": None,
                "significant_5pct": False,
            }
            continue
        pos = m > 0
        y_pos = y[pos]
        y_non = y[~pos]
        n1 = int(y_pos.count())
        n0 = int(y_non.count())
        m1 = float(y_pos.mean()) if n1 > 0 else float("nan")
        m0 = float(y_non.mean()) if n0 > 0 else float("nan")
        v1 = float(y_pos.var(ddof=1)) if n1 > 1 else float("nan")
        v0 = float(y_non.var(ddof=1)) if n0 > 1 else float("nan")
        spread = (m1 - m0) if (np.isfinite(m1) and np.isfinite(m0)) else float("nan")
        denom = (
            math.sqrt(max(1e-18, (v1 / max(1, n1)) + (v0 / max(1, n0))))
            if (np.isfinite(v1) and np.isfinite(v0))
            else float("nan")
        )
        t_stat = (
            (spread / denom)
            if (np.isfinite(spread) and np.isfinite(denom) and denom > 0)
            else float("nan")
        )
        pval = _pval_norm_2sided_from_t(t_stat)
        ic = float(m.corr(y)) if len(m) > 2 else float("nan")
        signif_by_code[c] = {
            "n": int(len(m)),
            "mom_positive_ratio": float(pos.mean()) if len(pos) else None,
            "mean_fwd_when_mom_pos": (None if not np.isfinite(m1) else float(m1)),
            "mean_fwd_when_mom_nonpos": (None if not np.isfinite(m0) else float(m0)),
            "effect_spread": (None if not np.isfinite(spread) else float(spread)),
            "t_stat": (None if not np.isfinite(t_stat) else float(t_stat)),
            "p_value": (None if not np.isfinite(pval) else float(pval)),
            "ic": (None if not np.isfinite(ic) else float(ic)),
            "significant_5pct": bool(
                np.isfinite(pval) and pval < 0.05 and np.isfinite(spread) and spread > 0
            ),
        }

    corr = ret[codes].corr().fillna(0.0).astype(float)

    min_obs = int(max(30, horizon + 10))
    eligible = {c: bool(int(obs_count.get(c, 0)) >= min_obs) for c in codes}
    eligible_codes = [c for c in codes if eligible.get(c, False)]
    if not eligible_codes:
        raise ValueError(
            f"insufficient samples: no assets have >= {min_obs} valid daily-return observations"
        )

    ordered = sorted(
        eligible_codes, key=lambda c: float(score_raw.get(c, 0.0)), reverse=True
    )
    selected: list[str] = []
    blocked_by: dict[str, str] = {}
    max_corr_to_sel: dict[str, float | None] = {}
    selected_set: set[str] = set()
    non_selected_reason: dict[str, str] = {
        c: "insufficient_samples" for c in codes if not eligible.get(c, False)
    }

    top_n_eff = min(int(top_n), len(eligible_codes))
    min_n_eff = min(int(min_n), len(eligible_codes))

    # Step-1: satisfy category quotas first (if provided), with low-corr constraint.
    if quotas:
        by_cat: dict[str, list[str]] = {}
        for c in ordered:
            cat = str(category_by_code.get(c, "OTHER")).upper()
            by_cat.setdefault(cat, []).append(c)
        for cat, q in sorted(quotas.items()):
            if q <= 0:
                continue
            for c in by_cat.get(cat, []):
                if len(selected) >= top_n_eff or q <= 0 or c in selected_set:
                    break
                ok, mx, with_code = _is_corr_ok(c, selected, corr, max_corr)
                max_corr_to_sel[c] = mx if np.isfinite(mx) else None
                if ok:
                    selected.append(c)
                    selected_set.add(c)
                    q -= 1
                else:
                    blocked_by[c] = with_code
                    non_selected_reason[c] = "blocked_by_correlation"
            # quota fallback: if still unmet, fill by score ignoring corr
            if q > 0:
                for c in by_cat.get(cat, []):
                    if len(selected) >= top_n_eff or q <= 0 or c in selected_set:
                        break
                    selected.append(c)
                    selected_set.add(c)
                    q -= 1
                    non_selected_reason.pop(c, None)

    # Step-2: global fill by score under corr constraint.
    for c in ordered:
        if len(selected) >= top_n_eff:
            break
        if c in selected_set:
            continue
        ok, mx, with_code = _is_corr_ok(c, selected, corr, max_corr)
        max_corr_to_sel[c] = mx if np.isfinite(mx) else None
        if ok:
            selected.append(c)
            selected_set.add(c)
            non_selected_reason.pop(c, None)
        else:
            blocked_by[c] = with_code
            non_selected_reason[c] = "blocked_by_correlation"

    # Ensure at least min_n names: fill from top scores if low-corr filter too strict.
    if len(selected) < min_n_eff:
        for c in ordered:
            if c in selected_set:
                continue
            selected.append(c)
            selected_set.add(c)
            non_selected_reason.pop(c, None)
            if len(selected) >= min_n_eff:
                break

    rows = []
    sel_set = set(selected)
    for c in ordered:
        reason = non_selected_reason.get(c)
        if (c not in sel_set) and (reason is None):
            reason = "not_in_topn_after_constraints"
        rows.append(
            {
                "code": c,
                "name": str(name_by_code.get(c, "")),
                "category": str(category_by_code.get(c, "OTHER")),
                "eligible": bool(eligible.get(c, False)),
                "obs_count": int(obs_count.get(c, 0)),
                "min_obs_required": int(min_obs),
                "selected": c in sel_set,
                "score": float(score_raw.get(c, 0.0)),
                "mom_63": (None if pd.isna(mom_63.get(c)) else float(mom_63.get(c))),
                "mom_126": (None if pd.isna(mom_126.get(c)) else float(mom_126.get(c))),
                "sharpe_like": (
                    None if pd.isna(sharpe.get(c)) else float(sharpe.get(c))
                ),
                "win_rate": (
                    None if pd.isna(win_rate.get(c)) else float(win_rate.get(c))
                ),
                "max_drawdown": (
                    None if pd.isna(mdd_s.get(c)) else float(mdd_s.get(c))
                ),
                "liquidity_proxy": (None if pd.isna(liq.get(c)) else float(liq.get(c))),
                "max_abs_corr_to_selected": (
                    None
                    if max_corr_to_sel.get(c) is None
                    else float(max_corr_to_sel[c])
                ),
                "blocked_by_code": blocked_by.get(c),
                "not_selected_reason": (None if c in sel_set else reason),
                "significance": signif_by_code.get(c, {}),
            }
        )

    for c in codes:
        if c in eligible_codes:
            continue
        rows.append(
            {
                "code": c,
                "name": str(name_by_code.get(c, "")),
                "category": str(category_by_code.get(c, "OTHER")),
                "eligible": False,
                "obs_count": int(obs_count.get(c, 0)),
                "min_obs_required": int(min_obs),
                "selected": False,
                "score": float(score_raw.get(c, 0.0)),
                "mom_63": (None if pd.isna(mom_63.get(c)) else float(mom_63.get(c))),
                "mom_126": (None if pd.isna(mom_126.get(c)) else float(mom_126.get(c))),
                "sharpe_like": (
                    None if pd.isna(sharpe.get(c)) else float(sharpe.get(c))
                ),
                "win_rate": (
                    None if pd.isna(win_rate.get(c)) else float(win_rate.get(c))
                ),
                "max_drawdown": (
                    None if pd.isna(mdd_s.get(c)) else float(mdd_s.get(c))
                ),
                "liquidity_proxy": (None if pd.isna(liq.get(c)) else float(liq.get(c))),
                "max_abs_corr_to_selected": None,
                "blocked_by_code": None,
                "not_selected_reason": "insufficient_samples",
                "significance": signif_by_code.get(c, {}),
            }
        )

    signif_rows = []
    for c in codes:
        s = signif_by_code.get(c, {})
        reason = non_selected_reason.get(c)
        if (c not in sel_set) and (reason is None):
            reason = "not_in_topn_after_constraints"
        signif_rows.append(
            {
                "code": c,
                "name": str(name_by_code.get(c, "")),
                "category": str(category_by_code.get(c, "OTHER")),
                "eligible": bool(eligible.get(c, False)),
                "obs_count": int(obs_count.get(c, 0)),
                "selected": c in sel_set,
                "n": int(s.get("n") or 0),
                "effect_spread": s.get("effect_spread"),
                "t_stat": s.get("t_stat"),
                "p_value": s.get("p_value"),
                "ic": s.get("ic"),
                "significant_5pct": bool(s.get("significant_5pct")),
                "not_selected_reason": (None if c in sel_set else reason),
            }
        )

    return {
        "meta": {
            "type": "rotation_candidate_screen",
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "adjust": str(inp.adjust or "hfq"),
            "lookback_days": int(lb),
            "top_n": int(top_n),
            "min_n": int(min_n),
            "max_pair_corr": float(max_corr),
            "factor_weights": {k: float(v) for k, v in factor_w.items()},
            "category_quotas": {k: int(v) for k, v in quotas.items()},
            "signif_horizon_days": int(horizon),
            "min_obs_required": int(min_obs),
            "input_count": int(len(codes)),
            "selected_count": int(len(selected)),
            "selected_by_category": {
                cat: int(
                    sum(
                        1
                        for c in selected
                        if str(category_by_code.get(c, "OTHER")) == cat
                    )
                )
                for cat in sorted(set(category_by_code.values()))
            },
        },
        "selected_codes": selected,
        "details": rows,
        "significance_report": {
            "summary": {
                "significant_count_5pct": int(
                    sum(1 for r in signif_rows if bool(r["significant_5pct"]))
                ),
                "selected_significant_count_5pct": int(
                    sum(
                        1
                        for r in signif_rows
                        if bool(r["selected"]) and bool(r["significant_5pct"])
                    )
                ),
            },
            "rows": signif_rows,
        },
    }
