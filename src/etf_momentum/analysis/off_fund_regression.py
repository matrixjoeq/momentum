from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass(frozen=True)
class RegressionFactorSpec:
    key: str
    label: str
    aliases: tuple[str, ...]


DEFAULT_CN_STOCK_FACTORS: tuple[RegressionFactorSpec, ...] = (
    RegressionFactorSpec(
        key="CSI300",
        label="沪深300",
        aliases=("000300", "510300"),
    ),
    RegressionFactorSpec(
        key="CSI500",
        label="中证500",
        aliases=("000905", "510500"),
    ),
    RegressionFactorSpec(
        key="CSI1000",
        label="中证1000",
        aliases=("000852", "159845"),
    ),
    RegressionFactorSpec(
        key="CSI2000",
        label="中证2000",
        aliases=("932000", "159533"),
    ),
)


def nav_to_returns(nav: pd.Series) -> pd.Series:
    s = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.sort_index()
    return s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def choose_factor_series(
    *,
    close_df: pd.DataFrame,
    factor_specs: list[RegressionFactorSpec],
    min_samples: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    out_cols: dict[str, pd.Series] = {}
    used: list[dict[str, Any]] = []
    warnings: list[str] = []
    for spec in factor_specs:
        chosen_code: str | None = None
        chosen: pd.Series | None = None
        for code in spec.aliases:
            if code not in close_df.columns:
                continue
            s = (
                pd.to_numeric(close_df[code], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if int(s.shape[0]) < max(3, min_samples // 2):
                continue
            chosen_code = code
            chosen = s
            break
        if chosen is None:
            warnings.append(f"factor {spec.key} missing data in aliases={list(spec.aliases)}")
            continue
        out_cols[spec.key] = chosen
        used.append(
            {
                "key": spec.key,
                "label": spec.label,
                "selected_code": chosen_code,
                "aliases": list(spec.aliases),
            }
        )
    if not out_cols:
        return pd.DataFrame(dtype=float), used, warnings
    out = pd.DataFrame(out_cols).sort_index()
    return out, used, warnings


def inspect_factor_availability(
    *,
    close_df: pd.DataFrame,
    factor_specs: list[RegressionFactorSpec],
    min_samples: int,
    rolling_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_threshold = max(3, int(min_samples // 2))
    required_days = max(int(min_samples), int(max(20, rolling_window) // 2))
    items: list[dict[str, Any]] = []
    for spec in factor_specs:
        alias_samples: dict[str, int] = {}
        selected_code: str | None = None
        selected_samples = 0
        for code in spec.aliases:
            if code not in close_df.columns:
                alias_samples[code] = 0
                continue
            s = pd.to_numeric(close_df[code], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            n = int(s.notna().sum())
            alias_samples[code] = n
            if selected_code is None and n >= use_threshold:
                selected_code = code
                selected_samples = n
        enough = selected_code is not None and int(selected_samples) >= required_days
        status = "ok" if enough else ("insufficient_data" if selected_code else "missing")
        items.append(
            {
                "key": spec.key,
                "label": spec.label,
                "aliases": list(spec.aliases),
                "selected_code": selected_code,
                "sample_days": int(selected_samples),
                "required_days": int(required_days),
                "enough": bool(enough),
                "status": status,
                "alias_samples": alias_samples,
            }
        )
    summary = {
        "required_days": int(required_days),
        "factor_count": int(len(items)),
        "enough_factor_count": int(sum(1 for x in items if bool(x.get("enough")))),
    }
    return items, summary


def _fit_nnls_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if x.ndim != 2 or y.ndim != 1:
        return None
    if x.shape[0] <= x.shape[1] + 2:
        return None
    w, _ = nnls(x, y)
    if not np.all(np.isfinite(w)):
        return None
    s = float(w.sum())
    if s > 1.0 and s > 1e-12:
        w = w / s
    return w


def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    if y.shape != yhat.shape or y.size == 0:
        return float("nan")
    sst = float(np.square(y - y.mean()).sum())
    if sst <= 1e-12:
        return float("nan")
    sse = float(np.square(y - yhat).sum())
    return float(1.0 - sse / sst)


def _label_from_exposure(
    *,
    avg_exposure: pd.Series,
    avg_r2: float,
    dominance_gap: float,
) -> tuple[str, str, str]:
    if avg_exposure.empty or not np.isfinite(avg_r2) or avg_r2 < 0.20:
        return "未分类（股票解释度低）", "LOW", "unclassified"
    ranked = avg_exposure.sort_values(ascending=False)
    top_key = str(ranked.index[0])
    top_v = float(ranked.iloc[0])
    second_v = float(ranked.iloc[1]) if ranked.shape[0] >= 2 else 0.0
    if top_v < 0.35 or (top_v - second_v) < float(max(0.0, dominance_gap)):
        return "A股均衡风格", "MEDIUM", "equity"
    mapping = {
        "CSI300": "A股大盘风格",
        "CSI500": "A股中盘风格",
        "CSI1000": "A股中小盘风格",
        "CSI2000": "A股小微盘风格",
    }
    label = mapping.get(top_key, f"A股{top_key}主导风格")
    conf = "HIGH" if avg_r2 >= 0.50 and top_v >= 0.45 else "MEDIUM"
    return label, conf, "equity"


def classify_fund_by_regression(
    *,
    fund_nav: pd.Series,
    factor_close_df: pd.DataFrame,
    rolling_window: int,
    min_samples: int,
    dominance_gap: float,
    include_series: bool,
    max_series_points: int,
) -> dict[str, Any]:
    ret_fund = nav_to_returns(fund_nav)
    fac_ret = factor_close_df.apply(nav_to_returns)
    common = ret_fund.index
    for c in fac_ret.columns:
        common = common.intersection(fac_ret[c].index)
    common = common.sort_values()
    factor_keys = [str(c) for c in fac_ret.columns]
    if len(factor_keys) < 2:
        return {
            "status": "insufficient_factors",
            "sample_days": 0,
            "avg_r2": None,
            "latest_r2": None,
            "label": "未分类（基准不足）",
            "confidence": "LOW",
            "primary_asset_class": "unclassified",
            "avg_exposures": {},
            "latest_exposures": {},
            "warnings": ["valid factor count < 2"],
            "series": [],
        }
    if common.shape[0] < max(int(min_samples), int(rolling_window // 2)):
        return {
            "status": "insufficient_samples",
            "sample_days": int(common.shape[0]),
            "avg_r2": None,
            "latest_r2": None,
            "label": "未分类（样本不足）",
            "confidence": "LOW",
            "primary_asset_class": "unclassified",
            "avg_exposures": {},
            "latest_exposures": {},
            "warnings": [
                f"sample days={int(common.shape[0])} < min required={int(min_samples)}"
            ],
            "series": [],
        }
    y_all = pd.to_numeric(ret_fund.reindex(common), errors="coerce").to_numpy(dtype=float)
    x_all = (
        fac_ret.reindex(common)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    win = int(max(20, min(int(rolling_window), int(common.shape[0]))))
    min_obs = max(int(min_samples), x_all.shape[1] + 6)
    rows: list[dict[str, Any]] = []
    for i in range(win - 1, int(common.shape[0])):
        l = i - win + 1
        y = y_all[l : i + 1]
        x = x_all[l : i + 1, :]
        valid = np.isfinite(y)
        valid &= np.all(np.isfinite(x), axis=1)
        if int(valid.sum()) < min_obs:
            continue
        y2 = y[valid]
        x2 = x[valid, :]
        w = _fit_nnls_weights(x2, y2)
        if w is None:
            continue
        yhat = x2 @ w
        rec: dict[str, Any] = {
            "trade_date": pd.Timestamp(common[i]).date().isoformat(),
            "r2": _r2_score(y2, yhat),
        }
        for j, key in enumerate(factor_keys):
            rec[key] = float(w[j])
        rec["cash"] = float(max(0.0, 1.0 - float(w.sum())))
        rows.append(rec)
    if not rows:
        return {
            "status": "regression_failed",
            "sample_days": int(common.shape[0]),
            "avg_r2": None,
            "latest_r2": None,
            "label": "未分类（回归失败）",
            "confidence": "LOW",
            "primary_asset_class": "unclassified",
            "avg_exposures": {},
            "latest_exposures": {},
            "warnings": ["no valid rolling regression window"],
            "series": [],
        }
    df = pd.DataFrame(rows)
    exposure_cols = factor_keys + ["cash"]
    avg_exposure = df[exposure_cols].mean(numeric_only=True)
    latest = df.iloc[-1]
    avg_r2 = float(df["r2"].mean(skipna=True))
    latest_r2 = float(latest.get("r2", np.nan))
    label, confidence, primary_asset_class = _label_from_exposure(
        avg_exposure=avg_exposure[factor_keys],
        avg_r2=avg_r2,
        dominance_gap=dominance_gap,
    )
    out_series: list[dict[str, Any]] = []
    if include_series:
        if max_series_points > 0 and df.shape[0] > max_series_points:
            df_out = df.iloc[-max_series_points:].copy()
        else:
            df_out = df
        for _, r in df_out.iterrows():
            exp = {
                k: float(r[k]) if np.isfinite(float(r[k])) else 0.0 for k in exposure_cols
            }
            out_series.append(
                {
                    "trade_date": str(r["trade_date"]),
                    "r2": float(r["r2"]) if np.isfinite(float(r["r2"])) else None,
                    "exposures": exp,
                }
            )
    return {
        "status": "ok",
        "sample_days": int(common.shape[0]),
        "effective_windows": int(df.shape[0]),
        "avg_r2": avg_r2 if np.isfinite(avg_r2) else None,
        "latest_r2": latest_r2 if np.isfinite(latest_r2) else None,
        "label": label,
        "confidence": confidence,
        "primary_asset_class": primary_asset_class,
        "avg_exposures": {
            k: float(avg_exposure[k]) if np.isfinite(float(avg_exposure[k])) else 0.0
            for k in exposure_cols
        },
        "latest_exposures": {
            k: float(latest[k]) if np.isfinite(float(latest[k])) else 0.0
            for k in exposure_cols
        },
        "warnings": [],
        "series": out_series,
    }
