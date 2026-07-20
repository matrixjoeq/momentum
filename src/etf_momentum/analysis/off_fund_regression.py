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
        aliases=("000852", "512100"),
    ),
    RegressionFactorSpec(
        key="CSI2000",
        label="中证2000",
        aliases=("932000", "563300"),
    ),
    RegressionFactorSpec(
        key="CNI2000",
        label="国证2000",
        aliases=("399303", "159907"),
    ),
    RegressionFactorSpec(
        key="CYB",
        label="创业板指",
        aliases=("399006", "159915"),
    ),
    RegressionFactorSpec(
        key="KCP50",
        label="科创50",
        aliases=("000688", "588000"),
    ),
    RegressionFactorSpec(
        key="CSIFCF",
        label="中证全指自由现金流",
        aliases=("932365", "159232"),
    ),
    RegressionFactorSpec(
        key="CSIHL",
        label="中证红利",
        aliases=("000922", "515180"),
    ),
    RegressionFactorSpec(
        key="CSI_300_GROWTH_INNOVATION",
        label="300成长创新",
        aliases=("931589", "159523"),
    ),
    RegressionFactorSpec(
        key="CSI_300_VALUE_STABILITY",
        label="300价值稳健",
        aliases=("931586", "159510"),
    ),
    RegressionFactorSpec(
        key="CSI_1000_GROWTH_INNOVATION",
        label="1000成长创新",
        aliases=("931591", "562520"),
    ),
    RegressionFactorSpec(
        key="CSI_1000_VALUE_STABILITY",
        label="1000价值稳健",
        aliases=("931588", "562530"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_ENERGY",
        label="中证全指能源",
        aliases=("000986", "159945"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_MATERIAL",
        label="中证全指材料",
        aliases=("000987", "159944"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_FINANCE",
        label="中证全指金融",
        aliases=("000992", "159940"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_ELECTRICITY",
        label="中证全指电力公用事业",
        aliases=("h30199", "159611"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_CONSUMER",
        label="中证主要消费",
        aliases=("000932", "159928"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_CONSUMER_SELECTIVE",
        label="中证全指可选消费",
        aliases=("000989", "159936"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_MEDICINE",
        label="中证全指医药",
        aliases=("000991", "159938"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_INFORMATION",
        label="中证全指信息",
        aliases=("000993", "159939"),
    ),
    RegressionFactorSpec(
        key="CSI_ALL_COMMUNICATION",
        label="中证全指通信",
        aliases=("931160", "515880"),
    ),
    RegressionFactorSpec(
        key="HSI",
        label="恒生指数",
        aliases=("HSI", "159920"),
    ),
    RegressionFactorSpec(
        key="HSI_TECH",
        label="港股科技",
        aliases=("HSI_TECH", "513980"),
    ),
    RegressionFactorSpec(
        key="HSI_DIVIDEND",
        label="港股红利",
        aliases=("HSI_DIVIDEND", "513690"),
    ),
    RegressionFactorSpec(
        key="GOLD_SPOT",
        label="黄金现货",
        aliases=("XAU", "518880"),
    ),
    RegressionFactorSpec(
        key="SILVER_FUTURES",
        label="白银期货",
        aliases=("SI", "161226"),
    ),
    RegressionFactorSpec(
        key="METALS_FUTURES",
        label="有色金属期货",
        aliases=("HG", "159980"),
    ),
    RegressionFactorSpec(
        key="OIL_FUTURES",
        label="原油期货",
        aliases=("CL", "501018"),
    ),
    RegressionFactorSpec(
        key="ENERGY_CHEMICAL_FUTURES",
        label="能源化工期货",
        aliases=("TA", "159981"),
    ),
    RegressionFactorSpec(
        key="SOYBEAN_MEAL_FUTURES",
        label="豆粕期货",
        aliases=("M", "159985"),
    ),
    RegressionFactorSpec(
        key="SP500",
        label="标普500",
        aliases=("SPY", "513500"),
    ),
    RegressionFactorSpec(
        key="NASDAQ100",
        label="纳斯达克100",
        aliases=("QQQ", "513100"),
    ),
    RegressionFactorSpec(
        key="DOW_JONES_INDUSTRIAL_INDEX",
        label="道琼斯工业指数",
        aliases=("DJI", "513400"),
    ),
    RegressionFactorSpec(
        key="DAX",
        label="德国DAX指数",
        aliases=("DAX", "513030"),
    ),
    RegressionFactorSpec(
        key="CAC40",
        label="法国CAC40指数",
        aliases=("CAC40", "513080"),
    ),
    RegressionFactorSpec(
        key="NIKKEI225",
        label="日经225指数",
        aliases=("N225", "513520"),
    ),
    RegressionFactorSpec(
        key="TOPIX",
        label="日本东证指数",
        aliases=("TOPIX", "513800"),
    ),
    RegressionFactorSpec(
        key="0-3Y_GOV_BOND",
        label="0-3年期国债",
        aliases=("0-3Y_GOV_BOND", "511580"),
    ),
    RegressionFactorSpec(
        key="5Y_GOV_BOND",
        label="5年国债",
        aliases=("5Y_GOV_BOND", "511010"),
    ),
    RegressionFactorSpec(
        key="10Y_GOV_BOND",
        label="10年国债",
        aliases=("10Y_GOV_BOND", "511260"),
    ),
    RegressionFactorSpec(
        key="30Y_GOV_BOND",
        label="30年国债",
        aliases=("30Y_GOV_BOND", "511090"),
    ),
    RegressionFactorSpec(
        key="CITY_DEBT",
        label="城投债",
        aliases=("CITY_DEBT", "511220"),
    ),
    RegressionFactorSpec(
        key="MONEY_MARKET_FUND",
        label="货币基金",
        aliases=("MONEY_MARKET_FUND", "511880"),
    ),
)


def nav_to_returns(nav: pd.Series) -> pd.Series:
    s = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index, errors="coerce")
    valid = ~pd.isna(idx)
    if not bool(np.all(valid)):
        s = s.iloc[np.asarray(valid, dtype=bool)]
        idx = idx[valid]
    if s.empty:
        return pd.Series(dtype=float)
    dti = pd.DatetimeIndex(idx)
    if dti.tz is not None:
        dti = dti.tz_localize(None)
    s.index = pd.DatetimeIndex(dti.date)
    if not s.index.is_unique:
        # Keep the latest observation if multiple rows map to the same trading day.
        s = s.groupby(level=0).last()
    s = s.sort_index()
    return s.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()


def _required_days(min_samples: int, rolling_window: int | None = None) -> int:
    if rolling_window is None:
        return max(3, int(min_samples))
    return max(int(min_samples), int(max(20, int(rolling_window)) // 2))


def choose_factor_series(
    *,
    close_df: pd.DataFrame,
    factor_specs: list[RegressionFactorSpec],
    min_samples: int,
    rolling_window: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    required_days = _required_days(
        min_samples=min_samples, rolling_window=rolling_window
    )
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
            if int(s.shape[0]) < required_days:
                continue
            chosen_code = code
            chosen = s
            break
        if chosen is None:
            warnings.append(
                f"factor {spec.key} missing enough data "
                f"(required_days={required_days}, aliases={list(spec.aliases)})"
            )
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
    out_df = pd.DataFrame(out_cols).sort_index()
    if out_df.shape[1] < 2:
        return out_df, used, warnings
    fac_ret = out_df.apply(nav_to_returns)
    active_keys = [str(c) for c in fac_ret.columns]

    def _common_days(keys: list[str]) -> int:
        if not keys:
            return 0
        idx = fac_ret[keys[0]].index
        for key in keys[1:]:
            idx = idx.intersection(fac_ret[key].index)
        return int(idx.shape[0])

    current_common = _common_days(active_keys)
    dropped: list[str] = []
    while len(active_keys) >= 3 and current_common < required_days:
        best_drop: str | None = None
        best_common = current_common
        for key in active_keys:
            remain = [x for x in active_keys if x != key]
            if len(remain) < 2:
                continue
            days = _common_days(remain)
            if days > best_common:
                best_common = days
                best_drop = key
        if best_drop is None:
            break
        active_keys = [x for x in active_keys if x != best_drop]
        dropped.append(best_drop)
        current_common = best_common
    if dropped:
        warnings.append(
            "dropped factors to improve common sample window: "
            f"{dropped}, common_days={current_common}, required_days={required_days}"
        )
    active_set = set(active_keys)
    kept_cols = [c for c in out_df.columns if str(c) in active_set]
    out_df = out_df.loc[:, kept_cols]
    used = [x for x in used if str(x.get("key") or "") in active_set]
    return out_df, used, warnings


def inspect_factor_availability(
    *,
    close_df: pd.DataFrame,
    factor_specs: list[RegressionFactorSpec],
    min_samples: int,
    rolling_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_threshold = max(3, int(min_samples // 2))
    required_days = _required_days(
        min_samples=min_samples, rolling_window=rolling_window
    )
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
        status = (
            "ok" if enough else ("insufficient_data" if selected_code else "missing")
        )
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
    fund_nav_clean = (
        pd.to_numeric(fund_nav, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ret_fund = nav_to_returns(fund_nav_clean)
    fac_ret = factor_close_df.apply(nav_to_returns)
    common = ret_fund.index
    for c in fac_ret.columns:
        common = common.intersection(fac_ret[c].index)
    common = common.sort_values()
    factor_keys = [str(c) for c in fac_ret.columns]
    required_days = max(int(min_samples), int(max(20, int(rolling_window)) // 2))
    factor_price_days = {
        str(c): int(
            pd.to_numeric(factor_close_df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .sum()
        )
        for c in factor_close_df.columns
    }
    factor_ret_days = {str(c): int(fac_ret[c].shape[0]) for c in fac_ret.columns}
    common_start = str(common[0]) if common.shape[0] > 0 else None
    common_end = str(common[-1]) if common.shape[0] > 0 else None
    debug_summary = (
        "debug:"
        f" fund_nav_days={int(fund_nav_clean.shape[0])},"
        f" fund_ret_days={int(ret_fund.shape[0])},"
        f" factor_price_days={factor_price_days},"
        f" factor_ret_days={factor_ret_days},"
        f" common_days={int(common.shape[0])},"
        f" required_days={int(required_days)},"
        f" common_range=({common_start},{common_end})"
    )
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
            "warnings": ["valid factor count < 2", debug_summary],
            "series": [],
        }
    if common.shape[0] < required_days:
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
                f"sample days={int(common.shape[0])} < required days={int(required_days)}",
                debug_summary,
            ],
            "series": [],
        }
    y_all = pd.to_numeric(ret_fund.reindex(common), errors="coerce").to_numpy(
        dtype=float
    )
    x_all = (
        fac_ret.reindex(common)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    win = int(max(20, min(int(rolling_window), int(common.shape[0]))))
    min_obs = max(int(min_samples), x_all.shape[1] + 6)
    rows: list[dict[str, Any]] = []
    for i in range(win - 1, int(common.shape[0])):
        left_idx = i - win + 1
        y = y_all[left_idx : i + 1]
        x = x_all[left_idx : i + 1, :]
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
            "warnings": ["no valid rolling regression window", debug_summary],
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
                k: float(r[k]) if np.isfinite(float(r[k])) else 0.0
                for k in exposure_cols
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
