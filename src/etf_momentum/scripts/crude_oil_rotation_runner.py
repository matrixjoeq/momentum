#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crude Oil Rotation Strategy Backtest Runner

原油相关ETF轮动策略的批量回测脚本，支持多因子策略、参数敏感性分析和策略筛选。
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_momentum.settings import get_settings
from etf_momentum.db.session import make_engine, make_session_factory, session_scope
from etf_momentum.db.init_db import init_db
from etf_momentum.db.repo import upsert_prices, get_price_date_range
from etf_momentum.data.multi_source_fetcher import FetchRequest, fetch_etf_daily_with_fallback
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation
from etf_momentum.strategy.crude_oil_rotation_config import (
    CRUDE_OIL_CODES,
    CrudeOilRotationConfig,
    FactorParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crude_oil_backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/crude_oil")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    """单次回测结果"""
    strategy_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "metrics": self.metrics,
        }


CRUDE_OIL_CODES = [
    "501018",  # 南方原油
    "160723",  # 嘉实原油
    "161129",  # 易方达原油
    "160416",  # 华安石油
    "162719",  # 广发石油
    "163208",  # 诺安油气
    "162411",  # 华宝油气
]


def fetch_crud_oil_data(db, codes: List[str], start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """获取原油ETF的日度行情数据"""
    from sqlalchemy import select
    from etf_momentum.db.models import EtfPrice
    
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.close)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == "qfq")
        .where(EtfPrice.trade_date >= start_date)
        .where(EtfPrice.trade_date <= end_date)
        .order_by(EtfPrice.trade_date.asc())
    )
    
    rows = db.execute(stmt).all()
    
    if not rows:
        logger.warning("No price data found for codes=%s", codes)
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["date", "code", "close"])
    df["date"] = pd.to_datetime(df["date"])
    
    pivot = df.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    pivot = pivot.sort_index()
    
    logger.info("Loaded price data: %d rows, %d codes, date range: %s to %s",
                len(pivot), len(pivot.columns),
                pivot.index.min().date(), pivot.index.max().date())
    
    return pivot


def get_common_date_range(db, codes: List[str]) -> Tuple[dt.date, dt.date]:
    """获取所有ETF的公共数据区间"""
    starts = []
    ends = []
    
    for code in codes:
        start, end = get_price_date_range(db, code=code, adjust="qfq")
        if start:
            starts.append(dt.datetime.strptime(start, "%Y%m%d").date())
        if end:
            ends.append(dt.datetime.strptime(end, "%Y%m%d").date())
    
    if not starts or not ends:
        return dt.date(2017, 1, 1), dt.date.today()
    
    common_start = max(starts)
    common_end = min(ends)
    
    logger.info("Common date range: %s to %s", common_start, common_end)
    
    return common_start, common_end


def calculate_metrics(nav: pd.Series, rf_rate: float = 0.025) -> Dict[str, float]:
    """计算绩效指标"""
    if len(nav) < 2:
        return {}
    
    daily_ret = nav.pct_change().dropna()
    
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(nav) - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / n_days) - 1
    
    ann_vol = daily_ret.std() * np.sqrt(252)
    
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()
    
    excess_ret = ann_ret - rf_rate
    sharpe = excess_ret / ann_vol if ann_vol > 0 else 0
    
    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = excess_ret / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    
    positive_days = (daily_ret > 0).sum()
    win_rate = positive_days / len(daily_ret) if len(daily_ret) > 0 else 0
    
    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "n_days": n_days,
    }


def compute_raw_momentum(close: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """计算原始动量因子"""
    return close.pct_change(periods=lookback)


def compute_sharpe_momentum(close: pd.DataFrame, mom_lookback: int = 90, vol_lookback: int = 20) -> pd.DataFrame:
    """计算夏普动量因子"""
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    sharpe = cum_ret / vol.replace(0, np.nan)
    return sharpe


def compute_sortino_momentum(close: pd.DataFrame, mom_lookback: int = 90, vol_lookback: int = 20) -> pd.DataFrame:
    """计算Sortino动量因子"""
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    downside = ret.where(ret < 0, 0)
    downside_vol = downside.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    sortino = cum_ret / downside_vol.replace(0, np.nan)
    return sortino


def compute_return_over_vol(close: pd.DataFrame, mom_lookback: int = 90, vol_lookback: int = 20) -> pd.DataFrame:
    """计算收益/波动率因子"""
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    rov = cum_ret / vol.replace(0, np.nan)
    return rov


def compute_sma_trend(close: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    """计算均线趋势因子"""
    sma = close.rolling(window=period, min_periods=period).mean()
    trend = close / sma - 1
    return trend


def compute_low_vol_factor(close: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    """计算低波动因子"""
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    low_vol = 1 / vol.replace(0, np.nan)
    return low_vol


def compute_multi_factor_score(
    close: pd.DataFrame,
    mom_weight: float = 0.4,
    trend_weight: float = 0.3,
    vol_weight: float = 0.3,
    mom_lookback: int = 90,
    vol_lookback: int = 20,
    sma_period: int = 50,
) -> pd.DataFrame:
    """计算多因子综合得分"""
    mom_score = compute_sharpe_momentum(close, mom_lookback, vol_lookback)
    trend_score = compute_sma_trend(close, sma_period)
    vol_score = compute_low_vol_factor(close, vol_lookback)
    
    def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)
    
    mom_norm = rank_normalize(mom_score)
    trend_norm = rank_normalize(trend_score)
    vol_norm = rank_normalize(vol_score)
    
    multi_factor = mom_weight * mom_norm + trend_weight * trend_norm + vol_weight * vol_norm
    return multi_factor


def compute_factor_score(close: pd.DataFrame, score_method: str, lookback: int, vol_lookback: int = 20, sma_period: int = 50) -> pd.DataFrame:
    """计算因子得分"""
    if score_method == "raw_mom":
        return compute_raw_momentum(close, lookback)
    elif score_method == "sharpe_mom":
        return compute_sharpe_momentum(close, lookback, vol_lookback)
    elif score_method == "sortino_mom":
        return compute_sortino_momentum(close, lookback, vol_lookback)
    elif score_method == "return_over_vol":
        return compute_return_over_vol(close, lookback, vol_lookback)
    elif score_method == "sma_trend":
        return compute_sma_trend(close, sma_period)
    elif score_method == "low_vol":
        return compute_low_vol_factor(close, vol_lookback)
    elif score_method == "multi_factor":
        return compute_multi_factor_score(close, mom_lookback=lookback, vol_lookback=vol_lookback, sma_period=sma_period)
    else:
        raise ValueError(f"Unknown score_method: {score_method}")


def should_be_cash(close: pd.DataFrame, sma_period: int = 50, rsi_period: int = 14, rsi_lower: float = 30.0) -> pd.Series:
    """判断是否应该空仓"""
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    above_sma = (close > sma).any(axis=1)
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_above_oversold = (rsi > rsi_lower).any(axis=1)
    
    should_cash = ~(above_sma & rsi_above_oversold)
    return should_cash


def apply_trend_filter(ranks: pd.DataFrame, close: pd.DataFrame, sma_period: int = 50) -> pd.DataFrame:
    """趋势过滤：只选择价格高于均线的标的"""
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    above_sma = close > sma
    filtered = ranks.where(above_sma)
    return filtered


def execute_rotation(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    top_k: int,
    rebalance: str,
    enable_trend_filter: bool = False,
    sma_period: int = 50,
    cost_bps: float = 3.0,
) -> pd.Series:
    """执行轮动策略"""
    ranks = scores.rank(axis=1, ascending=True)
    
    if enable_trend_filter:
        ranks = apply_trend_filter(ranks, close, sma_period)
    
    cash_mask = should_be_cash(close, sma_period=sma_period)
    
    if rebalance == "daily":
        rebalance_dates = close.index
    elif rebalance == "weekly":
        rebalance_dates = close.resample("W-FRI").first().index
    elif rebalance == "monthly":
        rebalance_dates = close.resample("ME").first().index
    elif rebalance == "quarterly":
        rebalance_dates = close.resample("QE").first().index
    else:
        rebalance_dates = close.index
    
    rebalance_set = set(rebalance_dates)
    
    portfolio_weights = []
    prev_weights = None
    
    for i, date in enumerate(close.index):
        if date not in ranks.index:
            weights = prev_weights if prev_weights is not None else pd.Series(0.0, index=close.columns)
            portfolio_weights.append(weights)
            continue
        
        if date in rebalance_set or i == 0:
            day_ranks = ranks.loc[date]
            
            if cash_mask.get(date, False):
                weights = pd.Series(0.0, index=close.columns)
            else:
                valid_ranks = day_ranks.dropna().sort_values(ascending=False)
                top_codes = valid_ranks.head(top_k).index.tolist()
                weights = pd.Series(0.0, index=close.columns)
                if top_codes:
                    weight = 1.0 / len(top_codes)
                    for code in top_codes:
                        weights.loc[code] = weight
        else:
            weights = prev_weights if prev_weights is not None else pd.Series(0.0, index=close.columns)
        
        portfolio_weights.append(weights)
        prev_weights = weights
    
    portfolio = pd.DataFrame(portfolio_weights, index=close.index)
    
    daily_ret = close.pct_change().fillna(0)
    
    prev_portfolio = portfolio.shift(1).fillna(0)
    weight_change = (portfolio - prev_portfolio).abs().sum(axis=1)
    
    portfolio_ret = (portfolio * daily_ret).sum(axis=1)
    portfolio_ret = portfolio_ret - weight_change * (cost_bps / 10000) / 2
    
    nav = (1 + portfolio_ret).cumprod()
    nav.iloc[0] = 1.0
    
    return nav


def backtest_strategy(
    close: pd.DataFrame,
    score_method: str,
    lookback_days: int,
    top_k: int,
    rebalance: str,
    enable_trend_filter: bool = False,
    sma_period: int = 50,
    mom_weight: float = 0.4,
    trend_weight: float = 0.3,
    vol_weight: float = 0.3,
    cost_bps: float = 3.0,
) -> BacktestResult:
    """回测单个策略"""
    vol_lookback = 20
    
    if score_method == "multi_factor":
        scores = compute_multi_factor_score(
            close,
            mom_weight=mom_weight,
            trend_weight=trend_weight,
            vol_weight=vol_weight,
            mom_lookback=lookback_days,
            vol_lookback=vol_lookback,
            sma_period=sma_period,
        )
        strategy_name = f"multi_{lookback_days}d_top{top_k}_{rebalance}_w{int(mom_weight*100)}{int(trend_weight*100)}{int(vol_weight*100)}"
    else:
        scores = compute_factor_score(close, score_method, lookback_days, vol_lookback, sma_period)
        strategy_name = f"{score_method}_{lookback_days}d_top{top_k}_{rebalance}"
    
    if enable_trend_filter:
        strategy_name += "_trend"
    
    nav = execute_rotation(
        close,
        scores,
        top_k=top_k,
        rebalance=rebalance,
        enable_trend_filter=enable_trend_filter,
        sma_period=sma_period,
        cost_bps=cost_bps,
    )
    
    metrics = calculate_metrics(nav)
    
    params = {
        "score_method": score_method,
        "lookback_days": lookback_days,
        "top_k": top_k,
        "rebalance": rebalance,
        "enable_trend_filter": enable_trend_filter,
        "sma_period": sma_period,
        "mom_weight": mom_weight,
        "trend_weight": trend_weight,
        "vol_weight": vol_weight,
        "cost_bps": cost_bps,
    }
    
    return BacktestResult(strategy_name, params, metrics)


def run_single_factor_grid_search(close: pd.DataFrame) -> List[BacktestResult]:
    """单因子策略网格搜索"""
    results = []
    
    score_methods = ["raw_mom", "sharpe_mom", "sortino_mom", "return_over_vol"]
    lookback_days_list = [60, 90, 120, 180]
    top_k_list = [1, 2, 3]
    rebalance_list = ["weekly", "monthly"]
    
    total = len(score_methods) * len(lookback_days_list) * len(top_k_list) * len(rebalance_list)
    count = 0
    
    for score_method in score_methods:
        for lookback in lookback_days_list:
            for top_k in top_k_list:
                for rebalance in rebalance_list:
                    count += 1
                    logger.info(f"[{count}/{total}] Testing: {score_method}_{lookback}d_top{top_k}_{rebalance}")
                    
                    result = backtest_strategy(
                        close,
                        score_method=score_method,
                        lookback_days=lookback,
                        top_k=top_k,
                        rebalance=rebalance,
                        cost_bps=3.0,
                    )
                    results.append(result)
    
    return results


def run_trend_filter_tests(close: pd.DataFrame) -> List[BacktestResult]:
    """趋势过滤测试"""
    results = []
    
    base_configs = [
        ("sharpe_mom", 90, 2, "weekly"),
        ("sharpe_mom", 120, 2, "weekly"),
        ("raw_mom", 90, 2, "weekly"),
        ("sortino_mom", 90, 2, "weekly"),
    ]
    
    sma_periods = [50, 100, 200]
    
    for score_method, lookback, top_k, rebalance in base_configs:
        for sma_period in sma_periods:
            logger.info(f"Testing trend filter: {score_method}_{lookback}d_sma{sma_period}")
            
            result = backtest_strategy(
                close,
                score_method=score_method,
                lookback_days=lookback,
                top_k=top_k,
                rebalance=rebalance,
                enable_trend_filter=True,
                sma_period=sma_period,
                cost_bps=3.0,
            )
            results.append(result)
    
    return results


def run_multi_factor_grid_search(close: pd.DataFrame) -> List[BacktestResult]:
    """多因子策略网格搜索"""
    results = []
    
    weight_combinations = [
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.4, 0.3, 0.3),
        (0.6, 0.2, 0.2),
        (0.33, 0.33, 0.34),
    ]
    
    lookback_days_list = [60, 90, 120]
    top_k_list = [1, 2]
    rebalance_list = ["weekly", "monthly"]
    
    total = len(weight_combinations) * len(lookback_days_list) * len(top_k_list) * len(rebalance_list)
    count = 0
    
    for mom_w, trend_w, vol_w in weight_combinations:
        for lookback in lookback_days_list:
            for top_k in top_k_list:
                for rebalance in rebalance_list:
                    count += 1
                    logger.info(f"[{count}/{total}] Multi-factor: w=({mom_w},{trend_w},{vol_w})_{lookback}d_top{top_k}_{rebalance}")
                    
                    result = backtest_strategy(
                        close,
                        score_method="multi_factor",
                        lookback_days=lookback,
                        top_k=top_k,
                        rebalance=rebalance,
                        enable_trend_filter=True,
                        sma_period=50,
                        mom_weight=mom_w,
                        trend_weight=trend_w,
                        vol_weight=vol_w,
                        cost_bps=3.0,
                    )
                    results.append(result)
    
    return results


def analyze_results(results: List[BacktestResult]) -> Dict[str, Any]:
    """分析回测结果"""
    if not results:
        return {"error": "No results to analyze"}
    
    df = pd.DataFrame([r.to_dict() for r in results])
    
    df_sorted = df.sort_values("metrics.sharpe_ratio", ascending=False)
    
    target = df[df["metrics.sharpe_ratio"].astype(float) >= 1.0]
    good = df[df["metrics.sharpe_ratio"].astype(float) >= 1.3]
    excellent = df[df["metrics.annualized_return"].astype(float) >= 0.28]
    low_drawdown = df[df["metrics.max_drawdown"].astype(float) >= -0.15]
    
    best_sharpe = df_sorted.iloc[0] if len(df_sorted) > 0 else None
    best_return = df.sort_values("metrics.annualized_return", ascending=False).iloc[0] if len(df) > 0 else None
    best_calmar = df.sort_values("metrics.calmar_ratio", ascending=False).iloc[0] if len(df) > 0 else None
    
    sensitivity = {}
    
    if "params.lookback_days" in df.columns:
        lookback_analysis = df.groupby("params.lookback_days")["metrics.sharpe_ratio"].mean()
        sensitivity["lookback_days"] = lookback_analysis.to_dict()
    
    if "params.top_k" in df.columns:
        topk_analysis = df.groupby("params.top_k")["metrics.sharpe_ratio"].mean()
        sensitivity["top_k"] = topk_analysis.to_dict()
    
    if "params.rebalance" in df.columns:
        reb_analysis = df.groupby("params.rebalance")["metrics.sharpe_ratio"].mean()
        sensitivity["rebalance"] = reb_analysis.to_dict()
    
    return {
        "total_strategies": len(df),
        "sharpe_ge_1.0": len(target),
        "sharpe_ge_1.3": len(good),
        "return_ge_28%": len(excellent),
        "maxdd_le_15%": len(low_drawdown),
        "best_by_sharpe": best_sharpe.to_dict() if best_sharpe is not None else None,
        "best_by_return": best_return.to_dict() if best_return is not None else None,
        "best_by_calmar": best_calmar.to_dict() if best_calmar is not None else None,
        "sensitivity": sensitivity,
        "top_strategies": df_sorted.head(20).to_dict(orient="records"),
    }


def save_results(results: List[BacktestResult], filename: str):
    """保存回测结果到CSV"""
    if not results:
        return
    
    df = pd.DataFrame([r.to_dict() for r in results])
    
    metrics_df = pd.json_normalize(df["metrics"])
    metrics_df.columns = [f"metrics.{c}" for c in metrics_df.columns]
    
    result_df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)
    
    filepath = OUTPUT_DIR / filename
    result_df.to_csv(filepath, index=False, encoding="utf-8")
    
    logger.info(f"Saved results to {filepath}")


def generate_report(analysis: Dict[str, Any], output_path: Path):
    """生成研究报告"""
    report = f"""# 原油轮动策略研究报告

## 执行摘要

本报告对7只原油相关ETF进行了全面的轮动策略研究，测试了多种单因子和多因子策略配置。

### 关键发现

- 测试策略总数: {analysis.get('total_strategies', 0)}
- 夏普比率 ≥ 1.0 的策略: {analysis.get('sharpe_ge_1.0', 0)}
- 夏普比率 ≥ 1.3 的策略: {analysis.get('sharpe_ge_1.3', 0)}
- 年化收益率 ≥ 28% 的策略: {analysis.get('return_ge_28%', 0)}
- 最大回撤 ≤ 15% 的策略: {analysis.get('maxdd_le_15%', 0)}

## 最佳策略

### 按夏普比率
```
策略名称: {analysis.get('best_by_sharpe', {}).get('strategy_name', 'N/A')}
夏普比率: {analysis.get('best_by_sharpe', {}).get('metrics.sharpe_ratio', 0):.3f}
年化收益率: {analysis.get('best_by_sharpe', {}).get('metrics.annualized_return', 0)*100:.1f}%
最大回撤: {analysis.get('best_by_sharpe', {}).get('metrics.max_drawdown', 0)*100:.1f}%
```

### 按年化收益率
```
策略名称: {analysis.get('best_by_return', {}).get('strategy_name', 'N/A')}
夏普比率: {analysis.get('best_by_return', {}).get('metrics.sharpe_ratio', 0):.3f}
年化收益率: {analysis.get('best_by_return', {}).get('metrics.annualized_return', 0)*100:.1f}%
最大回撤: {analysis.get('best_by_return', {}).get('metrics.max_drawdown', 0)*100:.1f}%
```

### 按卡尔马比率
```
策略名称: {analysis.get('best_by_calmar', {}).get('strategy_name', 'N/A')}
夏普比率: {analysis.get('best_by_calmar', {}).get('metrics.sharpe_ratio', 0):.3f}
年化收益率: {analysis.get('best_by_calmar', {}).get('metrics.annualized_return', 0)*100:.1f}%
最大回撤: {analysis.get('best_by_calmar', {}).get('metrics.max_drawdown', 0)*100:.1f}%
卡尔马比率: {analysis.get('best_by_calmar', {}).get('metrics.calmar_ratio', 0):.2f}
```

## 参数敏感性分析

### 回看天数
"""
    
    lookback_sens = analysis.get('sensitivity', {}).get('lookback_days', {})
    for k, v in lookback_sens.items():
        report += f"- {k}天: 夏普 {v:.3f}\n"
    
    report += """
### 持仓数量
"""
    topk_sens = analysis.get('sensitivity', {}).get('top_k', {})
    for k, v in topk_sens.items():
        report += f"- top_{k}: 夏普 {v:.3f}\n"
    
    report += """
### 再平衡频率
"""
    reb_sens = analysis.get('sensitivity', {}).get('rebalance', {})
    for k, v in reb_sens.items():
        report += f"- {k}: 夏普 {v:.3f}\n"
    
    report += """
## Top 20 策略绩效表

| 策略 | 夏普 | 年化收益 | 最大回撤 | 卡尔马 |
|------|------|----------|----------|--------|
"""
    
    for s in analysis.get("top_strategies", [])[:20]:
        m = s.get("metrics", {})
        report += f"| {s.get('strategy_name', 'N/A')} | {m.get('sharpe_ratio', 0):.3f} | {m.get('annualized_return', 0)*100:.1f}% | {m.get('max_drawdown', 0)*100:.1f}% | {m.get('calmar_ratio', 0):.2f} |\n"
    
    report += """
## 结论与建议

基于回测结果分析：

"""
    
    best = analysis.get("best_by_sharpe")
    if best:
        report += f"最优策略为 **{best.get('strategy_name', 'N/A')}**，夏普比率 {best.get('metrics.sharpe_ratio', 0):.3f}，"
        report += f"年化收益 {best.get('metrics.annualized_return', 0)*100:.1f}%，最大回撤 {best.get('metrics.max_drawdown', 0)*100:.1f}%。\n"
    
    report += f"""
---
报告生成时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    filepath = output_path / "crude_oil_rotation_report.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Generated report: {filepath}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Crude Oil Rotation Strategy Backtest")
    logger.info("=" * 60)
    
    settings = get_settings()
    engine = make_engine(db_url=settings.db_url)
    init_db(engine)
    SessionFactory = make_session_factory(engine)
    
    with session_scope(SessionFactory) as db:
        start_date, end_date = get_common_date_range(db, CRUDE_OIL_CODES)
        
        close = fetch_crud_oil_data(db, CRUDE_OIL_CODES, start_date, end_date)
        
        if close.empty:
            logger.error("No price data available. Exiting.")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: Single Factor Grid Search")
        logger.info("=" * 60)
        single_results = run_single_factor_grid_search(close)
        save_results(single_results, "single_factor_results.csv")
        
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Trend Filter Tests")
        logger.info("=" * 60)
        filter_results = run_trend_filter_tests(close)
        save_results(filter_results, "filter_results.csv")
        
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: Multi-Factor Strategy Search")
        logger.info("=" * 60)
        multi_results = run_multi_factor_grid_search(close)
        save_results(multi_results, "multi_factor_results.csv")
        
        all_results = single_results + filter_results + multi_results
        analysis = analyze_results(all_results)
        
        save_results(all_results, "all_results.csv")
        
        generate_report(analysis, OUTPUT_DIR)
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info("Total strategies tested: %d", len(all_results))
        logger.info("Sharpe >= 1.0: %d", analysis.get("sharpe_ge_1.0", 0))
        logger.info("Sharpe >= 1.3: %d", analysis.get("sharpe_ge_1.3", 0))
        logger.info("Return >= 28%%: %d", analysis.get("return_ge_28%", 0))
        logger.info("MaxDD <= 15%%: %d", analysis.get("maxdd_le_15%", 0))
        
        best = analysis.get("best_by_sharpe")
        if best:
            logger.info("\nBest Strategy by Sharpe:")
            logger.info("  Name: %s", best.get("strategy_name"))
            logger.info("  Sharpe: %.3f", best.get("metrics.sharpe_ratio", 0))
            logger.info("  Return: %.1f%%", best.get("metrics.annualized_return", 0) * 100)
            logger.info("  MaxDD: %.1f%%", best.get("metrics.max_drawdown", 0) * 100)
        
        logger.info("\nResults saved to: %s", OUTPUT_DIR)
        logger.info("Report: %s/crude_oil_rotation_report.md", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Backtest failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
