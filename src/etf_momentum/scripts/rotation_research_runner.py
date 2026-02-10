#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Rotation Research Runner

通用轮动策略研究脚本，支持任意标的组合的策略回测与分析。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from etf_momentum.settings import get_settings
from etf_momentum.db.session import make_engine, make_session_factory
from etf_momentum.db.init_db import init_db
from etf_momentum.db.repo import get_price_date_range
from etf_momentum.strategy.rotation_research_config import (
    UniverseConfig,
    RotationStrategyConfig,
    FactorParams,
    BacktestRules,
    get_preset_universe,
    validate_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rotation_research.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/rotation_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    strategy_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {"strategy_name": self.strategy_name, "params": self.params, "metrics": self.metrics}


def fetch_price_data(db, codes: List[str], start_date: dt.date, end_date: dt.date, adjust: str = "qfq") -> pd.DataFrame:
    """获取价格数据"""
    from sqlalchemy import select
    from etf_momentum.db.models import EtfPrice
    
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.close)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == adjust)
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
    pivot = df.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    
    logger.info("Loaded price data: %d rows, %d codes, date range: %s to %s",
                len(pivot), len(pivot.columns), pivot.index.min().date(), pivot.index.max().date())
    
    return pivot


def get_date_range(db, codes: List[str], adjust: str = "qfq") -> tuple[dt.date, dt.date]:
    """获取公共数据区间"""
    starts, ends = [], []
    for code in codes:
        start, end = get_price_date_range(db, code=code, adjust=adjust)
        if start:
            starts.append(dt.datetime.strptime(start, "%Y%m%d").date())
        if end:
            ends.append(dt.datetime.strptime(end, "%Y%m%d").date())
    
    if not starts or not ends:
        return dt.date(2017, 1, 1), dt.date.today()
    
    return max(starts), min(ends)


def calculate_metrics(nav: pd.Series, rf_rate: float = 0.025) -> Dict[str, float]:
    """计算绩效指标"""
    if len(nav) < 2:
        return {}
    
    daily_ret = nav.pct_change().dropna()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(nav) - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
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
    win_rate = (daily_ret > 0).sum() / len(daily_ret) if len(daily_ret) > 0 else 0
    
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
    return close.pct_change(periods=lookback)


def compute_sharpe_momentum(close: pd.DataFrame, mom_lookback: int = 90, vol_lookback: int = 20) -> pd.DataFrame:
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    return cum_ret / vol.replace(0, np.nan)


def compute_sma_trend(close: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    sma = close.rolling(window=period, min_periods=period).mean()
    return close / sma - 1


def compute_low_vol(close: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    return 1 / vol.replace(0, np.nan)


def compute_multi_factor(close: pd.DataFrame, mom_w: float, trend_w: float, vol_w: float, 
                       mom_lookback: int, vol_lookback: int, sma_period: int) -> pd.DataFrame:
    mom = compute_sharpe_momentum(close, mom_lookback, vol_lookback)
    trend = compute_sma_trend(close, sma_period)
    vol = compute_low_vol(close, vol_lookback)
    
    def rank_norm(df): return df.rank(axis=1, pct=True)
    
    return mom_w * rank_norm(mom) + trend_w * rank_norm(trend) + vol_w * rank_norm(vol)


def get_factor_score(close: pd.DataFrame, method: str, lookback: int, vol_lookback: int, sma_period: int) -> pd.DataFrame:
    if method == "raw_mom":
        return compute_raw_momentum(close, lookback)
    elif method == "sharpe_mom":
        return compute_sharpe_momentum(close, lookback, vol_lookback)
    elif method == "sma_trend":
        return compute_sma_trend(close, sma_period)
    elif method == "low_vol":
        return compute_low_vol(close, vol_lookback)
    elif method == "multi_factor":
        return compute_multi_factor(close, 0.4, 0.3, 0.3, lookback, vol_lookback, sma_period)
    else:
        return compute_raw_momentum(close, lookback)


def execute_rotation(close: pd.DataFrame, scores: pd.DataFrame, top_k: int, rebalance: str, 
                   enable_trend: bool, sma_period: int, cost_bps: float) -> pd.Series:
    """执行轮动策略"""
    ranks = scores.rank(axis=1, ascending=True)
    
    if enable_trend:
        sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
        above_sma = close > sma
        ranks = ranks.where(above_sma)
    
    if rebalance == "daily":
        rebalance_dates = close.index
    elif rebalance == "weekly":
        rebalance_dates = close.resample("W-FRI").first().index
    elif rebalance == "monthly":
        rebalance_dates = close.resample("ME").first().index
    else:
        rebalance_dates = close.index
    
    rebalance_set = set(rebalance_dates)
    
    weights_list = []
    prev_weights = None
    
    for i, date in enumerate(close.index):
        if date not in ranks.index:
            weights = prev_weights if prev_weights is not None else pd.Series(0.0, index=close.columns)
            weights_list.append(weights)
            continue
        
        if date in rebalance_set or i == 0:
            day_ranks = ranks.loc[date]
            valid_ranks = day_ranks.dropna().sort_values(ascending=False)
            top_codes = valid_ranks.head(top_k).index.tolist()
            weights = pd.Series(0.0, index=close.columns)
            if top_codes:
                w = 1.0 / len(top_codes)
                for code in top_codes:
                    weights.loc[code] = w
        else:
            weights = prev_weights if prev_weights is not None else pd.Series(0.0, index=close.columns)
        
        weights_list.append(weights)
        prev_weights = weights
    
    portfolio = pd.DataFrame(weights_list, index=close.index)
    daily_ret = close.pct_change().fillna(0)
    prev_portfolio = portfolio.shift(1).fillna(0)
    weight_change = (portfolio - prev_portfolio).abs().sum(axis=1)
    portfolio_ret = (portfolio * daily_ret).sum(axis=1) - weight_change * (cost_bps / 10000) / 2
    
    nav = (1 + portfolio_ret).cumprod()
    nav.iloc[0] = 1.0
    return nav


def backtest_strategy(close: pd.DataFrame, config: RotationStrategyConfig, cost_bps: float = 3.0) -> BacktestResult:
    """回测单个策略"""
    scores = get_factor_score(
        close, 
        config.score_method,
        config.lookback_days,
        config.factors.vol_lookback,
        config.factors.sma_period
    )
    
    strategy_name = f"{config.score_method}_{config.lookback_days}d_top{config.top_k}_{config.rebalance}"
    if config.enable_trend_filter:
        strategy_name += "_trend"
    
    nav = execute_rotation(
        close, scores, config.top_k, config.rebalance,
        config.enable_trend_filter, config.factors.sma_period, cost_bps
    )
    
    metrics = calculate_metrics(nav, config.rules.risk_free_rate)
    
    params = {
        "score_method": config.score_method,
        "lookback_days": config.lookback_days,
        "top_k": config.top_k,
        "rebalance": config.rebalance,
        "enable_trend_filter": config.enable_trend_filter,
        "universe": config.universe.name,
        "n_codes": len(config.universe.codes),
    }
    
    return BacktestResult(strategy_name, params, metrics)


def run_grid_search(close: pd.DataFrame, universe: UniverseConfig, cost_bps: float = 3.0) -> List[BacktestResult]:
    """运行网格搜索"""
    results = []
    
    score_methods = ["raw_mom", "sharpe_mom", "sma_trend"]
    lookback_days_list = [60, 90, 120]
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
                    
                    config = RotationStrategyConfig(
                        universe=universe,
                        score_method=score_method,
                        lookback_days=lookback,
                        top_k=top_k,
                        rebalance=rebalance,
                        enable_trend_filter=True,
                    )
                    
                    result = backtest_strategy(close, config, cost_bps)
                    results.append(result)
    
    return results


def run_parameter_search(close: pd.DataFrame, universe: UniverseConfig, cost_bps: float = 3.0) -> List[BacktestResult]:
    """运行参数敏感性搜索"""
    results = []
    
    lookback_days_list = [30, 45, 60, 90, 120, 180]
    top_k_list = [1, 2, 3]
    
    total = len(lookback_days_list) * len(top_k_list)
    count = 0
    
    for lookback in lookback_days_list:
        for top_k in top_k_list:
            count += 1
            logger.info(f"[{count}/{total}] Param search: lookback={lookback}d, top_k={top_k}")
            
            config = RotationStrategyConfig(
                universe=universe,
                score_method="sharpe_mom",
                lookback_days=lookback,
                top_k=top_k,
                rebalance="weekly",
                enable_trend_filter=True,
            )
            
            result = backtest_strategy(close, config, cost_bps)
            results.append(result)
    
    return results


def analyze_results(results: List[BacktestResult]) -> Dict[str, Any]:
    """分析回测结果"""
    if not results:
        return {"error": "No results"}
    
    df = pd.DataFrame([r.to_dict() for r in results])
    df_sorted = df.sort_values("metrics.sharpe_ratio", ascending=False)
    
    best_sharpe = df_sorted.iloc[0] if len(df_sorted) > 0 else None
    best_return = df.sort_values("metrics.annualized_return", ascending=False).iloc[0] if len(df) > 0 else None
    
    sensitivity = {}
    if "params.lookback_days" in df.columns:
        sensitivity["lookback"] = df.groupby("params.lookback_days")["metrics.sharpe_ratio"].mean().to_dict()
    if "params.top_k" in df.columns:
        sensitivity["top_k"] = df.groupby("params.top_k")["metrics.sharpe_ratio"].mean().to_dict()
    
    return {
        "total_strategies": len(df),
        "best_by_sharpe": best_sharpe.to_dict() if best_sharpe is not None else None,
        "best_by_return": best_return.to_dict() if best_return is not None else None,
        "sensitivity": sensitivity,
        "top_strategies": df_sorted.head(20).to_dict(orient="records"),
    }


def save_results(results: List[BacktestResult], filename: str, universe_name: str):
    """保存回测结果"""
    if not results:
        return
    
    df = pd.DataFrame([r.to_dict() for r in results])
    metrics_df = pd.json_normalize(df["metrics"])
    metrics_df.columns = [f"metrics.{c}" for c in metrics_df.columns]
    result_df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)
    
    subdir = OUTPUT_DIR / universe_name.replace(" ", "_").lower()
    subdir.mkdir(parents=True, exist_ok=True)
    filepath = subdir / filename
    result_df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Saved results to {filepath}")


def generate_report(analysis: Dict[str, Any], universe: UniverseConfig, output_dir: Path):
    """生成研究报告"""
    report = f"""# 轮动策略研究报告

## 标的池配置

**名称**: {universe.name}
**代码**: {', '.join(universe.codes)}
**描述**: {universe.description}

## 执行摘要

- 测试策略总数: {analysis.get('total_strategies', 0)}
- 最佳夏普比率: {analysis.get('best_by_sharpe', {}).get('metrics.sharpe_ratio', 'N/A')}
- 最佳年化收益: {analysis.get('best_by_return', {}).get('metrics.annualized_return', 'N/A')}

## 最佳策略（按夏普比率）

"""
    
    best = analysis.get("best_by_sharpe", {})
    m = best.get("metrics", {})
    report += f"""
| 指标 | 值 |
|------|-----|
| 策略名称 | {best.get('strategy_name', 'N/A')} |
| 夏普比率 | {m.get('sharpe_ratio', 'N/A')} |
| 年化收益率 | {m.get('annualized_return', 'N/A')} |
| 最大回撤 | {m.get('max_drawdown', 'N/A')} |
| 卡尔马比率 | {m.get('calmar_ratio', 'N/A')} |

## Top 20 策略

| 策略 | 夏普 | 年化收益 | 最大回撤 |
|------|------|----------|----------|
"""
    
    for s in analysis.get("top_strategies", [])[:20]:
        m = s.get("metrics", {})
        report += f"| {s.get('strategy_name', 'N/A')} | {m.get('sharpe_ratio', 0):.3f} | {m.get('annualized_return', 0)*100:.1f}% | {m.get('max_drawdown', 0)*100:.1f}% |\n"
    
    filepath = output_dir / f"{universe.name.replace(' ', '_').lower()}_report.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Generated report: {filepath}")


def run_research(
    universe: UniverseConfig,
    db,
    cost_bps: float = 3.0,
    run_full_grid: bool = True,
) -> Dict[str, Any]:
    """运行完整研究"""
    logger.info("=" * 60)
    logger.info(f"Rotation Research: {universe.name}")
    logger.info("=" * 60)
    
    start_date, end_date = get_date_range(db, universe.codes)
    close = fetch_price_data(db, universe.codes, start_date, end_date)
    
    if close.empty:
        logger.error("No price data available")
        return {"error": "No data"}
    
    if run_full_grid:
        results = run_grid_search(close, universe, cost_bps)
    else:
        results = run_parameter_search(close, universe, cost_bps)
    
    save_results(results, "all_results.csv", universe.name)
    analysis = analyze_results(results)
    
    subdir = OUTPUT_DIR / universe.name.replace(" ", "_").lower()
    generate_report(analysis, universe, subdir)
    
    return analysis


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generic Rotation Research")
    parser.add_argument("--universe", "-u", default="crude_oil", help="Universe name or JSON config")
    parser.add_argument("--codes", "-c", help="Comma-separated codes (overrides universe)")
    parser.add_argument("--name", "-n", default="Custom Universe", help="Universe name")
    parser.add_argument("--cost", "-k", type=float, default=3.0, help="Cost in bps")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick parameter search only")
    
    args = parser.parse_args()
    
    settings = get_settings()
    engine = make_engine(db_url=settings.db_url)
    init_db(engine)
    SessionFactory = make_session_factory(engine)
    
    db = SessionFactory()
    try:
        if args.codes:
            codes = [c.strip() for c in args.codes.split(",")]
            universe = UniverseConfig(name=args.name, codes=codes)
        else:
            universe = get_preset_universe(args.universe)
            if universe is None:
                universe = UniverseConfig(name=args.universe, codes=[])
        
        analysis = run_research(
            universe=universe,
            db=db,
            cost_bps=args.cost,
            run_full_grid=not args.quick,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("RESEARCH SUMMARY")
        logger.info("=" * 60)
        logger.info("Universe: %s", universe.name)
        logger.info("Total strategies: %s", analysis.get("total_strategies", 0))
        
        best = analysis.get("best_by_sharpe", {})
        if best:
            m = best.get("metrics", {})
            logger.info("Best Strategy: %s", best.get("strategy_name"))
            logger.info("  Sharpe: %.3f", m.get("sharpe_ratio", 0))
            logger.info("  Ann Return: %.1f%%", m.get("annualized_return", 0) * 100)
            logger.info("  Max Drawdown: %.1f%%", m.get("max_drawdown", 0) * 100)
        
        logger.info("\nResults saved to: %s", OUTPUT_DIR / universe.name.replace(" ", "_").lower())
    
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Research failed: %s", e)
        sys.exit(1)
