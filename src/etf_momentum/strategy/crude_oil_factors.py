"""
Crude Oil Factor Calculation Module

原油ETF轮动策略的因子计算模块，包含动量、趋势、波动率等因子的计算。
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_raw_momentum(close: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """
    计算原始动量因子
    
    公式: (Close(t) / Close(t-N) - 1)
    
    Args:
        close: 收盘价矩阵
        lookback: 回看天数
    
    Returns:
        动量因子矩阵
    """
    return close.pct_change(periods=lookback)


def compute_sharpe_momentum(
    close: pd.DataFrame, 
    mom_lookback: int = 90, 
    vol_lookback: int = 20
) -> pd.DataFrame:
    """
    计算夏普动量因子
    
    公式: 累计收益 / 波动率
    
    Args:
        close: 收盘价矩阵
        mom_lookback: 收益回看天数
        vol_lookback: 波动率回看天数
    
    Returns:
        夏普动量因子矩阵
    """
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    sharpe = cum_ret / vol.replace(0, np.nan)
    return sharpe


def compute_sortino_momentum(
    close: pd.DataFrame, 
    mom_lookback: int = 90, 
    vol_lookback: int = 20
) -> pd.DataFrame:
    """
    计算Sortino动量因子
    
    公式: 累计收益 / 下行波动率
    
    Args:
        close: 收盘价矩阵
        mom_lookback: 收益回看天数
        vol_lookback: 下行波动率回看天数
    
    Returns:
        Sortino动量因子矩阵
    """
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    
    downside = ret.where(ret < 0, 0)
    downside_vol = downside.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    sortino = cum_ret / downside_vol.replace(0, np.nan)
    return sortino


def compute_return_over_vol(
    close: pd.DataFrame, 
    mom_lookback: int = 90, 
    vol_lookback: int = 20
) -> pd.DataFrame:
    """
    计算收益/波动率因子
    
    公式: 累计收益 / 历史波动率
    
    Args:
        close: 收盘价矩阵
        mom_lookback: 收益回看天数
        vol_lookback: 波动率回看天数
    
    Returns:
        收益/波动率因子矩阵
    """
    ret = close.pct_change()
    cum_ret = close.pct_change(periods=mom_lookback)
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    rov = cum_ret / vol.replace(0, np.nan)
    return rov


def compute_sma_trend(close: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    """
    计算均线趋势因子
    
    公式: Close(t) / SMA(N) - 1
    
    Args:
        close: 收盘价矩阵
        period: 均线周期
    
    Returns:
        均线趋势因子矩阵
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    trend = close / sma - 1
    return trend


def compute_ema_trend(close: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """
    计算EMA趋势因子
    
    公式: Close(t) / EMA(N) - 1
    
    Args:
        close: 收盘价矩阵
        period: EMA周期
    
    Returns:
        EMA趋势因子矩阵
    """
    ema = close.ewm(span=period, adjust=False).mean()
    trend = close / ema - 1
    return trend


def compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算RSI因子
    
    使用Wilder平滑方法
    
    Args:
        close: 收盘价矩阵
        period: RSI周期
    
    Returns:
        RSI值矩阵 (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.clip(0, 100)
    return rsi


def compute_low_vol_factor(close: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    """
    计算低波动因子
    
    公式: 1 / 波动率
    
    Args:
        close: 收盘价矩阵
        vol_lookback: 波动率回看天数
    
    Returns:
        低波动因子矩阵
    """
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    low_vol = 1 / vol.replace(0, np.nan)
    return low_vol


def compute_volatility_momentum(close: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    """
    计算波动率动量因子
    
    公式: 波动率变化率
    
    Args:
        close: 收盘价矩阵
        vol_lookback: 波动率回看天数
    
    Returns:
        波动率动量因子矩阵
    """
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    vol_mom = vol.pct_change()
    return vol_mom


def compute_momentum_decay(
    close: pd.DataFrame, 
    short_lookback: int = 20, 
    long_lookback: int = 90
) -> pd.DataFrame:
    """
    计算动量衰减因子
    
    公式: 近N日收益 / 远N日收益
    
    Args:
        close: 收盘价矩阵
        short_lookback: 短期回看天数
        long_lookback: 长期回看天数
    
    Returns:
        动量衰减因子矩阵
    """
    short_ret = close.pct_change(periods=short_lookback)
    long_ret = close.pct_change(periods=long_lookback)
    decay = short_ret / long_ret.replace(0, np.nan)
    return decay


def compute_all_factors(
    close: pd.DataFrame,
    mom_lookback: int = 90,
    vol_lookback: int = 20,
    sma_period: int = 50,
    rsi_period: int = 14,
) -> Dict[str, pd.DataFrame]:
    """
    计算所有因子
    
    Args:
        close: 收盘价矩阵
        mom_lookback: 动量回看天数
        vol_lookback: 波动率回看天数
        sma_period: 均线周期
        rsi_period: RSI周期
    
    Returns:
        因子字典
    """
    factors = {
        "raw_mom": compute_raw_momentum(close, mom_lookback),
        "sharpe_mom": compute_sharpe_momentum(close, mom_lookback, vol_lookback),
        "sortino_mom": compute_sortino_momentum(close, mom_lookback, vol_lookback),
        "return_over_vol": compute_return_over_vol(close, mom_lookback, vol_lookback),
        "sma_trend": compute_sma_trend(close, sma_period),
        "ema_trend": compute_ema_trend(close, sma_period),
        "rsi": compute_rsi(close, rsi_period),
        "low_vol": compute_low_vol_factor(close, vol_lookback),
        "vol_mom": compute_volatility_momentum(close, vol_lookback),
    }
    return factors


def compute_multi_factor_score(
    close: pd.DataFrame,
    mom_weight: float = 0.4,
    trend_weight: float = 0.3,
    vol_weight: float = 0.3,
    mom_lookback: int = 90,
    vol_lookback: int = 20,
    sma_period: int = 50,
) -> pd.DataFrame:
    """
    计算多因子综合得分
    
    Args:
        close: 收盘价矩阵
        mom_weight: 动量因子权重
        trend_weight: 趋势因子权重
        vol_weight: 波动率因子权重
        mom_lookback: 动量回看天数
        vol_lookback: 波动率回看天数
        sma_period: 均线周期
    
    Returns:
        多因子综合得分矩阵
    """
    factors = compute_all_factors(
        close, 
        mom_lookback=mom_lookback,
        vol_lookback=vol_lookback,
        sma_period=sma_period,
    )
    
    mom_score = factors["sharpe_mom"]
    trend_score = factors["sma_trend"]
    vol_score = factors["low_vol"]
    
    def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """排名标准化"""
        return df.rank(axis=1, pct=True)
    
    mom_norm = rank_normalize(mom_score)
    trend_norm = rank_normalize(trend_score)
    vol_norm = rank_normalize(vol_score)
    
    multi_factor = (
        mom_weight * mom_norm + 
        trend_weight * trend_norm + 
        vol_weight * vol_norm
    )
    
    return multi_factor


def get_factor_ranks(
    close: pd.DataFrame,
    score_method: str = "sharpe_mom",
    lookback_days: int = 90,
    vol_lookback: int = 20,
    sma_period: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    获取因子得分和排名
    
    Args:
        close: 收盘价矩阵
        score_method: 评分方法
        lookback_days: 回看天数
        vol_lookback: 波动率回看天数
        sma_period: 均线周期
    
    Returns:
        (因子得分矩阵, 因子排名矩阵)
    """
    if score_method == "raw_mom":
        scores = compute_raw_momentum(close, lookback_days)
    elif score_method == "sharpe_mom":
        scores = compute_sharpe_momentum(close, lookback_days, vol_lookback)
    elif score_method == "sortino_mom":
        scores = compute_sortino_momentum(close, lookback_days, vol_lookback)
    elif score_method == "return_over_vol":
        scores = compute_return_over_vol(close, lookback_days, vol_lookback)
    elif score_method == "sma_trend":
        scores = compute_sma_trend(close, sma_period)
    elif score_method == "ema_trend":
        scores = compute_ema_trend(close, sma_period)
    elif score_method == "low_vol":
        scores = compute_low_vol_factor(close, vol_lookback)
    elif score_method == "multi_factor":
        scores = compute_multi_factor_score(
            close, 
            mom_weight=0.4, 
            trend_weight=0.3, 
            vol_weight=0.3,
            mom_lookback=lookback_days,
            vol_lookback=vol_lookback,
            sma_period=sma_period,
        )
    else:
        raise ValueError(f"Unknown score_method: {score_method}")
    
    ranks = scores.rank(axis=1, ascending=True)
    
    return scores, ranks


def filter_by_trend(
    ranks: pd.DataFrame,
    close: pd.DataFrame,
    sma_period: int = 50,
) -> pd.DataFrame:
    """
    趋势过滤：只选择价格高于均线的标的
    
    Args:
        ranks: 因子排名矩阵
        close: 收盘价矩阵
        sma_period: 均线周期
    
    Returns:
        过滤后的排名矩阵（不满足条件的设为NaN）
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    above_sma = close > sma
    filtered = ranks.where(above_sma)
    return filtered


def filter_by_rsi(
    ranks: pd.DataFrame,
    close: pd.DataFrame,
    rsi_period: int = 14,
    rsi_lower: float = 30.0,
    rsi_upper: float = 70.0,
    prefer_oversold: bool = True,
) -> pd.DataFrame:
    """
    RSI过滤：只选择RSI在合理区间的标的
    
    Args:
        ranks: 因子排名矩阵
        close: 收盘价矩阵
        rsi_period: RSI周期
        rsi_lower: RSI下界
        rsi_upper: RSI上界
        prefer_oversold: 是否优先选择超卖标的
    
    Returns:
        过滤后的排名矩阵
    """
    rsi = compute_rsi(close, rsi_period)
    
    if prefer_oversold:
        mask = (rsi >= rsi_lower) & (rsi <= rsi_upper)
    else:
        mask = (rsi >= rsi_lower) & (rsi <= rsi_upper)
    
    filtered = ranks.where(mask)
    return filtered


def filter_by_volatility(
    ranks: pd.DataFrame,
    close: pd.DataFrame,
    vol_threshold: float = 0.05,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    波动率过滤：排除波动率过高的标的
    
    Args:
        ranks: 因子排名矩阵
        close: 收盘价矩阵
        vol_threshold: 波动率阈值
        vol_lookback: 波动率回看天数
    
    Returns:
        过滤后的排名矩阵
    """
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback, min_periods=vol_lookback).std()
    low_vol_mask = vol <= vol_threshold
    filtered = ranks.where(low_vol_mask)
    return filtered


def apply_filters(
    ranks: pd.DataFrame,
    close: pd.DataFrame,
    enable_trend: bool = True,
    enable_rsi: bool = True,
    enable_vol: bool = False,
    sma_period: int = 50,
    rsi_period: int = 14,
    rsi_lower: float = 30.0,
    rsi_upper: float = 70.0,
    vol_threshold: Optional[float] = None,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    应用所有过滤器
    
    Args:
        ranks: 因子排名矩阵
        close: 收盘价矩阵
        enable_trend: 是否启用趋势过滤
        enable_rsi: 是否启用RSI过滤
        enable_vol: 是否启用波动率过滤
        sma_period: 均线周期
        rsi_period: RSI周期
        rsi_lower: RSI下界
        rsi_upper: RSI上界
        vol_threshold: 波动率阈值
        vol_lookback: 波动率回看天数
    
    Returns:
        过滤后的排名矩阵
    """
    filtered = ranks.copy()
    
    if enable_trend:
        filtered = filter_by_trend(filtered, close, sma_period)
    
    if enable_rsi:
        filtered = filter_by_rsi(
            filtered, 
            close, 
            rsi_period=rsi_period,
            rsi_lower=rsi_lower,
            rsi_upper=rsi_upper,
            prefer_oversold=True,
        )
    
    if enable_vol and vol_threshold is not None:
        filtered = filter_by_volatility(
            filtered,
            close,
            vol_threshold=vol_threshold,
            vol_lookback=vol_lookback,
        )
    
    return filtered


def should_be_cash(
    close: pd.DataFrame,
    sma_period: int = 50,
    rsi_period: int = 14,
    rsi_lower: float = 30.0,
) -> pd.Series:
    """
    判断是否应该空仓
    
    空仓条件（任一满足即空仓）：
    1. 所有标的的价格都低于均线
    2. 所有标的的RSI都低于超卖阈值
    
    Args:
        close: 收盘价矩阵
        sma_period: 均线周期
        rsi_period: RSI周期
        rsi_lower: RSI超卖阈值
    
    Returns:
        布尔Series，True表示应该空仓
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    above_sma = (close > sma).any(axis=1)
    
    rsi = compute_rsi(close, rsi_period)
    rsi_above_oversold = (rsi > rsi_lower).any(axis=1)
    
    should_cash = ~(above_sma & rsi_above_oversold)
    
    return should_cash
