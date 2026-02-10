"""
Crude Oil Rotation Strategy Configuration

原油相关ETF轮动策略的默认最优配置，包含标的池、因子定义、策略参数等。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


CRUDE_OIL_CODES = [
    "501018",  # 南方原油 - 2016-12-01成立
    "160723",  # 嘉实原油 - 2017-03-20成立
    "161129",  # 易方达原油 - 2017-04-21成立
    #"160416",  # 华安石油 - 2017-03-20成立
    "162719",  # 广发石油 - 2017-02-28成立
    "163208",  # 诺安油气 - 2017-03-20成立
    #"162411",  # 华宝油气 - 2017-01-23成立
]


@dataclass(frozen=True)
class FactorParams:
    """因子计算参数"""
    mom_lookback: int = 90
    vol_lookback: int = 20
    sma_period: int = 50
    ema_period: int = 12
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


@dataclass(frozen=True)
class BacktestRules:
    """回测基础规则"""
    cost_bps: float = 3.0
    risk_free_rate: float = 0.025
    adjust: str = "qfq"


@dataclass(frozen=True)
class CrudeOilRotationConfig:
    """原油轮动策略配置"""
    codes: Tuple[str, ...] = tuple(CRUDE_OIL_CODES)
    
    # 核心策略参数
    top_k: int = 2
    lookback_days: int = 90
    skip_days: int = 0
    rebalance: str = "weekly"
    score_method: str = "multi_factor"
    
    # 因子配置
    factors: FactorParams = field(default_factory=FactorParams)
    
    # 风险控制
    enable_trend_filter: bool = True
    enable_rsi_filter: bool = True
    enable_vol_filter: bool = False
    vol_threshold: Optional[float] = None
    
    # 回测规则
    rules: BacktestRules = field(default_factory=BacktestRules)
    
    # 因子权重（多因子合成时使用）
    factor_weights: Tuple[float, ...] = (0.4, 0.3, 0.3)
    
    @property
    def mom_weight(self) -> float:
        return self.factor_weights[0]
    
    @property
    def trend_weight(self) -> float:
        return self.factor_weights[1]
    
    @property
    def vol_weight(self) -> float:
        return self.factor_weights[2]


# 预设策略配置模板
PRESET_CONFIGS = {
    "raw_mom_90d_top2_weekly": CrudeOilRotationConfig(
        top_k=2,
        lookback_days=90,
        score_method="raw_mom",
        rebalance="weekly",
    ),
    "sharpe_mom_90d_top2_weekly": CrudeOilRotationConfig(
        top_k=2,
        lookback_days=90,
        score_method="sharpe_mom",
        rebalance="weekly",
    ),
    "multi_factor_90d_top2_weekly": CrudeOilRotationConfig(
        top_k=2,
        lookback_days=90,
        score_method="multi_factor",
        rebalance="weekly",
        factor_weights=(0.5, 0.3, 0.2),
    ),
    "multi_factor_120d_top1_monthly": CrudeOilRotationConfig(
        top_k=1,
        lookback_days=120,
        score_method="multi_factor",
        rebalance="monthly",
        factor_weights=(0.4, 0.4, 0.2),
    ),
}


def get_default_config() -> CrudeOilRotationConfig:
    """获取默认策略配置"""
    return CrudeOilRotationConfig()


def get_preset_config(name: str) -> Optional[CrudeOilRotationConfig]:
    """获取预设策略配置"""
    return PRESET_CONFIGS.get(name)
