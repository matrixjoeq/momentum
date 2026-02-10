"""
Generic Rotation Research Framework

通用轮动策略研究框架，支持任意标的组合的轮动策略回测与分析。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ScoreMethod(Enum):
    RAW_MOM = "raw_mom"
    SHARPE_MOM = "sharpe_mom"
    SORTINO_MOM = "sortino_mom"
    RETURN_OVER_VOL = "return_over_vol"
    SMA_TREND = "sma_trend"
    LOW_VOL = "low_vol"
    MULTI_FACTOR = "multi_factor"


class RebalanceFreq(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class FactorParams:
    mom_lookback: int = 90
    vol_lookback: int = 20
    sma_period: int = 50
    ema_period: int = 12
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


@dataclass
class BacktestRules:
    cost_bps: float = 3.0
    risk_free_rate: float = 0.025
    adjust: str = "qfq"


@dataclass
class UniverseConfig:
    name: str = "Custom Universe"
    codes: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "codes": self.codes, "description": self.description}


@dataclass 
class RotationStrategyConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    score_method: str = "multi_factor"
    lookback_days: int = 90
    top_k: int = 2
    skip_days: int = 0
    rebalance: str = "weekly"
    factors: FactorParams = field(default_factory=FactorParams)
    enable_trend_filter: bool = True
    enable_rsi_filter: bool = True
    enable_vol_filter: bool = False
    vol_threshold: Optional[float] = None
    enable_cash_protection: bool = True
    rules: BacktestRules = field(default_factory=BacktestRules)
    mom_weight: float = 0.4
    trend_weight: float = 0.3
    vol_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe": self.universe.to_dict(),
            "score_method": self.score_method,
            "lookback_days": self.lookback_days,
            "top_k": self.top_k,
            "rebalance": self.rebalance,
            "factor_weights": {"momentum": self.mom_weight, "trend": self.trend_weight, "volatility": self.vol_weight},
            "risk_control": {"trend": self.enable_trend_filter, "rsi": self.enable_rsi_filter},
        }


PRESET_UNIVERSES = {
    "crude_oil": UniverseConfig(name="原油ETF", codes=["501018", "160723", "161129", "160416", "162719", "163208", "162411"], description="7只原油/油气LOF"),
    "a_core": UniverseConfig(name="A股核心", codes=["510300", "510500", "510880", "510900", "511660"], description="核心A股ETF"),
    "sector": UniverseConfig(name="行业轮动", codes=["512880", "512690", "512760", "512800", "512780"], description="5大行业ETF"),
}


def get_preset_universe(name: str) -> Optional[UniverseConfig]:
    return PRESET_UNIVERSES.get(name)


def validate_config(config: RotationStrategyConfig) -> tuple[bool, List[str]]:
    errors = []
    if not config.universe.codes:
        errors.append("标的池不能为空")
    if config.top_k < 1:
        errors.append("持仓数量必须大于0")
    if config.lookback_days < 20:
        errors.append("回看天数至少20天")
    if config.top_k > len(config.universe.codes):
        errors.append("持仓数量不能超过标的池大小")
    weights = config.mom_weight + config.trend_weight + config.vol_weight
    if abs(weights - 1.0) > 0.01:
        errors.append("因子权重之和必须为1.0")
    return len(errors) == 0, errors
