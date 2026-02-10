# ETF轮动策略研究工具使用文档

## 概述

ETF轮动策略研究工具是一个通用的量化策略回测框架，支持对任意ETF组合进行多因子轮动策略的回测、参数敏感性分析和绩效评估。

### 主要功能

- 支持自定义ETF标的池
- 多种评分因子：动量、趋势、波动率等
- 灵活的参数配置
- 网格搜索最优参数
- 自动生成回测报告

---

## 快速开始

### 环境准备

```bash
# 确保已安装依赖
pip install -e .

# 进入项目目录
cd momentum
```

### 最简单的回测

```bash
# 使用预设的原油ETF标的池进行回测
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil
```

回测结果将保存在 `data/rotation_research/crude_oil/` 目录下。

---

## 命令行参数

### 基础参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--universe` | `-u` | 预设标的池名称 | `crude_oil` |
| `--codes` | `-c` | 自定义ETF代码（逗号分隔） | 无 |
| `--name` | `-n` | 研究名称 | `Custom Universe` |
| `--cost` | `-k` | 交易成本（bps） | `3.0` |
| `--quick` | `-q` | 快速参数敏感性测试 | `False` |

### 使用示例

```bash
# 1. 使用预设原油ETF标的池
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil

# 2. 使用预设A股核心标的池
python -m etf_momentum.scripts.rotation_research_runner -u a_core

# 3. 使用预设行业轮动标的池
python -m etf_momentum.scripts.rotation_research_runner -u sector

# 4. 自定义ETF组合
python -m etf_momentum.scripts.rotation_research_runner -c "510300,510500,510880" -n "A股三剑客"

# 5. 快速测试模式（仅参数敏感性分析）
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil -q

# 6. 设置不同的交易成本
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil -k 5.0
```

---

## 预设标的池

### 1. 原油ETF (crude_oil)

```bash
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil
```

包含7只原油/油气相关ETF：

| 代码 | 名称 |
|------|------|
| 501018 | 南方原油 |
| 160723 | 嘉实原油 |
| 161129 | 易方达原油 |
| 160416 | 华安石油 |
| 162719 | 广发石油 |
| 163208 | 诺安油气 |
| 162411 | 华宝油气 |

### 2. A股核心 (a_core)

```bash
python -m etf_momentum.scripts.rotation_research_runner -u a_core
```

包含5只核心A股ETF：

| 代码 | 名称 |
|------|------|
| 510300 | 沪深300 |
| 510500 | 中证500 |
| 510880 | 创业板指 |
| 510900 | 中证1000 |
| 511660 | 中证800 |

### 3. 行业轮动 (sector)

```bash
python -m etf_momentum.scripts.rotation_research_runner -u sector
```

包含5只行业ETF：

| 代码 | 名称 |
|------|------|
| 512880 | 证券ETF |
| 512690 | 银行ETF |
| 512760 | 券商ETF |
| 512800 | 保险ETF |
| 512780 | 地产ETF |

---

## 自定义标的池

### 方法一：命令行参数

```bash
# 多个ETF代码用逗号分隔
python -m etf_momentum.scripts.rotation_research_runner -c "510300,510500,510880,510900" -n "我的策略"

# 可以使用6位或完整的ETF代码
python -m etf_momentum.scripts.rotation_research_runner -c "159919,159938,510500" -n "ETF组合"
```

### 方法二：Python API

```python
from etf_momentum.scripts.rotation_research_runner import run_research
from etf_momentum.strategy.rotation_research_config import UniverseConfig

# 创建自定义标的池配置
universe = UniverseConfig(
    name="我的ETF策略",
    codes=["510300", "510500", "510880"],
    description="自定义的A股ETF组合"
)

# 运行回测
results = run_research(
    universe=universe,
    db=db,
    cost_bps=3.0,
    run_full_grid=True
)
```

---

## 评分因子

### 可用的评分方法

| 因子名称 | 说明 | 计算方式 |
|----------|------|----------|
| `raw_mom` | 原始动量 | 区间收益率 |
| `sharpe_mom` | 风险调整动量 | 收益率/波动率 |
| `sma_trend` | 均线趋势 | 价格/SMA偏离度 |
| `low_vol` | 低波动 | 1/波动率 |
| `multi_factor` | 多因子综合 | 动量+趋势+波动率加权 |

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 回看天数 | 90 | 计算动量的历史区间 |
| 持仓数量 | 2 | 每次轮动持有的ETF数量 |
| 再平衡频率 | weekly | 每周/每月调仓 |
| 均线周期 | 50 | SMA计算周期 |
| 波动率周期 | 20 | 波动率计算周期 |

### 默认网格搜索参数

```
评分方法: raw_mom, sharpe_mom, sma_trend
回看天数: 60, 90, 120
持仓数量: 1, 2, 3
再平衡频率: weekly, monthly

总计: 3 × 3 × 3 × 2 = 54 种策略配置
```

---

## 回测结果

### 输出文件

回测完成后，结果保存在 `data/rotation_research/{研究名称}/` 目录下：

```
data/rotation_research/crude_oil/
├── all_results.csv           # 所有策略回测结果
└── crude_oil_report.md      # 研究报告
```

### CSV字段说明

| 字段 | 说明 |
|------|------|
| strategy_name | 策略名称 |
| params | 策略参数JSON |
| metrics.total_return | 总收益率 |
| metrics.annualized_return | 年化收益率 |
| metrics.annualized_volatility | 年化波动率 |
| metrics.max_drawdown | 最大回撤 |
| metrics.sharpe_ratio | 夏普比率 |
| metrics.sortino_ratio | Sortino比率 |
| metrics.calmar_ratio | Calmar比率 |
| metrics.win_rate | 胜率 |
| metrics.n_days | 交易日数 |

### 报告示例

```markdown
# 轮动策略研究报告

## 标的池配置

**名称**: 原油ETF
**代码**: 501018, 160723, 161129, 160416, 162719, 163208, 162411
**描述**: 7只原油/油气LOF

## 执行摘要

- 测试策略总数: 54
- 最佳夏普比率: 0.344
- 最佳年化收益: 11.5%
```

---

## Web界面

### 启动方式

直接用浏览器打开研究网页：

```bash
# Windows
start src/etf_momentum/web/research_crude_oil.html

# macOS
open src/etf_momentum/web/research_crude_oil.html

# Linux
xdg-open src/etf_momentum/web/research_crude_oil.html
```

### Web界面功能

1. **策略概览** - 查看当前标的池和因子体系
2. **回测结果** - 查看所有策略的绩效指标
3. **参数敏感性** - 分析不同参数的效果
4. **策略配置** - 查看推荐策略配置
5. **运行回测** - 生成运行命令
6. **研究报告** - 查看完整报告

### 切换标的池

在网页右上角可以选择不同的预设标的池：
- 原油ETF (7只)
- A股核心 (5只)
- 行业轮动 (5只)

---

## 策略配置示例

### 推荐策略配置（原油ETF）

```json
{
  "strategy_name": "sharpe_mom_90d_top3_weekly",
  "codes": [
    "501018", "160723", "161129",
    "160416", "162719", "163208", "162411"
  ],
  "score_method": "sharpe_mom",
  "lookback_days": 90,
  "top_k": 3,
  "rebalance": "weekly",
  "enable_trend_filter": false,
  "sma_period": 50,
  "cost_bps": 3.0,
  "expected_metrics": {
    "sharpe_ratio": 0.344,
    "annualized_return": "11.5%",
    "max_drawdown": "-40.9%"
  }
}
```

### 高波动市场策略

```json
{
  "score_method": "sharpe_mom",
  "lookback_days": 120,
  "top_k": 2,
  "rebalance": "monthly",
  "enable_trend_filter": true,
  "sma_period": 200
}
```

---

## 常见问题

### Q1: 自定义ETF不工作？

确保ETF代码存在于数据库中：

```python
# 检查ETF代码
from etf_momentum.db.repo import get_price_date_range

start, end = get_price_date_range(db, code="510300", adjust="qfq")
print(f"510300: {start} ~ {end}")
```

### Q2: 如何只运行特定参数？

修改 `run_grid_search` 函数的参数范围：

```python
# 只测试90天回看
lookback_days_list = [90]
```

### Q3: 如何解读结果？

- **夏普比率 > 1.0**: 优秀的风险调整收益
- **最大回撤 < -15%**: 需要注意风险控制
- **胜率 > 50%**: 策略具有正向预期

### Q4: 如何添加新的标的池？

编辑 `src/etf_momentum/strategy/rotation_research_config.py`：

```python
PRESET_UNIVERSES = {
    # 现有...
    "new_universe": UniverseConfig(
        name="新标的池",
        codes=["510001", "510002", "510003"],
        description="自定义描述"
    ),
}
```

---

## 进阶使用

### 完整Python脚本

```python
from etf_momentum.scripts.rotation_research_runner import run_research
from etf_momentum.strategy.rotation_research_config import UniverseConfig
from etf_momentum.db.session import make_session_factory, make_engine
from etf_momentum.db.init_db import init_db
from etf_momentum.settings import get_settings

settings = get_settings()
engine = make_engine(db_url=settings.db_url)
init_db(engine)
SessionFactory = make_session_factory(engine)

db = SessionFactory()
try:
    # 创建自定义标的池
    universe = UniverseConfig(
        name="我的策略",
        codes=["510300", "510500", "510880"],
        description="测试策略"
    )
    
    # 运行回测
    results = run_research(
        universe=universe,
        db=db,
        cost_bps=3.0,
        run_full_grid=True
    )
    
    # 打印摘要
    print(f"测试策略数: {results['total_strategies']}")
    print(f"最佳夏普: {results['best_by_sharpe']['metrics']['sharpe_ratio']:.3f}")

finally:
    db.close()
```

### 分析回测结果

```python
import pandas as pd

# 读取结果
df = pd.read_csv("data/rotation_research/crude_oil/all_results.csv")

# 按夏普排序
df_sorted = df.sort_values("metrics.sharpe_ratio", ascending=False)

# 查看前10
print(df_sorted.head(10)[["strategy_name", "metrics.sharpe_ratio", "metrics.annualized_return", "metrics.max_drawdown"]])

# 筛选条件
filtered = df[
    (df["metrics.sharpe_ratio"] >= 0.3) & 
    (df["metrics.annualized_return"] >= 0.1)
]
```

---

## 输出目录结构

```
momentum/
├── data/
│   ├── crude_oil/                    # 原油ETF回测结果（原有）
│   │   ├── all_results.csv
│   │   └── crude_oil_rotation_report.md
│   └── rotation_research/            # 通用框架结果目录
│       ├── crude_oil/
│       │   ├── all_results.csv
│       │   └── crude_oil_report.md
│       ├── a_core/
│       │   ├── all_results.csv
│       │   └── a_core_report.md
│       ├── sector/
│       │   ├── all_results.csv
│       │   └── sector_report.md
│       └── custom_我的策略/
│           ├── all_results.csv
│           └── custom_我的策略_report.md
```

---

## 最佳实践

1. **先用Quick模式测试**：添加 `-q` 参数快速验证数据完整性
2. **从小范围开始**：先用2-3只ETF测试
3. **关注夏普比率**：比绝对收益更重要
4. **检查最大回撤**：确保在可承受范围内
5. **对比基准**：与买入持有策略对比

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/etf_momentum/strategy/rotation_research_config.py` | 策略配置模块 |
| `src/etf_momentum/scripts/rotation_research_runner.py` | 回测运行脚本 |
| `src/etf_momentum/web/research_crude_oil.html` | 研究结果Web页面 |
| `data/rotation_research/` | 回测结果目录 |
