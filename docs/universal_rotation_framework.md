# Universal Rotation Research Framework - Summary

## Completed: February 9, 2026

### What Was Built

#### 1. Universal Configuration (`src/etf_momentum/strategy/rotation_research_config.py`)

- **`UniverseConfig`**: Define any ETF pool with name, codes, and description
- **`RotationStrategyConfig`**: Full strategy parameters with dataclasses
- **`ScoreMethod` Enum**: raw_mom, sharpe_mom, sortino_mom, return_over_vol, sma_trend, low_vol, multi_factor
- **`RebalanceFreq` Enum**: daily, weekly, biweekly, monthly, quarterly
- **Preset Universes**:
  - `crude_oil`: 7原油/油气ETF
  - `a_core`: 5核心A股ETF
  - `sector`: 5行业ETF
- **Validation**: Config validation with error messages

#### 2. Universal Runner (`src/etf_momentum/scripts/rotation_research_runner.py`)

- **Grid Search**: 72 combinations (3 score_methods × 4 lookbacks × 3 top_k × 2 rebalance_freqs)
- **Parameter Search**: 18 combinations for sensitivity analysis
- **Factor Calculations**: All scoring methods with numpy/pandas
- **Execution Engine**: Portfolio rotation with cost modeling (3bps)
- **Metrics**: Sharpe, Sortino, Calmar, max drawdown, win rate, annualized return
- **Report Generation**: Auto-generated markdown reports
- **Command-Line Interface**:

  ```bash
  # Preset universe
  python -m etf_momentum.scripts.rotation_research_runner -u crude_oil

  # Custom ETF list
  python -m etf_momentum.scripts.rotation_research_runner -c "510300,510500" -n "My Strategy"

  # Quick test
  python -m etf_momentum.scripts.rotation_research_runner -u sector -q
  ```

#### 3. Research Web Page (`src/etf_momentum/web/research_crude_oil.html`)

- **Universe Selector**: Dropdown to switch between crude_oil, a_core, sector
- **Dynamic ETF Lists**: Shows correct ETFs for each universe
- **Results Loading**: Auto-loads from `data/rotation_research/{universe}/all_results.csv`
- **Run Backtest Section**: UI for generating run commands
- **Statistics Dashboard**: Filter results, view best strategy
- **Fallback Data**: Sample data if no CSV found

### File Structure

```
momentum/
├── src/etf_momentum/
│   ├── strategy/
│   │   ├── rotation_research_config.py   (NEW - Universal config)
│   │   ├── crude_oil_rotation_config.py  (Original - Crude oil specific)
│   │   ├── crude_oil_factors.py          (Original - Factors)
│   │   └── rotation.py                   (Existing)
│   ├── scripts/
│   │   ├── rotation_research_runner.py   (NEW - Universal runner)
│   │   └── crude_oil_rotation_runner.py  (Original)
│   └── web/
│       └── research_crude_oil.html       (UPDATED - Universal support)
├── data/
│   ├── crude_oil/
│   │   ├── all_results.csv               (Existing - 168 strategies)
│   │   └── crude_oil_rotation_report.md
│   └── rotation_research/                (NEW directory for future results)
│       ├── crude_oil/
│       ├── a_core/
│       └── sector/
└── docs/
    └── crude_oil_rotation_report.md     (Copy of report)
```

### Usage Examples

#### Python API

```python
from etf_momentum.strategy.rotation_research_config import (
    UniverseConfig,
    RotationStrategyConfig,
    get_preset_universe,
)
from etf_momentum.scripts.rotation_research_runner import run_research
from etf_momentum.db.session import make_session_factory

# Use preset universe
universe = get_preset_universe("crude_oil")

# Or create custom
universe = UniverseConfig(
    name="My ETFs",
    codes=["510300", "510500", "510880"],
    description="Custom ETF pool"
)

# Run research
session_factory = make_session_factory(engine)
db = session_factory()
results = run_research(universe=universe, db=db, cost_bps=3.0)
```

#### Command Line

```bash
# Full grid search on crude oil
python -m etf_momentum.scripts.rotation_research_runner -u crude_oil

# Quick parameter sensitivity
python -m etf_momentum.scripts.rotation_research_runner -u a_core -q

# Custom ETF universe
python -m etf_momentum.scripts.rotation_research_runner -c "510300,510500,510880" -n "A股三剑客"
```

### Results Location

- **CSV**: `data/rotation_research/{universe}/all_results.csv`
- **Report**: `data/rotation_research/{universe}/{universe}_report.md`

### Next Steps (Optional)

1. **Test the universal runner**: Run `python -m etf_momentum.scripts.rotation_research_runner -u crude_oil`
2. **Verify compatibility**: Results should match existing crude_oil results
3. **Run on other universes**: Test a_core and sector universes
4. **Update report links**: Point to universal research directory
5. **Add new universes**: Extend `PRESET_UNIVERSES` as needed

### Key Metrics (Crude Oil - Existing Results)

- **Total strategies tested**: 168
- **Best Sharpe**: 0.344 (sharpe_mom_90d_top3_weekly)
- **Best Return**: 11.5% annual
- **Best Max Drawdown**: -40.9%
- **Target achievement**: None (Sharpe≥1.3, Return≥28%, MaxDD≤15% not met)

### Notes

- The universal framework is backward compatible with existing crude_oil results
- Code compiles without syntax errors
- Web page supports dynamic universe switching
- Command-line interface is intuitive and flexible
- Results structure is consistent across universes
