# backtest-lab

Multi-asset portfolio backtesting engine for Python. Weights-based strategies, walk-forward analysis, and comprehensive metrics — with a dead-simple API.

## Features

- **Multi-asset** — backtest portfolios of any number of assets
- **Weights-based strategies** — strategies return target weight dicts, the engine handles execution
- **Walk-forward analysis** — rolling train/test validation with consistency and degradation metrics
- **Strategy comparison** — compare multiple strategies side-by-side in one call
- **Comprehensive metrics** — Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate, and more
- **Realistic execution** — commission, slippage, and sell-before-buy ordering
- **Flexible rebalancing** — daily, weekly, monthly, quarterly, or never
- **Pandas-native** — prices in, DataFrames out. No special formats
- **Zero magic** — no hidden state, no global config, no singletons

## Installation

```bash
pip install backtest-lab
```

Or install from source:

```bash
git clone https://github.com/TjTheDj2011/backtest-lab.git
cd backtest-lab
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from backtest_lab import Backtest, BacktestConfig

prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)

result = Backtest(
    prices,
    config=BacktestConfig(initial_capital=100_000, commission=0.001),
).run()

print(result.metrics.sharpe_ratio)   # 1.24
print(result.metrics.max_drawdown)   # -0.0832
print(result.equity_curve.tail())    # pandas DataFrame
```

## Custom Strategies

Subclass `Strategy` and implement `allocate()`. Return a dict of `{asset: weight}` — the engine handles everything else.

```python
from backtest_lab import Backtest, Strategy

class MomentumStrategy(Strategy):
    def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
        if len(prices) < 63:
            n = len(prices.columns)
            return {a: 1.0 / n for a in prices.columns}
        returns = prices.iloc[-1] / prices.iloc[-63] - 1
        top = returns.nlargest(3).index.tolist()
        return {asset: 1 / 3 for asset in top}

result = Backtest(prices, strategy=MomentumStrategy()).run()
```

## Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `EqualWeight()` | Equal allocation across all assets (default) |
| `BuyAndHold(weights)` | Allocate once, never rebalance |
| `InverseVolatility(lookback=60)` | Lower volatility gets more weight |

## Strategy Comparison

```python
from backtest_lab import Backtest, BuyAndHold, InverseVolatility

bt = Backtest(prices)
comparison = bt.compare(BuyAndHold(), InverseVolatility())
print(comparison)
#                    total_return_pct  sharpe_ratio  max_drawdown  total_trades
# strategy
# EqualWeight                 0.1245          1.24       -0.0832            36
# BuyAndHold                  0.1198          1.18       -0.0901             3
# InverseVolatility           0.1312          1.31       -0.0774            36
```

## Walk-Forward Analysis

Validate your strategy out-of-sample with rolling train/test windows:

```python
from backtest_lab import Backtest, WalkForwardConfig

bt = Backtest(prices)
wf = bt.walk_forward(WalkForwardConfig(
    train_periods=252,   # 1 year training
    test_periods=63,     # 3 months testing
    step_periods=63,     # Step forward 3 months
    min_windows=4,       # At least 4 windows
))

print(f"Consistency: {wf.consistency:.0%}")       # % of windows profitable OOS
print(f"Degradation: {wf.degradation:.2f}")       # Train Sharpe - Test Sharpe
print(f"OOS Sharpe:  {wf.oos_metrics.sharpe_ratio:.2f}")
```

## Rebalancing Frequencies

```python
from backtest_lab import BacktestConfig, RebalanceFrequency

config = BacktestConfig(
    rebalance_frequency=RebalanceFrequency.WEEKLY,  # or DAILY, MONTHLY, QUARTERLY, NEVER
)
```

## Metrics

Every backtest returns a `Metrics` object with:

| Metric | Description |
|--------|-------------|
| `total_return` | Total dollar return |
| `total_return_pct` | Total return as a percentage |
| `annualized_return` | CAGR |
| `volatility` | Annualized standard deviation |
| `sharpe_ratio` | Risk-adjusted return (annualized) |
| `sortino_ratio` | Downside-deviation-adjusted return |
| `max_drawdown` | Maximum peak-to-trough decline |
| `max_drawdown_duration` | Longest drawdown in periods |
| `calmar_ratio` | Return per unit of drawdown |
| `win_rate` | Fraction of positive-return periods |
| `profit_factor` | Gross profits / gross losses |
| `total_trades` | Number of trades executed |
| `total_commission` | Total commission paid |
| `total_slippage` | Total slippage cost |

```python
result.metrics.to_dict()  # Flat dict for serialization
```

## BacktestResult Extras

```python
result.equity_curve       # DataFrame: equity, cash, positions_value
result.returns            # Series: daily portfolio returns
result.trades             # List[Trade]: every trade executed
result.weights_history    # DataFrame: target weights over time
result.drawdown_series    # Series: drawdown time series
result.monthly_returns    # DataFrame: year x month pivot table
result.annual_returns     # Series: annual returns
```

## Configuration

```python
from backtest_lab import BacktestConfig, Frequency, RebalanceFrequency

config = BacktestConfig(
    initial_capital=100_000,                         # Starting portfolio value
    commission=0.001,                                # 0.1% per trade
    slippage=0.0005,                                 # 0.05% slippage
    rebalance_frequency=RebalanceFrequency.MONTHLY,  # When to rebalance
    frequency=Frequency.DAILY,                       # Data frequency (for annualization)
    risk_free_rate=0.05,                             # Annual risk-free rate
)
```

## Requirements

- Python 3.10+
- pandas >= 2.0
- numpy >= 1.24

## License

Apache 2.0
