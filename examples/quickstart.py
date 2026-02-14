"""Quick start: run a backtest with 5 lines of code."""

import numpy as np
import pandas as pd

from backtest_lab import Backtest, BacktestConfig

# Generate sample price data (3 assets, 1 year)
rng = np.random.default_rng(42)
dates = pd.bdate_range("2023-01-02", periods=252, freq="B")
prices = pd.DataFrame(
    {
        "AAPL": 150.0 * np.cumprod(1 + rng.normal(0.0004, 0.018, 252)),
        "GOOG": 90.0 * np.cumprod(1 + rng.normal(0.0003, 0.020, 252)),
        "MSFT": 250.0 * np.cumprod(1 + rng.normal(0.0005, 0.016, 252)),
    },
    index=dates,
)

# Run backtest with default equal-weight strategy
result = Backtest(
    prices,
    config=BacktestConfig(initial_capital=100_000, commission=0.001),
).run()

# Print results
print(result)
print(f"\nSharpe Ratio:   {result.metrics.sharpe_ratio:.2f}")
print(f"Total Return:   {result.metrics.total_return_pct:+.2%}")
print(f"Max Drawdown:   {result.metrics.max_drawdown:.2%}")
print(f"Win Rate:       {result.metrics.win_rate:.1%}")
print(f"Total Trades:   {result.metrics.total_trades}")
print(f"Commission:     ${result.metrics.total_commission:.2f}")

# Access the equity curve (a regular pandas DataFrame)
print(f"\nEquity curve shape: {result.equity_curve.shape}")
print(result.equity_curve.tail())
