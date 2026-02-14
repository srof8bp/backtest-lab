"""Compare different rebalancing frequencies and strategies."""

import numpy as np
import pandas as pd

from backtest_lab import (
    Backtest,
    BacktestConfig,
    BuyAndHold,
    EqualWeight,
    InverseVolatility,
    RebalanceFrequency,
)

# Generate sample data
rng = np.random.default_rng(42)
dates = pd.bdate_range("2022-01-03", periods=504, freq="B")
prices = pd.DataFrame(
    {
        "AAPL": 150.0 * np.cumprod(1 + rng.normal(0.0004, 0.018, 504)),
        "GOOG": 90.0 * np.cumprod(1 + rng.normal(0.0003, 0.020, 504)),
        "MSFT": 250.0 * np.cumprod(1 + rng.normal(0.0005, 0.016, 504)),
    },
    index=dates,
)

# ── Compare strategies ────────────────────────────────────────────────
print("=== Strategy Comparison ===\n")

bt = Backtest(prices, config=BacktestConfig(initial_capital=100_000))
comparison = bt.compare(
    BuyAndHold(),
    InverseVolatility(lookback=60),
)
print(comparison[["total_return_pct", "sharpe_ratio", "max_drawdown", "total_trades"]])

# ── Compare rebalance frequencies ─────────────────────────────────────
print("\n=== Rebalance Frequency Comparison ===\n")

for freq in [
    RebalanceFrequency.DAILY,
    RebalanceFrequency.WEEKLY,
    RebalanceFrequency.MONTHLY,
    RebalanceFrequency.QUARTERLY,
]:
    config = BacktestConfig(
        initial_capital=100_000,
        rebalance_frequency=freq,
    )
    result = Backtest(prices, config=config).run()
    print(
        f"  {freq.value:<12s}  "
        f"return={result.metrics.total_return_pct:+.2%}  "
        f"sharpe={result.metrics.sharpe_ratio:.2f}  "
        f"trades={result.metrics.total_trades:>4d}  "
        f"commission=${result.metrics.total_commission:.2f}"
    )
