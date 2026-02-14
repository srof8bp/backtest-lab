"""Custom strategy example: momentum-based allocation."""

import numpy as np
import pandas as pd

from backtest_lab import Backtest, BacktestConfig, Strategy


class MomentumStrategy(Strategy):
    """Allocate to the top-N assets by recent momentum.

    Ranks assets by their trailing return over a lookback window
    and equally weights the top performers.
    """

    def __init__(self, lookback: int = 63, top_n: int = 2) -> None:
        self.lookback = lookback
        self.top_n = top_n

    def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
        if len(prices) < self.lookback:
            # Not enough history — equal weight everything
            n = len(prices.columns)
            return {a: 1.0 / n for a in prices.columns} if n > 0 else {}

        # Trailing returns
        returns = prices.iloc[-1] / prices.iloc[-self.lookback] - 1
        top = returns.nlargest(self.top_n).index.tolist()
        weight = 1.0 / len(top)
        return {asset: weight for asset in top}


# Generate sample data (5 assets, 2 years)
rng = np.random.default_rng(42)
dates = pd.bdate_range("2022-01-03", periods=504, freq="B")
prices = pd.DataFrame(
    {
        "AAPL": 150.0 * np.cumprod(1 + rng.normal(0.0006, 0.018, 504)),
        "GOOG": 90.0 * np.cumprod(1 + rng.normal(0.0002, 0.022, 504)),
        "MSFT": 250.0 * np.cumprod(1 + rng.normal(0.0005, 0.016, 504)),
        "AMZN": 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.024, 504)),
        "TSLA": 200.0 * np.cumprod(1 + rng.normal(0.0001, 0.035, 504)),
    },
    index=dates,
)

# Run momentum backtest
result = Backtest(
    prices,
    strategy=MomentumStrategy(lookback=63, top_n=2),
    config=BacktestConfig(initial_capital=100_000),
).run()

print("Momentum Strategy Results:")
print(f"  Total Return:   {result.metrics.total_return_pct:+.2%}")
print(f"  Sharpe Ratio:   {result.metrics.sharpe_ratio:.2f}")
print(f"  Max Drawdown:   {result.metrics.max_drawdown:.2%}")
print(f"  Win Rate:       {result.metrics.win_rate:.1%}")
print(f"  Total Trades:   {result.metrics.total_trades}")
