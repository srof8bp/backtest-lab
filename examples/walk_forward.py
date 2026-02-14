"""Walk-forward analysis: train/test validation of a strategy."""

import numpy as np
import pandas as pd

from backtest_lab import Backtest, BacktestConfig, WalkForwardConfig

# Generate 3 years of data (enough for walk-forward windows)
rng = np.random.default_rng(42)
dates = pd.bdate_range("2021-01-04", periods=756, freq="B")
prices = pd.DataFrame(
    {
        "AAPL": 130.0 * np.cumprod(1 + rng.normal(0.0004, 0.018, 756)),
        "GOOG": 80.0 * np.cumprod(1 + rng.normal(0.0003, 0.022, 756)),
        "MSFT": 220.0 * np.cumprod(1 + rng.normal(0.0005, 0.016, 756)),
    },
    index=dates,
)

# Run walk-forward analysis
bt = Backtest(prices, config=BacktestConfig(initial_capital=100_000))
wf_result = bt.walk_forward(
    WalkForwardConfig(
        train_periods=252,  # 1 year training
        test_periods=63,    # 3 months testing
        step_periods=63,    # Step forward 3 months
        min_windows=4,      # At least 4 windows
    )
)

print(wf_result)
print(f"\nWindows:      {len(wf_result.windows)}")
print(f"Consistency:  {wf_result.consistency:.0%} of windows profitable OOS")
print(f"Degradation:  {wf_result.degradation:.2f} (train Sharpe - test Sharpe)")
print(f"OOS Sharpe:   {wf_result.oos_metrics.sharpe_ratio:.2f}")
print(f"OOS Return:   {wf_result.oos_metrics.total_return_pct:+.2%}")

print("\n--- Per-Window Results ---")
for w in wf_result.windows:
    train_sr = w.train_metrics.sharpe_ratio if w.train_metrics else 0.0
    test_sr = w.test_metrics.sharpe_ratio if w.test_metrics else 0.0
    test_ret = w.test_metrics.total_return_pct if w.test_metrics else 0.0
    print(
        f"  Window {w.window_id}: "
        f"train_sharpe={train_sr:+.2f}  "
        f"test_sharpe={test_sr:+.2f}  "
        f"test_return={test_ret:+.2%}"
    )
