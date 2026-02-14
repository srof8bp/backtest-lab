"""
backtest-lab: Multi-asset portfolio backtesting engine.

A lightweight, pandas-native backtesting library for multi-asset portfolio
strategies. Supports custom allocation strategies, walk-forward analysis,
and multi-strategy comparison with a dead-simple API.

Example::

    from backtest_lab import Backtest, BacktestConfig

    result = Backtest(
        prices,
        config=BacktestConfig(initial_capital=100_000),
    ).run()

    print(result.metrics.sharpe_ratio)
    print(result.equity_curve)
"""

from __future__ import annotations

from backtest_lab.data import compute_returns, validate_prices
from backtest_lab.engine import Backtest
from backtest_lab.metrics import compute_metrics
from backtest_lab.portfolio import Portfolio
from backtest_lab.strategy import BuyAndHold, EqualWeight, InverseVolatility, Strategy
from backtest_lab.types import (
    BacktestConfig,
    BacktestResult,
    Frequency,
    Metrics,
    Position,
    RebalanceFrequency,
    SizingMethod,
    Trade,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    Weights,
)

__all__ = [
    # Core
    "Backtest",
    "BacktestConfig",
    "BacktestResult",
    # Types
    "Frequency",
    "Metrics",
    "Position",
    "RebalanceFrequency",
    "SizingMethod",
    "Trade",
    "Weights",
    # Strategies
    "Strategy",
    "EqualWeight",
    "BuyAndHold",
    "InverseVolatility",
    # Portfolio
    "Portfolio",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    # Functions
    "compute_metrics",
    "compute_returns",
    "validate_prices",
]

__version__ = "0.1.0"
