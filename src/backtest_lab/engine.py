"""
backtest-lab: Main backtesting engine.

The ``Backtest`` class orchestrates the full backtest loop: validates data,
initializes the portfolio, iterates over trading days, rebalances on
schedule, and computes performance metrics.

Also provides walk-forward analysis and multi-strategy comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest_lab.data import validate_prices
from backtest_lab.metrics import compute_metrics
from backtest_lab.portfolio import Portfolio
from backtest_lab.strategy import EqualWeight, Strategy
from backtest_lab.types import (
    BacktestConfig,
    BacktestResult,
    Metrics,
    RebalanceFrequency,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
)

logger = logging.getLogger(__name__)


class Backtest:
    """Multi-asset portfolio backtesting engine.

    Accepts a DataFrame of prices and an optional strategy, runs the
    backtest, and returns comprehensive results with metrics, equity
    curve, and trade history.

    Example::

        from backtest_lab import Backtest, BacktestConfig

        result = Backtest(
            prices,
            config=BacktestConfig(initial_capital=100_000, commission=0.001),
        ).run()

        print(result.metrics.sharpe_ratio)
        print(result.equity_curve)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy | None = None,
        config: BacktestConfig | None = None,
        benchmark: pd.Series | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Initialize the backtester.

        Args:
            prices: DataFrame with DatetimeIndex, columns are asset symbols,
                values are closing prices.
            strategy: Allocation strategy. Defaults to EqualWeight.
            config: Backtest configuration. Uses defaults if not provided.
            benchmark: Optional benchmark price series for comparison.
        """
        self._prices = validate_prices(prices)
        self._strategy = strategy or EqualWeight()
        self._config = config or BacktestConfig()
        self._benchmark = benchmark

    # ── Core API ─────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Run the backtest and return results.

        Returns:
            BacktestResult with metrics, equity curve, trades, and returns.
        """
        return self._run_single(self._prices, self._strategy)

    def walk_forward(
        self,
        wf_config: WalkForwardConfig | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Splits the data into rolling train/test windows and runs backtests
        on each. Reports out-of-sample consistency and performance degradation.

        Args:
            wf_config: Walk-forward configuration. Uses defaults if not provided.

        Returns:
            WalkForwardResult with per-window and aggregate metrics.

        Raises:
            ValueError: If there aren't enough data points for the minimum
                number of windows.
        """
        cfg = wf_config or WalkForwardConfig()
        n = len(self._prices)
        required = cfg.train_periods + cfg.test_periods + (cfg.min_windows - 1) * cfg.step_periods
        if n < required:
            msg = (
                f"Need at least {required} periods for {cfg.min_windows} windows, "
                f"got {n}"
            )
            raise ValueError(msg)

        windows: list[WalkForwardWindow] = []
        oos_returns_list: list[pd.Series] = []  # type: ignore[type-arg]

        start = 0
        window_id = 0

        while start + cfg.train_periods + cfg.test_periods <= n:
            train_end = start + cfg.train_periods
            test_end = train_end + cfg.test_periods

            train_data = self._prices.iloc[start:train_end]
            test_data = self._prices.iloc[train_end:test_end]

            train_result = self._run_single(train_data, self._strategy)
            test_result = self._run_single(test_data, self._strategy)

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
            )
            windows.append(window)
            oos_returns_list.append(test_result.returns)

            start += cfg.step_periods
            window_id += 1

        if len(windows) < cfg.min_windows:
            msg = (
                f"Only generated {len(windows)} windows, "
                f"minimum required is {cfg.min_windows}"
            )
            raise ValueError(msg)

        # Combine out-of-sample returns
        oos_returns = pd.concat(oos_returns_list)

        # Build OOS equity curve from returns
        oos_equity = (1 + oos_returns).cumprod() * self._config.initial_capital
        oos_equity_curve = pd.DataFrame({"equity": oos_equity})

        oos_metrics = compute_metrics(oos_returns, oos_equity_curve, [], self._config)

        # Consistency: fraction of windows with positive OOS return
        positive_windows = sum(
            1 for w in windows
            if w.test_metrics is not None and w.test_metrics.total_return_pct > 0
        )
        consistency = positive_windows / len(windows) if windows else 0.0

        # Degradation: average (train Sharpe - test Sharpe)
        sharpe_diffs = [
            (w.train_metrics.sharpe_ratio - w.test_metrics.sharpe_ratio)
            for w in windows
            if w.train_metrics is not None and w.test_metrics is not None
        ]
        degradation = float(np.mean(sharpe_diffs)) if sharpe_diffs else 0.0

        return WalkForwardResult(
            windows=windows,
            oos_equity_curve=oos_equity,
            oos_metrics=oos_metrics,
            consistency=consistency,
            degradation=degradation,
        )

    def compare(self, *strategies: Strategy) -> pd.DataFrame:
        """Run multiple strategies and return a comparison table.

        Args:
            *strategies: Strategy instances to compare. The instance's
                strategy (from __init__) is always included first.

        Returns:
            DataFrame with one row per strategy and metrics as columns.
        """
        all_strategies: list[tuple[str, Strategy]] = [
            (self._strategy.__class__.__name__, self._strategy),
        ]
        for s in strategies:
            all_strategies.append((s.__class__.__name__, s))

        rows: list[dict[str, float | int | str]] = []
        for name, strat in all_strategies:
            result = self._run_single(self._prices, strat)
            row = {"strategy": name, **result.metrics.to_dict()}
            rows.append(row)

        return pd.DataFrame(rows).set_index("strategy")

    # ── Internal ─────────────────────────────────────────────────────

    def _run_single(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
    ) -> BacktestResult:
        """Run a single backtest on the given price data.

        This is the core engine loop.
        """
        portfolio = Portfolio(self._config)
        assets = prices.columns.tolist()
        rebalance_dates = self._get_rebalance_dates(prices.index)

        weights_records: list[dict[str, float]] = []
        first_rebalance_done = False

        for date in prices.index:
            # Update portfolio with today's prices
            current_prices = {
                asset: float(prices.loc[date, asset])
                for asset in assets
                if pd.notna(prices.loc[date, asset])
            }
            portfolio.update_prices(current_prices)

            # Check if this is a rebalance day
            if date in rebalance_dates:
                history = prices.loc[:date]
                raw_weights = strategy.allocate(history)

                # Skip if strategy returns empty (e.g., BuyAndHold after first call)
                if raw_weights or not first_rebalance_done:
                    weights = strategy.validate_weights(
                        raw_weights if raw_weights else {},
                        assets,
                    )
                    if weights:
                        portfolio.rebalance(weights, current_prices, date)
                        weights_records.append({"date": date, **weights})
                        first_rebalance_done = True

            # Record daily snapshot
            portfolio.record_snapshot(date)

        # Build results
        equity_curve = portfolio.get_equity_curve()

        if len(equity_curve) < 2:
            empty_returns: pd.Series = pd.Series(dtype=float)  # type: ignore[type-arg]
            return BacktestResult(
                metrics=Metrics(total_trades=len(portfolio.trades)),
                equity_curve=equity_curve,
                weights_history=pd.DataFrame(),
                trades=portfolio.trades,
                returns=empty_returns,
                config=self._config,
            )

        returns = equity_curve["equity"].pct_change().dropna()

        # Build weights history
        if weights_records:
            weights_df = pd.DataFrame(weights_records).set_index("date")
            weights_df = weights_df.reindex(equity_curve.index).ffill().fillna(0)
        else:
            weights_df = pd.DataFrame(index=equity_curve.index)

        metrics = compute_metrics(returns, equity_curve, portfolio.trades, self._config)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            weights_history=weights_df,
            trades=portfolio.trades,
            returns=returns,
            config=self._config,
        )

    def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> set[pd.Timestamp]:
        """Compute which dates in the index are rebalance dates."""
        freq = self._config.rebalance_frequency

        if freq == RebalanceFrequency.DAILY:
            return set(index)

        if freq == RebalanceFrequency.NEVER:
            # Only rebalance on the first date
            return {index[0]} if len(index) > 0 else set()

        # Group by period and take the first date in each group
        dates: set[pd.Timestamp] = set()

        if freq == RebalanceFrequency.WEEKLY:
            # First trading day of each week
            groups = pd.Series(index, index=index).groupby(
                [index.isocalendar().year, index.isocalendar().week]
            )
        elif freq == RebalanceFrequency.MONTHLY:
            groups = pd.Series(index, index=index).groupby(
                [index.year, index.month]
            )
        elif freq == RebalanceFrequency.QUARTERLY:
            groups = pd.Series(index, index=index).groupby(
                [index.year, (index.month - 1) // 3]
            )
        else:
            return set(index)

        for _, group in groups:
            dates.add(group.iloc[0])

        return dates
