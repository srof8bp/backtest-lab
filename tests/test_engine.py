"""Tests for the Backtest engine — integration + walk-forward."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.engine import Backtest
from backtest_lab.strategy import BuyAndHold, InverseVolatility, Strategy
from backtest_lab.types import (
    BacktestConfig,
    BacktestResult,
    RebalanceFrequency,
    WalkForwardConfig,
)


class TestBacktestRun:
    """Core run() method tests."""

    def test_basic_run(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.metrics.total_trades > 0

    def test_custom_config(self, prices_3asset: pd.DataFrame) -> None:
        config = BacktestConfig(
            initial_capital=50_000,
            commission=0.002,
            slippage=0.001,
        )
        result = Backtest(prices_3asset, config=config).run()
        assert result.config.initial_capital == 50_000
        assert result.metrics.total_commission > 0
        assert result.metrics.total_slippage > 0

    def test_equal_weight_default(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        # Should have weights for all 3 assets
        if not result.weights_history.empty:
            assert set(result.weights_history.columns) == {"AAPL", "GOOG", "MSFT"}

    def test_buy_and_hold(self, prices_2asset: pd.DataFrame) -> None:
        strategy = BuyAndHold({"SPY": 0.6, "TLT": 0.4})
        config = BacktestConfig(rebalance_frequency=RebalanceFrequency.NEVER)
        result = Backtest(prices_2asset, strategy=strategy, config=config).run()
        assert isinstance(result, BacktestResult)
        # With NEVER rebalance + BuyAndHold, should trade once at start
        assert result.metrics.total_trades >= 2  # at least buy SPY + buy TLT

    def test_inverse_volatility(self, prices_3asset: pd.DataFrame) -> None:
        strategy = InverseVolatility(lookback=10)
        result = Backtest(prices_3asset, strategy=strategy).run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_shape(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        assert len(result.equity_curve) == len(prices_3asset)
        assert "equity" in result.equity_curve.columns
        assert "cash" in result.equity_curve.columns

    def test_returns_series(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        assert len(result.returns) == len(prices_3asset) - 1
        assert isinstance(result.returns, pd.Series)

    def test_flat_prices_no_gain(self, prices_flat: pd.DataFrame) -> None:
        config = BacktestConfig(commission=0, slippage=0)
        result = Backtest(prices_flat, config=config).run()
        # Flat prices → returns near zero
        assert abs(result.metrics.total_return_pct) < 0.01

    def test_metrics_repr(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        r = repr(result.metrics)
        assert "return=" in r
        assert "sharpe=" in r

    def test_result_repr(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        r = repr(result)
        assert "BacktestResult" in r
        assert "periods" in r


class TestRebalanceFrequencies:
    """Rebalance scheduling."""

    def test_daily_rebalance(self, prices_3asset: pd.DataFrame) -> None:
        config = BacktestConfig(
            rebalance_frequency=RebalanceFrequency.DAILY,
            commission=0,
            slippage=0,
        )
        result = Backtest(prices_3asset, config=config).run()
        # Daily rebalance → many trades
        assert result.metrics.total_trades > 10

    def test_never_rebalance(self, prices_3asset: pd.DataFrame) -> None:
        config = BacktestConfig(
            rebalance_frequency=RebalanceFrequency.NEVER,
            commission=0,
            slippage=0,
        )
        result = Backtest(prices_3asset, config=config).run()
        # Only initial allocation
        assert result.metrics.total_trades >= 3  # 3 assets, 1 buy each

    def test_monthly_rebalance(self, prices_3asset: pd.DataFrame) -> None:
        config = BacktestConfig(
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            commission=0,
            slippage=0,
        )
        result = Backtest(prices_3asset, config=config).run()
        assert isinstance(result, BacktestResult)


class TestCompare:
    """Multi-strategy comparison."""

    def test_compare_two_strategies(self, prices_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_3asset)
        comparison = bt.compare(BuyAndHold(), InverseVolatility(lookback=10))
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3  # EqualWeight + BuyAndHold + InverseVolatility
        assert "sharpe_ratio" in comparison.columns
        assert "total_return_pct" in comparison.columns

    def test_compare_returns_metrics(self, prices_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_3asset)
        comparison = bt.compare(BuyAndHold())
        assert comparison.index.name == "strategy"
        assert "EqualWeight" in comparison.index
        assert "BuyAndHold" in comparison.index


class TestWalkForward:
    """Walk-forward analysis."""

    def test_basic_walk_forward(self, prices_long_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_long_3asset)
        wf_config = WalkForwardConfig(
            train_periods=126,
            test_periods=63,
            step_periods=63,
            min_windows=2,
        )
        result = bt.walk_forward(wf_config)
        assert len(result.windows) >= 2
        assert 0.0 <= result.consistency <= 1.0
        assert isinstance(result.degradation, float)

    def test_insufficient_data_raises(self, prices_3asset: pd.DataFrame) -> None:
        """20 data points is not enough for default walk-forward (252+63 periods)."""
        bt = Backtest(prices_3asset)
        with pytest.raises(ValueError, match="Need at least"):
            bt.walk_forward()

    def test_window_metrics(self, prices_long_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_long_3asset)
        wf_config = WalkForwardConfig(
            train_periods=126,
            test_periods=63,
            step_periods=63,
            min_windows=2,
        )
        result = bt.walk_forward(wf_config)
        for window in result.windows:
            assert window.train_metrics is not None
            assert window.test_metrics is not None
            assert window.train_start < window.train_end
            assert window.test_start < window.test_end

    def test_oos_equity_curve(self, prices_long_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_long_3asset)
        wf_config = WalkForwardConfig(
            train_periods=126,
            test_periods=63,
            step_periods=63,
            min_windows=2,
        )
        result = bt.walk_forward(wf_config)
        assert len(result.oos_equity_curve) > 0
        assert result.oos_metrics is not None

    def test_walk_forward_repr(self, prices_long_3asset: pd.DataFrame) -> None:
        bt = Backtest(prices_long_3asset)
        wf_config = WalkForwardConfig(
            train_periods=126,
            test_periods=63,
            step_periods=63,
            min_windows=2,
        )
        result = bt.walk_forward(wf_config)
        r = repr(result)
        assert "WalkForwardResult" in r
        assert "windows" in r


class TestCustomStrategy:
    """Verify custom strategies plug in correctly."""

    def test_momentum_strategy(self, prices_3asset: pd.DataFrame) -> None:
        class TopMomentum(Strategy):
            def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
                if len(prices) < 5:
                    n = len(prices.columns)
                    return {a: 1.0 / n for a in prices.columns} if n > 0 else {}
                ret = prices.pct_change(5).iloc[-1]
                top = ret.nlargest(2).index.tolist()
                return {a: 0.5 for a in top}

        result = Backtest(prices_3asset, strategy=TopMomentum()).run()
        assert isinstance(result, BacktestResult)
        assert result.metrics.total_trades > 0

    def test_cash_only_strategy(self, prices_3asset: pd.DataFrame) -> None:
        """A strategy that returns empty weights (all cash)."""

        class AllCash(Strategy):
            def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
                return {}

        config = BacktestConfig(commission=0, slippage=0)
        result = Backtest(prices_3asset, strategy=AllCash(), config=config).run()
        # No trades after initial (which is skipped because weights are empty)
        assert result.metrics.total_trades == 0
        # Equity stays at initial capital
        assert result.equity_curve["equity"].iloc[-1] == pytest.approx(100_000)


class TestBacktestResultProperties:
    """BacktestResult convenience properties."""

    def test_drawdown_series(self, prices_3asset: pd.DataFrame) -> None:
        result = Backtest(prices_3asset).run()
        dd = result.drawdown_series
        assert isinstance(dd, pd.Series)
        assert (dd <= 0).all()  # drawdowns are always <= 0


class TestImports:
    """Public API import tests."""

    def test_all_public_exports(self) -> None:
        import backtest_lab

        assert hasattr(backtest_lab, "Backtest")
        assert hasattr(backtest_lab, "BacktestConfig")
        assert hasattr(backtest_lab, "BacktestResult")
        assert hasattr(backtest_lab, "Strategy")
        assert hasattr(backtest_lab, "EqualWeight")
        assert hasattr(backtest_lab, "BuyAndHold")
        assert hasattr(backtest_lab, "InverseVolatility")
        assert hasattr(backtest_lab, "Portfolio")
        assert hasattr(backtest_lab, "Metrics")
        assert hasattr(backtest_lab, "Trade")
        assert hasattr(backtest_lab, "WalkForwardConfig")
        assert hasattr(backtest_lab, "WalkForwardResult")
        assert hasattr(backtest_lab, "__version__")

    def test_version_is_string(self) -> None:
        from backtest_lab import __version__

        assert isinstance(__version__, str)
        assert "." in __version__
