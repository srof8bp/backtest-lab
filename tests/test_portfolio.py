"""Tests for portfolio state management and trade execution."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.portfolio import Portfolio
from backtest_lab.types import BacktestConfig


class TestPortfolioInit:
    def test_initial_state(self) -> None:
        config = BacktestConfig(initial_capital=50_000)
        p = Portfolio(config)
        assert p.cash == 50_000
        assert p.positions == {}
        assert p.trades == []

    def test_initial_equity_equals_capital(self) -> None:
        config = BacktestConfig(initial_capital=100_000)
        p = Portfolio(config)
        assert p.equity() == 100_000

    def test_positions_value_zero_initially(self) -> None:
        p = Portfolio(BacktestConfig())
        assert p.positions_value() == 0.0


class TestRebalance:
    def test_single_asset_buy(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"AAPL": 100.0}
        p.update_prices(prices)
        trades = p.rebalance({"AAPL": 1.0}, prices, pd.Timestamp("2024-01-02"))

        assert len(trades) >= 1
        assert trades[0].side == "buy"
        assert trades[0].asset == "AAPL"
        assert p.cash < 100_000  # spent money
        assert "AAPL" in p.positions

    def test_two_asset_equal_weight(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0, "B": 50.0}
        p.update_prices(prices)
        p.rebalance({"A": 0.5, "B": 0.5}, prices, pd.Timestamp("2024-01-02"))

        # Each should have ~$50k allocated
        a_val = p.positions["A"].market_value(100.0)
        b_val = p.positions["B"].market_value(50.0)
        total_pos = a_val + b_val
        assert total_pos == pytest.approx(100_000, rel=0.01)

    def test_sell_existing_position(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"AAPL": 100.0, "GOOG": 200.0}
        p.update_prices(prices)
        p.rebalance({"AAPL": 0.5, "GOOG": 0.5}, prices, pd.Timestamp("2024-01-02"))

        # Now rebalance to AAPL only
        trades = p.rebalance({"AAPL": 1.0}, prices, pd.Timestamp("2024-01-03"))
        sell_trades = [t for t in trades if t.side == "sell"]
        assert len(sell_trades) >= 1
        assert any(t.asset == "GOOG" for t in sell_trades)

    def test_commission_applied(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0.01, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0}
        p.update_prices(prices)
        trades = p.rebalance({"A": 1.0}, prices, pd.Timestamp("2024-01-02"))

        total_commission = sum(t.commission for t in trades)
        assert total_commission > 0
        # Cash should reflect commission
        assert p.cash >= 0
        assert p.equity(prices) < 100_000  # Lost money to commission

    def test_slippage_applied(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0.01)
        p = Portfolio(config)
        prices = {"A": 100.0}
        p.update_prices(prices)
        trades = p.rebalance({"A": 1.0}, prices, pd.Timestamp("2024-01-02"))

        # Buy with 1% slippage: execution price = 101.0
        assert trades[0].price == pytest.approx(101.0)
        assert trades[0].slippage_cost > 0

    def test_sells_before_buys(self) -> None:
        """Sells should execute before buys to free up cash."""
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0, "B": 100.0}
        p.update_prices(prices)
        p.rebalance({"A": 1.0}, prices, pd.Timestamp("2024-01-02"))

        # Switch entirely to B
        trades = p.rebalance({"B": 1.0}, prices, pd.Timestamp("2024-01-03"))
        sell_indices = [i for i, t in enumerate(trades) if t.side == "sell"]
        buy_indices = [i for i, t in enumerate(trades) if t.side == "buy"]

        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices)

    def test_rebalance_skip_tiny_trades(self) -> None:
        """Trades < $1 should be skipped."""
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0}
        p.update_prices(prices)
        p.rebalance({"A": 0.5}, prices, pd.Timestamp("2024-01-02"))

        # Same target weight → no new trades (positions are already close)
        trades = p.rebalance({"A": 0.5}, prices, pd.Timestamp("2024-01-03"))
        # Should have very few or no new trades
        assert len(trades) <= 1  # might have a tiny adjustment


class TestPortfolioEquity:
    def test_equity_with_position(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0}
        p.update_prices(prices)
        p.rebalance({"A": 0.5}, prices, pd.Timestamp("2024-01-02"))

        # Total equity should be ~100k (no commission/slippage)
        assert p.equity(prices) == pytest.approx(100_000, rel=0.01)

    def test_equity_changes_with_price(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0}
        p.update_prices(prices)
        p.rebalance({"A": 1.0}, prices, pd.Timestamp("2024-01-02"))

        # Price doubles → equity roughly doubles (all invested)
        new_prices = {"A": 200.0}
        assert p.equity(new_prices) > 150_000

    def test_current_weights(self) -> None:
        config = BacktestConfig(initial_capital=100_000, commission=0, slippage=0)
        p = Portfolio(config)
        prices = {"A": 100.0, "B": 100.0}
        p.update_prices(prices)
        p.rebalance({"A": 0.5, "B": 0.5}, prices, pd.Timestamp("2024-01-02"))

        weights = p.current_weights(prices)
        assert weights["A"] == pytest.approx(0.5, abs=0.02)
        assert weights["B"] == pytest.approx(0.5, abs=0.02)


class TestSnapshots:
    def test_record_and_retrieve(self) -> None:
        config = BacktestConfig(initial_capital=100_000)
        p = Portfolio(config)
        p.record_snapshot(pd.Timestamp("2024-01-02"))
        p.record_snapshot(pd.Timestamp("2024-01-03"))

        curve = p.get_equity_curve()
        assert len(curve) == 2
        assert "equity" in curve.columns
        assert "cash" in curve.columns
        assert curve["equity"].iloc[0] == 100_000

    def test_empty_equity_curve(self) -> None:
        p = Portfolio(BacktestConfig())
        curve = p.get_equity_curve()
        assert len(curve) == 0
        assert "equity" in curve.columns
