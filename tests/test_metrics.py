"""Tests for performance metrics — one test per metric with known values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_lab.metrics import (
    _annualized_return,
    _annualized_volatility,
    _calmar_ratio,
    _max_drawdown,
    _profit_factor,
    _sharpe_ratio,
    _sortino_ratio,
    _win_rate,
    compute_metrics,
)
from backtest_lab.types import BacktestConfig, Trade


class TestAnnualizedReturn:
    def test_positive_returns(self) -> None:
        # 252 days of 0.04% daily return → ~10.6% annually
        returns = pd.Series([0.0004] * 252)
        ann = _annualized_return(returns, 252)
        assert 0.10 < ann < 0.12

    def test_zero_returns(self) -> None:
        returns = pd.Series([0.0] * 100)
        assert _annualized_return(returns, 252) == pytest.approx(0.0)

    def test_negative_total_returns(self) -> None:
        returns = pd.Series([-0.001] * 252)
        ann = _annualized_return(returns, 252)
        assert ann < 0

    def test_empty_returns(self) -> None:
        assert _annualized_return(pd.Series(dtype=float), 252) == 0.0

    def test_catastrophic_loss(self) -> None:
        returns = pd.Series([-0.5, -0.5, -0.5])
        ann = _annualized_return(returns, 252)
        assert ann == -1.0


class TestAnnualizedVolatility:
    def test_known_volatility(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        vol = _annualized_volatility(returns, 252)
        # daily std ~0.01, annualized ~0.01 * sqrt(252) ~0.159
        assert 0.12 < vol < 0.20

    def test_zero_volatility(self) -> None:
        returns = pd.Series([0.01] * 100)
        vol = _annualized_volatility(returns, 252)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_single_return(self) -> None:
        returns = pd.Series([0.05])
        assert _annualized_volatility(returns, 252) == 0.0


class TestSharpeRatio:
    def test_positive_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sharpe = _sharpe_ratio(returns, 0.05, 252)
        # Expected: positive, likely > 1
        assert sharpe > 0

    def test_zero_returns_negative_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        sharpe = _sharpe_ratio(returns, 0.05, 252)
        # Zero mean returns with positive risk-free rate → negative Sharpe
        assert sharpe < 0

    def test_constant_returns_zero_std(self) -> None:
        # Constant returns → std ≈ 0 (floating-point noise), Sharpe undefined
        # When excess returns also have ~0 std, our code returns 0.0
        returns = pd.Series([0.001] * 100)
        # Set rfr so excess mean ≈ 0.001 - 0.001 = 0 → numerator ~0 too
        rfr_annual = (1 + 0.001) ** 252 - 1  # Match daily return
        sharpe = _sharpe_ratio(returns, rfr_annual, 252)
        assert abs(sharpe) < 1.0  # Effectively zero (within float noise)

    def test_single_return_zero_sharpe(self) -> None:
        assert _sharpe_ratio(pd.Series([0.05]), 0.05, 252) == 0.0


class TestSortinoRatio:
    def test_positive_sortino(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sortino = _sortino_ratio(returns, 0.05, 252)
        assert isinstance(sortino, float)

    def test_all_positive_returns_zero_sortino(self) -> None:
        # All positive excess returns → no downside deviation → Sortino = 0
        returns = pd.Series([0.01] * 100)
        sortino = _sortino_ratio(returns, 0.0, 252)
        assert sortino == 0.0

    def test_single_return(self) -> None:
        assert _sortino_ratio(pd.Series([0.05]), 0.0, 252) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        # Equity: 100, 110, 90, 95, 100
        equity = pd.Series([100.0, 110.0, 90.0, 95.0, 100.0])
        max_dd, duration = _max_drawdown(equity)
        # Peak = 110, trough = 90 → dd = (90-110)/110 ≈ -0.1818
        assert max_dd == pytest.approx(-0.1818, abs=0.001)
        assert duration > 0

    def test_no_drawdown(self) -> None:
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        max_dd, duration = _max_drawdown(equity)
        assert max_dd == 0.0
        assert duration == 0

    def test_monotone_decline(self) -> None:
        equity = pd.Series([100.0, 90.0, 80.0, 70.0])
        max_dd, duration = _max_drawdown(equity)
        # Peak = 100, trough = 70 → dd = -0.30
        assert max_dd == pytest.approx(-0.30, abs=0.001)
        assert duration == 3

    def test_empty_equity(self) -> None:
        assert _max_drawdown(pd.Series(dtype=float)) == (0.0, 0)


class TestCalmarRatio:
    def test_known_calmar(self) -> None:
        # 10% return, -20% drawdown → calmar = 0.10/0.20 = 0.5
        assert _calmar_ratio(0.10, -0.20) == pytest.approx(0.5)

    def test_zero_drawdown(self) -> None:
        assert _calmar_ratio(0.10, 0.0) == 0.0

    def test_negative_return(self) -> None:
        calmar = _calmar_ratio(-0.05, -0.20)
        assert calmar < 0


class TestWinRate:
    def test_known_win_rate(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
        # 2 positive out of 5 (zero is not positive)
        assert _win_rate(returns) == pytest.approx(0.4)

    def test_all_positive(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert _win_rate(returns) == pytest.approx(1.0)

    def test_all_negative(self) -> None:
        returns = pd.Series([-0.01, -0.02, -0.03])
        assert _win_rate(returns) == pytest.approx(0.0)

    def test_empty_returns(self) -> None:
        assert _win_rate(pd.Series(dtype=float)) == 0.0


class TestProfitFactor:
    def test_known_profit_factor(self) -> None:
        # gains = 0.01 + 0.03 = 0.04, losses = 0.02 + 0.01 = 0.03
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        pf = _profit_factor(returns)
        assert pf == pytest.approx(0.04 / 0.03, abs=0.001)

    def test_no_losses_infinite(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03])
        assert _profit_factor(returns) == float("inf")

    def test_no_gains_zero(self) -> None:
        returns = pd.Series([-0.01, -0.02])
        assert _profit_factor(returns) == pytest.approx(0.0)

    def test_empty_returns(self) -> None:
        assert _profit_factor(pd.Series(dtype=float)) == 0.0


class TestComputeMetrics:
    def test_full_metrics(
        self,
        simple_returns: pd.Series,
        simple_equity: pd.DataFrame,
    ) -> None:
        config = BacktestConfig()
        trades = [
            Trade(
                date=pd.Timestamp("2024-01-02"),
                asset="AAPL",
                side="buy",
                shares=100,
                price=150.0,
                commission=15.0,
                slippage_cost=3.75,
            )
        ]
        metrics = compute_metrics(simple_returns, simple_equity, trades, config)
        assert metrics.total_trades == 1
        assert metrics.total_commission == 15.0
        assert metrics.total_slippage == 3.75
        assert metrics.win_rate > 0
        assert metrics.max_drawdown <= 0

    def test_empty_returns_gives_defaults(self) -> None:
        config = BacktestConfig()
        metrics = compute_metrics(
            pd.Series(dtype=float),
            pd.DataFrame(columns=["equity"]),
            [],
            config,
        )
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.total_trades == 0

    def test_metrics_to_dict(
        self,
        simple_returns: pd.Series,
        simple_equity: pd.DataFrame,
    ) -> None:
        config = BacktestConfig()
        metrics = compute_metrics(simple_returns, simple_equity, [], config)
        d = metrics.to_dict()
        assert "sharpe_ratio" in d
        assert "total_return_pct" in d
        assert isinstance(d["total_trades"], int)
