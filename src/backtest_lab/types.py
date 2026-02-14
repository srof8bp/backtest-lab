"""
backtest-lab: Shared data types, enums, and dataclasses.

All public types used across the library are defined here. This module
has no dependencies on other backtest_lab modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import pandas as pd

# ── Enums ────────────────────────────────────────────────────────────────


class Frequency(IntEnum):
    """Trading calendar frequency for annualization.

    The integer value is the number of trading periods per year.
    """

    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12


class RebalanceFrequency(Enum):
    """How often the portfolio rebalances to target weights."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    NEVER = "never"


class SizingMethod(Enum):
    """Portfolio weight allocation methods."""

    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    RISK_PARITY = "risk_parity"
    CUSTOM = "custom"


# ── Type Aliases ─────────────────────────────────────────────────────────

Weights = dict[str, float]
"""Asset name -> target weight (0.0 to 1.0). Remainder is cash."""


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio value in dollars.
        commission: Per-trade commission as a fraction (0.001 = 0.1%).
        slippage: Per-trade slippage as a fraction (0.0005 = 0.05%).
        rebalance_frequency: How often to rebalance to target weights.
        frequency: Data frequency for annualization calculations.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        weight_tolerance: Minimum drift before triggering rebalance (0.05 = 5%).
    """

    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    frequency: Frequency = Frequency.DAILY
    risk_free_rate: float = 0.05
    weight_tolerance: float = 0.05


# ── Trade ────────────────────────────────────────────────────────────────


@dataclass
class Trade:
    """A completed trade (buy or sell) in the backtest.

    Attributes:
        date: Date the trade was executed.
        asset: Asset symbol (e.g., "AAPL").
        side: "buy" or "sell".
        shares: Number of shares traded (always positive).
        price: Execution price per share (after slippage).
        commission: Commission paid for this trade.
        slippage_cost: Total slippage cost for this trade.
    """

    date: pd.Timestamp
    asset: str
    side: str
    shares: float
    price: float
    commission: float = 0.0
    slippage_cost: float = 0.0

    @property
    def notional(self) -> float:
        """Total dollar value of the trade."""
        return self.shares * self.price

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return self.commission + self.slippage_cost


# ── Position ─────────────────────────────────────────────────────────────


@dataclass
class Position:
    """An open position in the portfolio.

    Attributes:
        asset: Asset symbol.
        shares: Number of shares held.
        avg_cost: Volume-weighted average cost basis per share.
    """

    asset: str
    shares: float
    avg_cost: float

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.shares * self.avg_cost

    def market_value(self, price: float) -> float:
        """Current market value at the given price."""
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        """Unrealized profit/loss at the given price."""
        return (price - self.avg_cost) * self.shares


# ── Metrics ──────────────────────────────────────────────────────────────


@dataclass
class Metrics:
    """Comprehensive performance metrics from a backtest.

    Attributes:
        total_return: Total dollar return.
        total_return_pct: Total return as a percentage.
        annualized_return: Annualized return percentage.
        volatility: Annualized volatility (std dev of returns).
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio (downside deviation).
        max_drawdown: Maximum peak-to-trough drawdown as a fraction (negative).
        max_drawdown_duration: Maximum drawdown duration in periods.
        calmar_ratio: Annualized return / abs(max drawdown).
        win_rate: Fraction of periods with positive return.
        profit_factor: Gross profits / gross losses.
        total_trades: Total number of trades executed.
        total_commission: Total commission paid.
        total_slippage: Total slippage cost.
        turnover: Average portfolio turnover per period.
    """

    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    turnover: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to a flat dictionary."""
        return {
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return": round(self.annualized_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": round(self.calmar_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "total_commission": round(self.total_commission, 2),
            "total_slippage": round(self.total_slippage, 2),
            "turnover": round(self.turnover, 4),
        }

    def __repr__(self) -> str:
        return (
            f"Metrics(return={self.total_return_pct:+.2%}, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown:.2%}, "
            f"trades={self.total_trades})"
        )


# ── Backtest Result ──────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    Attributes:
        metrics: Computed performance metrics.
        equity_curve: DataFrame with columns [equity, cash, positions_value].
        weights_history: DataFrame of target weights over time (assets as columns).
        trades: List of all trades executed.
        returns: Series of portfolio period returns.
        config: Configuration used for this backtest.
    """

    metrics: Metrics
    equity_curve: pd.DataFrame
    weights_history: pd.DataFrame
    trades: list[Trade]
    returns: pd.Series  # type: ignore[type-arg]
    config: BacktestConfig

    @property
    def monthly_returns(self) -> pd.DataFrame:
        """Monthly returns pivot table (years as rows, months as columns)."""
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table = pd.DataFrame(
            {"year": monthly.index.year, "month": monthly.index.month, "return": monthly.values}
        )
        return table.pivot(index="year", columns="month", values="return")

    @property
    def annual_returns(self) -> pd.Series:  # type: ignore[type-arg]
        """Annual returns series."""
        return self.returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    @property
    def drawdown_series(self) -> pd.Series:  # type: ignore[type-arg]
        """Drawdown time series (negative values = underwater)."""
        equity = self.equity_curve["equity"]
        running_max = equity.expanding().max()
        return (equity - running_max) / running_max

    def __repr__(self) -> str:
        n_days = len(self.equity_curve)
        return (
            f"BacktestResult({n_days} periods, {self.metrics.total_trades} trades, "
            f"return={self.metrics.total_return_pct:+.2%}, "
            f"sharpe={self.metrics.sharpe_ratio:.2f})"
        )


# ── Walk-Forward Types ───────────────────────────────────────────────────


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        train_periods: Number of periods in the training window.
        test_periods: Number of periods in the test window.
        step_periods: Number of periods to advance the window each step.
        min_windows: Minimum number of train/test windows required.
    """

    train_periods: int = 252
    test_periods: int = 63
    step_periods: int = 63
    min_windows: int = 4


@dataclass
class WalkForwardWindow:
    """A single train/test window result.

    Attributes:
        window_id: Sequential window identifier.
        train_start: Start date of training period.
        train_end: End date of training period.
        test_start: Start date of test period.
        test_end: End date of test period.
        train_metrics: Metrics from the training backtest.
        test_metrics: Metrics from the test backtest.
    """

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Metrics | None = None
    test_metrics: Metrics | None = None


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward analysis result.

    Attributes:
        windows: List of individual window results.
        oos_equity_curve: Combined out-of-sample equity curve.
        oos_metrics: Metrics computed on combined out-of-sample returns.
        consistency: Fraction of windows with positive out-of-sample returns.
        degradation: Average train Sharpe minus test Sharpe.
    """

    windows: list[WalkForwardWindow]
    oos_equity_curve: pd.Series  # type: ignore[type-arg]
    oos_metrics: Metrics
    consistency: float = 0.0
    degradation: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult({len(self.windows)} windows, "
            f"consistency={self.consistency:.0%}, "
            f"oos_sharpe={self.oos_metrics.sharpe_ratio:.2f})"
        )
