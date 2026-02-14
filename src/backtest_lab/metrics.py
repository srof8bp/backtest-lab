"""
backtest-lab: Performance metrics calculator.

All metric functions are pure, stateless, and operate on NumPy arrays
or Pandas Series for vectorized performance. No loops over individual
returns — everything is array math.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest_lab.types import BacktestConfig, Metrics, Trade

logger = logging.getLogger(__name__)


def compute_metrics(
    returns: pd.Series,  # type: ignore[type-arg]
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    config: BacktestConfig,
) -> Metrics:
    """Compute comprehensive performance metrics from backtest results.

    Args:
        returns: Series of portfolio period returns.
        equity_curve: DataFrame with 'equity' column.
        trades: List of all trades executed.
        config: Backtest configuration (for frequency and risk-free rate).

    Returns:
        Metrics dataclass with all computed values.
    """
    freq = int(config.frequency)
    rfr = config.risk_free_rate

    if len(returns) == 0:
        return Metrics(total_trades=len(trades))

    equity = equity_curve["equity"]
    initial = equity.iloc[0]
    final = equity.iloc[-1]

    total_return = final - initial
    total_return_pct = total_return / initial if initial != 0 else 0.0

    ann_return = _annualized_return(returns, freq)
    vol = _annualized_volatility(returns, freq)
    sharpe = _sharpe_ratio(returns, rfr, freq)
    sortino = _sortino_ratio(returns, rfr, freq)
    max_dd, max_dd_dur = _max_drawdown(equity)
    calmar = _calmar_ratio(ann_return, max_dd)
    wr = _win_rate(returns)
    pf = _profit_factor(returns)

    total_comm = sum(t.commission for t in trades)
    total_slip = sum(t.slippage_cost for t in trades)

    return Metrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=ann_return,
        volatility=vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        calmar_ratio=calmar,
        win_rate=wr,
        profit_factor=pf,
        total_trades=len(trades),
        total_commission=total_comm,
        total_slippage=total_slip,
    )


# ── Individual Metric Functions ──────────────────────────────────────────


def _annualized_return(returns: pd.Series, frequency: int) -> float:  # type: ignore[type-arg]
    """Annualized return from a series of period returns.

    Uses the compound annual growth rate (CAGR) formula:
        (1 + total_return) ^ (frequency / n_periods) - 1
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total = float((1 + returns).prod() - 1)
    if total <= -1.0:
        return -1.0
    return float((1 + total) ** (frequency / n) - 1)


def _annualized_volatility(returns: pd.Series, frequency: int) -> float:  # type: ignore[type-arg]
    """Annualized volatility (standard deviation of returns * sqrt(frequency))."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(frequency))


def _sharpe_ratio(
    returns: pd.Series, risk_free_rate: float, frequency: int  # type: ignore[type-arg]
) -> float:
    """Annualized Sharpe ratio.

    Formula: (mean_excess_return / std_excess_return) * sqrt(frequency)
    where excess_return = return - risk_free_rate_per_period
    """
    if len(returns) < 2:
        return 0.0

    rfr_per_period = (1 + risk_free_rate) ** (1 / frequency) - 1
    excess = returns - rfr_per_period
    std = float(excess.std())

    if std == 0:
        return 0.0

    return float(excess.mean() / std * np.sqrt(frequency))


def _sortino_ratio(
    returns: pd.Series, risk_free_rate: float, frequency: int  # type: ignore[type-arg]
) -> float:
    """Annualized Sortino ratio (uses downside deviation only).

    Same as Sharpe but the denominator only considers negative returns,
    so strategies with asymmetric upside are not penalized.
    """
    if len(returns) < 2:
        return 0.0

    rfr_per_period = (1 + risk_free_rate) ** (1 / frequency) - 1
    excess = returns - rfr_per_period
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = float(np.sqrt((downside**2).mean()))
    if downside_std == 0:
        return 0.0

    return float(excess.mean() / downside_std * np.sqrt(frequency))


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:  # type: ignore[type-arg]
    """Maximum drawdown and its duration in periods.

    Returns:
        Tuple of (max_drawdown_fraction, max_drawdown_duration).
        Drawdown is negative (e.g., -0.15 means 15% drawdown).
        Duration is the number of periods from peak to recovery
        (or to end of series if not yet recovered).
    """
    if len(equity) < 2:
        return 0.0, 0

    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    max_dd = float(drawdown.min())

    # Calculate duration: longest streak below the running max
    is_underwater = drawdown < 0
    if not is_underwater.any():
        return 0.0, 0

    # Find streaks of being underwater
    groups = (~is_underwater).cumsum()
    underwater_groups = groups[is_underwater]
    if len(underwater_groups) == 0:
        return max_dd, 0

    max_duration = int(underwater_groups.value_counts().max())
    return max_dd, max_duration


def _calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calmar ratio: annualized return / abs(max drawdown).

    A higher Calmar means better return per unit of drawdown risk.
    """
    if max_drawdown == 0:
        return 0.0
    return annualized_return / abs(max_drawdown)


def _win_rate(returns: pd.Series) -> float:  # type: ignore[type-arg]
    """Fraction of periods with positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def _profit_factor(returns: pd.Series) -> float:  # type: ignore[type-arg]
    """Gross profits divided by gross losses.

    A profit factor > 1.0 means the strategy is profitable.
    Returns float('inf') if there are no losses.
    """
    if len(returns) == 0:
        return 0.0

    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return gains / losses
