"""
backtest-lab: Strategy base class and built-in allocation strategies.

A strategy receives the full price history up to the current rebalance
date and returns target portfolio weights. Weights are a dict mapping
asset names to fractions (0.0 to 1.0). Any remainder is held as cash.

Built-in strategies:
    - EqualWeight: Equal allocation across all assets
    - BuyAndHold: Allocate once using initial weights, never rebalance
    - InverseVolatility: Weight inversely proportional to recent volatility
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from backtest_lab.types import Weights

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for portfolio allocation strategies.

    Subclass and implement ``allocate()`` to define custom strategies.

    Example::

        class MomentumStrategy(Strategy):
            def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
                returns = prices.pct_change(252).iloc[-1]
                top = returns.nlargest(3).index.tolist()
                return {asset: 1 / 3 for asset in top}

        result = Backtest(prices, strategy=MomentumStrategy()).run()
    """

    @abstractmethod
    def allocate(self, prices: pd.DataFrame) -> Weights:
        """Return target weights for each asset.

        Args:
            prices: Price DataFrame with DatetimeIndex. Contains all
                history up to (and including) the current rebalance date.

        Returns:
            Dict mapping asset names to target weights (0.0 to 1.0).
            Weights should sum to <= 1.0. Remainder is held as cash.
        """

    def validate_weights(self, weights: Weights, assets: list[str]) -> Weights:
        """Validate and clamp weights returned by allocate().

        Called by the engine after each ``allocate()`` call. Ensures
        weights are non-negative, apply only to known assets, and
        sum to at most 1.0.

        Args:
            weights: Raw weights from allocate().
            assets: Valid asset names.

        Returns:
            Cleaned weights dict.
        """
        cleaned: Weights = {}
        for asset, w in weights.items():
            if asset not in assets:
                logger.warning("Unknown asset '%s' in weights, skipping", asset)
                continue
            if w < 0:
                logger.warning("Negative weight for '%s' clamped to 0", asset)
                w = 0.0
            cleaned[asset] = w

        total = sum(cleaned.values())
        if total > 1.0:
            # Scale down proportionally
            for asset in cleaned:
                cleaned[asset] /= total

        return cleaned


class EqualWeight(Strategy):
    """Equal weight across all assets.

    Divides the portfolio equally among all available assets
    at each rebalance.

    Example::

        result = Backtest(prices, strategy=EqualWeight()).run()
    """

    def allocate(self, prices: pd.DataFrame) -> Weights:
        assets = prices.columns.tolist()
        n = len(assets)
        if n == 0:
            return {}
        w = 1.0 / n
        return {asset: w for asset in assets}


class BuyAndHold(Strategy):
    """Buy-and-hold with optional initial weights.

    Allocates on the first call using the provided weights (or equal
    weight if none given), then returns empty weights on subsequent
    calls to prevent rebalancing.

    Args:
        initial_weights: Starting allocation. If None, uses equal weight.

    Example::

        strategy = BuyAndHold({"AAPL": 0.6, "GOOG": 0.4})
        result = Backtest(prices, strategy=strategy).run()
    """

    def __init__(self, initial_weights: Weights | None = None) -> None:
        self._initial_weights = initial_weights
        self._allocated = False

    def allocate(self, prices: pd.DataFrame) -> Weights:
        if self._allocated:
            return {}  # No rebalancing

        self._allocated = True

        if self._initial_weights is not None:
            return dict(self._initial_weights)

        # Default to equal weight
        assets = prices.columns.tolist()
        n = len(assets)
        if n == 0:
            return {}
        w = 1.0 / n
        return {asset: w for asset in assets}


class InverseVolatility(Strategy):
    """Inverse volatility weighting.

    Assets with lower recent volatility receive a larger weight.
    This is a simple risk-parity-like approach.

    Args:
        lookback: Number of periods to compute volatility over.

    Example::

        strategy = InverseVolatility(lookback=60)
        result = Backtest(prices, strategy=strategy).run()
    """

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def allocate(self, prices: pd.DataFrame) -> Weights:
        if len(prices) < 2:
            # Not enough data — fall back to equal weight
            assets = prices.columns.tolist()
            n = len(assets)
            return {a: 1.0 / n for a in assets} if n > 0 else {}

        window = prices.iloc[-self._lookback:] if len(prices) >= self._lookback else prices
        returns = window.pct_change().dropna()

        if len(returns) < 2:
            assets = prices.columns.tolist()
            n = len(assets)
            return {a: 1.0 / n for a in assets} if n > 0 else {}

        vol = returns.std()

        # Replace zero volatility with a small number to avoid division by zero
        vol = vol.replace(0, np.nan).fillna(vol[vol > 0].min() if (vol > 0).any() else 1.0)

        inv_vol = 1.0 / vol
        total = inv_vol.sum()

        if total == 0:
            assets = prices.columns.tolist()
            n = len(assets)
            return {a: 1.0 / n for a in assets} if n > 0 else {}

        weights = inv_vol / total
        return {asset: float(w) for asset, w in weights.items()}
