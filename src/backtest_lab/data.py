"""
backtest-lab: Data validation and preprocessing.

Validates price DataFrames before backtesting. Catches common data issues
early with clear error messages rather than letting them cause cryptic
errors deep in the engine.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a price DataFrame for backtesting.

    Performs the following checks and transformations:
    - Ensures the index is a DatetimeIndex
    - Sorts by date (ascending)
    - Removes duplicate dates (keeps last)
    - Checks all values are numeric
    - Forward-fills NaN gaps (warns if > 5% missing)
    - Drops any columns that are entirely NaN
    - Verifies at least 2 rows remain

    Args:
        prices: DataFrame with DatetimeIndex and asset prices as columns.

    Returns:
        Cleaned and validated DataFrame.

    Raises:
        TypeError: If prices is not a DataFrame or index is not datetime.
        ValueError: If the DataFrame is empty or has fewer than 2 rows.
    """
    if not isinstance(prices, pd.DataFrame):
        msg = f"Expected pd.DataFrame, got {type(prices).__name__}"
        raise TypeError(msg)

    if prices.empty:
        msg = "Price DataFrame is empty"
        raise ValueError(msg)

    # Ensure DatetimeIndex
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        except (ValueError, TypeError) as exc:
            msg = (
                "Index must be a DatetimeIndex or convertible to one. "
                f"Got index dtype: {prices.index.dtype}"
            )
            raise TypeError(msg) from exc

    prices = prices.copy()

    # Sort by date
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    # Remove duplicate dates
    if prices.index.duplicated().any():
        n_dups = prices.index.duplicated().sum()
        logger.warning("Removed %d duplicate dates from price data", n_dups)
        prices = prices[~prices.index.duplicated(keep="last")]

    # Drop all-NaN columns
    all_nan_cols = prices.columns[prices.isna().all()]
    if len(all_nan_cols) > 0:
        logger.warning("Dropped all-NaN columns: %s", list(all_nan_cols))
        prices = prices.drop(columns=all_nan_cols)

    # Check for non-numeric columns
    non_numeric = prices.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        msg = f"Non-numeric columns found: {list(non_numeric)}"
        raise ValueError(msg)

    # Forward-fill NaN gaps with warning
    nan_pct = prices.isna().sum().sum() / prices.size
    if nan_pct > 0:
        if nan_pct > 0.05:
            warnings.warn(
                f"{nan_pct:.1%} of price data is missing. "
                "Forward-filling, but results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        prices = prices.ffill()
        # Drop any leading NaNs that can't be forward-filled
        prices = prices.dropna()

    # Final size check
    if len(prices) < 2:
        msg = f"Need at least 2 data points, got {len(prices)}"
        raise ValueError(msg)

    if len(prices.columns) == 0:
        msg = "No valid asset columns remain after validation"
        raise ValueError(msg)

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple period returns from a price DataFrame.

    Args:
        prices: Validated price DataFrame with DatetimeIndex.

    Returns:
        DataFrame of simple returns (same shape, first row dropped).
    """
    returns = prices.pct_change()
    return returns.iloc[1:]  # Drop the NaN first row
