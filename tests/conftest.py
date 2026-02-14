"""Shared fixtures for backtest-lab tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dates_short() -> pd.DatetimeIndex:
    """20 business days starting 2024-01-02."""
    return pd.bdate_range("2024-01-02", periods=20, freq="B")


@pytest.fixture()
def dates_long() -> pd.DatetimeIndex:
    """504 business days (~2 years) starting 2022-01-03."""
    return pd.bdate_range("2022-01-03", periods=504, freq="B")


@pytest.fixture()
def prices_3asset(dates_short: pd.DatetimeIndex) -> pd.DataFrame:
    """3-asset price DataFrame with mild upward trend.

    Assets: AAPL (starts 150), GOOG (starts 2800), MSFT (starts 300).
    Each has a small daily return of ~0.1% with fixed seed noise.
    """
    rng = np.random.default_rng(42)
    n = len(dates_short)
    aapl = 150.0 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    goog = 2800.0 * np.cumprod(1 + rng.normal(0.001, 0.012, n))
    msft = 300.0 * np.cumprod(1 + rng.normal(0.001, 0.013, n))
    return pd.DataFrame(
        {"AAPL": aapl, "GOOG": goog, "MSFT": msft},
        index=dates_short,
    )


@pytest.fixture()
def prices_2asset(dates_short: pd.DatetimeIndex) -> pd.DataFrame:
    """2-asset price DataFrame. SPY (starts 450), TLT (starts 100)."""
    rng = np.random.default_rng(99)
    n = len(dates_short)
    spy = 450.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    tlt = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    return pd.DataFrame({"SPY": spy, "TLT": tlt}, index=dates_short)


@pytest.fixture()
def prices_flat(dates_short: pd.DatetimeIndex) -> pd.DataFrame:
    """2-asset DataFrame with constant prices (zero returns)."""
    n = len(dates_short)
    return pd.DataFrame(
        {"A": np.full(n, 100.0), "B": np.full(n, 50.0)},
        index=dates_short,
    )


@pytest.fixture()
def prices_long_3asset(dates_long: pd.DatetimeIndex) -> pd.DataFrame:
    """3-asset DataFrame with ~2 years of data for walk-forward tests."""
    rng = np.random.default_rng(123)
    n = len(dates_long)
    aapl = 150.0 * np.cumprod(1 + rng.normal(0.0004, 0.018, n))
    goog = 2800.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    msft = 300.0 * np.cumprod(1 + rng.normal(0.0005, 0.016, n))
    return pd.DataFrame(
        {"AAPL": aapl, "GOOG": goog, "MSFT": msft},
        index=dates_long,
    )


@pytest.fixture()
def simple_returns() -> pd.Series:
    """10-element return series with known values for metric tests."""
    return pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, -0.003, 0.018],
        index=pd.bdate_range("2024-01-02", periods=10, freq="B"),
    )


@pytest.fixture()
def simple_equity() -> pd.DataFrame:
    """Equity curve matching simple_returns (starting at 100_000)."""
    returns = pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, -0.003, 0.018],
        index=pd.bdate_range("2024-01-02", periods=10, freq="B"),
    )
    equity = 100_000 * (1 + returns).cumprod()
    return pd.DataFrame({"equity": equity})
