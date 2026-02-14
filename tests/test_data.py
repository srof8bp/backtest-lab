"""Tests for data validation and preprocessing."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from backtest_lab.data import compute_returns, validate_prices


class TestValidatePrices:
    """validate_prices() edge cases and transformations."""

    def test_valid_dataframe_passes(self, prices_3asset: pd.DataFrame) -> None:
        result = validate_prices(prices_3asset)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(prices_3asset)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            validate_prices([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_prices(pd.DataFrame())

    def test_converts_string_index_to_datetime(self) -> None:
        df = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0]},
            index=["2024-01-02", "2024-01-03", "2024-01-04"],
        )
        result = validate_prices(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_rejects_non_datetime_index(self) -> None:
        df = pd.DataFrame({"A": [100.0, 101.0]}, index=["foo", "bar"])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_prices(df)

    def test_sorts_unsorted_dates(self) -> None:
        dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"A": [102.0, 100.0, 101.0]}, index=dates)
        result = validate_prices(df)
        assert result.index.is_monotonic_increasing

    def test_removes_duplicate_dates(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({"A": [100.0, 100.5, 101.0]}, index=dates)
        result = validate_prices(df)
        assert not result.index.duplicated().any()
        assert len(result) == 2

    def test_drops_all_nan_columns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [np.nan] * 5},
            index=dates,
        )
        result = validate_prices(df)
        assert "B" not in result.columns
        assert "A" in result.columns

    def test_rejects_non_numeric_columns(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        df = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": ["x", "y", "z"]}, index=dates)
        with pytest.raises(ValueError, match="Non-numeric"):
            validate_prices(df)

    def test_forward_fills_nan_values(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [100.0, np.nan, 102.0, np.nan, 104.0]},
            index=dates,
        )
        result = validate_prices(df)
        assert not result.isna().any().any()
        assert result.loc[dates[1], "A"] == 100.0  # forward-filled

    def test_warns_on_high_nan_percentage(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        values = [100.0] + [np.nan] * 8 + [110.0]
        df = pd.DataFrame({"A": values}, index=dates)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_prices(df)
            nan_warnings = [x for x in w if "missing" in str(x.message).lower()]
            assert len(nan_warnings) >= 1

    def test_rejects_fewer_than_2_rows(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=1, freq="B")
        df = pd.DataFrame({"A": [100.0]}, index=dates)
        with pytest.raises(ValueError, match="at least 2"):
            validate_prices(df)


class TestComputeReturns:
    """compute_returns() basic behavior."""

    def test_returns_shape(self, prices_3asset: pd.DataFrame) -> None:
        returns = compute_returns(prices_3asset)
        assert len(returns) == len(prices_3asset) - 1
        assert list(returns.columns) == list(prices_3asset.columns)

    def test_returns_values(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        df = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=dates)
        returns = compute_returns(df)
        assert abs(returns.iloc[0]["A"] - 0.10) < 1e-10
        assert abs(returns.iloc[1]["A"] - (-0.10)) < 1e-10
