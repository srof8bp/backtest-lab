"""Tests for strategy base class and built-in strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_lab.strategy import BuyAndHold, EqualWeight, InverseVolatility, Strategy


class TestStrategyABC:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_custom_strategy_works(self) -> None:
        class FixedWeight(Strategy):
            def allocate(self, prices: pd.DataFrame) -> dict[str, float]:
                return {"A": 0.7, "B": 0.3}

        s = FixedWeight()
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame({"A": [100.0] * 5, "B": [50.0] * 5}, index=dates)
        weights = s.allocate(prices)
        assert weights == {"A": 0.7, "B": 0.3}


class TestValidateWeights:
    def test_filters_unknown_assets(self) -> None:
        s = EqualWeight()
        weights = {"A": 0.5, "UNKNOWN": 0.3}
        cleaned = s.validate_weights(weights, ["A", "B"])
        assert "UNKNOWN" not in cleaned
        assert "A" in cleaned

    def test_clamps_negative_to_zero(self) -> None:
        s = EqualWeight()
        weights = {"A": -0.5, "B": 0.8}
        cleaned = s.validate_weights(weights, ["A", "B"])
        assert cleaned["A"] == 0.0
        assert cleaned["B"] == 0.8

    def test_scales_down_over_one(self) -> None:
        s = EqualWeight()
        weights = {"A": 0.8, "B": 0.6}  # sum = 1.4
        cleaned = s.validate_weights(weights, ["A", "B"])
        total = sum(cleaned.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_preserves_valid_weights(self) -> None:
        s = EqualWeight()
        weights = {"A": 0.4, "B": 0.4}
        cleaned = s.validate_weights(weights, ["A", "B"])
        assert cleaned == {"A": 0.4, "B": 0.4}


class TestEqualWeight:
    def test_equal_weights(self) -> None:
        s = EqualWeight()
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame(
            {"A": [100.0] * 5, "B": [200.0] * 5, "C": [50.0] * 5},
            index=dates,
        )
        weights = s.allocate(prices)
        assert len(weights) == 3
        for w in weights.values():
            assert w == pytest.approx(1 / 3)

    def test_single_asset(self) -> None:
        s = EqualWeight()
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame({"ONLY": [100.0, 101.0, 102.0]}, index=dates)
        weights = s.allocate(prices)
        assert weights == {"ONLY": 1.0}

    def test_empty_returns_empty(self) -> None:
        s = EqualWeight()
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame(index=dates)
        weights = s.allocate(prices)
        assert weights == {}


class TestBuyAndHold:
    def test_allocates_once(self) -> None:
        s = BuyAndHold({"A": 0.6, "B": 0.4})
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame({"A": [100.0] * 5, "B": [50.0] * 5}, index=dates)

        first = s.allocate(prices)
        assert first == {"A": 0.6, "B": 0.4}

        # Second call returns empty (no rebalancing)
        second = s.allocate(prices)
        assert second == {}

    def test_default_equal_weight(self) -> None:
        s = BuyAndHold()
        dates = pd.bdate_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame({"X": [100.0] * 3, "Y": [200.0] * 3}, index=dates)
        weights = s.allocate(prices)
        assert weights["X"] == pytest.approx(0.5)
        assert weights["Y"] == pytest.approx(0.5)


class TestInverseVolatility:
    def test_lower_vol_gets_more_weight(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        # A is low vol, B is high vol
        a_prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, 100))
        b_prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.030, 100))
        prices = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)

        s = InverseVolatility(lookback=60)
        weights = s.allocate(prices)
        assert weights["A"] > weights["B"]

    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        prices = pd.DataFrame(
            {
                "A": 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 100)),
                "B": 200.0 * np.cumprod(1 + rng.normal(0, 0.02, 100)),
                "C": 50.0 * np.cumprod(1 + rng.normal(0, 0.015, 100)),
            },
            index=dates,
        )
        s = InverseVolatility()
        weights = s.allocate(prices)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)

    def test_insufficient_data_equal_weight(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=1, freq="B")
        prices = pd.DataFrame({"A": [100.0], "B": [200.0]}, index=dates)
        s = InverseVolatility()
        weights = s.allocate(prices)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)
