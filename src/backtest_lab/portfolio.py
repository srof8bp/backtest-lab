"""
backtest-lab: Portfolio state management and trade execution.

Tracks cash, positions, and trades. Handles rebalancing by computing
the delta between current and target weights, then executing sells
first (to free cash) followed by buys. Applies commission and slippage
to each trade.
"""

from __future__ import annotations

import logging

import pandas as pd

from backtest_lab.types import BacktestConfig, Position, Trade, Weights

logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks portfolio state: cash, positions, trades, equity curve.

    This is the accounting engine. It receives target weights from a
    strategy and executes rebalancing trades, applying commission
    and slippage to each trade.

    Example::

        portfolio = Portfolio(BacktestConfig(initial_capital=100_000))
        trades = portfolio.rebalance(
            {"AAPL": 0.5, "GOOG": 0.5},
            {"AAPL": 150.0, "GOOG": 2800.0},
            pd.Timestamp("2024-01-02"),
        )
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._cash = config.initial_capital
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._snapshots: list[dict[str, float]] = []
        self._current_prices: dict[str, float] = {}

    # ── State Properties ─────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Current open positions (read-only view)."""
        return dict(self._positions)

    @property
    def trades(self) -> list[Trade]:
        """All executed trades."""
        return list(self._trades)

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Total portfolio value (cash + positions at current prices)."""
        p = prices or self._current_prices
        positions_value = sum(
            pos.market_value(p.get(asset, pos.avg_cost))
            for asset, pos in self._positions.items()
        )
        return self._cash + positions_value

    def positions_value(self, prices: dict[str, float] | None = None) -> float:
        """Total value of all open positions."""
        p = prices or self._current_prices
        return sum(
            pos.market_value(p.get(asset, pos.avg_cost))
            for asset, pos in self._positions.items()
        )

    def current_weights(self, prices: dict[str, float] | None = None) -> Weights:
        """Current portfolio weights based on market values."""
        total = self.equity(prices)
        if total == 0:
            return {}
        p = prices or self._current_prices
        return {
            asset: pos.market_value(p.get(asset, pos.avg_cost)) / total
            for asset, pos in self._positions.items()
        }

    # ── Core Operations ──────────────────────────────────────────────

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current market prices for all assets."""
        self._current_prices = prices

    def rebalance(
        self,
        target_weights: Weights,
        prices: dict[str, float],
        date: pd.Timestamp,
    ) -> list[Trade]:
        """Rebalance the portfolio to target weights.

        Sells first (to free cash), then buys. Each trade has commission
        and slippage applied.

        Args:
            target_weights: Target allocation {asset: weight}.
            prices: Current prices {asset: price}.
            date: Current date for trade records.

        Returns:
            List of trades executed during this rebalance.
        """
        self._current_prices = prices
        total_equity = self.equity(prices)
        rebalance_trades: list[Trade] = []

        # Compute target shares for each asset
        deltas: dict[str, float] = {}
        for asset, target_w in target_weights.items():
            if asset not in prices:
                logger.warning("No price for %s, skipping", asset)
                continue

            price = prices[asset]
            if price <= 0:
                continue

            target_value = total_equity * target_w
            target_shares = target_value / price
            current_shares = self._positions[asset].shares if asset in self._positions else 0.0
            delta = target_shares - current_shares

            if abs(delta) * price > 1.0:  # Skip tiny trades (< $1)
                deltas[asset] = delta

        # Sell positions not in target
        for asset in list(self._positions):
            if asset not in target_weights or target_weights.get(asset, 0) == 0:
                pos = self._positions[asset]
                price = prices.get(asset, pos.avg_cost)
                trade = self._execute_trade(asset, -pos.shares, price, date)
                if trade:
                    rebalance_trades.append(trade)

        # Execute sells first (delta < 0), then buys (delta > 0)
        sells = {a: d for a, d in deltas.items() if d < 0}
        buys = {a: d for a, d in deltas.items() if d > 0}

        for asset, delta in sells.items():
            trade = self._execute_trade(asset, delta, prices[asset], date)
            if trade:
                rebalance_trades.append(trade)

        for asset, delta in buys.items():
            trade = self._execute_trade(asset, delta, prices[asset], date)
            if trade:
                rebalance_trades.append(trade)

        self._trades.extend(rebalance_trades)
        return rebalance_trades

    def record_snapshot(self, date: pd.Timestamp) -> None:
        """Record a point-in-time snapshot of portfolio state."""
        eq = self.equity()
        pos_val = self.positions_value()
        self._snapshots.append({
            "date": date,
            "equity": eq,
            "cash": self._cash,
            "positions_value": pos_val,
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """Build equity curve DataFrame from recorded snapshots."""
        if not self._snapshots:
            return pd.DataFrame(columns=["equity", "cash", "positions_value"])

        df = pd.DataFrame(self._snapshots)
        df = df.set_index("date")
        return df

    # ── Internal ─────────────────────────────────────────────────────

    def _execute_trade(
        self,
        asset: str,
        shares_delta: float,
        price: float,
        date: pd.Timestamp,
    ) -> Trade | None:
        """Execute a single trade (buy or sell).

        Args:
            asset: Asset symbol.
            shares_delta: Positive for buy, negative for sell.
            price: Market price before slippage.
            date: Trade date.

        Returns:
            Trade record, or None if the trade is too small.
        """
        if abs(shares_delta) < 1e-10:
            return None

        side = "buy" if shares_delta > 0 else "sell"
        abs_shares = abs(shares_delta)

        # Apply slippage
        exec_price = self._apply_slippage(price, side)
        slippage_cost = abs(exec_price - price) * abs_shares

        # Apply commission
        notional = exec_price * abs_shares
        commission = self._apply_commission(notional)

        if side == "buy":
            total_cost = notional + commission
            if total_cost > self._cash:
                # Reduce shares to what we can afford
                affordable = (self._cash - commission) / exec_price
                if affordable <= 0:
                    return None
                abs_shares = affordable
                notional = exec_price * abs_shares
                commission = self._apply_commission(notional)
                total_cost = notional + commission

            self._cash -= total_cost

            # Update position
            if asset in self._positions:
                pos = self._positions[asset]
                new_shares = pos.shares + abs_shares
                pos.avg_cost = (
                    (pos.avg_cost * pos.shares + exec_price * abs_shares) / new_shares
                )
                pos.shares = new_shares
            else:
                self._positions[asset] = Position(
                    asset=asset, shares=abs_shares, avg_cost=exec_price
                )
        else:
            # Sell
            sell_shares = min(abs_shares, self._positions.get(asset, Position(asset, 0, 0)).shares)
            if sell_shares <= 0:
                return None
            abs_shares = sell_shares

            proceeds = exec_price * abs_shares - commission
            self._cash += proceeds

            # Update position
            if asset in self._positions:
                self._positions[asset].shares -= abs_shares
                if self._positions[asset].shares < 1e-10:
                    del self._positions[asset]

        return Trade(
            date=date,
            asset=asset,
            side=side,
            shares=abs_shares,
            price=exec_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to a price. Buys pay more, sells receive less."""
        slippage = self._config.slippage
        if side == "buy":
            return price * (1 + slippage)
        return price * (1 - slippage)

    def _apply_commission(self, notional: float) -> float:
        """Calculate commission from the notional trade value."""
        return notional * self._config.commission
