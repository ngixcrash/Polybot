"""Orderbook microstructure indicators for Polymarket YES/NO books.
All signals normalized to +/-1.0 (T3 tier weight).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from orderbook import OrderbookManager

log = logging.getLogger(__name__)


@dataclass
class MicrostructureAnalyzer:
    orderbook: Optional[OrderbookManager] = None

    def get_signal_components(self, direction_hint: str = "") -> dict[str, float]:
        """Turn orderbook state into directional scores for all 4 micro indicators."""
        if not self.orderbook:
            return {}

        components = {}
        components["micro_depth_imbalance"] = self._analyze_depth_imbalance()
        components["micro_spread_dynamics"] = self._analyze_spread_dynamics(direction_hint)
        components["micro_microprice"] = self._analyze_microprice()
        components["micro_smart_money"] = self._analyze_smart_money()

        return components

    def _analyze_depth_imbalance(self) -> float:
        """Compare YES vs NO bid depth to gauge directional bias."""
        yes_book = self.orderbook.yes_book
        no_book = self.orderbook.no_book

        yes_imb = yes_book.depth_imbalance  # +1 = bid-heavy (buyers)
        no_imb = no_book.depth_imbalance

        # If YES has more bid depth (buyers) → bullish (+)
        # If NO has more bid depth (buyers) → bearish (-)
        # Combine: YES buyer pressure minus NO buyer pressure
        combined = (yes_imb - no_imb) / 2.0

        if abs(combined) < 0.15:
            return 0.0
        return max(-1.0, min(1.0, combined * 2.0))

    def _analyze_spread_dynamics(self, direction_hint: str) -> float:
        """Check if the spread is tightening or widening on the relevant book."""
        # Narrowing = conviction, widening = uncertainty
        if direction_hint == "up":
            book = self.orderbook.yes_book
        elif direction_hint == "down":
            book = self.orderbook.no_book
        else:
            book = self.orderbook.yes_book

        spread_trend = book.get_spread_trend(lookback_seconds=30.0)

        if spread_trend < -0.1:
            return 1.0   # Narrowing — conviction, good for direction
        elif spread_trend > 0.1:
            return -1.0  # Widening — uncertainty, bad for direction
        elif spread_trend < -0.03:
            return 0.5
        elif spread_trend > 0.03:
            return -0.5
        return 0.0

    def _analyze_microprice(self) -> float:
        """Microprice vs midprice — microprice > midprice = buying pressure."""
        yes_book = self.orderbook.yes_book
        no_book = self.orderbook.no_book

        yes_micro = yes_book.microprice
        yes_mid = yes_book.mid_price
        no_micro = no_book.microprice
        no_mid = no_book.mid_price

        signal = 0.0

        # YES microprice above mid → buy pressure on YES → bullish
        if yes_micro is not None and yes_mid is not None and yes_mid > 0:
            yes_dev = (yes_micro - yes_mid) / yes_mid
            if yes_dev > 0.005:
                signal += 0.5  # Microprice tilted toward asks (thin asks, price up)
            elif yes_dev < -0.005:
                signal -= 0.5

        # NO microprice above mid → buy pressure on NO → bearish
        if no_micro is not None and no_mid is not None and no_mid > 0:
            no_dev = (no_micro - no_mid) / no_mid
            if no_dev > 0.005:
                signal -= 0.5  # Buyers on NO = bearish
            elif no_dev < -0.005:
                signal += 0.5

        return max(-1.0, min(1.0, signal))

    def _analyze_smart_money(self) -> float:
        """Aggregate large order events from both books into a directional signal."""
        # Large bids appearing on YES = bullish smart money
        now = time.time()
        lookback = 60.0  # Last 60 seconds

        bullish_score = 0.0
        bearish_score = 0.0

        # YES book events
        for event in self.orderbook.yes_book.large_order_events:
            if event["timestamp"] < now - lookback:
                continue
            if event["type"] == "appear" and event["side"] == "bid":
                bullish_score += 1.0  # Large buyer on YES
            elif event["type"] == "appear" and event["side"] == "ask":
                bearish_score += 1.0  # Large seller on YES
            elif event["type"] == "vanish" and event["side"] == "bid":
                bearish_score += 0.5  # Large buyer pulled from YES
            elif event["type"] == "vanish" and event["side"] == "ask":
                bullish_score += 0.5  # Large seller pulled from YES

        # NO book events (inverse interpretation)
        for event in self.orderbook.no_book.large_order_events:
            if event["timestamp"] < now - lookback:
                continue
            if event["type"] == "appear" and event["side"] == "bid":
                bearish_score += 1.0  # Large buyer on NO = bearish
            elif event["type"] == "appear" and event["side"] == "ask":
                bullish_score += 1.0  # Large seller on NO = bullish
            elif event["type"] == "vanish" and event["side"] == "bid":
                bullish_score += 0.5
            elif event["type"] == "vanish" and event["side"] == "ask":
                bearish_score += 0.5

        total = bullish_score + bearish_score
        if total <= 0:
            return 0.0

        net = (bullish_score - bearish_score) / total
        return max(-1.0, min(1.0, net))
