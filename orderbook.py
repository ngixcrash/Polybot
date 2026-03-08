"""Polymarket orderbook WebSocket stream -- tracks bids/asks, spread, and arbitrage."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import websockets

from config import cfg

log = logging.getLogger(__name__)


@dataclass
class OrderbookSide:
    prices: dict[float, float] = field(default_factory=dict)  # price -> size

    @property
    def best(self) -> Optional[tuple[float, float]]:
        if not self.prices:
            return None
        return None

    def update(self, price: float, size: float):
        if size <= 0:
            self.prices.pop(price, None)
        else:
            self.prices[price] = size

    def clear(self):
        self.prices.clear()


@dataclass
class TokenBook:
    token_id: str = ""
    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)  # price -> size
    last_update: float = 0.0

    # Microstructure tracking
    spread_history: list[tuple[float, float]] = field(default_factory=list)  # (ts, spread)
    mid_history: list[tuple[float, float]] = field(default_factory=list)     # (ts, mid)
    _max_history: int = 200
    _prev_bids: dict[float, float] = field(default_factory=dict)
    _prev_asks: dict[float, float] = field(default_factory=dict)
    large_order_events: list[dict] = field(default_factory=list)
    _max_events: int = 50

    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    @property
    def best_bid_size(self) -> float:
        if self.best_bid is not None:
            return self.bids[self.best_bid]
        return 0.0

    @property
    def best_ask_size(self) -> float:
        if self.best_ask is not None:
            return self.asks[self.best_ask]
        return 0.0

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return bid or ask

    @property
    def spread(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None

    @property
    def microprice(self) -> Optional[float]:
        """Depth-weighted mid price -- shifts toward the thinner side."""
        # microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return self.mid_price
        bid_sz = self.best_bid_size
        ask_sz = self.best_ask_size
        total_sz = bid_sz + ask_sz
        if total_sz <= 0:
            return (bid + ask) / 2.0
        return (bid * ask_sz + ask * bid_sz) / total_sz

    @property
    def depth_imbalance(self) -> float:
        """Top-5-level bid/ask imbalance, -1.0 (bearish) to +1.0 (bullish)."""
        top_bids = sorted(self.bids.keys(), reverse=True)[:5]
        top_asks = sorted(self.asks.keys())[:5]
        total_bid = sum(self.bids[p] for p in top_bids)
        total_ask = sum(self.asks[p] for p in top_asks)
        total = total_bid + total_ask
        if total <= 0:
            return 0.0
        return (total_bid - total_ask) / total

    @property
    def spread_bps(self) -> Optional[float]:
        s = self.spread
        m = self.mid_price
        if s is not None and m and m > 0:
            return s / m * 10000
        return None

    def record_snapshot(self) -> None:
        now = time.time()
        s = self.spread
        m = self.mid_price
        if s is not None:
            self.spread_history.append((now, s))
            if len(self.spread_history) > self._max_history:
                self.spread_history = self.spread_history[-self._max_history:]
        if m is not None:
            self.mid_history.append((now, m))
            if len(self.mid_history) > self._max_history:
                self.mid_history = self.mid_history[-self._max_history:]

        self._detect_large_orders(now)

        self._prev_bids = dict(self.bids)
        self._prev_asks = dict(self.asks)

    def _detect_large_orders(self, now: float, threshold_mult: float = 3.0) -> None:
        if not self._prev_bids and not self._prev_asks:
            return  # First snapshot, nothing to compare

        all_sizes = list(self.bids.values()) + list(self.asks.values())
        if not all_sizes:
            return
        avg_size = sum(all_sizes) / len(all_sizes)
        threshold = avg_size * threshold_mult

        for price, size in self.bids.items():
            prev_size = self._prev_bids.get(price, 0)
            delta = size - prev_size
            if delta > threshold:
                self.large_order_events.append({
                    "type": "appear", "side": "bid", "price": price,
                    "size_delta": delta, "timestamp": now,
                })

        for price, size in self.asks.items():
            prev_size = self._prev_asks.get(price, 0)
            delta = size - prev_size
            if delta > threshold:
                self.large_order_events.append({
                    "type": "appear", "side": "ask", "price": price,
                    "size_delta": delta, "timestamp": now,
                })

        for price, prev_size in self._prev_bids.items():
            new_size = self.bids.get(price, 0)
            if prev_size > threshold and new_size < prev_size * 0.3:
                self.large_order_events.append({
                    "type": "vanish", "side": "bid", "price": price,
                    "size_delta": -(prev_size - new_size), "timestamp": now,
                })

        for price, prev_size in self._prev_asks.items():
            new_size = self.asks.get(price, 0)
            if prev_size > threshold and new_size < prev_size * 0.3:
                self.large_order_events.append({
                    "type": "vanish", "side": "ask", "price": price,
                    "size_delta": -(prev_size - new_size), "timestamp": now,
                })

        if len(self.large_order_events) > self._max_events:
            self.large_order_events = self.large_order_events[-self._max_events:]

    def get_spread_trend(self, lookback_seconds: float = 30.0) -> float:
        """Positive = spread widening, negative = narrowing, zero = no data."""
        if len(self.spread_history) < 5:
            return 0.0
        cutoff = time.time() - lookback_seconds
        recent = [(t, s) for t, s in self.spread_history if t >= cutoff]
        if len(recent) < 3:
            return 0.0
        first_third = recent[:len(recent)//3]
        last_third = recent[-len(recent)//3:]
        avg_early = sum(s for _, s in first_third) / len(first_third)
        avg_late = sum(s for _, s in last_third) / len(last_third)
        if avg_early <= 0:
            return 0.0
        return (avg_late - avg_early) / avg_early

    def update_bid(self, price: float, size: float):
        if size <= 0:
            self.bids.pop(price, None)
        else:
            self.bids[price] = size
        self.last_update = time.time()

    def update_ask(self, price: float, size: float):
        if size <= 0:
            self.asks.pop(price, None)
        else:
            self.asks[price] = size
        self.last_update = time.time()

    def clear(self):
        self.bids.clear()
        self.asks.clear()


@dataclass
class OrderbookManager:

    yes_book: TokenBook = field(default_factory=TokenBook)
    no_book: TokenBook = field(default_factory=TokenBook)
    _ws: Optional[object] = field(default=None, repr=False)
    _running: bool = False
    _connected: bool = False
    _subscribed_tokens: set = field(default_factory=set)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def has_arbitrage(self) -> Optional[float]:
        """Returns profit margin if yes_ask + no_ask < 1.0, else None."""
        yes_ask = self.yes_book.best_ask
        no_ask = self.no_book.best_ask
        if yes_ask is not None and no_ask is not None:
            total = yes_ask + no_ask
            if total < 1.0:
                return 1.0 - total
        return None

    async def start(self):
        self._running = True
        asyncio.create_task(self._run_ws())
        log.info("Orderbook manager started")

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()
        log.info("Orderbook manager stopped")

    async def subscribe(self, yes_token_id: str, no_token_id: str):
        self.yes_book = TokenBook(token_id=yes_token_id)
        self.no_book = TokenBook(token_id=no_token_id)

        new_tokens = {yes_token_id, no_token_id}

        if self._ws and self._connected and self._subscribed_tokens:
            old_tokens = self._subscribed_tokens - new_tokens
            if old_tokens:
                unsub_msg = {
                    "type": "unsubscribe",
                    "assets_ids": list(old_tokens),
                }
                try:
                    await self._ws.send(json.dumps(unsub_msg))
                except Exception:
                    pass

        if self._ws and self._connected:
            sub_msg = {
                "assets_ids": list(new_tokens),
                "type": "market",
            }
            try:
                await self._ws.send(json.dumps(sub_msg))
                self._subscribed_tokens = new_tokens
                log.info(f"Subscribed to orderbook: YES={yes_token_id[:12]}... NO={no_token_id[:12]}...")
            except Exception as e:
                log.error(f"Failed to subscribe: {e}")
        else:
            self._subscribed_tokens = new_tokens

    async def _run_ws(self):
        while self._running:
            try:
                async with websockets.connect(cfg.ws_orderbook_url) as ws:
                    self._ws = ws
                    self._connected = True
                    log.info("Polymarket orderbook WS connected")

                    if self._subscribed_tokens:
                        sub_msg = {
                            "assets_ids": list(self._subscribed_tokens),
                            "type": "market",
                        }
                        await ws.send(json.dumps(sub_msg))
                        log.info("Re-subscribed to orderbook after reconnect")

                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_message(msg)

            except websockets.ConnectionClosed:
                log.warning("Polymarket orderbook WS disconnected, reconnecting...")
            except Exception as e:
                log.error(f"Polymarket orderbook WS error: {e}")
            finally:
                self._connected = False

            if self._running:
                await asyncio.sleep(2)

    def _process_message(self, raw: str):
        try:
            data = json.loads(raw)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._handle_single_message(item)
                    else:
                        log.debug(f"Skipping non-dict list item: {type(item)}")
            elif isinstance(data, dict):
                self._handle_single_message(data)
            else:
                log.debug(f"Unexpected WS payload type: {type(data)}")

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            log.debug(f"Orderbook parse error: {e}")

    def _handle_single_message(self, data: dict):
        msg_type = data.get("type", "")

        if msg_type == "book":
            self._handle_book_snapshot(data)
        elif msg_type == "price_change":
            self._handle_price_change(data)
        elif msg_type == "tick_size_change":
            pass  # Ignore tick size changes
        elif msg_type == "last_trade_price":
            pass  # Optional: track last trade
        else:
            if "market" in data or "asset_id" in data:
                self._handle_generic_update(data)

    def _handle_book_snapshot(self, data: dict):
        asset_id = data.get("asset_id", "")
        book = self._get_book(asset_id)
        if not book:
            return

        book.clear()

        for bid in data.get("bids", []):
            price = float(bid.get("price", 0))
            size = float(bid.get("size", 0))
            book.update_bid(price, size)

        for ask in data.get("asks", []):
            price = float(ask.get("price", 0))
            size = float(ask.get("size", 0))
            book.update_ask(price, size)

        book.record_snapshot()
        log.debug(
            f"Book snapshot: token={asset_id[:12]}... "
            f"bid={book.best_bid} ask={book.best_ask}"
        )

    def _handle_price_change(self, data: dict):
        changes = data.get("changes", [])
        updated_books = set()
        for change in changes:
            asset_id = change.get("asset_id", "")
            book = self._get_book(asset_id)
            if not book:
                continue

            side = change.get("side", "")
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))

            if side == "BUY" or side == "bid":
                book.update_bid(price, size)
            elif side == "SELL" or side == "ask":
                book.update_ask(price, size)

            updated_books.add(id(book))

        for book in [self.yes_book, self.no_book]:
            if id(book) in updated_books:
                book.record_snapshot()

    def _handle_generic_update(self, data: dict):
        asset_id = data.get("asset_id", "")
        book = self._get_book(asset_id)
        if not book:
            return

        if "bids" in data:
            for bid in data["bids"]:
                price = float(bid.get("price", bid.get("p", 0)))
                size = float(bid.get("size", bid.get("s", 0)))
                book.update_bid(price, size)

        if "asks" in data:
            for ask in data["asks"]:
                price = float(ask.get("price", ask.get("p", 0)))
                size = float(ask.get("size", ask.get("s", 0)))
                book.update_ask(price, size)

    def _get_book(self, asset_id: str) -> Optional[TokenBook]:
        if asset_id == self.yes_book.token_id:
            return self.yes_book
        elif asset_id == self.no_book.token_id:
            return self.no_book
        return None
