"""Binance WebSocket feed — multi-timeframe klines, order flow, and trade ticks for BTC."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import websockets

from config import cfg

log = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: int  # open time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float = 0.0  # taker buy base-asset volume
    is_closed: bool = False

    @property
    def taker_sell_volume(self) -> float:
        return max(0.0, self.volume - self.taker_buy_volume)

    @property
    def buy_sell_ratio(self) -> float:
        sell = self.taker_sell_volume
        if sell <= 0:
            return 2.0  # All buys
        return self.taker_buy_volume / sell


@dataclass
class BinanceFeed:

    # 1-minute candles (primary)
    candles: list[Candle] = field(default_factory=list)
    max_candles: int = 120

    # 5-minute candles
    candles_5m: list[Candle] = field(default_factory=list)
    max_candles_5m: int = 120

    # 15-minute candles
    candles_15m: list[Candle] = field(default_factory=list)
    max_candles_15m: int = 60

    current_price: float = 0.0
    tick_prices: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, price)
    max_ticks: int = 500
    _ws_kline: Optional[object] = field(default=None, repr=False)
    _ws_trade: Optional[object] = field(default=None, repr=False)
    _running: bool = False
    _kline_connected: bool = False
    _trade_connected: bool = False
    _last_rest_fetch: float = 0.0

    async def start(self):
        self._running = True
        await self._bootstrap_candles()
        asyncio.create_task(self._run_kline_ws())
        log.info("Binance feed started (multi-timeframe)")

    async def stop(self):
        self._running = False
        if self._ws_kline:
            await self._ws_kline.close()
        if self._ws_trade:
            await self._ws_trade.close()
        log.info("Binance feed stopped")

    async def start_tick_stream(self):
        if not self._trade_connected:
            asyncio.create_task(self._run_trade_ws())

    async def stop_tick_stream(self):
        if self._ws_trade:
            await self._ws_trade.close()
            self._ws_trade = None
            self._trade_connected = False

    def get_window_open_price(self, window_ts: int) -> Optional[float]:
        """Find the 1m candle open price at the given 5-min-aligned timestamp."""
        target_ms = window_ts * 1000
        for candle in self.candles:
            if candle.timestamp == target_ms:
                return candle.open
        for candle in self.candles:
            if abs(candle.timestamp - target_ms) < 60_000:
                return candle.open
        return None

    def get_recent_candles(self, n: int = 30) -> list[Candle]:
        closed = [c for c in self.candles if c.is_closed]
        return closed[-n:]

    def get_recent_candles_5m(self, n: int = 20) -> list[Candle]:
        closed = [c for c in self.candles_5m if c.is_closed]
        return closed[-n:]

    def get_recent_candles_15m(self, n: int = 10) -> list[Candle]:
        closed = [c for c in self.candles_15m if c.is_closed]
        return closed[-n:]

    def get_order_flow_delta(self, n: int = 5) -> float:
        """Buy/sell imbalance from last n candles, normalized to [-1.0, 1.0]."""
        candles = self.get_recent_candles(n)
        if not candles:
            return 0.0

        total_buy = sum(c.taker_buy_volume for c in candles)
        total_sell = sum(c.taker_sell_volume for c in candles)
        total = total_buy + total_sell

        if total <= 0:
            return 0.0

        return (total_buy - total_sell) / total

    def get_tick_trend(self, lookback_seconds: float = 20.0) -> tuple[float, float]:
        """Returns (direction_pct, price_change_pct) from recent ticks."""
        if len(self.tick_prices) < 5:
            return 0.0, 0.0

        cutoff = time.time() - lookback_seconds
        recent = [(t, p) for t, p in self.tick_prices if t >= cutoff]
        if len(recent) < 5:
            return 0.0, 0.0

        ups = 0
        downs = 0
        for i in range(1, len(recent)):
            if recent[i][1] > recent[i - 1][1]:
                ups += 1
            elif recent[i][1] < recent[i - 1][1]:
                downs += 1

        total_moves = ups + downs
        if total_moves == 0:
            return 0.0, 0.0

        direction_pct = max(ups, downs) / total_moves
        price_change_pct = (recent[-1][1] - recent[0][1]) / recent[0][1] * 100

        return direction_pct, price_change_pct

    # ── Bootstrap ─────────────────────────────────────────────

    async def _bootstrap_candles(self):
        """Bootstrap candle history from Binance REST for all timeframes."""
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._fetch_rest_candles(session, "1m", cfg.strategy.bootstrap_1m),
                    self._fetch_rest_candles(session, "5m", cfg.strategy.bootstrap_5m),
                    self._fetch_rest_candles(session, "15m", cfg.strategy.bootstrap_15m),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            if isinstance(results[0], list):
                self.candles = results[0]
                log.info(f"Bootstrapped {len(self.candles)} x 1m candles")
            else:
                log.error(f"1m bootstrap failed: {results[0]}")

            if isinstance(results[1], list):
                self.candles_5m = results[1]
                log.info(f"Bootstrapped {len(self.candles_5m)} x 5m candles")
            else:
                log.error(f"5m bootstrap failed: {results[1]}")

            if isinstance(results[2], list):
                self.candles_15m = results[2]
                log.info(f"Bootstrapped {len(self.candles_15m)} x 15m candles")
            else:
                log.error(f"15m bootstrap failed: {results[2]}")

            if self.candles:
                self.current_price = self.candles[-1].close

            self._last_rest_fetch = time.time()
            log.info(f"Bootstrap complete, price={self.current_price:.2f}")

        except Exception as e:
            log.error(f"Failed to bootstrap candles: {e}")

    async def _fetch_rest_candles(
        self, session: aiohttp.ClientSession, interval: str, limit: int
    ) -> list[Candle]:
        url = f"{cfg.binance_rest_url}?symbol=BTCUSDT&interval={interval}&limit={limit}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Binance REST error {resp.status} for {interval}")
            data = await resp.json()

        candles = []
        for k in data:
            candle = Candle(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                taker_buy_volume=float(k[9]),  # Taker buy base-asset volume
                is_closed=True,
            )
            candles.append(candle)
        return candles

    async def refresh_candles_rest(self):
        if time.time() - self._last_rest_fetch < 2.0:
            return
        await self._bootstrap_candles()

    # ── WebSocket streams ─────────────────────────────────────

    async def _run_kline_ws(self):
        """Combined kline WS with auto-reconnect."""
        while self._running:
            try:
                async with websockets.connect(cfg.binance_ws_url) as ws:
                    self._ws_kline = ws
                    self._kline_connected = True
                    log.info("Binance combined kline WS connected (1m/5m/15m)")

                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_kline(msg)

            except websockets.ConnectionClosed:
                log.warning("Binance kline WS disconnected, reconnecting...")
            except Exception as e:
                log.error(f"Binance kline WS error: {e}")
            finally:
                self._kline_connected = False

            if self._running:
                await asyncio.sleep(2)

    async def _run_trade_ws(self):
        while self._running:
            try:
                async with websockets.connect(cfg.binance_trade_ws_url) as ws:
                    self._ws_trade = ws
                    self._trade_connected = True
                    log.info("Binance trade WS connected")

                    async for msg in ws:
                        if not self._running or not self._trade_connected:
                            break
                        self._process_trade(msg)

            except websockets.ConnectionClosed:
                log.debug("Binance trade WS disconnected")
            except Exception as e:
                log.error(f"Binance trade WS error: {e}")
            finally:
                self._trade_connected = False

            if self._running and self._trade_connected:
                await asyncio.sleep(1)
            else:
                break

    def _process_kline(self, raw: str):
        """Parse kline message (combined or single stream) and update the right candle store."""
        try:
            msg = json.loads(raw)

            if "stream" in msg:
                stream_name = msg["stream"]
                data = msg.get("data", {})
            else:
                stream_name = "btcusdt@kline_1m"
                data = msg

            k = data.get("k", {})
            if not k:
                return

            candle = Candle(
                timestamp=int(k["t"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                taker_buy_volume=float(k.get("V", 0)),  # Taker buy base-asset volume
                is_closed=k.get("x", False),
            )

            self.current_price = candle.close

            if "kline_1m" in stream_name:
                self._update_candle_store(self.candles, candle, self.max_candles)
            elif "kline_5m" in stream_name:
                self._update_candle_store(self.candles_5m, candle, self.max_candles_5m)
            elif "kline_15m" in stream_name:
                self._update_candle_store(self.candles_15m, candle, self.max_candles_15m)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.debug(f"Kline parse error: {e}")

    def _update_candle_store(self, store: list[Candle], candle: Candle, max_len: int):
        if store and store[-1].timestamp == candle.timestamp:
            store[-1] = candle
        else:
            store.append(candle)

        if len(store) > max_len:
            del store[: len(store) - max_len]

    def _process_trade(self, raw: str):
        try:
            data = json.loads(raw)
            price = float(data["p"])
            ts = time.time()

            self.current_price = price
            self.tick_prices.append((ts, price))

            if len(self.tick_prices) > self.max_ticks:
                self.tick_prices = self.tick_prices[-self.max_ticks:]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.debug(f"Trade parse error: {e}")
