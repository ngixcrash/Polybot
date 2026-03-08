"""External sentiment feeds (Binance L/S ratio, OI, Fear & Greed) for supplementary signals."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from config import cfg

log = logging.getLogger(__name__)

# API endpoints
LONG_SHORT_URL = "https://fapi.binance.com/fapi/v1/globalLongShortAccountRatio"
OPEN_INTEREST_URL = "https://fapi.binance.com/fapi/v1/openInterest"
FEAR_GREED_URL = "https://api.alternative.me/fng/"


@dataclass
class SentimentSnapshot:
    long_short_ratio: float = 1.0       # >1 means more longs, <1 more shorts
    long_pct: float = 50.0              # % of accounts that are long
    short_pct: float = 50.0             # % of accounts that are short
    open_interest: float = 0.0          # BTC futures open interest in BTC
    oi_change_pct: float = 0.0          # OI change vs previous reading
    fear_greed_index: int = 50          # 0=extreme fear, 100=extreme greed
    fear_greed_label: str = "Neutral"
    timestamp: float = 0.0


@dataclass
class SentimentFeed:
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _running: bool = False
    current: SentimentSnapshot = field(default_factory=SentimentSnapshot)
    history: list[SentimentSnapshot] = field(default_factory=list)
    max_history: int = 50
    _last_oi: float = 0.0              # Previous OI reading for change calc
    _fetch_interval: float = 60.0       # Fetch every 60s (conservative rate limit)
    _last_fetch: float = 0.0

    async def start(self):
        self._session = aiohttp.ClientSession()
        self._running = True
        # Do an initial fetch, then start background polling
        await self.fetch_now()
        asyncio.create_task(self._poll_loop())
        log.info("[SENTIMENT] Feed started")

    async def stop(self):
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None
        log.info("[SENTIMENT] Feed stopped")

    async def fetch_now(self) -> SentimentSnapshot:
        snapshot = SentimentSnapshot(timestamp=time.time())

        tasks = [
            self._fetch_long_short_ratio(),
            self._fetch_open_interest(),
            self._fetch_fear_greed(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        if isinstance(results[0], tuple):
            snapshot.long_short_ratio, snapshot.long_pct, snapshot.short_pct = results[0]
        else:
            log.debug(f"Long/short fetch failed: {results[0]}")

        if isinstance(results[1], float):
            snapshot.open_interest = results[1]
            if self._last_oi > 0:
                snapshot.oi_change_pct = (results[1] - self._last_oi) / self._last_oi * 100
            self._last_oi = results[1]
        else:
            log.debug(f"OI fetch failed: {results[1]}")

        if isinstance(results[2], tuple):
            snapshot.fear_greed_index, snapshot.fear_greed_label = results[2]
        else:
            log.debug(f"Fear/greed fetch failed: {results[2]}")

        self.current = snapshot
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self._last_fetch = time.time()

        log.info(
            f"[SENTIMENT] L/S={snapshot.long_short_ratio:.2f} "
            f"({snapshot.long_pct:.1f}%L/{snapshot.short_pct:.1f}%S) "
            f"OI={snapshot.open_interest:.1f} (chg={snapshot.oi_change_pct:+.2f}%) "
            f"F&G={snapshot.fear_greed_index} ({snapshot.fear_greed_label})"
        )

        return snapshot

    async def _poll_loop(self):
        while self._running:
            await asyncio.sleep(self._fetch_interval)
            if not self._running:
                break
            try:
                await self.fetch_now()
            except Exception as e:
                log.debug(f"Sentiment poll error: {e}")

    async def _fetch_long_short_ratio(self) -> tuple[float, float, float]:
        if not self._session:
            raise RuntimeError("Session not initialized")

        params = {"symbol": "BTCUSDT", "period": "5m", "limit": 1}
        async with self._session.get(LONG_SHORT_URL, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Status {resp.status}")
            data = await resp.json()

        if not data:
            raise RuntimeError("Empty response")

        entry = data[0] if isinstance(data, list) else data
        ratio = float(entry.get("longShortRatio", 1.0))
        long_pct = float(entry.get("longAccount", 0.5)) * 100
        short_pct = float(entry.get("shortAccount", 0.5)) * 100

        return ratio, long_pct, short_pct

    async def _fetch_open_interest(self) -> float:
        if not self._session:
            raise RuntimeError("Session not initialized")

        params = {"symbol": "BTCUSDT"}
        async with self._session.get(OPEN_INTEREST_URL, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Status {resp.status}")
            data = await resp.json()

        return float(data.get("openInterest", 0))

    async def _fetch_fear_greed(self) -> tuple[int, str]:
        if not self._session:
            raise RuntimeError("Session not initialized")

        async with self._session.get(FEAR_GREED_URL) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Status {resp.status}")
            data = await resp.json()

        entries = data.get("data", [])
        if not entries:
            raise RuntimeError("No data in response")

        entry = entries[0]
        index = int(entry.get("value", 50))
        label = entry.get("value_classification", "Neutral")

        return index, label

    def get_signal_components(self) -> dict[str, float]:
        """Turn current sentiment snapshot into directional scores."""
        s = self.current
        components = {}
        sent_cfg = cfg.sentiment

        # 1. Long/Short Ratio — contrarian at extremes
        # More longs = potential for squeeze down, more shorts = squeeze up
        ls_signal = 0.0
        if s.long_short_ratio > sent_cfg.ls_ratio_extreme_long:
            ls_signal = -1.0  # Too many longs → expect reversal down
        elif s.long_short_ratio < sent_cfg.ls_ratio_extreme_short:
            ls_signal = 1.0   # Too many shorts → expect squeeze up
        elif s.long_short_ratio > 1.3:
            ls_signal = -0.5
        elif s.long_short_ratio < 0.7:
            ls_signal = 0.5
        components["sentiment_ls_ratio"] = ls_signal

        # 2. Open Interest Change — rising OI with direction = conviction
        oi_signal = 0.0
        if abs(s.oi_change_pct) > sent_cfg.oi_change_threshold * 100:
            # OI rising = new money entering (confirms trend)
            # OI falling = positions closing (weakening)
            if s.oi_change_pct > 0:
                oi_signal = 0.5   # New money = bullish bias
            else:
                oi_signal = -0.5  # Money leaving = bearish bias
        components["sentiment_oi_change"] = oi_signal

        # 3. Fear & Greed Index — contrarian at extremes (slow-moving regime filter)
        fg_signal = 0.0
        if s.fear_greed_index < sent_cfg.fear_greed_extreme_fear:
            fg_signal = 1.0   # Extreme fear → contrarian bullish
        elif s.fear_greed_index > sent_cfg.fear_greed_extreme_greed:
            fg_signal = -1.0  # Extreme greed → contrarian bearish
        elif s.fear_greed_index < 35:
            fg_signal = 0.5
        elif s.fear_greed_index > 65:
            fg_signal = -0.5
        components["sentiment_fear_greed"] = fg_signal

        return components
