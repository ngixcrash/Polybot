"""Market discovery — slug calculation and Gamma API token resolution for 5-min BTC windows."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from config import cfg

log = logging.getLogger(__name__)

WINDOW_DURATION = 300  # 5 minutes in seconds


@dataclass
class MarketWindow:
    window_ts: int           # Start of the 5-min window (Unix seconds)
    close_ts: int            # End of the 5-min window
    slug: str                # Polymarket event slug
    condition_id: str = ""   # CTF condition ID (for merging)
    yes_token_id: str = ""   # Token ID for "Up" / YES
    no_token_id: str = ""    # Token ID for "Down" / NO
    event_id: str = ""       # Polymarket event ID
    market_id: str = ""      # Polymarket market ID
    resolved: bool = False
    outcome: Optional[str] = None  # "up" or "down" after resolution


def calc_current_window() -> tuple[int, int]:
    now = int(time.time())
    window_ts = now - (now % WINDOW_DURATION)
    close_ts = window_ts + WINDOW_DURATION
    return window_ts, close_ts


def calc_next_window() -> tuple[int, int]:
    window_ts, _ = calc_current_window()
    next_ts = window_ts + WINDOW_DURATION
    return next_ts, next_ts + WINDOW_DURATION


def build_slug(window_ts: int) -> str:
    return f"btc-updown-5m-{window_ts}"


def seconds_until_close() -> float:
    _, close_ts = calc_current_window()
    return max(0.0, close_ts - time.time())


def seconds_into_window() -> float:
    window_ts, _ = calc_current_window()
    return time.time() - window_ts


@dataclass
class MarketDiscovery:
    _cache: dict[int, MarketWindow] = field(default_factory=dict)
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)

    async def start(self):
        self._session = aiohttp.ClientSession()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def get_market(self, window_ts: int) -> Optional[MarketWindow]:
        """Fetch or return cached market data, resolving token IDs from Gamma."""
        if window_ts in self._cache and self._cache[window_ts].yes_token_id:
            return self._cache[window_ts]

        slug = build_slug(window_ts)
        close_ts = window_ts + WINDOW_DURATION
        window = MarketWindow(window_ts=window_ts, close_ts=close_ts, slug=slug)

        try:
            url = f"{cfg.gamma_url}/events?slug={slug}"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    log.warning(f"Gamma API error {resp.status} for slug={slug}")
                    return None
                data = await resp.json()

            if not data:
                log.debug(f"No event found for slug={slug}")
                return None

            event = data[0] if isinstance(data, list) else data
            window.event_id = str(event.get("id", ""))

            markets = event.get("markets", [])
            if not markets:
                log.warning(f"No markets in event for slug={slug}")
                return None

            market = markets[0]
            window.market_id = str(market.get("id", ""))
            window.condition_id = str(market.get("conditionId", ""))

            # clobTokenIds: [YES_token, NO_token] — may be JSON string or list
            token_ids = market.get("clobTokenIds", [])
            if isinstance(token_ids, str):
                import json
                try:
                    token_ids = json.loads(token_ids)
                except (json.JSONDecodeError, TypeError):
                    token_ids = []
            if len(token_ids) >= 2:
                window.yes_token_id = token_ids[0]
                window.no_token_id = token_ids[1]
            elif len(token_ids) == 1:
                window.yes_token_id = token_ids[0]

            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                import json
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []
            if outcomes:
                log.debug(f"Market outcomes: {outcomes}")

            self._cache[window_ts] = window
            log.info(
                f"Market discovered: slug={slug} "
                f"yes_token={window.yes_token_id[:12]}... "
                f"no_token={window.no_token_id[:12]}..."
            )
            return window

        except Exception as e:
            log.error(f"Market discovery failed for slug={slug}: {e}")
            return None

    async def get_current_market(self) -> Optional[MarketWindow]:
        window_ts, _ = calc_current_window()
        return await self.get_market(window_ts)

    def cleanup_old(self, keep_last: int = 10):
        if len(self._cache) > keep_last * 2:
            sorted_keys = sorted(self._cache.keys())
            for key in sorted_keys[:-keep_last]:
                del self._cache[key]
