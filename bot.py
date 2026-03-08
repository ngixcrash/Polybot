"""Main bot engine -- timing, lifecycle, order management for 5-min windows."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

from adaptive import AdaptiveWeightEngine
from binance_feed import BinanceFeed
from config import cfg
from execution import Executor, OrderSide
from market import (
    MarketDiscovery,
    MarketWindow,
    WINDOW_DURATION,
    calc_current_window,
    seconds_into_window,
    seconds_until_close,
)
from merge import Merger
from microstructure import MicrostructureAnalyzer
from orderbook import OrderbookManager
from persistence import TradeStore
from risk import RiskManager
from sentiment import SentimentFeed
from strategy import Signal, Strategy, calc_dynamic_taker_fee, estimate_token_price

# ---------------------------------------------------------------------------
# Custom log handler that clears the live status line before emitting output,
# preventing garbled text when log messages and \r-based progress overlap.
# ---------------------------------------------------------------------------
class _StatusAwareHandler(logging.StreamHandler):

    def emit(self, record):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80
        # Overwrite the status line with blanks, then carriage-return
        try:
            sys.stdout.write("\r" + " " * term_width + "\r")
            sys.stdout.flush()
        except UnicodeEncodeError:
            pass
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Windows cp1252 can't handle Unicode — fall back to ASCII
            record.msg = record.msg.encode("ascii", "replace").decode("ascii")
            super().emit(record)


_handler = _StatusAwareHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

log = logging.getLogger(__name__)


class TradingBot:

    def __init__(
        self,
        mode: str = "safe",
        dry_run: bool = True,
        run_once: bool = False,
        max_trades: int = 0,
    ):
        self.mode = mode
        self.dry_run = dry_run
        self.run_once = run_once
        self.max_trades = max_trades
        self.trade_count = 0

        self.feed = BinanceFeed()
        self.market = MarketDiscovery()
        self.orderbook = OrderbookManager()
        self.executor = Executor(dry_run=dry_run)
        self.merger = Merger(dry_run=dry_run)

        # Persistence + ML
        self.trade_store = TradeStore()
        self.adaptive = AdaptiveWeightEngine()
        self.risk = RiskManager(mode=mode, _trade_store=self.trade_store)

        # Sentiment + Microstructure
        self.sentiment = SentimentFeed()
        self.micro = MicrostructureAnalyzer(orderbook=self.orderbook)

        # Strategy (wired with all subsystems)
        self.strategy = Strategy(
            self.feed,
            adaptive=self.adaptive,
            sentiment=self.sentiment,
            microstructure=self.micro,
        )

        self._running = False
        self._current_window: Optional[MarketWindow] = None
        self._window_open_price: float = 0.0
        self._order_placed_this_window = False
        self._best_signal: Optional[Signal] = None
        self._last_score: float = 0.0
        self._last_arb_logged: float = 0.0      # last arb margin we logged
        self._last_arb_time: float = 0.0         # timestamp of last arb log

    async def start(self):
        log.info(
            f"{'='*60}\n"
            f"  Polymarket BTC 5-Min Up/Down Trading Bot\n"
            f"  Mode: {self.mode} | Dry Run: {self.dry_run}\n"
            f"  Bankroll: ${self.risk.bankroll:.2f}\n"
            f"{'='*60}"
        )

        self._running = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(sig, self._shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: asyncio.create_task(self._async_shutdown()))

        await self.feed.start()
        await self.market.start()
        await self.orderbook.start()
        await self.executor.start()
        await self.merger.start()
        await self.sentiment.start()

        log.info("Waiting for initial Binance price data...")
        for _ in range(30):
            if self.feed.current_price > 0:
                break
            await asyncio.sleep(1)

        if self.feed.current_price <= 0:
            log.error("Failed to get initial BTC price — check Binance connection")
            await self.stop()
            return

        log.info(f"BTC price: ${self.feed.current_price:,.2f}")

        if not self.dry_run:
            actual_balance = await self._get_usdc_balance()
            if actual_balance is not None and actual_balance > 0:
                if abs(actual_balance - self.risk.bankroll) > 1.0:
                    log.warning(
                        f"[BALANCE SYNC] Wallet=${actual_balance:.2f} "
                        f"vs internal=${self.risk.bankroll:.2f} — using wallet balance"
                    )
                self.risk.bankroll = actual_balance
                self.risk.starting_bankroll = actual_balance
                log.info(f"Live bankroll synced: ${actual_balance:.2f}")

        try:
            await self._main_loop()
        except Exception as e:
            log.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        log.info("Shutting down...")
        self._running = False

        cancelled = await self.executor.cancel_all()
        if cancelled:
            log.info(f"Cancelled {cancelled} open orders")

        await self.feed.stop()
        await self.market.stop()
        await self.orderbook.stop()
        await self.executor.stop()
        await self.merger.stop()
        await self.sentiment.stop()

        # Save adaptive model state on shutdown
        self.adaptive.save_to_disk()

        self.risk.print_stats()

        # Log adaptive weight summary
        if self.adaptive.total_resolved > 0:
            summary = self.adaptive.get_weight_summary()
            log.info(f"[ADAPTIVE] Final weights after {self.adaptive.total_resolved} trades:")
            for name, info in summary.items():
                if info["total"] > 0:
                    log.info(f"  {name}: weight={info['weight']:.3f} accuracy={info['accuracy']:.3f} ({info['total']} samples)")

        log.info("Bot stopped.")

    def _shutdown(self):
        asyncio.create_task(self._async_shutdown())

    async def _async_shutdown(self):
        self._running = False

    def _print_status(self, phase: str, remaining: float, total: float, extra: str = ""):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80

        elapsed = total - remaining
        pct = max(0.0, min(1.0, elapsed / total)) if total > 0 else 0.0
        bar_width = min(20, term_width - 60)
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        mins, secs = divmod(int(remaining), 60)
        countdown = f"{mins}:{secs:02d}"

        btc = self.feed.current_price
        delta_str = ""
        if self._window_open_price > 0 and btc > 0:
            delta = (btc - self._window_open_price) / self._window_open_price * 100
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
            delta_str = f" {arrow}{abs(delta):.3f}%"

        line = f"\r  {phase} [{bar}] {countdown} | BTC ${btc:,.2f}{delta_str}"
        if extra:
            line += f" | {extra}"

        line = line[:term_width].ljust(term_width)
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            # Windows cp1252 can't handle Unicode emojis — strip them
            sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()

    def _clear_status(self):
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80
        sys.stdout.write("\r" + " " * term_width + "\r")
        sys.stdout.flush()

    async def _main_loop(self):
        while self._running:
            if self.max_trades and self.trade_count >= self.max_trades:
                log.info(f"Max trades reached ({self.max_trades}). Stopping.")
                break

            paused, reason = self.risk.is_paused()
            if paused:
                log.warning(f"[PAUSED] {reason}")
                await asyncio.sleep(10)
                continue

            try:
                await self._run_window()
            except Exception as e:
                log.error(f"Error in window: {e}", exc_info=True)
                await asyncio.sleep(5)

            if self.run_once:
                log.info("Single window mode — stopping.")
                break

            await asyncio.sleep(1)

    # ---- Price-aware entry thresholds ----
    # The closer the ask is to $0.50 (fair value), the less confidence we need.
    # At $0.50, even a tiny edge is profitable.  At $0.90 we need near certainty.
    _PRICE_CONFIDENCE_TABLE = [
        (0.52, 0.25),   # ask ≤ $0.52: need 25% confidence (tiny edge is fine)
        (0.55, 0.30),   # ask ≤ $0.55: need 30%
        (0.58, 0.35),   # ask ≤ $0.58: need 35%
        (0.60, 0.40),   # ask ≤ $0.60: need 40%
        (0.65, 0.50),   # ask ≤ $0.65: need 50%
        (0.70, 0.55),   # ask ≤ $0.70: need 55%
        (0.75, 0.60),   # ask ≤ $0.75: need 60%
        (0.80, 0.65),   # ask ≤ $0.80: need 65%
        (0.85, 0.70),   # ask ≤ $0.85: need 70%
        (0.90, 0.80),   # ask ≤ $0.90: need 80%
    ]

    def _min_confidence_for_price(self, ask_price: float) -> float:
        """Lower ask = lower confidence needed; returns >1.0 if price is too expensive."""
        for threshold, min_conf in self._PRICE_CONFIDENCE_TABLE:
            if ask_price <= threshold:
                return min_conf
        return 1.01  # Unreachable — skip this price

    # Maximum seconds into a window before we skip it and wait for next
    MAX_LATE_ENTRY = 30  # Must catch window within first 30s

    async def _run_window(self):
        """Execute one 5-min window: discover market, poll for entry, place orders, resolve."""
        # ── Phase 0: Skip stale windows ────────────────────────────────
        elapsed = seconds_into_window()
        if elapsed > self.MAX_LATE_ENTRY:
            log.info(
                f"[SKIP WINDOW] Already {elapsed:.0f}s into window "
                f"(max {self.MAX_LATE_ENTRY}s) -- waiting for next fresh window"
            )
            await self._wait_for_next_window()

        # Now we're at the start of a fresh window
        window_ts, close_ts = calc_current_window()
        self._order_placed_this_window = False
        self._best_signal = None
        self._last_score = 0.0
        self._last_arb_logged = 0.0
        self._last_arb_time = 0.0

        log.info(f"\n{'='*50}")
        log.info(
            f"[WINDOW] ts={window_ts} T+{seconds_into_window():.0f}s "
            f"closes_in={seconds_until_close():.0f}s"
        )

        # ── Phase 1: Discover market ─────────────────────────────────────
        market = await self.market.get_market(window_ts)
        if not market or not market.yes_token_id:
            log.warning(f"Market not found for window {window_ts}, waiting for next window...")
            await self._wait_for_next_window()
            return

        self._current_window = market

        # Subscribe to orderbook WS (for arb detection, not primary pricing)
        await self.orderbook.subscribe(market.yes_token_id, market.no_token_id)

        # Pre-fetch fee rates, neg_risk flag, and orderbook metadata
        if not self.dry_run:
            await self.executor.get_fee_rate(market.yes_token_id)
            await self.executor.get_fee_rate(market.no_token_id)
            # Pre-fetch neg_risk (determines exchange contract for signing)
            neg_risk = await self.executor.get_neg_risk(market.yes_token_id)
            log.info(f"[MARKET] neg_risk={neg_risk} (MINT matching {'enabled' if neg_risk else 'disabled'})")
            # Pre-fetch orderbook to cache tick_size
            await self.executor.get_order_book_raw(market.yes_token_id)

        self._window_open_price = self.feed.get_window_open_price(window_ts)
        if not self._window_open_price:
            self._window_open_price = self.feed.current_price
            log.warning(f"Using current price as window open: ${self._window_open_price:,.2f}")
        else:
            log.info(f"Window open price: ${self._window_open_price:,.2f}")

        await self.feed.start_tick_stream()

        # ── Phase 2: Wait for initial BTC data (20s) ────────────────────
        EARLY_WAIT = 20  # seconds to gather initial price data
        total_wait = seconds_until_close()
        while self._running and seconds_into_window() < EARLY_WAIT:
            remaining = seconds_until_close()
            self._print_status("⏳ GATHERING DATA", remaining, total_wait)
            await asyncio.sleep(1)
        self._clear_status()

        log.info(f"[PHASE 3] Unified entry loop -- trading EARLY at good prices")

        # ── Phase 3: UNIFIED ENTRY LOOP (T+20s to T-5s) ─────────────────
        hard_deadline = cfg.timing.hard_deadline_seconds
        entry_poll_interval = 5  # Check every 5s (REST calls have latency)
        last_log_time = 0.0

        while self._running and seconds_until_close() > hard_deadline:
            if self._order_placed_this_window:
                break

            remaining = seconds_until_close()
            elapsed = seconds_into_window()

            sig = self.strategy.analyze(self._window_open_price)

            if self._best_signal is None or abs(sig.score) > abs(self._best_signal.score):
                self._best_signal = sig
            score_jump = abs(sig.score - self._last_score)
            self._last_score = sig.score

            if sig.direction == "up":
                buy_token = market.yes_token_id
                complement_token = market.no_token_id
            else:
                buy_token = market.no_token_id
                complement_token = market.yes_token_id

            real_book = await self._fetch_effective_book(buy_token, complement_token)
            if real_book:
                ask_price = real_book["best_ask"]
                ask_depth = real_book["ask_depth"]
                book_source = real_book.get("source", "?")
            else:
                # Dry run or API failure — estimate price from BTC delta
                ask_price = estimate_token_price(abs(sig.delta_pct))
                ask_depth = 999  # Assume unlimited in dry run
                book_source = "estimated"

            # Check entry conditions
            if ask_price and ask_price > 0:
                min_conf = self._min_confidence_for_price(ask_price)

                phase = "🎯 EARLY" if elapsed < 120 else "📊 MONITOR" if remaining > 30 else "⚡ SNIPE"
                self._print_status(
                    phase, remaining, total_wait,
                    f"ask=${ask_price:.2f} conf={sig.confidence:.2f} need={min_conf:.2f}",
                )

                # Log periodically (every 15s) to avoid spam
                now = time.time()
                if now - last_log_time > 15:
                    self._clear_status()
                    log.info(
                        f"[CHECK] T+{elapsed:.0f}s | ask=${ask_price:.2f} ({book_source}) "
                        f"conf={sig.confidence:.2f} need>={min_conf:.2f} "
                        f"dir={sig.direction} score={sig.score:.1f} "
                        f"depth={ask_depth:.0f}"
                    )
                    last_log_time = now

                # ENTRY DECISION: confidence meets price-adjusted threshold
                if sig.confidence >= min_conf:
                    self._clear_status()
                    log.info(
                        f"[ENTRY] Confidence {sig.confidence:.2f} >= {min_conf:.2f} "
                        f"at ask=${ask_price:.2f} -- placing order!"
                    )
                    await self._place_order(sig, market)
                    if self._order_placed_this_window:
                        break

                # Spike detection: big score jump → immediate entry
                if score_jump >= 1.5 and sig.confidence >= 0.4 and ask_price < 0.85:
                    self._clear_status()
                    log.info(f"[SPIKE] Score jumped {score_jump:.1f} -- immediate order")
                    await self._place_order(sig, market)
                    if self._order_placed_this_window:
                        break

            arb = self.orderbook.has_arbitrage()
            if arb and arb > 0.01:
                now = time.time()
                margin_changed = abs(arb - self._last_arb_logged) > 0.02
                cooldown_expired = (now - self._last_arb_time) > 30
                if margin_changed or cooldown_expired:
                    self._clear_status()
                    log.info(f"[ARB] Arbitrage detected: {arb:.4f} margin")
                    self._last_arb_logged = arb
                    self._last_arb_time = now

            if remaining < 15:
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(entry_poll_interval)

        self._clear_status()

        # ── Phase 4: Hard deadline fallback ──────────────────────────────
        if not self._order_placed_this_window and self._best_signal and self._running:
            remaining = seconds_until_close()
            if remaining > 1 and remaining <= hard_deadline:
                if self._best_signal.confidence >= 0.5:
                    log.info(
                        f"[DEADLINE] Hard deadline -- placing best signal "
                        f"(conf={self._best_signal.confidence:.2f})"
                    )
                    await self._place_order(self._best_signal, market)
                else:
                    log.info(
                        f"[SKIP] No trade -- best signal too weak "
                        f"(conf={self._best_signal.confidence:.2f})"
                    )

        await self.feed.stop_tick_stream()

        await self._wait_for_window_close()

        if not self.dry_run and self.executor.get_active_orders():
            cancelled = await self.executor.cancel_all()
            if cancelled:
                log.info(f"[CLEANUP] Cancelled {cancelled} unfilled orders after window close")

        # ── Phase 5: Resolution ──────────────────────────────────────────
        await self._resolve_window(market)

    async def _place_order(self, sig: Signal, market: MarketWindow):
        """Place BUY order using effective pricing (includes complement-token MINT matching)."""
        if self._order_placed_this_window:
            return

        if sig.direction == "up":
            buy_token = market.yes_token_id
            complement_token = market.no_token_id
        else:
            buy_token = market.no_token_id
            complement_token = market.yes_token_id

        total_usd = self.risk.get_bet_size(sig.confidence)
        if total_usd <= 0:
            log.debug("Bet size is 0 -- skipping")
            return

        buy_usd = total_usd

        ebook = await self._fetch_effective_book(buy_token, complement_token)
        if ebook:
            buy_ask = ebook["best_ask"]
            buy_ask_size = ebook["ask_depth"]
            buy_bid = ebook["best_bid"]
            book_source = ebook.get("source", "?")
            log.info(
                f"[BOOK] EFFECTIVE: ask=${buy_ask:.2f} ({book_source}, "
                f"{buy_ask_size:.0f} shares) bid=${buy_bid:.2f} | dir={sig.direction}"
            )
        else:
            # Dry run or API failure — use WS or estimates
            buy_book = self.orderbook.yes_book if sig.direction == "up" else self.orderbook.no_book
            buy_ask = buy_book.best_ask
            buy_ask_size = 999
            buy_bid = buy_book.best_bid
            book_source = "ws"

        buy_target = buy_ask if (buy_ask and buy_ask > 0) else estimate_token_price(sig.delta_pct)

        min_conf = self._min_confidence_for_price(buy_target)
        if min_conf > 1.0:
            log.warning(f"[SKIP] Effective ask ${buy_target:.2f} too high -- no profitable entry")
            return

        results = []
        buy_shares = max(5, int(buy_usd / buy_target))
        if buy_ask_size > 0:
            buy_shares = min(buy_shares, max(5, int(buy_ask_size)))

        mp = round(buy_target, 2)
        mp = max(0.01, min(0.99, mp))

        log.info(
            f"[EXEC] FOK BUY {buy_shares} shares @ ${mp:.2f} "
            f"(${buy_usd:.2f} total, conf={sig.confidence:.2f}, src={book_source})"
        )
        buy_result = await self.executor.place_taker_order(
            token_id=buy_token, price=mp,
            size=buy_shares, side=OrderSide.BUY,
        )

        if not buy_result.success:
            mp2 = min(0.99, mp + 0.02)
            log.info(f"[FALLBACK 1] FOK failed -- trying +2c @ ${mp2:.2f}")
            buy_result = await self.executor.place_taker_order(
                token_id=buy_token, price=mp2,
                size=buy_shares, side=OrderSide.BUY,
            )

        if not buy_result.success:
            log.info(f"[FALLBACK 2] FOK failed -- GTC maker @ ${mp:.2f} (wait for MINT match)")
            buy_result = await self.executor.place_maker_order(
                token_id=buy_token, price=mp,
                size=buy_shares, side=OrderSide.BUY,
            )

        if not buy_result.success:
            log.info(f"[FALLBACK 3] GTC failed -- market order ${buy_usd:.2f}")
            buy_result = await self.executor.place_market_order(
                token_id=buy_token, amount=buy_usd, side=OrderSide.BUY,
            )

        results.append(("BUY", buy_result))

        any_success = False
        total_cost = 0.0
        total_shares = 0.0
        buy_size = 0.0
        buy_cost = 0.0

        for side_label, result in results:
            if result.success:
                any_success = True
                cost = result.size * result.price
                total_cost += cost
                total_shares += result.size
                buy_size += result.size
                buy_cost += cost
                log.info(
                    f"[ORDER PLACED] {side_label} {sig.direction.upper()} "
                    f"conf={sig.confidence:.2f} price=${result.price:.4f} "
                    f"shares={result.size} cost=${cost:.2f} "
                    f"strategy={result.strategy}"
                )

        if any_success:
            self._order_placed_this_window = True
            self.trade_count += 1
            avg_price = total_cost / total_shares if total_shares > 0 else 0

            self.risk.record_trade(
                window_ts=market.window_ts,
                direction=sig.direction,
                confidence=sig.confidence,
                entry_price=avg_price,
                size=total_shares,
                cost=total_cost,
                buy_size=buy_size,
                buy_cost=buy_cost,
                sell_size=0.0,
                sell_cost=0.0,
                signal_components=sig.components,
                score=sig.score,
                delta_pct=sig.delta_pct,
            )

    async def _place_market_making_orders(self, sig: Signal, market: MarketWindow):
        """Post maker orders on both sides to earn rebates with minimal directional risk."""
        bet_usd = self.risk.get_bet_size(sig.confidence)
        if bet_usd <= 0:
            return

        half_usd = bet_usd / 2.0
        yes_book = self.orderbook.yes_book
        no_book = self.orderbook.no_book

        log.info(f"[MM] Market making both sides (conf={sig.confidence:.2f}, ${bet_usd:.2f})")

        # BUY YES at slightly below best bid (or estimated)
        yes_bid = yes_book.best_bid
        yes_price = round(yes_bid - 0.01, 2) if yes_bid else 0.48
        yes_price = max(0.01, min(0.99, yes_price))
        yes_shares = max(5, int(half_usd / yes_price))

        # BUY NO at slightly below best bid (or estimated)
        no_bid = no_book.best_bid
        no_price = round(no_bid - 0.01, 2) if no_bid else 0.48
        no_price = max(0.01, min(0.99, no_price))
        no_shares = max(5, int(half_usd / no_price))

        results = []

        if yes_shares >= 5:
            r = await self.executor.place_maker_order(
                token_id=market.yes_token_id, price=yes_price,
                size=yes_shares, side=OrderSide.BUY,
            )
            if r.success:
                results.append(r)
                log.info(f"[MM] BUY YES {yes_shares} @ ${yes_price:.2f}")

        if no_shares >= 5:
            r = await self.executor.place_maker_order(
                token_id=market.no_token_id, price=no_price,
                size=no_shares, side=OrderSide.BUY,
            )
            if r.success:
                results.append(r)
                log.info(f"[MM] BUY NO  {no_shares} @ ${no_price:.2f}")

        if results:
            self._order_placed_this_window = True
            self.trade_count += 1
            total_cost = sum(r.size * r.price for r in results)
            total_shares = sum(r.size for r in results)

            # MM: both sides are BUY — record all as buy_size/buy_cost
            self.risk.record_trade(
                window_ts=market.window_ts,
                direction="mm",  # market making — no directional bias
                confidence=sig.confidence,
                entry_price=total_cost / total_shares if total_shares else 0,
                size=total_shares,
                cost=total_cost,
                buy_size=total_shares,
                buy_cost=total_cost,
            )

    async def _fetch_rest_orderbook(self, token_id: str) -> Optional[dict]:
        """Fetch single-token orderbook via REST. DEPRECATED: Use _fetch_effective_book() instead."""
        if not self.executor._client:
            return None
        try:
            book = await asyncio.to_thread(self.executor._client.get_order_book, token_id)
            if not book or not book.asks:
                return None
            # CRITICAL: asks are sorted DESCENDING (worst first), so we must find min
            best_ask = min(float(a.price) for a in book.asks)
            # Sum depth at the best (lowest) few ask levels
            ask_prices = sorted([(float(a.price), float(a.size)) for a in book.asks])
            ask_depth = sum(size for _, size in ask_prices[:3])
            # Bids are sorted ASCENDING (worst first), so find max
            best_bid = max(float(b.price) for b in book.bids) if book.bids else 0.0
            return {"best_ask": best_ask, "ask_depth": ask_depth, "best_bid": best_bid}
        except Exception as e:
            log.debug(f"REST orderbook fetch failed: {e}")
            return None

    async def _fetch_effective_book(
        self, target_token: str, complement_token: str
    ) -> Optional[dict]:
        """Fetch orderbook combining direct asks + synthetic asks from complement-token MINT matching."""
        if not self.executor._client:
            return None

        try:
            ebook = await self.executor.get_effective_book(target_token, complement_token)
            if not ebook or (ebook.best_ask <= 0 and ebook.best_bid <= 0):
                return None

            # Determine if best ask came from direct or synthetic
            source = "direct"
            if ebook.synthetic_asks and ebook.direct_asks:
                if ebook.synthetic_asks[0].price < ebook.direct_asks[0].price:
                    source = "synthetic"
            elif ebook.synthetic_asks and not ebook.direct_asks:
                source = "synthetic"

            return {
                "best_ask": ebook.best_ask,
                "ask_depth": ebook.ask_depth,
                "best_bid": ebook.best_bid,
                "bid_depth": ebook.bid_depth,
                "neg_risk": ebook.neg_risk,
                "tick_size": ebook.tick_size,
                "source": source,
            }
        except Exception as e:
            log.debug(f"Effective book fetch failed: {e}")
            return None

    async def _resolve_window(self, market: MarketWindow):
        # Compare Binance prices: open vs close
        close_price = self.feed.current_price
        open_price = self._window_open_price

        # If close price is stale/zero, try refreshing from REST
        if close_price <= 0:
            await self.feed.refresh_candles_rest()
            close_price = self.feed.current_price

        if open_price <= 0 or close_price <= 0:
            log.warning(
                f"[RESOLVE] Cannot resolve — missing price data "
                f"(open=${open_price:,.2f} close=${close_price:,.2f})"
            )
            return

        actual_direction = "up" if close_price > open_price else "down"
        delta_pct = (close_price - open_price) / open_price * 100

        log.info(
            f"[RESOLVE] BTC {actual_direction.upper()} "
            f"open=${open_price:,.2f} close=${close_price:,.2f} "
            f"delta={delta_pct:+.4f}%"
        )

        market.outcome = actual_direction
        market.resolved = True

        # Resolve the last trade if it belongs to this window
        if self.risk.trades:
            last_trade = self.risk.trades[-1]
            if last_trade.window_ts == market.window_ts and last_trade.outcome is None:
                if last_trade.direction == "mm":
                    # Market-making: bought YES and NO — one pays $1, other $0
                    self.risk.resolve_mm_trade(last_trade)
                else:
                    # Directional trade (BUY winner + SELL loser)
                    won = last_trade.direction == actual_direction
                    self.risk.resolve_trade(last_trade, won)

                    # Update adaptive ML weights from outcome
                    if last_trade.signal_components:
                        self.adaptive.update_after_resolution(
                            components=last_trade.signal_components,
                            actual_direction=actual_direction,
                            predicted_direction=last_trade.direction,
                            won=won,
                        )
                        self.adaptive.save_to_disk()

        # Check for position merging opportunities
        if not self.dry_run:
            await self.merger.check_and_merge(market.condition_id)

        if self.trade_count % 10 == 0 and self.trade_count > 0:
            self.risk.print_stats()

        self.market.cleanup_old()

    async def _wait_for_window_close(self):
        _, close_ts = calc_current_window()
        total = max(0.0, close_ts - time.time())
        while self._running:
            remaining = close_ts - time.time()
            if remaining <= 0:
                break
            status = "✅ WAITING" if self._order_placed_this_window else "⏸️  IDLE"
            self._print_status(status, remaining, total)
            await asyncio.sleep(0.5)
        self._clear_status()
        await asyncio.sleep(2)

    async def _get_usdc_balance(self) -> Optional[float]:
        try:
            if not self.executor._session:
                return None
            url = f"{cfg.clob_url}/balance"
            headers = {}
            if self.executor._client:
                try:
                    creds = self.executor._client.creds
                    headers = {
                        "POLY_API_KEY": creds.api_key,
                        "POLY_PASSPHRASE": creds.api_passphrase,
                        "POLY_SECRET": creds.api_secret,
                    }
                except Exception:
                    pass
            async with self.executor._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict):
                        return float(data.get("balance", data.get("cash", 0)))
                    return float(data) if data else None
                else:
                    log.debug(f"Balance query returned {resp.status}")
        except Exception as e:
            log.debug(f"Balance query failed: {e}")
        return None

    async def _wait_for_next_window(self):
        """Wait for next window to start, pre-fetching market data near the end."""
        from market import calc_next_window
        _, close_ts = calc_current_window()
        total = max(0.0, close_ts - time.time())
        log.info(f"Waiting {total:.0f}s for next window...")

        prefetched = False
        while self._running:
            remaining = close_ts - time.time()
            if remaining <= 0:
                break
            self._print_status(">> NEXT WINDOW", remaining, total)

            if not prefetched and remaining < 10:
                next_ts, _ = calc_next_window()
                try:
                    next_market = await self.market.get_market(next_ts)
                    if next_market and next_market.yes_token_id:
                        log.info(
                            f"[PREFETCH] Next window market ready: "
                            f"slug={next_market.slug}"
                        )
                        if not self.dry_run:
                            await self.executor.get_fee_rate(next_market.yes_token_id)
                            await self.executor.get_fee_rate(next_market.no_token_id)
                    prefetched = True
                except Exception as e:
                    log.debug(f"Prefetch failed: {e}")

            await asyncio.sleep(1)
        self._clear_status()
        await asyncio.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5-Min Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["safe", "aggressive", "degen"],
        default=None,
        help="Trading mode (default: from .env or 'safe')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Simulate trades without placing real orders",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live trading (override DRY_RUN=true in .env)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run for a single window then exit",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=0,
        help="Maximum number of trades before stopping",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Determine dry_run setting
    if args.live:
        dry_run = False
    elif args.dry_run is not None:
        dry_run = args.dry_run
    else:
        dry_run = cfg.bot.dry_run

    mode = args.mode or cfg.bot.mode

    bot = TradingBot(
        mode=mode,
        dry_run=dry_run,
        run_once=args.once,
        max_trades=args.max_trades,
    )

    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
