"""Order execution — fee queries, order placement, cancellation, signing.

CLOB asks sort DESCENDING, bids ASCENDING — always re-sort for real best prices.
Supports complementary MINT matching: BUY UP + BUY DOWN -> MINT pair ($1.00).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aiohttp

from config import cfg

log = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStrategy(str, Enum):
    MAKER_GTC = "maker_gtc"     # Good-til-cancelled limit order (zero fees)
    TAKER_FOK = "taker_fok"     # Fill-or-kill (pays dynamic fee)


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    error: str = ""
    price: float = 0.0
    size: float = 0.0
    side: str = ""
    token_id: str = ""
    strategy: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class EffectiveBook:
    """Effective book combining direct + complement MINT/MERGE liquidity for a token."""
    token_id: str
    # Direct orderbook
    direct_bids: list[BookLevel] = field(default_factory=list)
    direct_asks: list[BookLevel] = field(default_factory=list)
    # Complement-derived (via MINT matching)
    synthetic_asks: list[BookLevel] = field(default_factory=list)  # from complement bids
    synthetic_bids: list[BookLevel] = field(default_factory=list)  # from complement asks
    # Combined effective
    best_ask: float = 0.0       # min of direct + synthetic asks
    best_bid: float = 0.0       # max of direct + synthetic bids
    ask_depth: float = 0.0      # total shares available at best few ask levels
    bid_depth: float = 0.0
    neg_risk: bool = False
    tick_size: str = "0.01"


@dataclass
class Executor:
    """Handles all order operations against the Polymarket CLOB."""

    _client: Optional[object] = field(default=None, repr=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _fee_cache: dict[str, tuple[float, int]] = field(default_factory=dict)  # token_id -> (timestamp, fee_bps)
    _fee_cache_ttl: float = 10.0  # Cache fee rates for 10 seconds
    _neg_risk_cache: dict[str, bool] = field(default_factory=dict)  # token_id -> neg_risk
    _tick_size_cache: dict[str, str] = field(default_factory=dict)  # token_id -> tick_size
    _active_orders: dict[str, dict] = field(default_factory=dict)
    dry_run: bool = True

    async def start(self):
        self._session = aiohttp.ClientSession()

        if not self.dry_run:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                self._client = ClobClient(
                    cfg.clob_url,
                    key=cfg.creds.private_key,
                    chain_id=cfg.chain_id,
                    creds=ApiCreds(
                        api_key=cfg.creds.api_key,
                        api_secret=cfg.creds.api_secret,
                        api_passphrase=cfg.creds.api_passphrase,
                    ),
                    signature_type=cfg.creds.signature_type,
                    funder=cfg.creds.funder_address or None,
                )
                log.info("CLOB client initialized for LIVE trading")
            except ImportError:
                log.error("py-clob-client not installed — falling back to dry run")
                self.dry_run = True
            except Exception as e:
                log.error(f"CLOB client init failed: {e} — falling back to dry run")
                self.dry_run = True
        else:
            log.info("Executor in DRY RUN mode")

    async def stop(self):
        if not self.dry_run:
            await self.cancel_all()
        if self._session:
            await self._session.close()

    async def get_fee_rate(self, token_id: str) -> int:
        """Get fee rate (bps) for a token — MUST be called before every order, it's part of the signed payload."""
        if token_id in self._fee_cache:
            cached_ts, cached_fee = self._fee_cache[token_id]
            if time.time() - cached_ts < self._fee_cache_ttl:
                return cached_fee

        # Use the CLOB client's built-in method (handles auth)
        if self._client and not self.dry_run:
            try:
                fee_bps = await asyncio.to_thread(self._client.get_fee_rate_bps, token_id)
                fee_bps = int(fee_bps)
                self._fee_cache[token_id] = (time.time(), fee_bps)
                log.debug(f"Fee rate for {token_id[:12]}...: {fee_bps} bps")
                return fee_bps
            except Exception as e:
                log.warning(f"Fee rate query error: {e}")

        try:
            url = f"{cfg.clob_url}/fee-rate?tokenID={token_id}"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    log.debug(f"Fee rate HTTP fallback failed: {resp.status}")
                    return 0
                data = await resp.json()

            fee_bps = int(data.get("fee_rate_bps", data.get("feeRateBps", 0)))
            self._fee_cache[token_id] = (time.time(), fee_bps)
            return fee_bps

        except Exception as e:
            log.debug(f"Fee rate fallback error: {e}")
            return 0

    async def get_neg_risk(self, token_id: str) -> bool:
        """Check if token uses neg-risk exchange contract (supports MINT matching across complement tokens)."""
        if token_id in self._neg_risk_cache:
            return self._neg_risk_cache[token_id]

        if self._client and not self.dry_run:
            try:
                neg_risk = await asyncio.to_thread(self._client.get_neg_risk, token_id)
                self._neg_risk_cache[token_id] = bool(neg_risk)
                log.info(f"[NEG RISK] token={token_id[:12]}... neg_risk={neg_risk}")
                return bool(neg_risk)
            except Exception as e:
                log.debug(f"get_neg_risk failed: {e}")

        # Fallback: assume neg_risk=False — BTC 5-min markets are NOT neg_risk
        return False

    async def get_order_book_raw(self, token_id: str) -> Optional[object]:
        """Fetch raw OrderBookSummary from the CLOB for a token."""
        if not self._client or self.dry_run:
            return None
        try:
            book = await asyncio.to_thread(self._client.get_order_book, token_id)
            if book:
                if hasattr(book, 'neg_risk') and book.neg_risk is not None:
                    self._neg_risk_cache[token_id] = bool(book.neg_risk)
                if hasattr(book, 'tick_size') and book.tick_size:
                    self._tick_size_cache[token_id] = str(book.tick_size)
            return book
        except Exception as e:
            log.debug(f"get_order_book_raw failed for {token_id[:12]}...: {e}")
            return None

    async def get_effective_book(
        self, target_token: str, complement_token: str
    ) -> EffectiveBook:
        """Build effective orderbook combining direct asks/bids with complement MINT/MERGE liquidity."""
        result = EffectiveBook(token_id=target_token)

        # Fetch both orderbooks in parallel
        target_book, complement_book = await asyncio.gather(
            self.get_order_book_raw(target_token),
            self.get_order_book_raw(complement_token),
        )

        if target_book:
            result.neg_risk = bool(getattr(target_book, 'neg_risk', False))
            result.tick_size = str(getattr(target_book, 'tick_size', '0.01'))

            if target_book.asks:
                for a in target_book.asks:
                    result.direct_asks.append(BookLevel(float(a.price), float(a.size)))
            if target_book.bids:
                for b in target_book.bids:
                    result.direct_bids.append(BookLevel(float(b.price), float(b.size)))

        # Build synthetic asks from complement's bids (MINT matching)
        # If someone wants to BUY complement at $X, we can BUY target at $(1-X)
        if complement_book and complement_book.bids:
            for b in complement_book.bids:
                synthetic_price = round(1.0 - float(b.price), 2)
                if 0.01 <= synthetic_price <= 0.99:
                    result.synthetic_asks.append(
                        BookLevel(synthetic_price, float(b.size))
                    )

        # Build synthetic bids from complement's asks (MERGE matching)
        # If someone wants to SELL complement at $X, we can SELL target at $(1-X)
        if complement_book and complement_book.asks:
            for a in complement_book.asks:
                synthetic_price = round(1.0 - float(a.price), 2)
                if 0.01 <= synthetic_price <= 0.99:
                    result.synthetic_bids.append(
                        BookLevel(synthetic_price, float(a.size))
                    )

        all_asks = result.direct_asks + result.synthetic_asks
        all_asks.sort(key=lambda x: x.price)  # lowest first
        if all_asks:
            result.best_ask = all_asks[0].price
            seen_prices = set()
            for a in all_asks:
                seen_prices.add(a.price)
                if len(seen_prices) <= 3:
                    result.ask_depth += a.size

        all_bids = result.direct_bids + result.synthetic_bids
        all_bids.sort(key=lambda x: x.price, reverse=True)  # highest first
        if all_bids:
            result.best_bid = all_bids[0].price
            seen_prices = set()
            for b in all_bids:
                seen_prices.add(b.price)
                if len(seen_prices) <= 3:
                    result.bid_depth += b.size

        log.info(
            f"[EFFECTIVE BOOK] token={target_token[:12]}... "
            f"ask=${result.best_ask:.2f} (depth={result.ask_depth:.0f}) "
            f"bid=${result.best_bid:.2f} (depth={result.bid_depth:.0f}) "
            f"direct_asks={len(result.direct_asks)} synth_asks={len(result.synthetic_asks)} "
            f"neg_risk={result.neg_risk}"
        )

        return result

    async def place_maker_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide = OrderSide.BUY,
    ) -> OrderResult:
        """Place a GTC maker limit order (zero fees). In neg-risk markets, can MINT-match against complement buyers."""
        log.info(
            f"[ORDER] MAKER GTC {side.value} {size:.1f} shares @ ${price:.4f} "
            f"token={token_id[:12]}..."
        )

        if self.dry_run:
            return OrderResult(
                success=True,
                order_id=f"dry_{int(time.time()*1000)}",
                price=price,
                size=size,
                side=side.value,
                token_id=token_id,
                strategy=OrderStrategy.MAKER_GTC.value,
            )

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            fee_bps = await self.get_fee_rate(token_id)
            neg_risk = await self.get_neg_risk(token_id)

            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY if side == OrderSide.BUY else SELL,
                token_id=token_id,
                fee_rate_bps=fee_bps,
            )

            # Pass neg_risk option so the order is signed against the correct exchange contract
            options = PartialCreateOrderOptions(neg_risk=neg_risk)
            tick_size = self._tick_size_cache.get(token_id)
            if tick_size:
                try:
                    from py_clob_client.clob_types import TickSize
                    options.tick_size = TickSize(tick_size)
                except Exception:
                    pass

            signed_order = await asyncio.to_thread(
                self._client.create_order, order_args, options
            )
            resp = await asyncio.to_thread(self._client.post_order, signed_order, OrderType.GTC)

            order_id = resp.get("orderID", resp.get("id", "")) if isinstance(resp, dict) else ""
            if order_id:
                self._active_orders[order_id] = {
                    "token_id": token_id,
                    "price": price,
                    "size": size,
                    "side": side.value,
                    "strategy": OrderStrategy.MAKER_GTC.value,
                    "timestamp": time.time(),
                }
                log.info(f"[POSTED] Maker order posted (may not fill): id={order_id}")

                # Poll for fill status — maker orders aren't instantly filled
                filled_size = await self._poll_order_fill(order_id, timeout=8.0)
                if filled_size > 0:
                    self._active_orders.pop(order_id, None)
                    log.info(f"[FILL] Maker order filled: {filled_size} shares")
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        price=price,
                        size=filled_size,
                        side=side.value,
                        token_id=token_id,
                        strategy=OrderStrategy.MAKER_GTC.value,
                    )
                else:
                    log.warning(f"[UNFILLED] Maker order not filled within timeout")
                    await self.cancel_order(order_id)
                    return OrderResult(success=False, error="unfilled")
            else:
                error = resp.get("error", resp.get("message", str(resp))) if isinstance(resp, dict) else str(resp)
                log.error(f"[ORDER FAILED] {error}")
                return OrderResult(success=False, error=error)

        except Exception as e:
            log.error(f"[ORDER ERROR] Maker order failed: {e}")
            return OrderResult(success=False, error=str(e))

    async def place_taker_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide = OrderSide.BUY,
    ) -> OrderResult:
        """Place a FOK taker order (pays dynamic fee). Use only at price extremes."""
        log.info(
            f"[ORDER] TAKER FOK {side.value} {size:.1f} shares @ ${price:.4f} "
            f"token={token_id[:12]}..."
        )

        if self.dry_run:
            return OrderResult(
                success=True,
                order_id=f"dry_fok_{int(time.time()*1000)}",
                price=price,
                size=size,
                side=side.value,
                token_id=token_id,
                strategy=OrderStrategy.TAKER_FOK.value,
            )

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            fee_bps = await self.get_fee_rate(token_id)
            neg_risk = await self.get_neg_risk(token_id)

            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY if side == OrderSide.BUY else SELL,
                token_id=token_id,
                fee_rate_bps=fee_bps,
            )

            options = PartialCreateOrderOptions(neg_risk=neg_risk)
            tick_size = self._tick_size_cache.get(token_id)
            if tick_size:
                try:
                    from py_clob_client.clob_types import TickSize
                    options.tick_size = TickSize(tick_size)
                except Exception:
                    pass

            signed_order = await asyncio.to_thread(
                self._client.create_order, order_args, options
            )
            resp = await asyncio.to_thread(self._client.post_order, signed_order, OrderType.FOK)

            order_id = resp.get("orderID", resp.get("id", "")) if isinstance(resp, dict) else ""
            if order_id:
                log.info(f"[FILL] Taker FOK order filled: id={order_id}")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    price=price,
                    size=size,
                    side=side.value,
                    token_id=token_id,
                    strategy=OrderStrategy.TAKER_FOK.value,
                )
            else:
                error = resp.get("error", resp.get("message", str(resp))) if isinstance(resp, dict) else str(resp)
                log.error(f"[ORDER FAILED] FOK: {error}")
                return OrderResult(success=False, error=error)

        except Exception as e:
            log.error(f"[ORDER ERROR] Taker order failed: {e}")
            return OrderResult(success=False, error=str(e))

    async def place_market_order(
        self,
        token_id: str,
        amount: float,
        side: OrderSide = OrderSide.BUY,
    ) -> OrderResult:
        """Place a market order — walks the book to find execution price for the given amount."""
        log.info(
            f"[ORDER] MARKET {side.value} ${amount:.2f} "
            f"token={token_id[:12]}..."
        )

        if self.dry_run:
            est_price = 0.50
            est_shares = amount / est_price
            return OrderResult(
                success=True,
                order_id=f"dry_mkt_{int(time.time()*1000)}",
                price=est_price,
                size=est_shares,
                side=side.value,
                token_id=token_id,
                strategy="market",
            )

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            fee_bps = await self.get_fee_rate(token_id)
            neg_risk = await self.get_neg_risk(token_id)

            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=BUY if side == OrderSide.BUY else SELL,
                fee_rate_bps=fee_bps,
            )

            options = PartialCreateOrderOptions(neg_risk=neg_risk)
            tick_size = self._tick_size_cache.get(token_id)
            if tick_size:
                try:
                    from py_clob_client.clob_types import TickSize
                    options.tick_size = TickSize(tick_size)
                except Exception:
                    pass

            signed_order = await asyncio.to_thread(
                self._client.create_market_order, market_args, options
            )
            resp = await asyncio.to_thread(
                self._client.post_order, signed_order, OrderType.FOK
            )

            order_id = resp.get("orderID", resp.get("id", "")) if isinstance(resp, dict) else ""
            if order_id:
                exec_price = 0.0
                exec_size = 0.0
                try:
                    order_data = getattr(signed_order, 'order', signed_order)
                    if isinstance(order_data, dict):
                        exec_price = float(order_data.get('price', 0))
                        exec_size = float(order_data.get('size', 0))
                    else:
                        exec_price = float(getattr(order_data, 'price', 0))
                        exec_size = float(getattr(order_data, 'size', 0))
                except Exception:
                    pass
                if exec_price <= 0:
                    exec_price = 0.50  # Fallback estimate
                if exec_size <= 0:
                    exec_size = amount / exec_price

                log.info(
                    f"[FILL] Market order filled: id={order_id} "
                    f"price=${exec_price:.4f} size={exec_size:.1f}"
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    price=exec_price,
                    size=exec_size,
                    side=side.value,
                    token_id=token_id,
                    strategy="market",
                )
            else:
                error = resp.get("error", resp.get("message", str(resp))) if isinstance(resp, dict) else str(resp)
                log.error(f"[ORDER FAILED] Market: {error}")
                return OrderResult(success=False, error=error)

        except Exception as e:
            log.error(f"[ORDER ERROR] Market order failed: {e}")
            return OrderResult(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        log.info(f"[CANCEL] order_id={order_id}")

        if self.dry_run:
            self._active_orders.pop(order_id, None)
            return True

        try:
            await asyncio.to_thread(self._client.cancel, order_id)
            self._active_orders.pop(order_id, None)
            log.info(f"[CANCELLED] order_id={order_id}")
            return True
        except Exception as e:
            log.error(f"Cancel failed for {order_id}: {e}")
            return False

    async def cancel_all(self) -> int:
        if self.dry_run:
            count = len(self._active_orders)
            self._active_orders.clear()
            return count

        try:
            await asyncio.to_thread(self._client.cancel_all)
            count = len(self._active_orders)
            self._active_orders.clear()
            log.info(f"[CANCEL ALL] Cancelled {count} orders")
            return count
        except Exception as e:
            log.error(f"Cancel all failed: {e}")
            return 0

    async def _poll_order_fill(self, order_id: str, timeout: float = 8.0) -> float:
        """Poll until filled or timeout, returns filled size."""
        if self.dry_run:
            return self._active_orders.get(order_id, {}).get("size", 0)

        start = time.time()
        while time.time() - start < timeout:
            try:
                order = await asyncio.to_thread(self._client.get_order, order_id)
                if not order or not isinstance(order, dict):
                    return 0

                status = order.get("status", "").upper()
                if status in ("MATCHED", "FILLED"):
                    filled = float(order.get("size_matched", order.get("sizeMatched", 0)))
                    return filled if filled > 0 else float(order.get("size", 0))
                elif status in ("CANCELLED", "EXPIRED"):
                    return 0

            except Exception as e:
                log.debug(f"Poll order {order_id}: {e}")

            await asyncio.sleep(1.0)

        return 0

    def get_active_orders(self) -> dict[str, dict]:
        return dict(self._active_orders)
