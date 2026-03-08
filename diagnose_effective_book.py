"""Diagnostic: test Executor.get_effective_book() against live CLOB + Gamma data."""

import asyncio
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


async def main():
    from execution import Executor, OrderSide
    import aiohttp

    GAMMA_URL = "https://gamma-api.polymarket.com"

    now = int(time.time())
    window_ts = now - (now % 300)
    slug = f"btc-updown-5m-{window_ts}"

    print(f"=== EFFECTIVE BOOK TEST (Production Code) ===")
    print(f"Window: {window_ts} (T+{now - window_ts}s)")
    print(f"Slug: {slug}")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{GAMMA_URL}/events?slug={slug}") as resp:
            if resp.status != 200:
                print(f"Gamma error: {resp.status}")
                return
            data = await resp.json()

    if not data:
        print("No event found")
        return

    event = data[0] if isinstance(data, list) else data
    market = event.get("markets", [{}])[0]
    token_ids = market.get("clobTokenIds", [])
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)

    yes_token = token_ids[0]
    no_token = token_ids[1]

    print(f"YES (UP): {yes_token[:20]}...")
    print(f"NO (DOWN): {no_token[:20]}...")
    print(f"Gamma: bid={market.get('bestBid')} ask={market.get('bestAsk')}")
    print()

    executor = Executor(dry_run=False)
    await executor.start()

    if not executor._client:
        print("CLOB client not initialized!")
        await executor.stop()
        return

    neg_risk = await executor.get_neg_risk(yes_token)
    print(f"neg_risk: {neg_risk}")
    print()

    print("--- Testing get_effective_book(UP, DOWN) ---")
    up_book = await executor.get_effective_book(yes_token, no_token)
    print(f"  best_ask: ${up_book.best_ask:.2f}")
    print(f"  best_bid: ${up_book.best_bid:.2f}")
    print(f"  ask_depth: {up_book.ask_depth:.1f} shares")
    print(f"  bid_depth: {up_book.bid_depth:.1f} shares")
    print(f"  direct_asks: {len(up_book.direct_asks)} levels")
    print(f"  synthetic_asks: {len(up_book.synthetic_asks)} levels")
    print(f"  Top 5 direct asks:")
    sorted_direct = sorted(up_book.direct_asks, key=lambda x: x.price)
    for a in sorted_direct[:5]:
        print(f"    ${a.price:.2f} x {a.size:.1f}")
    print(f"  Top 5 synthetic asks (from DOWN bids):")
    sorted_synth = sorted(up_book.synthetic_asks, key=lambda x: x.price)
    for a in sorted_synth[:5]:
        print(f"    ${a.price:.2f} x {a.size:.1f}")
    print()

    print("--- Testing get_effective_book(DOWN, UP) ---")
    down_book = await executor.get_effective_book(no_token, yes_token)
    print(f"  best_ask: ${down_book.best_ask:.2f}")
    print(f"  best_bid: ${down_book.best_bid:.2f}")
    print(f"  ask_depth: {down_book.ask_depth:.1f} shares")
    print()

    print("=== COMPARISON ===")
    print(f"Our effective UP ask:   ${up_book.best_ask:.2f}")
    print(f"Our effective DOWN ask: ${down_book.best_ask:.2f}")
    print(f"Gamma bestBid: {market.get('bestBid')}")
    print(f"Gamma bestAsk: {market.get('bestAsk')}")
    print(f"Sum of best asks: {up_book.best_ask + down_book.best_ask:.2f} (should be ~1.00)")
    print()

    from strategy import estimate_token_price
    delta_pct = 0.03  # Typical delta
    est = estimate_token_price(delta_pct)
    print(f"Strategy estimated price (delta=0.03%): ${est:.2f}")
    print(f"Real effective ask for UP: ${up_book.best_ask:.2f}")
    print(f"Real effective ask for DOWN: ${down_book.best_ask:.2f}")
    print()

    if up_book.best_ask < 0.90 and up_book.best_ask > 0.01:
        print("SUCCESS: Mid-range pricing available! Bot should be able to trade!")
    else:
        print("WARNING: Still no mid-range pricing")

    await executor.stop()


if __name__ == "__main__":
    asyncio.run(main())
