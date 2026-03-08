"""Diagnostic: verify effective pricing via complement token MINT matching vs direct orderbooks."""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def main():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds
    import aiohttp

    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    client = ClobClient(
        CLOB_URL,
        key=os.getenv("POLY_PRIVATE_KEY"),
        chain_id=137,
        creds=ApiCreds(
            api_key=os.getenv("POLY_API_KEY"),
            api_secret=os.getenv("POLY_API_SECRET"),
            api_passphrase=os.getenv("POLY_API_PASSPHRASE"),
        ),
        signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "1")),
        funder=os.getenv("POLY_FUNDER_ADDRESS") or None,
    )

    import time
    now = int(time.time())
    window_ts = now - (now % 300)
    slug = f"btc-updown-5m-{window_ts}"

    print(f"=== COMPLEMENT PRICING DIAGNOSTIC ===")
    print(f"Window: {window_ts} (T+{now - window_ts}s)")
    print(f"Slug: {slug}")
    print()

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{GAMMA_URL}/events?slug={slug}") as resp:
            if resp.status != 200:
                print(f"Gamma API error: {resp.status}")
                return
            data = await resp.json()

    if not data:
        print("No event found for this slug. Try a different window.")
        return

    event = data[0] if isinstance(data, list) else data
    markets = event.get("markets", [])
    if not markets:
        print("No markets found")
        return

    market = markets[0]
    token_ids = market.get("clobTokenIds", [])
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)

    if len(token_ids) < 2:
        print(f"Need 2 token IDs, got {len(token_ids)}")
        return

    yes_token = token_ids[0]  # UP
    no_token = token_ids[1]   # DOWN

    gamma_best_bid = market.get("bestBid", "?")
    gamma_best_ask = market.get("bestAsk", "?")
    gamma_last_trade = market.get("lastTradePrice", "?")

    print(f"YES (UP) token:  {yes_token[:20]}...")
    print(f"NO (DOWN) token: {no_token[:20]}...")
    print(f"Gamma bestBid: {gamma_best_bid}")
    print(f"Gamma bestAsk: {gamma_best_ask}")
    print(f"Gamma lastTrade: {gamma_last_trade}")
    print()

    print("--- NEG RISK CHECK ---")
    try:
        yes_neg_risk = client.get_neg_risk(yes_token)
        print(f"YES token neg_risk: {yes_neg_risk}")
    except Exception as e:
        print(f"YES token neg_risk error: {e}")
        yes_neg_risk = None

    try:
        no_neg_risk = client.get_neg_risk(no_token)
        print(f"NO token neg_risk:  {no_neg_risk}")
    except Exception as e:
        print(f"NO token neg_risk error: {e}")
        no_neg_risk = None
    print()

    print("--- YES (UP) TOKEN ORDERBOOK ---")
    yes_book = client.get_order_book(yes_token)
    print(f"  neg_risk: {getattr(yes_book, 'neg_risk', '?')}")
    print(f"  tick_size: {getattr(yes_book, 'tick_size', '?')}")

    if yes_book.asks:
        print(f"  Asks ({len(yes_book.asks)} levels):")
        for a in yes_book.asks[:8]:
            print(f"    ${float(a.price):.2f} x {float(a.size):.1f}")
    else:
        print("  No asks")

    if yes_book.bids:
        print(f"  Bids ({len(yes_book.bids)} levels):")
        for b in yes_book.bids[:8]:
            print(f"    ${float(b.price):.2f} x {float(b.size):.1f}")
    else:
        print("  No bids")
    print()

    print("--- NO (DOWN) TOKEN ORDERBOOK ---")
    no_book = client.get_order_book(no_token)
    print(f"  neg_risk: {getattr(no_book, 'neg_risk', '?')}")
    print(f"  tick_size: {getattr(no_book, 'tick_size', '?')}")

    if no_book.asks:
        print(f"  Asks ({len(no_book.asks)} levels):")
        for a in no_book.asks[:8]:
            print(f"    ${float(a.price):.2f} x {float(a.size):.1f}")
    else:
        print("  No asks")

    if no_book.bids:
        print(f"  Bids ({len(no_book.bids)} levels):")
        for b in no_book.bids[:8]:
            print(f"    ${float(b.price):.2f} x {float(b.size):.1f}")
    else:
        print("  No bids")
    print()

    print("=== EFFECTIVE PRICING (MINT MATCHING) ===")

    # Effective UP asks = direct UP asks + (1 - DOWN bids)
    effective_up_asks = []
    if yes_book.asks:
        for a in yes_book.asks:
            effective_up_asks.append(("direct", float(a.price), float(a.size)))

    if no_book.bids:
        for b in no_book.bids:
            synth_price = round(1.0 - float(b.price), 2)
            if 0.01 <= synth_price <= 0.99:
                effective_up_asks.append(("MINT", synth_price, float(b.size)))

    effective_up_asks.sort(key=lambda x: x[1])
    print(f"\nEffective UP (YES) asks (best available to BUY UP):")
    for src, price, size in effective_up_asks[:10]:
        print(f"  [{src:6s}] ${price:.2f} x {size:.1f}")

    # Effective DOWN asks = direct DOWN asks + (1 - UP bids)
    effective_down_asks = []
    if no_book.asks:
        for a in no_book.asks:
            effective_down_asks.append(("direct", float(a.price), float(a.size)))

    if yes_book.bids:
        for b in yes_book.bids:
            synth_price = round(1.0 - float(b.price), 2)
            if 0.01 <= synth_price <= 0.99:
                effective_down_asks.append(("MINT", synth_price, float(b.size)))

    effective_down_asks.sort(key=lambda x: x[1])
    print(f"\nEffective DOWN (NO) asks (best available to BUY DOWN):")
    for src, price, size in effective_down_asks[:10]:
        print(f"  [{src:6s}] ${price:.2f} x {size:.1f}")

    print(f"\n=== SUMMARY ===")
    if effective_up_asks:
        best_up_ask = effective_up_asks[0]
        print(f"Best effective UP ask:   ${best_up_ask[1]:.2f} ({best_up_ask[0]}, {best_up_ask[2]:.1f} shares)")
    if effective_down_asks:
        best_down_ask = effective_down_asks[0]
        print(f"Best effective DOWN ask: ${best_down_ask[1]:.2f} ({best_down_ask[0]}, {best_down_ask[2]:.1f} shares)")
    print(f"Gamma reports:          bid=${gamma_best_bid} ask=${gamma_best_ask}")
    print(f"\nIf effective prices are mid-range ($0.40-$0.60), MINT matching is working!")
    print(f"If still at $0.99, complement token has no mid-range bids either.")


if __name__ == "__main__":
    asyncio.run(main())
