# Polymarket BTC 5-Minute Up/Down Trading Bot

Automated trading bot for Polymarket's BTC 5-minute binary markets. Uses a hybrid maker + directional sniper strategy optimized for the February 2026 rule changes (no 500ms taker delay, dynamic taker fees).

> **Note:** This project was built with heavy assistance from AI (Claude). Architecture, strategy design, and domain logic are mine, implementation was largely AI-assisted.

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your private key

# Derive API credentials (one-time)
python setup_creds.py
# Add the output to your .env file
```

## Usage
```bash
# Dry run (simulated trades, real data)
python bot.py --dry-run --mode safe

# Single window test
python bot.py --dry-run --once

# Limit number of trades
python bot.py --dry-run --max-trades 50

# Live trading
python bot.py --mode safe

# Backtest
python backtest.py --hours 72 --output results.xlsx
```

## Trading Modes

| Mode | Bet Size | Min Confidence | Description |
|------|----------|---------------|-------------|
| safe | 25% bankroll | 30% | Default. Survives losing streaks. |
| aggressive | Profits only | 20% | Protects principal, compounds gains. |
| degen | 100% bankroll | 0% | Always trades. Will bust often. |

## Architecture

| File | Purpose |
|------|---------|
| `bot.py` | Main engine — timing, lifecycle, order management |
| `strategy.py` | Signal generation — 7 weighted indicators |
| `market.py` | Market discovery, slug calculation, token IDs |
| `orderbook.py` | WebSocket orderbook stream |
| `binance_feed.py` | Binance WebSocket for real-time BTC price |
| `execution.py` | Order placement, cancellation, fee queries |
| `risk.py` | Position sizing, bankroll management, circuit breakers |
| `merge.py` | YES/NO position merging |
| `config.py` | Configuration + .env loading |
| `setup_creds.py` | One-time credential derivation |
| `backtest.py` | Historical backtesting with Excel output |

## Strategy

Composite weighted signal from 7 indicators:

1. **Window Delta** (weight 5-7) — BTC price change vs window open
2. **Micro Momentum** (weight 2) — Last 2 candle direction
3. **Acceleration** (weight 1.5) — Building vs fading momentum
4. **EMA 9/21 Crossover** (weight 1) — Trend direction
5. **RSI 14** (weight 1-2) — Contrarian at extremes
6. **Volume Surge** (weight 1) — Confirms direction
7. **Tick Trend** (weight 2) — Real-time micro data

## 2026 Rule Compliance

- Maker orders pay zero fees and earn USDC rebates
- Dynamic taker fees queried before every order via `feeRateBps`
- Taker orders only used at price extremes where fees are negligible
- 500ms taker delay removed — instant execution
- Minimum 5 shares per order enforced