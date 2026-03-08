"""Configuration loader -- reads .env and exposes typed settings."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


@dataclass
class PolymarketCreds:
    private_key: str = field(default_factory=lambda: _env("POLY_PRIVATE_KEY"))
    api_key: str = field(default_factory=lambda: _env("POLY_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("POLY_API_SECRET"))
    api_passphrase: str = field(default_factory=lambda: _env("POLY_API_PASSPHRASE"))
    funder_address: str = field(default_factory=lambda: _env("POLY_FUNDER_ADDRESS"))
    signature_type: int = field(default_factory=lambda: _env_int("POLY_SIGNATURE_TYPE", 1))


@dataclass
class BotSettings:
    starting_bankroll: float = field(default_factory=lambda: _env_float("STARTING_BANKROLL", 50.0))
    min_bet: float = field(default_factory=lambda: _env_float("MIN_BET", 5.0))
    max_bet_pct: float = field(default_factory=lambda: _env_float("MAX_BET_PCT", 0.25))
    mode: str = field(default_factory=lambda: _env("BOT_MODE", "safe"))
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))


@dataclass
class RiskSettings:
    max_daily_loss_pct: float = field(default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 0.20))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 5))


@dataclass
class TimingSettings:
    snipe_seconds_before_close: int = field(default_factory=lambda: _env_int("SNIPE_SECONDS_BEFORE_CLOSE", 12))
    hard_deadline_seconds: int = field(default_factory=lambda: _env_int("HARD_DEADLINE_SECONDS", 5))
    poll_interval: int = field(default_factory=lambda: _env_int("POLL_INTERVAL", 2))


@dataclass
class StrategySettings:
    # Bootstrap candle counts
    bootstrap_1m: int = 100
    bootstrap_5m: int = 100
    bootstrap_15m: int = 50

    # Stochastic RSI
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # MACD (fast settings for 5m)
    macd_fast: int = 5
    macd_slow: int = 12
    macd_signal: int = 5

    # ATR
    atr_period: int = 14
    atr_low_mult: float = 0.5    # below this = low vol regime
    atr_high_mult: float = 1.5   # above this = high vol regime

    # VWAP
    vwap_lookback: int = 50

    # OBV
    obv_ema_period: int = 10

    # Confidence engine
    min_consensus_indicators: int = 3  # at least 3 must agree to trade

    # Order flow
    order_flow_strong: float = 0.15   # |flow| > this → ±3.0
    order_flow_moderate: float = 0.05  # |flow| > this → ±1.5

    # Window delta (%)
    delta_extreme: float = 0.10
    delta_strong: float = 0.03
    delta_moderate: float = 0.01
    delta_weak: float = 0.003

    # Stochastic RSI thresholds
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0

    # MACD histogram (basis points)
    macd_strong_bps: float = 0.5
    macd_weak_bps: float = 0.1

    # Bollinger band position
    bb_extreme_upper: float = 0.95
    bb_extreme_lower: float = 0.05
    bb_moderate_upper: float = 0.75
    bb_moderate_lower: float = 0.25

    # VWAP threshold
    vwap_threshold_pct: float = 0.02

    # Tick trend thresholds
    tick_dir_threshold: float = 0.60
    tick_change_threshold: float = 0.005

    # Confidence engine weights
    conf_consensus_weight: float = 0.60
    conf_magnitude_weight: float = 0.25
    conf_tier1_weight: float = 0.15

    # Volatility regime multipliers
    vol_low_mult: float = 0.80
    vol_high_mult: float = 1.10

    # Consensus gate multiplier
    consensus_gate_mult: float = 0.50


@dataclass
class SentimentSettings:
    ls_ratio_extreme_long: float = 2.0    # Above this = contrarian bearish
    ls_ratio_extreme_short: float = 0.5   # Below this = contrarian bullish
    oi_change_threshold: float = 0.02     # 2% OI change = significant
    fear_greed_extreme_fear: int = 20     # Below = contrarian bullish
    fear_greed_extreme_greed: int = 80    # Above = contrarian bearish
    fetch_interval: float = 60.0          # Seconds between API calls


@dataclass
class Config:
    creds: PolymarketCreds = field(default_factory=PolymarketCreds)
    bot: BotSettings = field(default_factory=BotSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    timing: TimingSettings = field(default_factory=TimingSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    sentiment: SentimentSettings = field(default_factory=SentimentSettings)

    # Polymarket endpoints
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    ws_orderbook_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Binance endpoints — combined stream for multi-timeframe
    binance_ws_url: str = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/btcusdt@kline_5m/btcusdt@kline_15m"
    binance_trade_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    binance_rest_url: str = "https://api.binance.com/api/v3/klines"

    # Chain ID for Polygon
    chain_id: int = 137


# Singleton config instance
cfg = Config()
