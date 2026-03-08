"""Multi-timeframe consensus strategy with 10 indicators (T1/T2/T3 tiers) and ATR-adjusted confidence."""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from binance_feed import BinanceFeed, Candle
from config import cfg

log = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────

@dataclass
class Signal:
    direction: str       # "up" or "down"
    confidence: float    # 0.0 to 1.0
    score: float         # raw weighted score
    delta_pct: float     # current price vs window open (%)
    components: dict     # individual indicator contributions


# ── Pure indicator functions ──────────────────────────────────

def _ema(prices: list[float], period: int) -> list[float]:
    if not prices:
        return []
    multiplier = 2.0 / (period + 1)
    ema_vals = [prices[0]]
    for price in prices[1:]:
        ema_vals.append(price * multiplier + ema_vals[-1] * (1 - multiplier))
    return ema_vals


def _rsi(prices: list[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_rsi(
    prices: list[float],
    rsi_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[Optional[float], Optional[float]]:
    """Stochastic oscillator applied to RSI; K% = stoch of RSI, D% = EMA of K%."""
    min_prices = rsi_period + k_period + d_period + 5
    if len(prices) < min_prices:
        return None, None

    rsi_series = []
    for i in range(rsi_period + 1, len(prices) + 1):
        r = _rsi(prices[:i], rsi_period)
        if r is not None:
            rsi_series.append(r)

    if len(rsi_series) < rsi_period:
        return None, None

    stoch_k_raw = []
    for i in range(rsi_period - 1, len(rsi_series)):
        window = rsi_series[i - rsi_period + 1 : i + 1]
        low = min(window)
        high = max(window)
        if high == low:
            stoch_k_raw.append(50.0)
        else:
            stoch_k_raw.append((rsi_series[i] - low) / (high - low) * 100.0)

    if len(stoch_k_raw) < k_period:
        return None, None

    k_ema = _ema(stoch_k_raw, k_period)
    if len(k_ema) < d_period:
        return None, None

    d_ema = _ema(k_ema, d_period)

    return k_ema[-1], d_ema[-1]


def _bollinger_bands(
    prices: list[float], period: int = 20, std_mult: float = 2.0
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < period:
        return None, None, None

    window = prices[-period:]
    middle = sum(window) / period
    variance = sum((p - middle) ** 2 for p in window) / period
    std = math.sqrt(variance)

    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def _macd(
    prices: list[float],
    fast: int = 5,
    slow: int = 12,
    signal_period: int = 5,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < slow + signal_period:
        return None, None, None

    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line[-1] - signal_line[-1]

    return macd_line[-1], signal_line[-1], histogram


def _obv(candles: list[Candle]) -> list[float]:
    if not candles:
        return []

    obv_series = [0.0]
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv_series.append(obv_series[-1] + candles[i].volume)
        elif candles[i].close < candles[i - 1].close:
            obv_series.append(obv_series[-1] - candles[i].volume)
        else:
            obv_series.append(obv_series[-1])

    return obv_series


def _atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high_low = candles[i].high - candles[i].low
        high_close = abs(candles[i].high - candles[i - 1].close)
        low_close = abs(candles[i].low - candles[i - 1].close)
        true_ranges.append(max(high_low, high_close, low_close))

    if len(true_ranges) < period:
        return None

    return sum(true_ranges[-period:]) / period


def _vwap(candles: list[Candle], lookback: int = 50) -> Optional[float]:
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    if not recent:
        return None

    total_pv = 0.0
    total_vol = 0.0
    for c in recent:
        typical_price = (c.high + c.low + c.close) / 3.0
        total_pv += typical_price * c.volume
        total_vol += c.volume

    if total_vol <= 0:
        return None
    return total_pv / total_vol


# ── Utility functions ─────────────────────────────────────────

def estimate_token_price(delta_pct: float) -> float:
    """Piecewise linear estimate of token price from BTC delta, used for dry-run P&L."""
    abs_delta = abs(delta_pct)
    if abs_delta < 0.005:
        return 0.50
    elif abs_delta < 0.02:
        return 0.50 + (abs_delta - 0.005) / (0.02 - 0.005) * 0.05
    elif abs_delta < 0.05:
        return 0.55 + (abs_delta - 0.02) / (0.05 - 0.02) * 0.10
    elif abs_delta < 0.10:
        return 0.65 + (abs_delta - 0.05) / (0.10 - 0.05) * 0.15
    elif abs_delta < 0.15:
        return 0.80 + (abs_delta - 0.10) / (0.15 - 0.10) * 0.12
    else:
        return min(0.92 + (abs_delta - 0.15) / 0.10 * 0.05, 0.97)


def calc_dynamic_taker_fee(probability: float, C: float = 1.0) -> float:
    # fee = C * 0.25 * (p * (1 - p))^2 — max ~1.56% at p=0.50, near-zero at extremes
    p = max(0.01, min(0.99, probability))
    return C * 0.25 * (p * (1.0 - p)) ** 2


# ── Strategy engine ───────────────────────────────────────────

# Max possible score per tier (for normalization)
_T1_MAX = 3.0  # 3 indicators x 3.0
_T2_MAX = 2.0  # 3 indicators x 2.0
_T3_MAX = 1.0  # 4 base + 4 micro + 3 sentiment = 11 indicators x 1.0
_MAX_SCORE = 3 * _T1_MAX + 3 * _T2_MAX + 11 * _T3_MAX  # 26.0


class Strategy:
    def __init__(self, feed: BinanceFeed, adaptive=None, sentiment=None, microstructure=None):
        self.feed = feed
        self.s = cfg.strategy  # Shortcut to StrategySettings
        self.adaptive = adaptive        # Optional[AdaptiveWeightEngine]
        self.sentiment = sentiment      # Optional[SentimentFeed]
        self.microstructure = microstructure  # Optional[MicrostructureAnalyzer]

    def analyze(self, window_open_price: float) -> Signal:
        """Run all indicators and return a scored Signal for the current window."""
        components = {}
        total_score = 0.0

        current_price = self.feed.current_price
        if current_price <= 0 or window_open_price <= 0:
            return Signal("up", 0.0, 0.0, 0.0, {})

        candles_1m = self.feed.get_recent_candles(60)
        candles_5m = self.feed.get_recent_candles_5m(20)
        candles_15m = self.feed.get_recent_candles_15m(10)

        delta_pct = (current_price - window_open_price) / window_open_price * 100
        closes_1m = [c.close for c in candles_1m] if candles_1m else []

        order_flow = self.feed.get_order_flow_delta(n=5)
        if abs(order_flow) > self.s.order_flow_strong:
            of_signal = 3.0 if order_flow > 0 else -3.0
        elif abs(order_flow) > self.s.order_flow_moderate:
            of_signal = 1.5 if order_flow > 0 else -1.5
        else:
            of_signal = 0.0
        components["order_flow"] = of_signal
        total_score += of_signal

        abs_delta = abs(delta_pct)
        if abs_delta > self.s.delta_extreme:
            delta_signal = 3.0
        elif abs_delta > self.s.delta_strong:
            delta_signal = 2.0
        elif abs_delta > self.s.delta_moderate:
            delta_signal = 1.0
        elif abs_delta > self.s.delta_weak:
            delta_signal = 0.5
        else:
            delta_signal = 0.0
        delta_dir = 1.0 if delta_pct > 0 else -1.0 if delta_pct < 0 else 0.0
        delta_signal *= delta_dir
        components["window_delta"] = delta_signal
        total_score += delta_signal

        mtf_signal = self._multi_tf_alignment(closes_1m, candles_5m, candles_15m)
        components["multi_tf"] = mtf_signal
        total_score += mtf_signal

        stoch_signal = 0.0
        if len(closes_1m) >= 30:
            k, d = _stoch_rsi(
                closes_1m,
                rsi_period=self.s.stoch_rsi_period,
                k_period=self.s.stoch_rsi_k,
                d_period=self.s.stoch_rsi_d,
            )
            if k is not None and d is not None:
                if k > self.s.stoch_overbought and d > self.s.stoch_overbought:
                    stoch_signal = -2.0  # Overbought
                elif k < self.s.stoch_oversold and d < self.s.stoch_oversold:
                    stoch_signal = 2.0   # Oversold
                elif k > d and k < self.s.stoch_overbought:
                    stoch_signal = 1.0   # Bullish crossover
                elif k < d and k > self.s.stoch_oversold:
                    stoch_signal = -1.0  # Bearish crossover
        components["stoch_rsi"] = stoch_signal
        total_score += stoch_signal

        macd_signal = 0.0
        if len(closes_1m) >= self.s.macd_slow + self.s.macd_signal:
            macd_line, sig_line, histogram = _macd(
                closes_1m,
                fast=self.s.macd_fast,
                slow=self.s.macd_slow,
                signal_period=self.s.macd_signal,
            )
            if histogram is not None:
                # Normalize histogram relative to price
                norm_hist = histogram / current_price * 10000  # basis points
                if norm_hist > self.s.macd_strong_bps:
                    macd_signal = 2.0
                elif norm_hist > self.s.macd_weak_bps:
                    macd_signal = 1.0
                elif norm_hist < -self.s.macd_strong_bps:
                    macd_signal = -2.0
                elif norm_hist < -self.s.macd_weak_bps:
                    macd_signal = -1.0
        components["macd"] = macd_signal
        total_score += macd_signal

        bb_signal = 0.0
        if len(closes_1m) >= self.s.bb_period:
            upper, middle, lower = _bollinger_bands(
                closes_1m,
                period=self.s.bb_period,
                std_mult=self.s.bb_std,
            )
            if upper is not None and upper != lower:
                bb_pos = (current_price - lower) / (upper - lower)  # 0=lower, 1=upper
                if bb_pos > self.s.bb_extreme_upper:
                    bb_signal = -2.0  # At upper band — expect reversion
                elif bb_pos < self.s.bb_extreme_lower:
                    bb_signal = 2.0   # At lower band — expect reversion
                elif bb_pos > self.s.bb_moderate_upper:
                    bb_signal = -1.0
                elif bb_pos < self.s.bb_moderate_lower:
                    bb_signal = 1.0
                # Between moderate_lower and moderate_upper: neutral
        components["bollinger"] = bb_signal
        total_score += bb_signal

        obv_signal = 0.0
        if len(candles_1m) >= self.s.obv_ema_period + 5:
            obv_series = _obv(candles_1m)
            if len(obv_series) >= self.s.obv_ema_period:
                obv_ema = _ema(obv_series, self.s.obv_ema_period)
                if obv_series[-1] > obv_ema[-1]:
                    obv_signal = 1.0  # Volume confirming uptrend
                else:
                    obv_signal = -1.0
        components["obv"] = obv_signal
        total_score += obv_signal

        vwap_signal = 0.0
        if candles_1m:
            vwap_val = _vwap(candles_1m, lookback=self.s.vwap_lookback)
            if vwap_val is not None and vwap_val > 0:
                vwap_pct = (current_price - vwap_val) / vwap_val * 100
                if vwap_pct > self.s.vwap_threshold_pct:
                    vwap_signal = 1.0   # Above VWAP — bullish
                elif vwap_pct < -self.s.vwap_threshold_pct:
                    vwap_signal = -1.0  # Below VWAP — bearish
        components["vwap"] = vwap_signal
        total_score += vwap_signal

        dir_pct, tick_change_pct = self.feed.get_tick_trend(lookback_seconds=20.0)
        tick_signal = 0.0
        if dir_pct >= self.s.tick_dir_threshold and abs(tick_change_pct) > self.s.tick_change_threshold:
            tick_signal = 1.0 if tick_change_pct > 0 else -1.0
        components["tick_trend"] = tick_signal
        total_score += tick_signal

        ema_signal = 0.0
        if len(closes_1m) >= 21:
            ema9 = _ema(closes_1m, 9)
            ema21 = _ema(closes_1m, 21)
            if ema9[-1] > ema21[-1]:
                ema_signal = 1.0
            else:
                ema_signal = -1.0
        components["ema_crossover"] = ema_signal
        total_score += ema_signal

        # ── PLUGGABLE: Microstructure signals (T3, ±1.0 each) ────
        if self.microstructure:
            try:
                micro_hint = "up" if total_score > 0 else "down"
                micro_components = self.microstructure.get_signal_components(
                    direction_hint=micro_hint
                )
                for key, value in micro_components.items():
                    components[key] = value
                    total_score += value
            except Exception as e:
                log.debug(f"Microstructure signals failed: {e}")

        # ── PLUGGABLE: Sentiment signals (T3, ±1.0 each) ────────
        if self.sentiment:
            try:
                sent_components = self.sentiment.get_signal_components()
                for key, value in sent_components.items():
                    components[key] = value
                    total_score += value
            except Exception as e:
                log.debug(f"Sentiment signals failed: {e}")

        # ── ADAPTIVE: Apply learned weights if engine is ready ───
        if self.adaptive and self.adaptive.is_ready:
            total_score = self.adaptive.get_adjusted_score(components)

        # ── Confidence engine ────────────────────────────────────

        confidence = self._compute_confidence(components, total_score, candles_1m)

        direction = "up" if total_score > 0 else "down"

        signal = Signal(
            direction=direction,
            confidence=confidence,
            score=total_score,
            delta_pct=delta_pct,
            components=components,
        )

        log.info(
            f"[SIGNAL] dir={direction} conf={confidence:.2f} "
            f"delta={delta_pct:.4f}% score={total_score:.1f} "
            f"components={components}"
        )

        return signal

    # ── Internal helpers ──────────────────────────────────────

    def _multi_tf_alignment(
        self,
        closes_1m: list[float],
        candles_5m: list[Candle],
        candles_15m: list[Candle],
    ) -> float:
        """Score how well 1m/5m/15m trends agree -- full alignment = +/-3.0."""
        trends = []

        # 1m trend: EMA 9 vs EMA 21
        if len(closes_1m) >= 21:
            ema9 = _ema(closes_1m, 9)
            ema21 = _ema(closes_1m, 21)
            trends.append(1 if ema9[-1] > ema21[-1] else -1)

        # 5m trend: last 3 candle direction
        if len(candles_5m) >= 3:
            c5 = candles_5m[-3:]
            if c5[-1].close > c5[0].open:
                trends.append(1)
            elif c5[-1].close < c5[0].open:
                trends.append(-1)
            else:
                trends.append(0)

        # 15m trend: last 2 candle direction
        if len(candles_15m) >= 2:
            c15 = candles_15m[-2:]
            if c15[-1].close > c15[0].open:
                trends.append(1)
            elif c15[-1].close < c15[0].open:
                trends.append(-1)
            else:
                trends.append(0)

        if not trends:
            return 0.0

        up_count = sum(1 for t in trends if t > 0)
        down_count = sum(1 for t in trends if t < 0)
        total = len(trends)

        if up_count == total:
            return 3.0  # All bullish
        elif down_count == total:
            return -3.0  # All bearish
        elif up_count >= 2:
            return 1.5
        elif down_count >= 2:
            return -1.5
        return 0.0

    def _compute_confidence(
        self, components: dict, total_score: float, candles_1m: list[Candle]
    ) -> float:
        """Blend consensus ratio, magnitude, and T1 conviction into 0-1 confidence."""
        # confidence = consensus_w * ratio + magnitude_w * mag + tier1_w * conviction
        direction = 1 if total_score > 0 else -1

        active_count = 0
        agree_count = 0
        for name, value in components.items():
            if value != 0.0:
                active_count += 1
                if (value > 0 and direction > 0) or (value < 0 and direction < 0):
                    agree_count += 1

        consensus_ratio = agree_count / max(active_count, 1)

        magnitude = min(abs(total_score) / _MAX_SCORE, 1.0)

        tier1_keys = ["order_flow", "window_delta", "multi_tf"]
        tier1_agrees = 0
        tier1_active = 0
        for k in tier1_keys:
            v = components.get(k, 0.0)
            if v != 0.0:
                tier1_active += 1
                if (v > 0 and direction > 0) or (v < 0 and direction < 0):
                    tier1_agrees += 1
        tier1_conviction = tier1_agrees / max(tier1_active, 1)

        raw_confidence = (
            self.s.conf_consensus_weight * consensus_ratio
            + self.s.conf_magnitude_weight * magnitude
            + self.s.conf_tier1_weight * tier1_conviction
        )

        # ── Volatility regime adjustment ─────────────────────
        vol_mult = 1.0
        if len(candles_1m) >= self.s.atr_period + 1:
            atr_val = _atr(candles_1m, self.s.atr_period)
            if atr_val is not None and atr_val > 0:
                all_atrs = []
                for i in range(self.s.atr_period + 1, len(candles_1m) + 1):
                    a = _atr(candles_1m[:i], self.s.atr_period)
                    if a is not None:
                        all_atrs.append(a)

                if all_atrs:
                    avg_atr = sum(all_atrs) / len(all_atrs)
                    atr_ratio = atr_val / avg_atr if avg_atr > 0 else 1.0

                    if atr_ratio < self.s.atr_low_mult:
                        vol_mult = self.s.vol_low_mult   # Low vol — reduce confidence
                    elif atr_ratio > self.s.atr_high_mult:
                        vol_mult = self.s.vol_high_mult  # High vol — slight boost

        # ── Consensus gate ───────────────────────────────────
        consensus_mult = 1.0
        if agree_count < self.s.min_consensus_indicators:
            consensus_mult = self.s.consensus_gate_mult  # Not enough indicators agree

        confidence = raw_confidence * vol_mult * consensus_mult
        confidence = max(0.0, min(confidence, 1.0))

        log.debug(
            f"[CONFIDENCE] raw={raw_confidence:.3f} "
            f"consensus={consensus_ratio:.2f} ({agree_count}/{active_count}) "
            f"magnitude={magnitude:.3f} tier1_conv={tier1_conviction:.2f} "
            f"vol_mult={vol_mult:.2f} cons_mult={consensus_mult:.2f} "
            f"final={confidence:.3f}"
        )

        return confidence
