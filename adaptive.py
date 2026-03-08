"""Adaptive indicator weighting -- EMA of per-indicator accuracy drives weight adjustments online."""

import logging
from dataclasses import dataclass, field

from persistence import TradeStore, ModelStore

log = logging.getLogger(__name__)

# Learning rate: how fast weights adapt (lower = more stable, higher = more reactive)
EMA_ALPHA = 0.05

# Don't adapt until we have at least this many resolved trades
MIN_TRADES_FOR_ADAPTATION = 20

# Default weights: all indicators start equal
DEFAULT_INDICATOR_NAMES = [
    "order_flow",
    "window_delta",
    "multi_tf",
    "stoch_rsi",
    "macd",
    "bollinger",
    "obv",
    "vwap",
    "tick_trend",
    "ema_crossover",
    # Microstructure (added by Feature 2)
    "micro_depth_imbalance",
    "micro_spread_dynamics",
    "micro_microprice",
    "micro_smart_money",
    # Sentiment (added by Feature 3)
    "sentiment_ls_ratio",
    "sentiment_oi_change",
    "sentiment_fear_greed",
]


@dataclass
class IndicatorStats:
    correct: int = 0
    total: int = 0
    ema_accuracy: float = 0.5  # Start at coin-flip assumption


@dataclass
class AdaptiveWeightEngine:
    """Adjusts indicator weights based on trade outcomes, bounded [0.2, 2.5]."""

    weights: dict[str, float] = field(default_factory=dict)
    stats: dict[str, IndicatorStats] = field(default_factory=dict)
    total_resolved: int = 0
    _trade_store: TradeStore = field(default_factory=TradeStore)
    _model_store: ModelStore = field(default_factory=ModelStore)

    def __post_init__(self):
        for name in DEFAULT_INDICATOR_NAMES:
            self.weights.setdefault(name, 1.0)
            self.stats.setdefault(name, IndicatorStats())

        self.load_from_disk()

    def load_from_disk(self) -> None:
        state = self._model_store.load()
        if state is None:
            log.info("[ADAPTIVE] No saved model state — starting fresh")
            return

        saved_weights = state.get("weights", {})
        for name, w in saved_weights.items():
            self.weights[name] = float(w)

        saved_stats = state.get("stats", {})
        for name, s in saved_stats.items():
            self.stats[name] = IndicatorStats(
                correct=int(s.get("correct", 0)),
                total=int(s.get("total", 0)),
                ema_accuracy=float(s.get("ema_accuracy", 0.5)),
            )

        self.total_resolved = int(state.get("total_resolved", 0))
        log.info(
            f"[ADAPTIVE] Loaded model: {self.total_resolved} resolved trades, "
            f"weights range [{min(self.weights.values()):.2f}, {max(self.weights.values()):.2f}]"
        )

    def save_to_disk(self) -> None:
        state = {
            "total_resolved": self.total_resolved,
            "weights": self.weights,
            "stats": {
                name: {
                    "correct": s.correct,
                    "total": s.total,
                    "ema_accuracy": s.ema_accuracy,
                }
                for name, s in self.stats.items()
            },
        }
        self._model_store.save(state)

    def update_after_resolution(
        self,
        components: dict[str, float],
        actual_direction: str,
        predicted_direction: str,
        won: bool,
    ) -> None:
        """Update EMA accuracy and recompute weights after a trade resolves."""
        if not components:
            return

        self.total_resolved += 1
        actual_up = actual_direction == "up"

        updated = []
        for name, value in components.items():
            if value == 0.0:
                continue  # Indicator was neutral — no update

            if name not in self.stats:
                self.stats[name] = IndicatorStats()
            if name not in self.weights:
                self.weights[name] = 1.0

            stat = self.stats[name]
            voted_up = value > 0
            correct = voted_up == actual_up

            stat.total += 1
            if correct:
                stat.correct += 1

            # ema = (1 - alpha) * ema + alpha * (1 if correct else 0)
            stat.ema_accuracy = (
                (1 - EMA_ALPHA) * stat.ema_accuracy
                + EMA_ALPHA * (1.0 if correct else 0.0)
            )

            # Map accuracy to weight:
            #   accuracy=0.5 (coin flip) → weight=1.0
            #   accuracy=0.7 (good) → weight=1.6
            #   accuracy=0.3 (bad) → weight=0.5
            #   accuracy=0.8 (great) → weight=1.9
            raw_weight = 0.5 + (stat.ema_accuracy - 0.5) * 3.0
            self.weights[name] = max(0.2, min(2.5, raw_weight))

            updated.append(f"{name}={self.weights[name]:.2f}")

        outcome_str = "WIN" if won else "LOSS"
        log.info(
            f"[ADAPTIVE] Trade #{self.total_resolved} {outcome_str} | "
            f"actual={actual_direction} predicted={predicted_direction} | "
            f"Updated: {', '.join(updated[:5])}{'...' if len(updated) > 5 else ''}"
        )

    def get_adjusted_score(self, components: dict[str, float]) -> float:
        """Multiply each raw indicator score by its learned weight and sum."""
        adjusted_total = 0.0
        for name, raw_value in components.items():
            weight = self.weights.get(name, 1.0)
            adjusted_total += raw_value * weight
        return adjusted_total

    def get_weight_summary(self) -> dict:
        summary = {}
        for name in sorted(self.weights.keys()):
            w = self.weights[name]
            s = self.stats.get(name, IndicatorStats())
            accuracy = s.correct / s.total if s.total > 0 else 0.0
            summary[name] = {
                "weight": round(w, 3),
                "accuracy": round(accuracy, 3),
                "ema_accuracy": round(s.ema_accuracy, 3),
                "total": s.total,
            }
        return summary

    @property
    def is_ready(self) -> bool:
        return self.total_resolved >= MIN_TRADES_FOR_ADAPTATION
