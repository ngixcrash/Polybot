"""Risk management -- position sizing, bankroll tracking, circuit breakers."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    window_ts: int
    direction: str
    confidence: float
    entry_price: float
    size: float
    cost: float
    buy_size: float = 0.0       # shares bought (winning token)
    buy_cost: float = 0.0       # cost of buy leg
    sell_size: float = 0.0      # shares sold (losing token)
    sell_cost: float = 0.0      # proceeds from sell leg
    outcome: Optional[str] = None  # "win" or "loss"
    pnl: float = 0.0
    timestamp: float = field(default_factory=time.time)
    # ML fields — persisted for adaptive learning
    signal_components: Optional[dict] = None  # per-indicator raw scores
    score: float = 0.0                        # total raw score
    delta_pct: float = 0.0                    # BTC delta % from window open


@dataclass
class RiskManager:

    starting_bankroll: float = 0.0
    bankroll: float = 0.0
    mode: str = "safe"
    trades: list[TradeRecord] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    _paused_until: float = 0.0
    _day_start: float = 0.0
    _trade_store: object = None  # Optional[TradeStore] — set externally

    def __post_init__(self):
        if self.starting_bankroll == 0:
            self.starting_bankroll = cfg.bot.starting_bankroll
        if self.bankroll == 0:
            self.bankroll = self.starting_bankroll
        if self.mode == "safe":
            self.mode = cfg.bot.mode
        self._day_start = self._start_of_day()

    @staticmethod
    def _start_of_day() -> float:
        now = time.time()
        return now - (now % 86400)

    def _check_new_day(self):
        current_day = self._start_of_day()
        if current_day > self._day_start:
            log.info(
                f"New day — resetting daily stats. "
                f"Yesterday P&L: ${self.daily_pnl:.2f} / {self.daily_trades} trades"
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self._day_start = current_day

    def is_paused(self) -> tuple[bool, str]:
        """Check if any circuit breaker is tripped and return (paused, reason)."""
        self._check_new_day()

        if time.time() < self._paused_until:
            remaining = self._paused_until - time.time()
            return True, f"Cooling down ({remaining:.0f}s remaining)"

        if self.bankroll < cfg.bot.min_bet:
            return True, f"Bankroll ${self.bankroll:.2f} below minimum bet ${cfg.bot.min_bet:.2f}"

        max_loss = self.starting_bankroll * cfg.risk.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            return True, f"Daily loss cap hit: ${self.daily_pnl:.2f} (max ${-max_loss:.2f})"

        if self.consecutive_losses >= cfg.risk.max_consecutive_losses:
            # Auto-pause for 3 windows (15 min)
            self._paused_until = time.time() + 900
            self.consecutive_losses = 0
            return True, f"Consecutive loss breaker: {cfg.risk.max_consecutive_losses} losses in a row"

        return False, ""

    def get_bet_size(self, confidence: float) -> float:
        """Figure out how much to bet given mode, confidence, and bankroll."""
        paused, reason = self.is_paused()
        if paused:
            log.warning(f"[RISK] Paused: {reason}")
            return 0.0

        min_conf = self._min_confidence()
        if confidence < min_conf:
            log.debug(f"[RISK] Confidence {confidence:.2f} < min {min_conf:.2f} for mode={self.mode}")
            return 0.0

        if self.mode == "safe":
            base = self.bankroll * cfg.bot.max_bet_pct
        elif self.mode == "aggressive":
            profits = max(0.0, self.bankroll - self.starting_bankroll)
            if profits < cfg.bot.min_bet:
                base = self.bankroll * cfg.bot.max_bet_pct * 0.5
            else:
                base = profits * cfg.bot.max_bet_pct
        elif self.mode == "degen":
            base = self.bankroll
        else:
            base = self.bankroll * cfg.bot.max_bet_pct

        scaled = base * min(confidence * 1.2, 1.0)

        bet = max(scaled, cfg.bot.min_bet)
        bet = min(bet, self.bankroll)

        if bet < cfg.bot.min_bet:
            return 0.0

        return round(bet, 2)

    def _min_confidence(self) -> float:
        if self.mode == "safe":
            return 0.65
        elif self.mode == "aggressive":
            return 0.50
        elif self.mode == "degen":
            return 0.0
        return 0.65

    def record_trade(
        self,
        window_ts: int,
        direction: str,
        confidence: float,
        entry_price: float,
        size: float,
        cost: float,
        buy_size: float = 0.0,
        buy_cost: float = 0.0,
        sell_size: float = 0.0,
        sell_cost: float = 0.0,
        signal_components: Optional[dict] = None,
        score: float = 0.0,
        delta_pct: float = 0.0,
    ) -> TradeRecord:
        trade = TradeRecord(
            window_ts=window_ts,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            size=size,
            cost=cost,
            buy_size=buy_size,
            buy_cost=buy_cost,
            sell_size=sell_size,
            sell_cost=sell_cost,
            signal_components=signal_components,
            score=score,
            delta_pct=delta_pct,
        )
        self.trades.append(trade)
        self.daily_trades += 1
        net_outflow = buy_cost - sell_cost
        self.bankroll -= net_outflow

        if self._trade_store:
            self._trade_store.append({
                "window_ts": window_ts,
                "direction": direction,
                "confidence": confidence,
                "entry_price": entry_price,
                "size": size,
                "cost": cost,
                "buy_size": buy_size,
                "buy_cost": buy_cost,
                "sell_size": sell_size,
                "sell_cost": sell_cost,
                "signal_components": signal_components,
                "score": score,
                "delta_pct": delta_pct,
                "timestamp": trade.timestamp,
            })

        log.info(
            f"[TRADE] {direction.upper()} conf={confidence:.2f} "
            f"price=${entry_price:.4f} size={size:.1f} "
            f"buy=${buy_cost:.2f} sell=${sell_cost:.2f} "
            f"bankroll=${self.bankroll:.2f}"
        )
        return trade

    def resolve_trade(self, trade: TradeRecord, won: bool):
        """Settle a trade using Polymarket binary resolution (winning token pays $1/share, losing token pays $0)."""
        if won:
            # BUY leg: winning token pays $1/share
            buy_payout = trade.buy_size * 1.0
            buy_profit = buy_payout - trade.buy_cost
            # SELL leg: losing token → $0, keep the sale proceeds
            sell_profit = trade.sell_cost
            trade.pnl = buy_profit + sell_profit
            trade.outcome = "win"
            # buy_cost already deducted at entry; receive buy payout now
            # sell proceeds already received at entry; token → $0, obligation cleared
            self.bankroll += buy_payout
            self.consecutive_losses = 0
        else:
            # BUY leg: token → $0, lose buy_cost (already deducted at entry)
            buy_loss = -trade.buy_cost
            # SELL leg: token → $1, owe $1/share to cover
            # Already received sell_cost at entry, now must pay sell_size × $1
            sell_loss = -(trade.sell_size * 1.0 - trade.sell_cost)
            trade.pnl = buy_loss + sell_loss
            trade.outcome = "loss"
            # Bankroll already has: -(buy_cost) + sell_cost from entry
            # Now owe sell_size × $1 to cover the short
            self.bankroll -= trade.sell_size * 1.0
            self.consecutive_losses += 1

        self.daily_pnl += trade.pnl

        if self._trade_store:
            self._trade_store.update_last({
                "outcome": trade.outcome,
                "pnl": trade.pnl,
            })

        log.info(
            f"[RESOLVE] {trade.outcome.upper()} pnl=${trade.pnl:+.2f} "
            f"bankroll=${self.bankroll:.2f} daily_pnl=${self.daily_pnl:+.2f} "
            f"consecutive_losses={self.consecutive_losses}"
        )

    def resolve_mm_trade(self, trade: TradeRecord):
        """Resolve MM trade -- one side pays $1, the other $0, roughly half the shares win."""
        # Approximate: half the shares win (the correct-direction token)
        winning_shares = trade.buy_size / 2.0
        payout = winning_shares * 1.0
        trade.pnl = payout - trade.cost
        trade.outcome = "mm"
        self.bankroll += payout
        self.daily_pnl += trade.pnl

        log.info(
            f"[RESOLVE] MM pnl=${trade.pnl:+.2f} "
            f"bankroll=${self.bankroll:.2f} daily_pnl=${self.daily_pnl:+.2f}"
        )

    def get_stats(self) -> dict:
        resolved = [t for t in self.trades if t.outcome]
        wins = [t for t in resolved if t.outcome == "win"]
        losses = [t for t in resolved if t.outcome == "loss"]

        total_pnl = sum(t.pnl for t in resolved)
        win_rate = len(wins) / len(resolved) if resolved else 0.0

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        peak = self.starting_bankroll
        max_dd = 0.0
        running = self.starting_bankroll
        for t in resolved:
            running += t.pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(resolved),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "bankroll": self.bankroll,
            "roi_pct": (self.bankroll - self.starting_bankroll) / self.starting_bankroll * 100,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "mode": self.mode,
        }

    def print_stats(self):
        s = self.get_stats()
        log.info(
            f"\n{'='*50}\n"
            f"  TRADING STATS ({s['mode']} mode)\n"
            f"{'='*50}\n"
            f"  Bankroll:    ${s['bankroll']:.2f}\n"
            f"  Total P&L:   ${s['total_pnl']:+.2f}\n"
            f"  ROI:         {s['roi_pct']:+.1f}%\n"
            f"  Win Rate:    {s['win_rate']*100:.1f}% ({s['wins']}W / {s['losses']}L)\n"
            f"  Profit Factor: {s['profit_factor']:.2f}\n"
            f"  Max Drawdown: {s['max_drawdown_pct']:.1f}%\n"
            f"  Today:       ${s['daily_pnl']:+.2f} ({s['daily_trades']} trades)\n"
            f"{'='*50}"
        )
