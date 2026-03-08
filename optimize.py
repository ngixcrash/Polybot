"""CLI for parameter optimization via backtesting (sweep, pretrain, walk-forward)."""

import argparse
import logging
import sys
import time

from backtest import (
    fetch_binance_candles,
    run_backtest_v2,
    parameter_sweep,
    compute_metrics,
    write_excel,
    WindowResult,
    BacktestCandle,
    WINDOW_DURATION,
)
from config import StrategySettings
from adaptive import AdaptiveWeightEngine

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Default sweep grid ──────────────────────────────────────────
# These are the parameters most likely to affect profitability.
# Each combination is tested independently.

DEFAULT_PARAM_GRID = {
    # Order flow thresholds: how strong does buy/sell imbalance need to be
    "order_flow_strong": [0.10, 0.15, 0.20],

    # Confidence engine weighting: consensus vs magnitude vs tier-1
    "conf_consensus_weight": [0.50, 0.60, 0.70],

    # Stochastic RSI: overbought/oversold thresholds
    "stoch_overbought": [75.0, 80.0, 85.0],

    # Minimum consensus: how many indicators must agree
    "min_consensus_indicators": [2, 3, 4],
}

# Smaller grid for quick runs
QUICK_PARAM_GRID = {
    "order_flow_strong": [0.10, 0.15, 0.20],
    "conf_consensus_weight": [0.50, 0.60, 0.70],
}


def pretrain_weights(
    candles: list[BacktestCandle],
    settings: StrategySettings = None,
    min_confidence: float = 0.0,
    starting_bankroll: float = 50.0,
    alpha: float = 0.05,
) -> AdaptiveWeightEngine:
    """Run backtest then feed trades through AdaptiveWeightEngine to learn indicator weights."""
    import adaptive as adaptive_module

    original_alpha = adaptive_module.EMA_ALPHA
    adaptive_module.EMA_ALPHA = alpha

    log.info("Running V2 backtest to generate training data...")
    results = run_backtest_v2(
        candles=candles,
        settings=settings or StrategySettings(),
        starting_bankroll=starting_bankroll,
        min_confidence=min_confidence,
    )

    if not results:
        log.error("No trades generated from backtest -- cannot pretrain")
        adaptive_module.EMA_ALPHA = original_alpha
        return AdaptiveWeightEngine()

    log.info(f"Training on {len(results)} historical trades...")

    engine = AdaptiveWeightEngine()
    engine.total_resolved = 0
    for name in engine.stats:
        engine.stats[name].correct = 0
        engine.stats[name].total = 0
        engine.stats[name].ema_accuracy = 0.5
    for name in engine.weights:
        engine.weights[name] = 1.0

    wins = 0
    for r in results:
        if not r.components:
            continue  # Skip trades without component data

        engine.update_after_resolution(
            components=r.components,
            actual_direction=r.actual_direction,
            predicted_direction=r.predicted_direction,
            won=r.won,
        )
        if r.won:
            wins += 1

    engine.save_to_disk()
    adaptive_module.EMA_ALPHA = original_alpha

    print(f"\n{'=' * 70}")
    print(f"  PRE-TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Trained on: {len(results)} trades ({len(candles)} candles)")
    print(f"  Overall Win Rate: {wins / len(results) * 100:.1f}%")
    print(f"  Weights saved to: data/indicator_weights.json")
    print(f"\n  {'Indicator':<28} {'Weight':>8}  {'Accuracy':>10}  {'EMA':>8}  {'Samples':>8}")
    print(f"  {'-' * 28} {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    summary = engine.get_weight_summary()
    sorted_indicators = sorted(summary.items(), key=lambda x: x[1]["weight"], reverse=True)

    for name, info in sorted_indicators:
        if info["total"] == 0:
            continue  # Skip indicators that never fired (e.g. micro/sentiment)
        w = info["weight"]
        acc = info["accuracy"] * 100
        ema = info["ema_accuracy"] * 100
        total = info["total"]

        marker = "**" if w >= 1.5 else "  " if w >= 0.8 else "vv"
        print(f"  {marker} {name:<26} {w:>7.3f}  {acc:>9.1f}%  {ema:>7.1f}%  {total:>7}")

    print(f"\n  ** = boosted (>1.5x)  |  vv = dampened (<0.8x)")
    print(f"\n  Next time you run the bot, it will load these weights automatically.")
    print()

    return engine


def walk_forward_validation(
    candles: list[BacktestCandle],
    train_hours: int = 48,
    test_hours: int = 24,
    settings: StrategySettings = None,
    min_confidence: float = 0.30,
    starting_bankroll: float = 50.0,
) -> list[dict]:
    """Walk-forward validation: train on N hours, test on M hours, slide forward to prevent overfitting."""
    if settings is None:
        settings = StrategySettings()

    train_candles = train_hours * 60  # 1m candles
    test_candles = test_hours * 60

    if len(candles) < train_candles + test_candles:
        log.error(
            f"Not enough data for walk-forward: need {train_candles + test_candles} "
            f"candles, have {len(candles)}"
        )
        return []

    folds = []
    fold_num = 0
    cursor = 0

    while cursor + train_candles + test_candles <= len(candles):
        fold_num += 1
        train_slice = candles[cursor : cursor + train_candles]
        test_slice = candles[cursor + train_candles : cursor + train_candles + test_candles]

        # Train: run backtest on training period
        train_results = run_backtest_v2(
            candles=train_slice,
            settings=settings,
            starting_bankroll=starting_bankroll,
            min_confidence=min_confidence,
        )
        train_metrics = compute_metrics(train_results, starting_bankroll)

        # Test: run backtest on test period (out-of-sample)
        test_results = run_backtest_v2(
            candles=test_slice,
            settings=settings,
            starting_bankroll=starting_bankroll,
            min_confidence=min_confidence,
        )
        test_metrics = compute_metrics(test_results, starting_bankroll)

        folds.append({
            "fold": fold_num,
            "train_start": train_slice[0].timestamp if train_slice else 0,
            "test_start": test_slice[0].timestamp if test_slice else 0,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        })

        log.info(
            f"  Fold {fold_num}: "
            f"Train WR={train_metrics.get('win_rate', 0) * 100:.1f}% "
            f"PnL=${train_metrics.get('total_pnl', 0):.2f} | "
            f"Test WR={test_metrics.get('win_rate', 0) * 100:.1f}% "
            f"PnL=${test_metrics.get('total_pnl', 0):.2f}"
        )

        cursor += test_candles

    return folds


def print_sweep_results(results: list[tuple[dict, dict]], top_n: int = 5):
    print(f"\n{'=' * 80}")
    print(f"  TOP {min(top_n, len(results))} PARAMETER COMBINATIONS")
    print(f"{'=' * 80}\n")

    for rank, (params, metrics) in enumerate(results[:top_n], 1):
        trades = metrics.get("total_trades", 0)
        if trades == 0:
            continue

        wr = metrics.get("win_rate", 0) * 100
        pnl = metrics.get("total_pnl", 0)
        roi = metrics.get("roi_pct", 0)
        pf = metrics.get("profit_factor", 0)
        dd = metrics.get("max_drawdown_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        print(f"  #{rank}")
        print(f"    Parameters: {params}")
        print(
            f"    Trades: {trades} | Win Rate: {wr:.1f}% | P&L: ${pnl:.2f} | "
            f"ROI: {roi:.1f}%"
        )
        print(
            f"    Profit Factor: {pf:.2f} | Max DD: {dd:.1f}% | "
            f"Sharpe: {sharpe:.2f}"
        )
        print()

    # Also show worst for comparison
    if len(results) > top_n:
        print(f"  --- Worst combo ---")
        params, metrics = results[-1]
        trades = metrics.get("total_trades", 0)
        if trades > 0:
            print(f"    Parameters: {params}")
            print(
                f"    Trades: {trades} | Win Rate: {metrics.get('win_rate', 0) * 100:.1f}% | "
                f"P&L: ${metrics.get('total_pnl', 0):.2f}"
            )
        print()


def print_walk_forward_results(folds: list[dict]):
    print(f"\n{'=' * 80}")
    print(f"  WALK-FORWARD VALIDATION ({len(folds)} folds)")
    print(f"{'=' * 80}\n")

    total_test_pnl = 0
    total_test_trades = 0
    total_test_wins = 0

    for fold in folds:
        fm = fold["test_metrics"]
        trades = fm.get("total_trades", 0)
        total_test_trades += trades
        total_test_wins += fm.get("wins", 0)
        total_test_pnl += fm.get("total_pnl", 0)

        print(
            f"  Fold {fold['fold']}: "
            f"Test: {trades} trades, "
            f"WR={fm.get('win_rate', 0) * 100:.1f}%, "
            f"PnL=${fm.get('total_pnl', 0):.2f}"
        )

    print(f"\n  --- Aggregate OOS Results ---")
    if total_test_trades > 0:
        agg_wr = total_test_wins / total_test_trades * 100
        print(
            f"  Total OOS Trades: {total_test_trades} | "
            f"Win Rate: {agg_wr:.1f}% | "
            f"Total PnL: ${total_test_pnl:.2f}"
        )
    else:
        print("  No out-of-sample trades produced.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket BTC Bot — Parameter Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py --hours 72                    # V2 backtest, default settings
  python optimize.py --hours 168 --sweep           # Full grid search (7 days data)
  python optimize.py --hours 72 --sweep --quick    # Quick sweep (fewer combos)
  python optimize.py --hours 168 --walk-forward    # Walk-forward validation
  python optimize.py --hours 72 --sweep --top 10   # Show top 10 results
  python optimize.py --hours 72 --output opt.xlsx  # Save to Excel
  python optimize.py --hours 168 --pretrain        # Pre-train weights from 7 days
  python optimize.py --hours 336 --pretrain --alpha 0.03  # 14 days, slower learning
        """,
    )

    parser.add_argument(
        "--hours", type=int, default=72,
        help="Hours of historical data to fetch (default: 72)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run parameter sweep (grid search)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use smaller parameter grid for faster sweep",
    )
    parser.add_argument(
        "--pretrain", action="store_true",
        help="Pre-train adaptive weights from historical data",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="EMA learning rate for pretrain (default: 0.05, lower=more stable)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward validation",
    )
    parser.add_argument(
        "--train-hours", type=int, default=48,
        help="Training window for walk-forward (default: 48)",
    )
    parser.add_argument(
        "--test-hours", type=int, default=24,
        help="Test window for walk-forward (default: 24)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=50.0,
        help="Starting bankroll (default: 50.0)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.30,
        help="Minimum confidence to enter a trade (default: 0.30)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top results to display (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output Excel file (optional)",
    )

    args = parser.parse_args()

    start_time = time.time()
    log.info(f"Fetching {args.hours} hours of historical 1m candles...")
    candles = fetch_binance_candles(args.hours)

    if not candles:
        log.error("No candle data — aborting")
        sys.exit(1)

    fetch_time = time.time() - start_time
    log.info(f"Fetched {len(candles)} candles in {fetch_time:.1f}s")

    if args.sweep:
        grid = QUICK_PARAM_GRID if args.quick else DEFAULT_PARAM_GRID

        total_combos = 1
        for v in grid.values():
            total_combos *= len(v)

        log.info(
            f"Starting parameter sweep: {total_combos} combinations, "
            f"{len(grid)} parameters"
        )
        sweep_start = time.time()

        results = parameter_sweep(
            candles=candles,
            param_grid=grid,
            starting_bankroll=args.bankroll,
            min_confidence=args.min_confidence,
        )

        sweep_time = time.time() - sweep_start
        log.info(f"Sweep completed in {sweep_time:.1f}s")

        print_sweep_results(results, top_n=args.top)

        # Optionally save to Excel
        if args.output and results:
            best_params = results[0][0]
            best_settings = StrategySettings()
            for k, v in best_params.items():
                setattr(best_settings, k, v)

            best_results = run_backtest_v2(
                candles=candles,
                settings=best_settings,
                starting_bankroll=args.bankroll,
                min_confidence=args.min_confidence,
            )
            best_metrics = compute_metrics(best_results, args.bankroll)

            default_results = run_backtest_v2(
                candles=candles,
                settings=StrategySettings(),
                starting_bankroll=args.bankroll,
                min_confidence=args.min_confidence,
            )
            default_metrics = compute_metrics(default_results, args.bankroll)

            all_results = {
                "best_optimized": best_results,
                "default": default_results,
            }
            all_metrics = {
                "best_optimized": best_metrics,
                "default": default_metrics,
            }

            write_excel(all_results, all_metrics, args.bankroll, args.output)
            log.info(f"Results saved to {args.output}")

    elif args.pretrain:
        log.info(
            f"Pre-training adaptive weights from {args.hours}h of data "
            f"(alpha={args.alpha})..."
        )

        engine = pretrain_weights(
            candles=candles,
            min_confidence=0.0,  # Train on ALL windows for max data
            starting_bankroll=args.bankroll,
            alpha=args.alpha,
        )

    elif args.walk_forward:
        log.info(
            f"Starting walk-forward validation: "
            f"train={args.train_hours}h, test={args.test_hours}h"
        )

        folds = walk_forward_validation(
            candles=candles,
            train_hours=args.train_hours,
            test_hours=args.test_hours,
            min_confidence=args.min_confidence,
            starting_bankroll=args.bankroll,
        )

        print_walk_forward_results(folds)

    else:
        log.info("Running V2 backtest with default settings...")

        results = run_backtest_v2(
            candles=candles,
            settings=StrategySettings(),
            starting_bankroll=args.bankroll,
            min_confidence=args.min_confidence,
        )
        metrics = compute_metrics(results, args.bankroll)

        if metrics["total_trades"] == 0:
            log.warning("No trades generated — try lowering --min-confidence")
        else:
            print(f"\n{'=' * 60}")
            print(f"  V2 BACKTEST RESULTS ({args.hours}h)")
            print(f"{'=' * 60}")
            print(f"  Trades:        {metrics['total_trades']}")
            print(f"  Wins:          {metrics['wins']} ({metrics['win_rate'] * 100:.1f}%)")
            print(f"  Losses:        {metrics['losses']}")
            print(f"  Total P&L:     ${metrics['total_pnl']:.2f}")
            print(f"  Final Bankroll: ${metrics['final_bankroll']:.2f}")
            print(f"  ROI:           {metrics['roi_pct']:.1f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Max Drawdown:  {metrics['max_drawdown_pct']:.1f}%")
            print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
            print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
            print()

        # Save to Excel if requested
        if args.output and results:
            all_results = {"v2_default": results}
            all_metrics = {"v2_default": metrics}
            write_excel(all_results, all_metrics, args.bankroll, args.output)

    total_time = time.time() - start_time
    log.info(f"Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
