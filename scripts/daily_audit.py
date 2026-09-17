r"""
Daily accuracy job — run this on a schedule.

It does three things, in order:

  1. Refreshes market data for every tracked ticker.
  2. Scores any past prediction whose target day has now closed. This is
     the hindsight-free track record, and it is the point of the job.
  3. Re-runs the walk-forward backtest and stores today's scores, so you
     can watch the model drift over time.

Run it by hand:
    .venv\Scripts\python.exe -m scripts.daily_audit
    .venv\Scripts\python.exe -m scripts.daily_audit --symbols AAPL MSFT NVDA

Exit codes: 0 = ok, 1 = every symbol failed (so a scheduler can alert).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow `python scripts\daily_audit.py` as well as `-m scripts.daily_audit`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import repository as repo          # noqa: E402
from app.config import PROJECT_ROOT         # noqa: E402
from app.db import init_db, session_scope   # noqa: E402
from app.scoring import audit_universe, track_record  # noqa: E402
from app.service import get_client          # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"


def configure_logging(quiet: bool) -> None:
    """Log to console and to logs/daily_audit.log.

    The file matters: a scheduled job runs with no one watching, so the
    only evidence it ran correctly is what it wrote down.
    """
    LOG_DIR.mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(LOG_DIR / "daily_audit.log", encoding="utf-8")
    ]
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily accuracy audit.")
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Tickers to audit. Defaults to everything already tracked.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Log to file only, not the console.",
    )
    args = parser.parse_args(argv)

    configure_logging(args.quiet)
    log = logging.getLogger("stockapp.audit")

    init_db()

    with session_scope() as session:
        report = audit_universe(session, get_client(), symbols=args.symbols)

        log.info("=" * 68)
        log.info("AUDIT %s", report.run_date)
        log.info("=" * 68)

        for run in sorted(report.runs, key=lambda r: r.mae_ratio):
            log.info(
                "  %-6s mae %5.2f%%  naive %5.2f%%  ratio %.3f  dir %4.1f%%  base %4.1f%%%s",
                run.symbol, run.mae_pct, run.naive_mae_pct, run.mae_ratio,
                run.directional_accuracy * 100, run.baseline_accuracy * 100,
                "  <- beats naive" if run.beats_naive else "",
            )

        for symbol, reason in report.failures:
            log.warning("  %-6s SKIPPED: %s", symbol, reason)

        if report.n:
            accuracy, total, z = report.pooled_direction
            log.info("-" * 68)
            log.info("  tickers audited:        %d", report.n)
            log.info("  beat no-change:         %d/%d", report.beat_naive, report.n)
            log.info("  beat always-up:         %d/%d", report.beat_direction, report.n)
            log.info("  median MAE ratio:       %.3f  (1.0 = tied with naive)", report.median_ratio)
            log.info("  mean directional:       %.1f%%  (baseline %.1f%%)",
                     report.mean_directional * 100, report.mean_baseline * 100)
            log.info("  pooled direction:       %.1f%% of %d, z=%.2f  (%s)",
                     accuracy * 100, total, z,
                     "significant" if abs(z) > 1.96 else "not significant")

        # --- the measurement that actually accumulates ---
        log.info("-" * 68)
        log.info("  predictions newly scored: %d", report.scored)

        record = track_record(session)
        if record.n:
            log.info("  LIVE TRACK RECORD (%d scored predictions):", record.n)
            log.info("    mae %.2f%%  vs naive %.2f%%  -> %s",
                     record.mae_pct, record.naive_mae_pct,
                     "beats naive" if record.beats_naive else "loses to naive")
            log.info("    directional accuracy: %.1f%%", record.directional_accuracy * 100)
            log.info("    mean signed error:    %+.2f%% (%s)",
                     record.bias_pct,
                     "overshoots" if record.bias_pct > 0 else "undershoots")
            if not record.is_meaningful:
                log.info("    NOTE: under 100 samples — not yet statistically meaningful.")
        else:
            log.info("  LIVE TRACK RECORD: none yet — predictions are scored the")
            log.info("    session after they are made, so this fills in from tomorrow.")

        history = repo.get_accuracy_runs(session, since=date.today() - timedelta(days=30))
        log.info("  stored audit rows (30d):  %d", len(history))
        log.info("=" * 68)

        if not report.n and report.failures:
            log.error("every symbol failed")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
