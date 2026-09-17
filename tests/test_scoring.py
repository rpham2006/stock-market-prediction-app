"""
Accuracy-tracking tests.

The distinction under test: *scoring* compares a stored prediction to the
close that actually happened, while *auditing* re-runs the backtest. The
first is hindsight-free; the second is not. They must not be conflated.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app import repository as repo
from app import scoring, service
from app.models import AccuracyRun, Prediction


def _seed(session, client, symbol="AAPL"):
    """Load bars for a symbol and return them."""
    service.get_dashboard(session, symbol, client)
    return repo.get_bars(session, symbol)


def _prediction(symbol, base_bar, target_day, predicted_close) -> Prediction:
    return Prediction(
        symbol=symbol,
        target_day=target_day,
        base_day=base_bar.day,
        base_close=base_bar.close,
        predicted_close=predicted_close,
        predicted_return_pct=(predicted_close / base_bar.close - 1) * 100,
        interval_low=predicted_close * 0.98,
        interval_high=predicted_close * 1.02,
        model_version="ridge-v1",
    )


# ---------------------------------------------------------------
# Scoring against reality
# ---------------------------------------------------------------

def test_scoring_fills_in_the_outcome(session, client):
    bars = _seed(session, client)
    base, target = bars[-3], bars[-2]

    repo.save_prediction(session, _prediction("AAPL", base, target.day, target.close * 1.01))
    scored = scoring.score_predictions(session)

    assert len(scored) == 1
    prediction = scored[0]
    assert prediction.actual_close == pytest.approx(target.close)
    assert prediction.is_scored
    assert prediction.scored_at is not None
    # Predicted 1% above the real close, so the model overshot by ~1%.
    assert prediction.error_pct == pytest.approx(1.0, abs=0.01)


def test_error_sign_distinguishes_over_from_undershoot(session, client):
    bars = _seed(session, client)
    base, target = bars[-3], bars[-2]

    repo.save_prediction(session, _prediction("AAPL", base, target.day, target.close * 0.95))
    scoring.score_predictions(session)

    assert repo.scored_predictions(session)[0].error_pct < 0


def test_direction_correct_when_both_move_the_same_way(session, client):
    bars = _seed(session, client)
    base, target = bars[-3], bars[-2]

    # Predict a move in whichever direction actually happened.
    actual_up = target.close > base.close
    predicted = base.close * (1.02 if actual_up else 0.98)

    repo.save_prediction(session, _prediction("AAPL", base, target.day, predicted))
    scoring.score_predictions(session)

    assert repo.scored_predictions(session)[0].direction_correct is True


def test_direction_wrong_when_they_disagree(session, client):
    bars = _seed(session, client)
    base, target = bars[-3], bars[-2]

    actual_up = target.close > base.close
    predicted = base.close * (0.98 if actual_up else 1.02)   # deliberately opposite

    repo.save_prediction(session, _prediction("AAPL", base, target.day, predicted))
    scoring.score_predictions(session)

    assert repo.scored_predictions(session)[0].direction_correct is False


def test_future_predictions_are_left_alone(session, client):
    """A target day that hasn't happened has nothing to score against."""
    bars = _seed(session, client)
    future = bars[-1].day + timedelta(days=30)

    repo.save_prediction(session, _prediction("AAPL", bars[-1], future, 123.0))
    assert scoring.score_predictions(session) == []
    assert repo.unscored_predictions(session)


def test_market_holiday_scores_against_the_next_session(session, client):
    """A target landing on a closed day uses the session that did happen."""
    bars = _seed(session, client)
    base, next_session = bars[-3], bars[-2]

    # Aim at the calendar day right after `base`, which has no bar if
    # the following session is further out. on-or-after picks the real one.
    holiday = base.day + timedelta(days=1)
    if holiday == next_session.day:
        pytest.skip("no gap in this fixture to simulate a holiday")

    repo.save_prediction(session, _prediction("AAPL", base, holiday, base.close))
    scored = scoring.score_predictions(session)

    assert len(scored) == 1
    assert scored[0].actual_close == pytest.approx(next_session.close)


def test_scoring_is_idempotent(session, client):
    bars = _seed(session, client)
    repo.save_prediction(
        session, _prediction("AAPL", bars[-3], bars[-2].day, bars[-2].close)
    )

    assert len(scoring.score_predictions(session)) == 1
    assert scoring.score_predictions(session) == []      # nothing left to do


# ---------------------------------------------------------------
# Track record
# ---------------------------------------------------------------

def test_empty_track_record(session):
    record = scoring.track_record(session)
    assert record.n == 0
    assert not record.is_meaningful


def test_track_record_aggregates_scored_predictions(session, client):
    bars = _seed(session, client)

    for i in range(5, 25):
        base, target = bars[-i - 1], bars[-i]
        repo.save_prediction(
            session, _prediction("AAPL", base, target.day, target.close * 1.005)
        )
    scoring.score_predictions(session)

    record = scoring.track_record(session)
    assert record.n == 20
    assert record.mae_pct == pytest.approx(0.5, abs=0.05)
    # Every prediction was 0.5% high, so the bias is systematically positive.
    assert record.bias_pct > 0
    assert not record.is_meaningful          # 20 samples is not evidence


def test_track_record_filters_by_symbol(session, client):
    bars_a = _seed(session, client, "AAPL")
    bars_m = _seed(session, client, "MSFT")

    repo.save_prediction(
        session, _prediction("AAPL", bars_a[-3], bars_a[-2].day, bars_a[-2].close)
    )
    repo.save_prediction(
        session, _prediction("MSFT", bars_m[-3], bars_m[-2].day, bars_m[-2].close)
    )
    scoring.score_predictions(session)

    assert scoring.track_record(session).n == 2
    assert scoring.track_record(session, "AAPL").n == 1


# ---------------------------------------------------------------
# The audit
# ---------------------------------------------------------------

def test_audit_stores_a_row_per_symbol(session, client):
    report = scoring.audit_universe(session, client, symbols=["AAPL", "MSFT"])

    assert report.n == 2
    assert {run.symbol for run in report.runs} == {"AAPL", "MSFT"}
    assert all(run.mae_pct > 0 for run in report.runs)
    assert all(run.n_test > 0 for run in report.runs)
    assert len(repo.get_accuracy_runs(session)) == 2


def test_audit_rerun_same_day_overwrites(session, client):
    scoring.audit_universe(session, client, symbols=["AAPL"])
    scoring.audit_universe(session, client, symbols=["AAPL"])

    # One row per (day, symbol) — the table stays a clean time series.
    assert len(repo.get_accuracy_runs(session)) == 1


def test_audit_keeps_separate_days_apart(session, client):
    today = date(2026, 9, 3)
    scoring.audit_universe(session, client, symbols=["AAPL"], run_date=today)
    scoring.audit_universe(session, client, symbols=["AAPL"], run_date=today - timedelta(days=1))

    assert len(repo.get_accuracy_runs(session)) == 2
    assert len(repo.audit_run_dates(session)) == 2


def test_audit_records_unknown_symbols_as_failures(session, client):
    report = scoring.audit_universe(session, client, symbols=["AAPL", "NOPE"])

    assert report.n == 1
    assert [symbol for symbol, _ in report.failures] == ["NOPE"]


def test_audit_creates_predictions_to_score_later(session, client):
    """Without this the live track record would never start accumulating."""
    scoring.audit_universe(session, client, symbols=["AAPL"])
    assert repo.latest_prediction(session, "AAPL") is not None


def test_audit_report_aggregates(session, client):
    report = scoring.audit_universe(session, client, symbols=["AAPL", "MSFT", "KO"])

    assert 0 <= report.beat_naive <= report.n
    assert report.median_ratio > 0

    accuracy, total, z = report.pooled_direction
    assert total == sum(run.n_test for run in report.runs)
    assert 0 <= accuracy <= 1
    assert abs(z) < 10          # a sane z-score, not an artefact


def test_audit_defaults_to_tracked_symbols(session, client):
    service.get_dashboard(session, "AAPL", client)
    report = scoring.audit_universe(session, client)

    assert [run.symbol for run in report.runs] == ["AAPL"]


def test_mae_ratio_matches_its_inputs(session, client):
    run = scoring.audit_universe(session, client, symbols=["AAPL"]).runs[0]
    assert run.mae_ratio == pytest.approx(run.mae_pct / run.naive_mae_pct)
    assert run.beats_naive == (run.mae_ratio < 1.0)


# ---------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------

def test_missing_columns_are_added_to_an_existing_table(tmp_path, monkeypatch):
    """A database created before the outcome columns existed must survive."""
    import sqlalchemy as sa

    db_file = tmp_path / "legacy.db"
    engine = sa.create_engine(f"sqlite:///{db_file}")

    # Build a predictions table missing every outcome column.
    with engine.begin() as connection:
        connection.execute(sa.text("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                target_day DATE NOT NULL,
                base_day DATE NOT NULL,
                base_close FLOAT NOT NULL,
                predicted_close FLOAT NOT NULL,
                predicted_return_pct FLOAT NOT NULL,
                interval_low FLOAT NOT NULL,
                interval_high FLOAT NOT NULL,
                model_version VARCHAR(40) NOT NULL,
                created_at DATETIME NOT NULL
            )
        """))

    from app import db as db_module

    monkeypatch.setattr(db_module, "engine", engine)
    added = db_module._add_missing_columns()

    assert "predictions.actual_close" in added
    assert "predictions.direction_correct" in added

    columns = {c["name"] for c in sa.inspect(engine).get_columns("predictions")}
    assert {"actual_close", "error_pct", "direction_correct", "scored_at"} <= columns
    engine.dispose()
