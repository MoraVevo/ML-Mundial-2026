from __future__ import annotations

import random
from datetime import date

import pytest

from kinela.extra_time_model import ExtraTimePredictor
from kinela.worldcup_2026 import WorldCup2026Simulator


def test_equal_split_is_neutral_even_with_different_90_minute_xg() -> None:
    predictor = ExtraTimePredictor(
        {"total_goal_lambda": 0.62, "allocation_exponent": 0.0}
    )

    expected_a, expected_b = predictor.expected_goals(2.4, 0.6)
    outcomes = predictor.outcome_probabilities(2.4, 0.6)

    assert expected_a == pytest.approx(0.31)
    assert expected_b == pytest.approx(0.31)
    assert outcomes.team_a_win == pytest.approx(outcomes.team_b_win)
    assert outcomes.team_a_win + outcomes.draw + outcomes.team_b_win == pytest.approx(1.0)


def test_advance_probability_combines_extra_time_and_penalties() -> None:
    predictor = ExtraTimePredictor(
        {"total_goal_lambda": 0.62, "allocation_exponent": 0.0}
    )

    assert predictor.team_a_advance_probability(0.5, 1.8, 1.2) == pytest.approx(0.5)
    assert predictor.team_a_advance_probability(0.6, 1.8, 1.2) > 0.5


class _ExtraTimeStub:
    @staticmethod
    def expected_goals(_xg_a: float, _xg_b: float) -> tuple[float, float]:
        return 0.31, 0.31


class _PenaltyStub:
    @staticmethod
    def team_a_probability(_team_a: str, _team_b: str, _date: date) -> float:
        return 0.5


def test_extra_time_resolution_keeps_90_minute_score_clean(monkeypatch) -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.extra_time_model = _ExtraTimeStub()
    simulator.penalty_model = _PenaltyStub()
    simulator.rng = random.Random(42)
    simulator.current_match_records = []
    recorded_history: list[tuple[int, int, str]] = []
    simulator._record_simulated_match = lambda *args, **kwargs: recorded_history.append(
        (args[3], args[4], kwargs["decided_by"])
    )
    samples = iter([1, 0])
    monkeypatch.setattr("kinela.worldcup_2026._poisson_sample", lambda *_args: next(samples))

    result = simulator._resolve_knockout_draw_after_90(
        "France",
        "Spain",
        date(2026, 7, 14),
        "SEMI_FINALS",
        101,
        None,
        1,
        1,
        1.6,
        1.6,
    )

    assert result == (1, 1, "France")
    assert recorded_history == [(1, 1, "extra_time")]
    trace = simulator.current_match_records[0]
    assert trace["team_a_goals"] == 1
    assert trace["team_b_goals"] == 1
    assert trace["total_goals"] == 2
    assert trace["team_a_extra_time_goals"] == 1
    assert trace["team_b_extra_time_goals"] == 0
    assert trace["team_a_goals_after_extra_time"] == 2
    assert trace["team_b_goals_after_extra_time"] == 1
    assert trace["decided_by"] == "extra_time"
