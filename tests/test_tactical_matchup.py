import pytest

from kinela.tactical_matchup import (
    tactical_history_summary,
    tactical_matchup_features,
)


def _low_block_result(*, opponent_elo: float, goals_against: float = 0.0) -> dict[str, float]:
    return {
        "ball_possession_pct": 31.0,
        "total_passes": 270.0,
        "total_shots": 5.0,
        "shots_on_goal": 2.0,
        "team_elo": 1500.0,
        "opponent_elo": opponent_elo,
        "goals_for": 1.0,
        "goals_against": goals_against,
        "points": 3.0 if goals_against == 0.0 else 1.0,
    }


def _possession_win() -> dict[str, float]:
    return {
        "ball_possession_pct": 68.0,
        "total_passes": 640.0,
        "total_shots": 18.0,
        "shots_on_goal": 7.0,
        "team_elo": 1750.0,
        "opponent_elo": 1500.0,
        "goals_for": 3.0,
        "goals_against": 0.0,
        "points": 3.0,
    }


def test_low_block_survival_weights_stronger_opponents_more() -> None:
    strong = tactical_history_summary(
        [_low_block_result(opponent_elo=1800.0)] * 3,
        window=20,
    )
    equal = tactical_history_summary(
        [_low_block_result(opponent_elo=1500.0)] * 3,
        window=20,
    )

    assert strong["low_block_sample_strength"] > equal["low_block_sample_strength"]
    assert strong["low_block_reliable_survival"] > equal["low_block_reliable_survival"]


def test_possession_productivity_requires_repeated_dominant_matches() -> None:
    one = tactical_history_summary([_possession_win()], window=20)
    five = tactical_history_summary([_possession_win()] * 5, window=20)

    assert five["possession_habit_strength"] > one["possession_habit_strength"]
    assert (
        five["possession_reliable_productivity"]
        > one["possession_reliable_productivity"]
    )


def test_matchup_features_are_symmetric_when_teams_are_swapped() -> None:
    team_a = tactical_history_summary([_possession_win()] * 4, window=20)
    team_b = tactical_history_summary(
        [_low_block_result(opponent_elo=1800.0)] * 4,
        window=20,
    )
    forward = tactical_matchup_features(
        team_a,
        team_b,
        team_a_elo=1800.0,
        team_b_elo=1500.0,
        team_a_counter_threat=0.0,
        team_b_counter_threat=0.2,
    )
    reverse = tactical_matchup_features(
        team_b,
        team_a,
        team_a_elo=1500.0,
        team_b_elo=1800.0,
        team_a_counter_threat=0.2,
        team_b_counter_threat=0.0,
    )

    for field in (
        "low_block_survival_edge",
        "low_block_result_survival_edge",
        "possession_productivity_edge",
        "possession_goal_productivity_edge",
        "possession_win_productivity_edge",
        "block_breaking_matchup_edge",
        "counter_dominance_matchup_edge",
        "compatibility_v2_edge",
    ):
        assert forward[field] == pytest.approx(-reverse[field])
    for field in ("tactical_stalemate_pressure", "tactical_matchup_intensity"):
        assert forward[field] == pytest.approx(reverse[field])
