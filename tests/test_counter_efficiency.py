from kinela.counter_efficiency import (
    counter_history_summary,
    counter_match_profile,
    current_counter_threat,
)


def _counter_win() -> dict[str, float]:
    return {
        "ball_possession_pct": 31.0,
        "total_passes": 266.0,
        "total_shots": 3.0,
        "shots_on_goal": 2.0,
        "passes_pct": 82.0,
        "team_elo": 1600.0,
        "opponent_elo": 1700.0,
        "goals_for": 2.0,
        "points": 3.0,
    }


def test_low_possession_efficient_win_is_a_counter_profile() -> None:
    profile = counter_match_profile(_counter_win())
    summary = counter_history_summary([_counter_win()], window=10)

    assert profile["counter_applied"] == 1.0
    assert summary["counter_scoring_efficiency"] > 0.0
    assert summary["counter_reliable_success"] > 0.0


def test_high_possession_match_is_not_counted_as_counter() -> None:
    item = {
        **_counter_win(),
        "ball_possession_pct": 68.0,
        "total_passes": 650.0,
    }
    summary = counter_history_summary([item], window=10)

    assert summary["counter_applied_matches"] == 0.0
    assert summary["counter_habit_strength"] == 0.0


def test_repeated_success_increases_reliability_without_changing_raw_efficiency_much() -> None:
    one = counter_history_summary([_counter_win()], window=10)
    five = counter_history_summary([_counter_win()] * 5, window=10)

    assert five["counter_sample_strength"] > one["counter_sample_strength"]
    assert five["counter_reliable_scoring"] > one["counter_reliable_scoring"]


def test_current_threat_requires_both_habit_and_matchup_fit() -> None:
    summary = counter_history_summary([_counter_win()] * 3, window=10)

    assert current_counter_threat(summary, 1.0) > current_counter_threat(summary, 0.2)
