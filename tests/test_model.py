import pandas as pd

from kinela.lightgbm_model import add_score_timing_to_clean_frame
from kinela.model import (
    _deduplicate_real_matches,
    _valid_goal_events,
    _normalise_stage_or_round,
    _rolling_team_averages,
    _update_rolling_team_stats,
)


def test_normalise_stage_or_round_group_stage_labels() -> None:
    assert _normalise_stage_or_round("Group Stage") == "GROUP_STAGE"
    assert _normalise_stage_or_round("GROUP_STAGE") == "GROUP_STAGE"
    assert _normalise_stage_or_round("Group E - 2") == "GROUP_STAGE"


def test_normalise_stage_or_round_knockout_labels() -> None:
    assert _normalise_stage_or_round("Round of 16") == "LAST_16"
    assert _normalise_stage_or_round("Quarter-finals") == "QUARTER_FINALS"
    assert _normalise_stage_or_round("Semi-finals") == "SEMI_FINALS"
    assert _normalise_stage_or_round("Final") == "FINAL"


def test_rolling_team_average_uses_only_already_recorded_matches() -> None:
    stats: dict[str, dict[str, float]] = {}

    assert _rolling_team_averages(stats.get("guatemala", {})) == (0.0, 0.0, 0)
    _update_rolling_team_stats(stats, "guatemala", 2, 1)
    assert _rolling_team_averages(stats["guatemala"]) == (2.0, 1.0, 1)
    _update_rolling_team_stats(stats, "guatemala", 0, 3)
    assert _rolling_team_averages(stats["guatemala"]) == (1.0, 2.0, 2)


def test_score_timing_excludes_missed_and_shootout_penalties() -> None:
    events = [
        {
            "type": "Goal",
            "detail": "Normal Goal",
            "comments": None,
            "time": {"elapsed": 20},
            "team": {"name": "Alpha"},
        },
        {
            "type": "Goal",
            "detail": "Missed Penalty",
            "comments": None,
            "time": {"elapsed": 50},
            "team": {"name": "Beta"},
        },
        {
            "type": "Goal",
            "detail": "Penalty",
            "comments": "Penalty Shootout",
            "time": {"elapsed": 120, "extra": 1},
            "team": {"name": "Alpha"},
        },
    ]

    valid = _valid_goal_events(events, "Alpha", "Beta", 1, 0)

    assert valid == [events[0]]
    assert _valid_goal_events(events, "Alpha", "Beta", 2, 0) is None


def test_score_control_window_counts_matches_without_timeline(
    monkeypatch,
    tmp_path,
) -> None:
    training = pd.DataFrame(
        [
            {
                "match_id": f"m{index}",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_opponent_elo_pre": 1500.0,
                "away_opponent_elo_pre": 1500.0,
            }
            for index in range(1, 9)
        ]
    )
    timing = {
        "m1": {
            "alpha": {"score_control_value": 1.0},
            "beta": {"score_control_value": -1.0},
        }
    }
    monkeypatch.setattr(
        "kinela.lightgbm_model.load_score_timing_metrics",
        lambda *_args, **_kwargs: timing,
    )

    treated = add_score_timing_to_clean_frame(
        tmp_path,
        pd.DataFrame(index=training.index),
        training,
        window=6,
    )

    assert treated.loc[6, "home_recent6_score_control_value"] == 1.0 / 6.0
    assert treated.loc[6, "home_recent6_score_timing_coverage"] == 1.0 / 6.0
    assert treated.loc[7, "home_recent6_score_control_value"] == 0.0
    assert treated.loc[7, "home_recent6_score_timing_coverage"] == 0.0


def test_real_match_dedup_prefers_manual_row_and_merges_api_detail() -> None:
    rows = [
        {
            "source": "api-football",
            "match_id": "api:10",
            "timestamp": "100",
            "date": "2026-06-13",
            "home_team": "USA",
            "away_team": "Paraguay",
            "home_goals": "4",
            "away_goals": "1",
            "competition_name": "World Cup",
            "home_actual_total_shots": 16.0,
        },
        {
            "source": "manual-worldcup-2026",
            "match_id": "fd:20",
            "timestamp": "101",
            "date": "2026-06-13",
            "home_team": "United States",
            "away_team": "Paraguay",
            "home_goals": "4",
            "away_goals": "1",
            "competition_name": "FIFA World Cup",
            "home_actual_total_shots": None,
        },
    ]

    deduped = _deduplicate_real_matches(rows)

    assert len(deduped) == 1
    assert deduped[0]["match_id"] == "fd:20"
    assert deduped[0]["competition_name"] == "FIFA World Cup"
    assert deduped[0]["home_actual_total_shots"] == 16.0


def test_real_match_dedup_ignores_shootout_total_in_provider_score() -> None:
    rows = [
        {
            "source": "api-football",
            "match_id": "api:10",
            "timestamp": "100",
            "date": "2024-07-05",
            "home_team": "Portugal",
            "away_team": "France",
            "home_goals": "0",
            "away_goals": "0",
            "home_penalty_goals": 3,
            "away_penalty_goals": 5,
            "result": "draw",
        },
        {
            "source": "football-data.org",
            "match_id": "fd:20",
            "timestamp": "101",
            "date": "2024-07-05",
            "home_team": "Portugal",
            "away_team": "France",
            "home_goals": "3",
            "away_goals": "5",
            "result": "away",
        },
    ]

    deduped = _deduplicate_real_matches(rows)

    assert len(deduped) == 1
    assert deduped[0]["match_id"] == "api:10"
    assert deduped[0]["home_goals"] == "0"
    assert deduped[0]["away_goals"] == "0"
    assert deduped[0]["home_penalty_goals"] == 3
    assert deduped[0]["away_penalty_goals"] == 5


def test_real_match_dedup_aligns_detail_from_reversed_provider_row() -> None:
    rows = [
        {
            "source": "api-football",
            "match_id": "api:10",
            "timestamp": "100",
            "date": "2023-06-21",
            "home_team": "Ecuador",
            "away_team": "Costa Rica",
            "home_goals": "3",
            "away_goals": "1",
            "result": "home",
            "home_actual_total_shots": None,
            "away_actual_total_shots": None,
        },
        {
            "source": "api-football",
            "match_id": "api:11",
            "timestamp": "101",
            "date": "2023-06-21",
            "home_team": "Costa Rica",
            "away_team": "Ecuador",
            "home_goals": "1",
            "away_goals": "3",
            "result": "away",
            "home_actual_total_shots": 7.0,
            "away_actual_total_shots": 15.0,
        },
    ]

    deduped = _deduplicate_real_matches(rows)

    assert len(deduped) == 1
    assert deduped[0]["home_team"] == "Ecuador"
    assert deduped[0]["away_team"] == "Costa Rica"
    assert deduped[0]["home_actual_total_shots"] == 15.0
    assert deduped[0]["away_actual_total_shots"] == 7.0
