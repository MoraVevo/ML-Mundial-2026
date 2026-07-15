import pandas as pd

from kinela.lightgbm_model import add_score_timing_to_clean_frame
from kinela.model import (
    _deduplicate_real_matches,
    _manual_goal_events_from_notes,
    _valid_goal_events,
    _normalise_stage_or_round,
    _rolling_team_averages,
    _update_rolling_team_stats,
    load_late85_points_swing_metrics,
    load_score_timing_metrics,
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


def test_manual_goal_notes_parse_multiple_minutes_and_ignore_disallowed_goals() -> None:
    notes = (
        "Final: South Korea 2-1 Czechia. Czechia led through Ladislav Krejci 59'; "
        "South Korea scored through Hwang In-beom 67' and Oh Hyeon-gyu 80'. "
        "Czechia had a Tomas Soucek goal ruled out for offside at 76'."
    )

    events = _manual_goal_events_from_notes(notes, "South Korea", "Czechia", 2, 1)
    valid = _valid_goal_events(events, "South Korea", "Czechia", 2, 1)

    assert valid is not None
    assert [
        (event["team"]["name"], event["time"]["elapsed"], event["time"].get("extra"))
        for event in valid
    ] == [
        ("Czechia", 59, None),
        ("South Korea", 67, None),
        ("South Korea", 80, None),
    ]


def test_manual_goal_notes_parse_shutout_multi_goal_clause() -> None:
    notes = (
        "Final: Argentina 3-0 Algeria. Goals: Lionel Messi 17' 60' and 76' "
        "for Argentina. No red cards reported by source."
    )

    events = _manual_goal_events_from_notes(notes, "Argentina", "Algeria", 3, 0)
    valid = _valid_goal_events(events, "Argentina", "Algeria", 3, 0)

    assert valid is not None
    assert [event["time"]["elapsed"] for event in valid] == [17, 60, 76]
    assert {event["team"]["name"] for event in valid} == {"Argentina"}


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


def test_espn_summary_goal_timeline_feeds_score_timing(tmp_path) -> None:
    manual = tmp_path / "static" / "worldcup_2026_manual_results.csv"
    summary = tmp_path / "raw" / "espn" / "worldcup_2026" / "summary_760493.json"
    manual.parent.mkdir(parents=True)
    summary.parent.mkdir(parents=True)
    manual.write_text(
        "\n".join(
            [
                "match_id,date,stage,group,team_a,team_b,team_a_goals,team_b_goals,winner,source,notes",
                "82,2026-07-01,ROUND_OF_32,,Belgium,Senegal,3,2,Belgium,ESPN event 760493,",
            ]
        ),
        encoding="utf-8",
    )
    summary.write_text(
        """
        {
          "keyEvents": [
            {"scoringPlay": true, "shootout": false, "type": {"type": "goal", "text": "Goal"}, "clock": {"displayValue": "24'"}, "team": {"displayName": "Senegal"}},
            {"scoringPlay": true, "shootout": false, "type": {"type": "goal", "text": "Goal"}, "clock": {"displayValue": "51'"}, "team": {"displayName": "Senegal"}},
            {"scoringPlay": true, "shootout": false, "type": {"type": "goal", "text": "Goal"}, "clock": {"displayValue": "86'"}, "team": {"displayName": "Belgium"}},
            {"scoringPlay": true, "shootout": false, "type": {"type": "goal", "text": "Goal"}, "clock": {"displayValue": "89'"}, "team": {"displayName": "Belgium"}},
            {"scoringPlay": true, "shootout": false, "type": {"type": "goal", "text": "Penalty - Scored"}, "clock": {"displayValue": "120'+5'"}, "team": {"displayName": "Belgium"}},
            {"scoringPlay": true, "shootout": true, "type": {"type": "goal", "text": "Penalty - Scored"}, "clock": {"displayValue": "120'+6'"}, "team": {"displayName": "Senegal"}}
          ]
        }
        """,
        encoding="utf-8",
    )

    training_rows = [
        {
            "match_id": "fd:82",
            "source": "manual-worldcup-2026",
            "home_team": "Belgium",
            "away_team": "Senegal",
            "home_goals": 3,
            "away_goals": 2,
        }
    ]

    timing = load_score_timing_metrics(tmp_path, training_rows)
    late = load_late85_points_swing_metrics(tmp_path, training_rows)

    assert timing["fd:82"]["belgium"]["state_change_swing"] == 2.0
    assert timing["fd:82"]["senegal"]["score_control_value"] > 0.5
    assert late["fd:82"]["belgium"]["late85_points_swing_edge"] == 3.0
    assert late["fd:82"]["senegal"]["late85_points_swing_edge"] == -3.0


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
