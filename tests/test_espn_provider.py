from kinela.providers.espn import coverage_by_field, extract_strategic_team_rows, normalize_team_name


def test_extract_strategic_team_rows_includes_tactical_stats_and_signals() -> None:
    payload = {
        "header": {"id": "760435"},
        "boxscore": {
            "teams": [
                {
                    "team": {"displayName": "Congo DR"},
                    "statistics": [
                        {"name": "possessionPct", "displayValue": "24.6"},
                        {"name": "totalPasses", "displayValue": "249"},
                        {"name": "accuratePasses", "displayValue": "195"},
                        {"name": "passPct", "displayValue": "78.3"},
                        {"name": "wonCorners", "displayValue": "4"},
                        {"name": "offsides", "displayValue": "2"},
                        {"name": "totalLongBalls", "displayValue": "40"},
                        {"name": "accurateLongBalls", "displayValue": "11"},
                        {"name": "totalClearance", "displayValue": "27"},
                        {"name": "interceptions", "displayValue": "5"},
                        {"name": "totalShots", "displayValue": "8"},
                        {"name": "shotsOnTarget", "displayValue": "2"},
                    ],
                }
            ]
        },
    }

    rows = extract_strategic_team_rows(payload, match_id="537403")

    assert rows == [
        {
            "match_id": "537403",
            "espn_event_id": "760435",
            "team": "Congo DR",
            "fouls": "",
            "yellow_cards": "",
            "red_cards": "",
            "offsides": "2",
            "corner_kicks": "4",
            "goalkeeper_saves": "",
            "ball_possession_pct": "24.6",
            "total_shots": "8",
            "shots_on_goal": "2",
            "shot_pct": "",
            "penalty_kick_goals": "",
            "penalty_kick_shots": "",
            "passes_accurate": "195",
            "total_passes": "249",
            "passes_pct": "78.3",
            "crosses_accurate": "",
            "total_crosses": "",
            "crosses_pct": "",
            "long_balls_accurate": "11",
            "total_long_balls": "40",
            "long_balls_pct": "",
            "blocked_shots": "",
            "tackles_effective": "",
            "total_tackles": "",
            "tackles_pct": "",
            "interceptions": "5",
            "clearances_effective": "",
            "total_clearances": "27",
            "low_block_profile_score": "0.768",
            "direct_play_share": "0.161",
            "clearances_per_30pct_opp_possession": "10.743",
            "shot_on_goal_rate": "0.250",
        }
    ]


def test_coverage_by_field_counts_non_empty_values() -> None:
    rows = [
        {"match_id": "1", "espn_event_id": "11", "team": "A", "total_passes": "400"},
        {"match_id": "1", "espn_event_id": "11", "team": "B", "total_passes": ""},
    ]

    assert coverage_by_field(rows)["total_passes"] == 1


def test_normalize_team_name_uses_aliases() -> None:
    assert normalize_team_name("Cabo Verde") == "cape verde islands"
    assert normalize_team_name("Côte d'Ivoire") == "ivory coast"
