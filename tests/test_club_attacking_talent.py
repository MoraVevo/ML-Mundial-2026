from collections import deque
from datetime import date

import kinela.club_attacking_talent as club_attacking_talent
from kinela.club_attacking_talent import (
    _player_rate,
    _team_summary,
    build_club_talent_state,
    club_talent_summary_before,
    load_football_data_scorer_attack_profiles,
)


def test_player_rate_rewards_club_attacking_production() -> None:
    strong = {
        "weighted_minutes": 1800.0,
        "raw_minutes": 1800.0,
        "weighted_contributions": 18.0,
    }
    quiet = {
        "weighted_minutes": 1800.0,
        "raw_minutes": 1800.0,
        "weighted_contributions": 2.0,
    }

    assert _player_rate(strong, "F") > _player_rate(quiet, "F")


def test_team_summary_treats_missing_profiles_as_missing_not_zero_talent() -> None:
    lineups = deque(
        [
            {
                9: {"position": "F", "selection_weight": 1.0},
                10: {"position": "M", "selection_weight": 1.0},
            }
            for _ in range(6)
        ],
        maxlen=6,
    )
    profiles = {
        9: {
            "weighted_minutes": 1800.0,
            "raw_minutes": 1800.0,
            "weighted_contributions": 15.0,
        }
    }

    summary = _team_summary(lineups, profiles, global_rate=0.25)

    assert summary["club_attack_talent_signal"] > 0.0
    assert 0.0 < summary["club_attack_coverage"] < 1.0


def test_football_data_scorer_fallback_matches_abbreviated_lineup_name(tmp_path) -> None:
    scorer_dir = tmp_path / "processed" / "football_data"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "scorers.csv").write_text(
        "\n".join(
            [
                "competition_code,competition_name,player_id,player,nationality,team_id,team,played_matches,goals,assists,penalties",
                "PL,Premier League,3257,Bruno Fernandes,Portugal,66,Manchester United FC,35,9,21,4",
                "EC,European Championship,3257,Bruno Fernandes,Portugal,765,Portugal,5,1,,",
            ]
        ),
        encoding="utf-8",
    )
    fallback = load_football_data_scorer_attack_profiles(tmp_path)
    lineups = deque(
        [
            {
                1485: {
                    "name": "B. Fernandes",
                    "position": "M",
                    "selection_weight": 1.0,
                },
            }
            for _ in range(6)
        ],
        maxlen=6,
    )

    summary = _team_summary(
        lineups,
        profiles={},
        global_rate=0.25,
        fallback_profiles=fallback,
        team_key="portugal",
    )

    assert summary["club_star_finisher_signal"] > 0.0
    assert summary["club_attack_coverage"] == 1.0


def test_football_data_scorer_fallback_drops_ambiguous_names(tmp_path) -> None:
    scorer_dir = tmp_path / "processed" / "football_data"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "scorers.csv").write_text(
        "\n".join(
            [
                "competition_code,competition_name,player_id,player,nationality,team_id,team,played_matches,goals,assists,penalties",
                "SA,Serie A,1,Giovanni Simeone,Argentina,1,Club A,20,8,2,1",
                "SA,Serie A,2,Giuliano Simeone,Argentina,2,Club B,20,6,3,0",
                "PL,Premier League,3,Alex Silva,Brazil,3,Club C,10,1,1,0",
                "PL,Premier League,4,Alex Silva,Brazil,4,Club D,10,2,1,0",
            ]
        ),
        encoding="utf-8",
    )

    fallback = load_football_data_scorer_attack_profiles(tmp_path)

    assert ("argentina", "g simeone") not in fallback
    assert ("brazil", "alex silva") not in fallback


def test_football_data_scorer_fallback_excludes_open_future_seasons(
    tmp_path,
) -> None:
    scorer_dir = tmp_path / "processed" / "football_data"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "scorers.csv").write_text(
        "\n".join(
            [
                "competition_code,competition_name,season_start,season_start_date,season_end_date,player_id,player,nationality,team_id,team,played_matches,goals,assists,penalties",
                "BSA,Campeonato Brasileiro Série A,2026,2026-01-28,2026-12-02,10,Pedro Silva,Brazil,1,Club A,12,9,3,1",
                "PL,Premier League,2025,2025-08-15,2026-05-24,11,Bruno Fernandes,Portugal,66,Manchester United FC,35,9,21,4",
            ]
        ),
        encoding="utf-8",
    )

    fallback = load_football_data_scorer_attack_profiles(tmp_path)

    assert ("brazil", "pedro silva") not in fallback
    assert ("portugal", "bruno fernandes") in fallback


def test_fallback_is_not_used_at_initial_club_profile_availability(
    tmp_path,
    monkeypatch,
) -> None:
    scorer_dir = tmp_path / "processed" / "football_data"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "scorers.csv").write_text(
        "\n".join(
            [
                "competition_code,competition_name,player_id,player,nationality,team_id,team,played_matches,goals,assists,penalties",
                "PL,Premier League,3257,Bruno Fernandes,Portugal,66,Manchester United FC,35,9,21,4",
            ]
        ),
        encoding="utf-8",
    )
    details = [
        {
            "fixture": {"date": "2025-06-15T00:00:00+00:00"},
            "teams": {
                "home": {"name": "Portugal"},
                "away": {"name": "Spain"},
            },
            "lineups": [
                {
                    "team": {"name": "Portugal"},
                    "startXI": [
                        {
                            "player": {
                                "id": 1485,
                                "name": "B. Fernandes",
                                "pos": "M",
                            }
                        }
                    ],
                    "substitutes": [],
                }
            ],
        },
        {
            "fixture": {"date": "2025-07-10T00:00:00+00:00"},
            "teams": {
                "home": {"name": "Portugal"},
                "away": {"name": "Croatia"},
            },
            "lineups": [],
        },
        {
            "fixture": {"date": "2026-06-02T00:00:00+00:00"},
            "teams": {
                "home": {"name": "Portugal"},
                "away": {"name": "Croatia"},
            },
            "lineups": [],
        },
    ]
    monkeypatch.setattr(
        club_attacking_talent,
        "load_finished_api_details",
        lambda _data_root: details,
    )

    _, timelines = build_club_talent_state(tmp_path)

    before_fallback = club_talent_summary_before(
        timelines,
        "Portugal",
        date(2025, 7, 10),
    )
    after_fallback = club_talent_summary_before(
        timelines,
        "Portugal",
        date(2026, 6, 3),
    )

    assert before_fallback["club_star_finisher_signal"] == 0.0
    assert before_fallback["club_attack_coverage"] == 0.0
    assert after_fallback["club_attack_coverage"] > 0.0


def test_pre_fallback_global_rate_ignores_fallback_profiles(
    tmp_path,
    monkeypatch,
) -> None:
    scorer_dir = tmp_path / "processed" / "football_data"
    scorer_dir.mkdir(parents=True)
    (scorer_dir / "scorers.csv").write_text(
        "\n".join(
            [
                "competition_code,competition_name,season_start,season_start_date,season_end_date,player_id,player,nationality,team_id,team,played_matches,goals,assists,penalties",
                "PL,Premier League,2025,2025-08-15,2026-05-24,99,Huge Star,Portugal,66,Manchester United FC,35,35,20,1",
            ]
        ),
        encoding="utf-8",
    )
    api_profile = {
        "weighted_minutes": 900.0,
        "raw_minutes": 900.0,
        "weighted_contributions": 5.0,
    }
    details = [
        {
            "fixture": {"date": "2025-06-15T00:00:00+00:00"},
            "teams": {
                "home": {"name": "Portugal"},
                "away": {"name": "Spain"},
            },
            "lineups": [
                {
                    "team": {"name": "Portugal"},
                    "startXI": [
                        {
                            "player": {
                                "id": 1,
                                "name": "Known Forward",
                                "pos": "F",
                            }
                        }
                    ],
                    "substitutes": [],
                }
            ],
        },
        {
            "fixture": {"date": "2025-07-10T00:00:00+00:00"},
            "teams": {
                "home": {"name": "Portugal"},
                "away": {"name": "Croatia"},
            },
            "lineups": [],
        },
    ]
    monkeypatch.setattr(
        club_attacking_talent,
        "load_finished_api_details",
        lambda _data_root: details,
    )
    monkeypatch.setattr(
        club_attacking_talent,
        "load_club_player_attack_profiles",
        lambda _data_root: {1: api_profile},
    )

    _, timelines = build_club_talent_state(tmp_path)
    before_fallback = club_talent_summary_before(
        timelines,
        "Portugal",
        date(2025, 7, 10),
    )

    assert before_fallback["club_attack_coverage"] > 0.0
    assert abs(before_fallback["club_star_finisher_signal"]) < 1e-12
