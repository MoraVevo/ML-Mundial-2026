from datetime import date
from types import SimpleNamespace

import pandas as pd

from kinela.penalty_model import (
    PenaltyShootoutPredictor,
    build_penalty_feature_frame,
    filter_fifa_affiliated_data,
)
from kinela.worldcup_2026 import PenaltyShootoutModel


def test_non_fifa_shootouts_are_removed_before_feature_building() -> None:
    shootouts = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "winner": "Alpha",
                "first_shooter": "",
            },
            {
                "date": "2020-02-01",
                "home_team": "Alpha",
                "away_team": "Outsider",
                "winner": "Outsider",
                "first_shooter": "",
            },
        ]
    )
    results = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Test Cup",
            },
            {
                "date": "2020-02-01",
                "home_team": "Alpha",
                "away_team": "Outsider",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Alternative Cup",
            },
        ]
    )

    filtered_shootouts, filtered_results, audit = filter_fifa_affiliated_data(
        shootouts,
        results,
        {"alpha", "beta"},
    )

    assert len(filtered_shootouts) == 1
    assert len(filtered_results) == 1
    assert audit["shootouts_excluded"] == 1
    frame, team_records, _, _ = build_penalty_feature_frame(
        filtered_shootouts,
        filtered_results,
    )
    assert len(frame) == 1
    assert "outsider" not in team_records


def test_shootout_features_use_only_strictly_prior_dates() -> None:
    shootouts = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "winner": "Alpha",
                "first_shooter": "Alpha",
            },
            {
                "date": "2020-01-01",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "winner": "Alpha",
                "first_shooter": "Gamma",
            },
            {
                "date": "2021-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "winner": "Beta",
                "first_shooter": "Beta",
            },
        ]
    )
    results = pd.DataFrame(
        [
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": 1,
                "away_score": 1,
                "tournament": "Test Cup",
            }
            for row in shootouts.to_dict("records")
        ]
    )

    frame, _, _, _ = build_penalty_feature_frame(shootouts, results)

    assert frame.loc[0, "career_rate_edge"] == 0.0
    assert frame.loc[1, "career_rate_edge"] == 0.0
    assert frame.loc[2, "career_rate_edge"] > 0.0


def test_coin_flip_artifact_is_neutral_and_symmetric() -> None:
    predictor = PenaltyShootoutPredictor(
        {"model_type": "coin_flip", "features": [], "probability": 0.5}
    )

    forward = predictor.probability("France", "Spain", date(2026, 7, 14))
    reverse = predictor.probability("Spain", "France", date(2026, 7, 14))

    assert forward == 0.5
    assert forward + reverse == 1.0


def test_worldcup_penalty_model_has_honest_coin_fallback(tmp_path) -> None:
    simulator = SimpleNamespace(data_root=tmp_path)
    model = PenaltyShootoutModel(simulator)

    assert model.team_a_probability("England", "Argentina", date(2026, 7, 15)) == 0.5
