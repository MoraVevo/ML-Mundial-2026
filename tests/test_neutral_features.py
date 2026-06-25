import math

import pandas as pd

from kinela.lightgbm_model import (
    NEUTRAL_FEATURES,
    _calibrated_classifier_importances,
    add_neutral_treated_features,
)


def test_rating_threat_uses_historical_pre_match_ranking() -> None:
    frame = pd.DataFrame(
        [
            {
                "elo_diff": 0.0,
                "fifa_points_diff": 500.0,
                "team_a_fifa_rank": 10.0,
                "team_b_fifa_rank": 100.0,
                "historical_fifa_points_diff": -200.0,
                "live_fifa_points_diff": -180.0,
                "team_a_historical_fifa_rank": 80.0,
                "team_b_historical_fifa_rank": 30.0,
                "confederation_strength_diff": 0.0,
                "team_a_attack_vs_b_defense_avg": 1.0,
                "team_b_attack_vs_a_defense_avg": 1.0,
                "recent6_team_a_attack_vs_b_defense_avg": 1.0,
                "recent6_team_b_attack_vs_a_defense_avg": 1.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "rating_threat_edge"] == treated.loc[
        0,
        "historical_rating_threat_edge",
    ]
    assert treated.loc[0, "rating_guardrail_edge"] == treated.loc[
        0,
        "historical_rating_guardrail_edge",
    ]
    assert treated.loc[0, "rating_drift_edge"] == (
        treated.loc[0, "live_fifa_anchor_edge"]
        - treated.loc[0, "historical_fifa_anchor_edge"]
    )
    assert "rating_drift_abs" in NEUTRAL_FEATURES
    assert "rating_drift_edge" not in NEUTRAL_FEATURES


def test_recent_points_form_is_direct_and_quality_form_remains_contextual() -> None:
    frame = pd.DataFrame(
        [
            {
                "recent6_points_diff": -0.25,
                "recent6_quality_result_points_diff": 0.75,
                "recent6_opponent_elo_diff": 1000.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "recent_points_form_edge"] == -0.25
    assert treated.loc[0, "quality_form_edge"] == 0.75
    assert "quality_form_edge" in NEUTRAL_FEATURES
    assert "recent_points_form_edge" not in NEUTRAL_FEATURES


def test_calibrated_classifier_importances_are_reported() -> None:
    class Estimator:
        feature_importances_ = [2.0, 5.0]

    class CalibratedClassifier:
        estimator = Estimator()

    class Model:
        calibrated_classifiers_ = [CalibratedClassifier()]

    importances = _calibrated_classifier_importances(
        Model(),
        ["slow_signal", "strong_signal"],
    )

    assert importances == [
        {"feature": "strong_signal", "importance": 5.0},
        {"feature": "slow_signal", "importance": 2.0},
    ]


def test_goal_balance_blends_causal_long_and_recent_form_equally() -> None:
    frame = pd.DataFrame(
        [
            {
                "goals_for_diff": 2.0,
                "goals_against_diff": 0.5,
                "recent6_goals_for_diff": 1.0,
                "recent6_goals_against_diff": 0.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "goal_balance_edge"] == 1.25


def test_draw_pressure_uses_historical_pre_match_ranking() -> None:
    common = {
        "historical_fifa_points_diff": 0.0,
        "team_a_historical_fifa_rank": 50.0,
        "team_b_historical_fifa_rank": 50.0,
        "team_a_train_matches": 6,
        "team_b_train_matches": 6,
        "tempo_index": math.log1p(2.5),
    }
    frame = pd.DataFrame(
        [
            {
                **common,
                "fifa_points_diff": 500.0,
                "team_a_fifa_rank": 1.0,
                "team_b_fifa_rank": 100.0,
            },
            {
                **common,
                "fifa_points_diff": -500.0,
                "team_a_fifa_rank": 100.0,
                "team_b_fifa_rank": 1.0,
            },
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "draw_pressure_index"] == treated.loc[
        1,
        "draw_pressure_index",
    ]


def test_draw_pressure_shrinks_cold_start_tempo_to_goal_prior() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_train_matches": 0,
                "team_b_train_matches": 0,
                "tempo_index": 0.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    expected = 1.0 / (1.0 + math.log1p(2.5))
    assert math.isclose(
        treated.loc[0, "draw_pressure_index"],
        expected,
        rel_tol=1e-12,
    )


def test_rating_guardrail_neutralises_missing_fifa_ranking() -> None:
    frame = pd.DataFrame(
        [
            {
                "historical_fifa_points_diff": 900.0,
                "team_a_historical_fifa_rank": 1.0,
                "team_b_historical_fifa_rank": 236.0,
                "live_fifa_points_diff": 950.0,
                "team_a_historical_fifa_observed": 1.0,
                "team_b_historical_fifa_observed": 0.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "historical_fifa_anchor_edge"] == 0.0
    assert treated.loc[0, "live_fifa_anchor_edge"] == 0.0
    assert treated.loc[0, "rating_drift_edge"] == 0.0
    assert treated.loc[0, "rating_guardrail_edge"] == -treated.loc[
        0,
        "rating_threat_edge",
    ]


def test_rating_guardrail_is_fifa_disagreement_not_duplicate_blend() -> None:
    frame = pd.DataFrame(
        [
            {
                "historical_fifa_points_diff": 200.0,
                "team_a_historical_fifa_rank": 20.0,
                "team_b_historical_fifa_rank": 70.0,
                "team_a_historical_fifa_observed": 1.0,
                "team_b_historical_fifa_observed": 1.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert math.isclose(
        treated.loc[0, "rating_guardrail_edge"],
        treated.loc[0, "historical_fifa_anchor_edge"]
        - treated.loc[0, "rating_threat_edge"],
        rel_tol=1e-12,
    )


def test_match_script_active_signal_uses_bilateral_quality_adjustment() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_tactical_detail_coverage": 1.0,
                "team_b_tactical_detail_coverage": 1.0,
                "team_a_score_timing_coverage": 1.0,
                "team_b_score_timing_coverage": 1.0,
                "team_a_recent6_opponent_elo_avg": 1750.0,
                "team_b_recent6_opponent_elo_avg": 1400.0,
                "team_a_elo_pre": 1600.0,
                "team_b_elo_pre": 1600.0,
                "team_a_recent6_low_block_breaking_strength": 0.8,
                "team_b_recent6_low_block_profile": 0.7,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "match_script_compatibility_edge"] != 0.0
    assert math.isclose(
        treated.loc[0, "match_script_compatibility_edge"],
        treated.loc[0, "match_script_quality_adjusted_edge"],
        rel_tol=1e-12,
    )


def test_clinical_finishing_targets_opponent_low_block() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_clinical_finishing": 0.6,
                "team_b_clinical_finishing": -0.2,
                "team_a_clinical_coverage": 0.75,
                "team_b_clinical_coverage": 0.75,
                "team_a_low_block_profile_aligned": 0.1,
                "team_b_low_block_profile_aligned": 0.8,
                "team_a_low_block_coverage_aligned": 0.75,
                "team_b_low_block_coverage_aligned": 0.75,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert math.isclose(
        treated.loc[0, "clinical_low_block_matchup_edge"],
        0.75 * (0.6 * 0.8 - (-0.2 * 0.1)),
        rel_tol=1e-12,
    )
