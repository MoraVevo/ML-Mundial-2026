import math

import pandas as pd

from kinela.lightgbm_model import (
    NEUTRAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    NEUTRAL_RESULT_FEATURES,
    _calibrated_classifier_importances,
    add_neutral_treated_features,
)


def test_default_neutral_recipe_is_v10_fifa_sum_live_without_custom_elo() -> None:
    assert (
        NEUTRAL_MODEL_RECIPE
        == "neutral_worldcup_v10_fifa_sum_live_no_custom_elo_depth4_fotmob_xg_probability_ensemble"
    )
    assert NEUTRAL_FEATURES == [
        "competition_family",
        "stage_or_round",
        "rating_threat_edge",
        "quality_form_edge",
        "goal_balance_edge",
        "draw_pressure_index",
        "score_timing_edge",
        "club_star_finisher_edge",
        "worldcup_fotmob_current_story_edge",
    ]
    assert NEUTRAL_RESULT_FEATURES == [
        *NEUTRAL_FEATURES,
        "worldcup_fotmob_xg_matchup_team_a",
        "worldcup_fotmob_xg_matchup_team_b",
    ]
    assert "match_script_compatibility_edge" not in NEUTRAL_FEATURES
    assert "clinical_low_block_matchup_edge" not in NEUTRAL_FEATURES


def test_rating_threat_uses_single_observed_fifa_sum_live_signal() -> None:
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

    assert math.isclose(
        treated.loc[0, "rating_threat_edge"],
        math.tanh(-180.0 / 190.0),
        rel_tol=1e-12,
    )
    assert treated.loc[0, "rating_guardrail_edge"] == 0.0
    assert treated.loc[0, "rating_drift_edge"] == (
        treated.loc[0, "live_fifa_anchor_edge"]
        - treated.loc[0, "historical_fifa_anchor_edge"]
    )
    assert "rating_drift_abs" not in NEUTRAL_FEATURES
    assert "rating_drift_edge" not in NEUTRAL_FEATURES
    assert "worldcup_points_memory_edge" not in NEUTRAL_FEATURES


def test_quality_form_is_plain_recent_points_without_custom_elo() -> None:
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
    assert treated.loc[0, "quality_form_edge"] == -0.25
    assert treated.loc[0, "opponent_adjusted_quality_form_edge"] == 0.75
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
    assert treated.loc[0, "rating_guardrail_edge"] == 0.0


def test_rating_guardrail_is_disabled_to_avoid_duplicate_rating_signal() -> None:
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

    assert treated.loc[0, "rating_guardrail_edge"] == 0.0
    assert "rating_guardrail_edge" not in NEUTRAL_FEATURES


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


def test_club_star_finisher_edge_is_active_without_total_attack_duplicate() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_club_star_finisher_signal": 0.65,
                "team_b_club_star_finisher_signal": 0.20,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert math.isclose(
        treated.loc[0, "club_star_finisher_edge"],
        0.45,
        rel_tol=1e-12,
    )
    assert "club_star_finisher_edge" in NEUTRAL_FEATURES
    assert "club_attack_talent_edge" not in NEUTRAL_FEATURES


def test_worldcup_detail_flow_edge_uses_espn_leader_coverage_gate() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_espn_leader_detail_coverage": 1.0,
                "team_b_espn_leader_detail_coverage": 1.0,
                "team_a_recent6_espn_top_xg": 1.1,
                "team_b_recent6_espn_top_xg": 0.2,
                "team_a_recent6_espn_top_big_chances_created": 1.0,
                "team_b_recent6_espn_top_big_chances_created": 0.0,
                "team_a_recent6_espn_top_big_chances_missed": 0.0,
                "team_b_recent6_espn_top_big_chances_missed": 0.0,
                "team_a_recent6_espn_top_duels_won": 7.0,
                "team_b_recent6_espn_top_duels_won": 3.0,
                "team_a_recent6_espn_keeper_xg_conceded": 0.4,
                "team_b_recent6_espn_keeper_xg_conceded": 1.4,
                "team_a_recent6_shots_on_goal": 5.0,
                "team_b_recent6_shots_on_goal": 2.0,
                "team_a_recent6_passes_accurate": 560.0,
                "team_b_recent6_passes_accurate": 300.0,
                "team_a_recent6_passes_pct": 88.0,
                "team_b_recent6_passes_pct": 73.0,
                "team_a_recent6_ball_possession_pct": 60.0,
                "team_b_recent6_ball_possession_pct": 42.0,
                "team_a_recent6_goalkeeper_saves": 1.0,
                "team_b_recent6_goalkeeper_saves": 4.0,
                "team_a_recent6_fouls": 9.0,
                "team_b_recent6_fouls": 15.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "espn_leader_detail_coverage_pair"] == 1.0
    assert treated.loc[0, "worldcup_chance_quality_edge"] > 0.0
    assert treated.loc[0, "worldcup_detail_flow_edge"] > 0.0
    assert "worldcup_detail_flow_edge" not in NEUTRAL_FEATURES


def test_worldcup_detail_flow_edge_is_zero_without_pair_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_espn_leader_detail_coverage": 1.0,
                "team_b_espn_leader_detail_coverage": 0.0,
                "team_a_recent6_espn_top_xg": 2.0,
                "team_b_recent6_espn_top_xg": 0.0,
            }
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "espn_leader_detail_coverage_pair"] == 0.0
    assert treated.loc[0, "worldcup_detail_flow_edge"] == 0.0


def test_fotmob_worldcup_chance_edge_requires_bilateral_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "team_a_recent6_fotmob_detail_coverage": 1.0,
                "team_b_recent6_fotmob_detail_coverage": 1.0,
                "team_a_recent6_fotmob_expected_goals": 2.0,
                "team_a_recent6_fotmob_expected_goals_conceded": 0.7,
                "team_a_recent6_fotmob_expected_goals_on_target": 1.5,
                "team_a_recent6_fotmob_big_chances": 4.0,
                "team_a_recent6_fotmob_big_chances_conceded": 1.0,
                "team_a_recent6_fotmob_big_chances_missed": 1.0,
                "team_a_recent6_fotmob_total_shots": 15.0,
                "team_a_recent6_fotmob_shots_on_target": 6.0,
                "team_a_recent6_fotmob_touches_opp_box": 35.0,
                "team_b_recent6_fotmob_expected_goals": 0.8,
                "team_b_recent6_fotmob_expected_goals_conceded": 1.6,
                "team_b_recent6_fotmob_expected_goals_on_target": 0.4,
                "team_b_recent6_fotmob_big_chances": 1.0,
                "team_b_recent6_fotmob_big_chances_conceded": 4.0,
                "team_b_recent6_fotmob_big_chances_missed": 3.0,
                "team_b_recent6_fotmob_total_shots": 7.0,
                "team_b_recent6_fotmob_shots_on_target": 2.0,
                "team_b_recent6_fotmob_touches_opp_box": 10.0,
            },
            {
                "team_a_recent6_fotmob_detail_coverage": 1.0,
                "team_b_recent6_fotmob_detail_coverage": 0.0,
                "team_a_recent6_fotmob_expected_goals": 2.0,
                "team_b_recent6_fotmob_expected_goals": 0.0,
            },
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "worldcup_fotmob_chance_coverage_pair"] == 1.0
    assert treated.loc[0, "worldcup_fotmob_xg_balance_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_chance_pressure_edge"] > 0.0
    assert treated.loc[1, "worldcup_fotmob_chance_coverage_pair"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_xg_balance_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_chance_pressure_edge"] == 0.0
    assert "worldcup_fotmob_chance_pressure_edge" not in NEUTRAL_FEATURES


def test_interpreted_fotmob_worldcup_edges_require_bilateral_coverage() -> None:
    def story_columns(side: str, *, strong: bool, coverage: float) -> dict[str, float]:
        prefix = f"team_{side}_worldcup_recent6"
        if strong:
            values = {
                "fotmob_underlying_threat": 0.78,
                "fotmob_finishing_signal": 0.22,
                "fotmob_waste_signal": 0.15,
                "fotmob_low_possession_punch": 0.52,
                "fotmob_sterile_control_risk": 0.05,
                "fotmob_defensive_resistance": 0.70,
                "fotmob_chance_control_signal": 0.68,
                "fotmob_unrewarded_pressure": 0.46,
                "fotmob_clinical_chance_signal": 0.31,
                "fotmob_expected_goals": 1.9,
                "fotmob_expected_goals_conceded": 0.7,
            }
        else:
            values = {
                "fotmob_underlying_threat": 0.32,
                "fotmob_finishing_signal": -0.18,
                "fotmob_waste_signal": 0.60,
                "fotmob_low_possession_punch": 0.10,
                "fotmob_sterile_control_risk": 0.40,
                "fotmob_defensive_resistance": 0.25,
                "fotmob_chance_control_signal": 0.08,
                "fotmob_unrewarded_pressure": 0.05,
                "fotmob_clinical_chance_signal": -0.34,
                "fotmob_expected_goals": 0.7,
                "fotmob_expected_goals_conceded": 1.8,
            }
        return {
            f"{prefix}_{key}": value
            for key, value in {
                "fotmob_detail_coverage": coverage,
                **values,
            }.items()
        }

    frame = pd.DataFrame(
        [
            {
                **story_columns("a", strong=True, coverage=1.0),
                **story_columns("b", strong=False, coverage=1.0),
                "team_a_low_block_coverage_aligned": 1.0,
                "team_b_low_block_coverage_aligned": 1.0,
                "team_a_low_block_profile_aligned": 0.25,
                "team_b_low_block_profile_aligned": 0.85,
            },
            {
                **story_columns("a", strong=True, coverage=1.0),
                **story_columns("b", strong=False, coverage=0.0),
                "team_a_low_block_coverage_aligned": 1.0,
                "team_b_low_block_coverage_aligned": 1.0,
                "team_a_low_block_profile_aligned": 0.25,
                "team_b_low_block_profile_aligned": 0.85,
            },
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "worldcup_fotmob_low_block_solution_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_transition_punch_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_unrewarded_pressure_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_finishing_discipline_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_interpreted_edge"] > 0.0
    assert treated.loc[1, "worldcup_fotmob_low_block_solution_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_transition_punch_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_unrewarded_pressure_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_finishing_discipline_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_interpreted_edge"] == 0.0
    assert "worldcup_fotmob_interpreted_edge" not in NEUTRAL_FEATURES


def test_current_worldcup_fotmob_story_requires_bilateral_coverage() -> None:
    def story_columns(side: str, *, strong: bool, coverage: float) -> dict[str, float]:
        prefix = f"team_{side}_current_worldcup_recent6"
        if strong:
            values = {
                "fotmob_underlying_threat": 0.72,
                "fotmob_finishing_signal": 0.18,
                "fotmob_waste_signal": 0.18,
                "fotmob_low_possession_punch": 0.45,
                "fotmob_sterile_control_risk": 0.07,
                "fotmob_defensive_resistance": 0.66,
                "fotmob_chance_control_signal": 0.61,
                "fotmob_unrewarded_pressure": 0.38,
                "fotmob_clinical_chance_signal": 0.24,
                "fotmob_expected_goals": 1.7,
                "fotmob_expected_goals_conceded": 0.8,
            }
        else:
            values = {
                "fotmob_underlying_threat": 0.29,
                "fotmob_finishing_signal": -0.12,
                "fotmob_waste_signal": 0.52,
                "fotmob_low_possession_punch": 0.08,
                "fotmob_sterile_control_risk": 0.34,
                "fotmob_defensive_resistance": 0.31,
                "fotmob_chance_control_signal": 0.11,
                "fotmob_unrewarded_pressure": 0.03,
                "fotmob_clinical_chance_signal": -0.25,
                "fotmob_expected_goals": 0.6,
                "fotmob_expected_goals_conceded": 1.5,
            }
        return {
            f"{prefix}_{key}": value
            for key, value in {
                "fotmob_detail_coverage": coverage,
                **values,
            }.items()
        }

    frame = pd.DataFrame(
        [
            {
                **story_columns("a", strong=True, coverage=1.0),
                **story_columns("b", strong=False, coverage=1.0),
                "team_a_low_block_coverage_aligned": 1.0,
                "team_b_low_block_coverage_aligned": 1.0,
                "team_a_low_block_profile_aligned": 0.30,
                "team_b_low_block_profile_aligned": 0.80,
            },
            {
                **story_columns("a", strong=True, coverage=1.0),
                **story_columns("b", strong=False, coverage=0.0),
                "team_a_low_block_coverage_aligned": 1.0,
                "team_b_low_block_coverage_aligned": 1.0,
                "team_a_low_block_profile_aligned": 0.30,
                "team_b_low_block_profile_aligned": 0.80,
            },
        ]
    )

    treated = add_neutral_treated_features(frame)

    assert treated.loc[0, "worldcup_fotmob_current_chance_pressure_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_current_low_block_solution_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_current_transition_punch_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_current_unrewarded_pressure_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_current_controlled_dominance_edge"] > 0.0
    assert treated.loc[0, "worldcup_fotmob_current_story_edge"] > 0.0
    current_story = (
        0.52 * treated.loc[0, "worldcup_fotmob_current_controlled_dominance_edge"]
        + 0.20 * treated.loc[0, "worldcup_fotmob_current_chance_pressure_edge"]
        + 0.18 * treated.loc[0, "worldcup_fotmob_current_low_block_solution_edge"]
        + 0.07 * treated.loc[0, "worldcup_fotmob_current_transition_punch_edge"]
        + 0.03 * treated.loc[0, "worldcup_fotmob_current_unrewarded_pressure_edge"]
    )
    expected_story = (
        0.55 * treated.loc[0, "worldcup_fotmob_interpreted_edge"]
        + 0.45 * current_story
    )
    assert math.isclose(
        treated.loc[0, "worldcup_fotmob_current_story_edge"],
        expected_story,
        rel_tol=1e-12,
    )
    assert treated.loc[1, "worldcup_fotmob_current_chance_pressure_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_current_low_block_solution_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_current_transition_punch_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_current_unrewarded_pressure_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_current_controlled_dominance_edge"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_current_story_edge"] == 0.0
    assert "worldcup_fotmob_current_story_edge" in NEUTRAL_FEATURES


def test_xg_matchup_uses_prior_worldcup_creation_and_opponent_concession() -> None:
    def xg_columns(prefix: str, *, xg: float, xgc: float, coverage: float) -> dict[str, float]:
        return {
            f"team_a_{prefix}_recent6_fotmob_expected_goals": xg,
            f"team_a_{prefix}_recent6_fotmob_expected_goals_conceded": xgc,
            f"team_a_{prefix}_recent6_fotmob_detail_coverage": coverage,
        }

    frame = pd.DataFrame(
        [
            {
                **xg_columns("worldcup", xg=2.0, xgc=0.6, coverage=1.0),
                "team_b_worldcup_recent6_fotmob_expected_goals": 0.8,
                "team_b_worldcup_recent6_fotmob_expected_goals_conceded": 1.4,
                "team_b_worldcup_recent6_fotmob_detail_coverage": 1.0,
                **xg_columns("current_worldcup", xg=1.8, xgc=0.7, coverage=1.0),
                "team_b_current_worldcup_recent6_fotmob_expected_goals": 0.9,
                "team_b_current_worldcup_recent6_fotmob_expected_goals_conceded": 1.2,
                "team_b_current_worldcup_recent6_fotmob_detail_coverage": 1.0,
            },
            {
                **xg_columns("worldcup", xg=2.0, xgc=0.6, coverage=0.0),
                "team_b_worldcup_recent6_fotmob_expected_goals": 0.8,
                "team_b_worldcup_recent6_fotmob_expected_goals_conceded": 1.4,
                "team_b_worldcup_recent6_fotmob_detail_coverage": 0.0,
                **xg_columns("current_worldcup", xg=1.8, xgc=0.7, coverage=0.0),
                "team_b_current_worldcup_recent6_fotmob_expected_goals": 0.9,
                "team_b_current_worldcup_recent6_fotmob_expected_goals_conceded": 1.2,
                "team_b_current_worldcup_recent6_fotmob_detail_coverage": 0.0,
            },
        ]
    )

    treated = add_neutral_treated_features(frame)

    history_a = 0.50 * (2.0 + 1.4)
    history_b = 0.50 * (0.8 + 0.6)
    current_a = 0.50 * (1.8 + 1.2)
    current_b = 0.50 * (0.9 + 0.7)
    assert math.isclose(
        treated.loc[0, "worldcup_fotmob_xg_matchup_team_a"],
        0.55 * history_a + 0.45 * current_a,
        rel_tol=1e-12,
    )
    assert math.isclose(
        treated.loc[0, "worldcup_fotmob_xg_matchup_team_b"],
        0.55 * history_b + 0.45 * current_b,
        rel_tol=1e-12,
    )
    assert treated.loc[1, "worldcup_fotmob_xg_matchup_team_a"] == 0.0
    assert treated.loc[1, "worldcup_fotmob_xg_matchup_team_b"] == 0.0
