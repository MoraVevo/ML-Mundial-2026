from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from kinela.counter_efficiency import (
    COUNTER_EFFICIENCY_WINDOW,
    counter_history_summary,
    current_counter_threat,
    current_underdog_fit,
)
from kinela.clinical_finishing import (
    CLINICAL_FINISHING_WINDOW,
    clinical_finishing_summary,
)
from kinela.club_attacking_talent import (
    build_club_talent_state,
    club_talent_summary_before,
)
from kinela.fifa_ranking import normalise_team_name
from kinela.model import (
    BASE_ELO,
    CORE_DETAIL_STAT_FEATURES,
    DETAIL_STAT_FEATURES,
    ESPN_LEADER_DETAIL_FEATURES,
    FOTMOB_WORLD_CUP_DETAIL_FEATURES,
    RECENT_FORM_WINDOW,
    _infer_group_keys,
    _load_match_rows,
    _opponent_quality_factor,
    _with_recent_features,
    load_late85_points_swing_metrics,
    load_score_timing_metrics,
)
from kinela.player_attacking_personnel import (
    build_personnel_state,
    personnel_summary_before,
)


TARGET_RESULT_MAP = {"home": 0, "draw": 1, "away": 2}
NEUTRAL_RESULT_LABELS = ["team_a", "draw", "team_b"]
DRAW_PRESSURE_TOTAL_GOALS_PRIOR = 2.5

CATEGORICAL_FEATURES = [
    "competition_family",
    "stage_or_round",
]

BASE_BOOLEAN_FEATURES = [
    "is_qualifier",
    "same_confederation",
]

BOOLEAN_FEATURES = BASE_BOOLEAN_FEATURES.copy()
BOOLEAN_FEATURES.extend(["home_is_tournament_host", "away_is_tournament_host"])

for _side in ("home", "away"):
    for _index in range(1, 7):
        BOOLEAN_FEATURES.extend(
            [
                f"{_side}_worldcup_last6_{_index}_win",
                f"{_side}_worldcup_last6_{_index}_draw",
                f"{_side}_worldcup_last6_{_index}_loss",
            ]
        )

NUMERIC_FEATURES = [
    "global_home_goals_train",
    "global_away_goals_train",
    "competition_train_matches",
    "competition_home_goals_avg",
    "competition_away_goals_avg",
    "phase_train_matches",
    "phase_home_goals_avg",
    "phase_away_goals_avg",
    "home_group_matches_pre",
    "home_group_points_pre",
    "home_group_goal_diff_pre",
    "home_group_goals_for_pre",
    "home_group_goals_against_pre",
    "home_group_position_pre",
    "away_group_matches_pre",
    "away_group_points_pre",
    "away_group_goal_diff_pre",
    "away_group_goals_for_pre",
    "away_group_goals_against_pre",
    "away_group_position_pre",
    "group_points_diff_pre",
    "group_goal_diff_diff_pre",
    "group_position_diff_pre",
    "home_team_train_matches",
    "home_team_goals_for_avg",
    "home_team_goals_against_avg",
    "home_confederation_strength",
    "home_cross_confederation_matches",
    "home_cross_confederation_strength",
    "away_team_train_matches",
    "away_team_goals_for_avg",
    "away_team_goals_against_avg",
    "away_confederation_strength",
    "away_cross_confederation_matches",
    "away_cross_confederation_strength",
    "home_recent6_matches",
    "home_recent6_goals_for_avg",
    "home_recent6_goals_against_avg",
    "home_recent6_points_avg",
    "home_recent6_win_rate",
    "home_recent6_goal_diff_avg",
    "home_recent6_opponent_elo_avg",
    "home_recent6_quality_result_points_avg",
    "home_recent6_quality_goal_balance_avg",
    "away_recent6_matches",
    "away_recent6_goals_for_avg",
    "away_recent6_goals_against_avg",
    "away_recent6_points_avg",
    "away_recent6_win_rate",
    "away_recent6_goal_diff_avg",
    "away_recent6_opponent_elo_avg",
    "away_recent6_quality_result_points_avg",
    "away_recent6_quality_goal_balance_avg",
    "home_rest_days",
    "away_rest_days",
    "home_elo_pre",
    "home_opponent_elo_pre",
    "away_elo_pre",
    "away_opponent_elo_pre",
    "home_worldcup_recent6_win_rate",
    "away_worldcup_recent6_win_rate",
    "home_fifa_rank",
    "home_fifa_points",
    "home_historical_fifa_rank",
    "home_historical_fifa_points",
    "home_historical_fifa_observed",
    "away_fifa_rank",
    "away_fifa_points",
    "away_historical_fifa_rank",
    "away_historical_fifa_points",
    "away_historical_fifa_observed",
]
for _side in ("home", "away"):
    NUMERIC_FEATURES.extend(f"{_side}_recent6_{_feature}_avg" for _feature in DETAIL_STAT_FEATURES)
    NUMERIC_FEATURES.extend(
        f"{_side}_worldcup_recent6_{_feature}_avg"
        for _feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
    )
    NUMERIC_FEATURES.extend(
        f"{_side}_current_worldcup_recent6_{_feature}_avg"
        for _feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
    )

NEUTRAL_NUMERIC_FEATURES = [
    "competition_train_matches",
    "competition_goals_avg",
    "phase_train_matches",
    "phase_goals_avg",
    "host_home_advantage_diff",
    "team_a_train_matches",
    "team_a_goals_for_avg",
    "team_a_goals_against_avg",
    "team_a_confederation_strength",
    "team_a_cross_confederation_matches",
    "team_a_cross_confederation_strength",
    "team_b_train_matches",
    "team_b_goals_for_avg",
    "team_b_goals_against_avg",
    "teams_goals_for_sum",
    "teams_goals_against_sum",
    "team_a_attack_vs_b_defense_avg",
    "team_b_attack_vs_a_defense_avg",
    "match_attack_defense_volume",
    "team_b_confederation_strength",
    "team_b_cross_confederation_matches",
    "team_b_cross_confederation_strength",
    "team_a_recent6_matches",
    "team_b_recent6_matches",
    "recent6_goals_for_sum",
    "recent6_goals_against_sum",
    "recent6_team_a_attack_vs_b_defense_avg",
    "recent6_team_b_attack_vs_a_defense_avg",
    "recent6_match_attack_defense_volume",
    "team_b_recent6_points_avg",
    "recent6_quality_result_points_diff",
    "team_a_rest_days",
    "team_b_rest_days",
    "goals_for_diff",
    "goals_against_diff",
    "confederation_strength_diff",
    "cross_confederation_strength_diff",
    "cross_confederation_matches_diff",
    "recent6_goals_for_diff",
    "recent6_goals_against_diff",
    "recent6_opponent_elo_diff",
    "rest_days_diff",
    "team_a_elo_pre",
    "team_b_elo_pre",
    "elo_diff",
    "team_a_recent6_opponent_elo_avg",
    "team_b_recent6_opponent_elo_avg",
    "team_a_worldcup_recent6_win_rate",
    "team_b_worldcup_recent6_win_rate",
    "worldcup_recent6_win_rate_diff",
    "team_a_fifa_rank",
    "team_b_fifa_rank",
    "fifa_points_diff",
    "team_a_historical_fifa_rank",
    "team_b_historical_fifa_rank",
    "historical_fifa_points_diff",
    "team_a_historical_fifa_observed",
    "team_b_historical_fifa_observed",
    "team_a_live_fifa_points",
    "team_b_live_fifa_points",
    "live_fifa_points_diff",
    "h2h_recent_2y_matches",
    "h2h_recent_2y_days_since_last",
    "h2h_recent_2y_team_a_goals_avg",
    "h2h_recent_2y_team_b_goals_avg",
    "h2h_recent_2y_team_a_points_avg",
    "h2h_recent_2y_team_b_points_avg",
    "h2h_recent_2y_draw_rate",
    "tempo_index",
    "worldcup_memory_edge",
    "recent6_fouls_sum",
    "recent6_yellow_cards_sum",
    "recent6_fouls_diff",
    "recent6_yellow_cards_diff",
    "recent6_physical_coverage",
    "team_a_espn_leader_detail_coverage",
    "team_b_espn_leader_detail_coverage",
    "espn_leader_detail_coverage_pair",
]
# Detailed API-Football stats are kept in the curated matrix, but excluded from
# model features until coverage is broad enough. Sparse tactical stats were
# adding noise to the national-only split.

NEUTRAL_BASE_FEATURES = CATEGORICAL_FEATURES + BASE_BOOLEAN_FEATURES + NEUTRAL_NUMERIC_FEATURES
NEUTRAL_HELPER_NUMERIC_FEATURES = [
    "team_a_late85_points_swing",
    "team_b_late85_points_swing",
    "team_a_score_state_value",
    "team_b_score_state_value",
    "team_a_score_control_value",
    "team_b_score_control_value",
    "team_a_scoring_quickness",
    "team_b_scoring_quickness",
    "team_a_score_control_quality",
    "team_b_score_control_quality",
    "team_a_narrow_lead_hold",
    "team_b_narrow_lead_hold",
    "team_a_comfortable_lead",
    "team_b_comfortable_lead",
    "team_a_game_state_friction",
    "team_b_game_state_friction",
    "team_a_state_change_swing",
    "team_b_state_change_swing",
    "team_a_early_state_change_swing",
    "team_b_early_state_change_swing",
    "team_a_quality_score_control_value",
    "team_b_quality_score_control_value",
    "team_a_quality_state_change_swing",
    "team_b_quality_state_change_swing",
    "team_a_quality_early_state_change_swing",
    "team_b_quality_early_state_change_swing",
    "team_a_score_timing_coverage",
    "team_b_score_timing_coverage",
    "team_a_counter_current_threat",
    "team_b_counter_current_threat",
    "team_a_tactical_detail_coverage",
    "team_b_tactical_detail_coverage",
    "team_a_clinical_finishing",
    "team_b_clinical_finishing",
    "team_a_clinical_coverage",
    "team_b_clinical_coverage",
    "team_a_low_block_profile_aligned",
    "team_b_low_block_profile_aligned",
    "team_a_low_block_coverage_aligned",
    "team_b_low_block_coverage_aligned",
    "team_a_attacking_personnel_signal",
    "team_b_attacking_personnel_signal",
    "team_a_star_finisher_signal",
    "team_b_star_finisher_signal",
    "team_a_attack_core_signal",
    "team_b_attack_core_signal",
    "team_a_personnel_coverage",
    "team_b_personnel_coverage",
    "team_a_club_attack_talent_signal",
    "team_b_club_attack_talent_signal",
    "team_a_club_star_finisher_signal",
    "team_b_club_star_finisher_signal",
    "team_a_club_attack_coverage",
    "team_b_club_attack_coverage",
]
NEUTRAL_HELPER_NUMERIC_FEATURES.extend(
    f"team_{side}_recent6_{feature}" for side in ("a", "b") for feature in DETAIL_STAT_FEATURES
)
NEUTRAL_HELPER_NUMERIC_FEATURES.extend(
    f"team_{side}_worldcup_recent6_{feature}"
    for side in ("a", "b")
    for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
)
NEUTRAL_HELPER_NUMERIC_FEATURES.extend(
    f"team_{side}_current_worldcup_recent6_{feature}"
    for side in ("a", "b")
    for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
)
PARSIMONIOUS_NEUTRAL_FEATURES = [
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
NEUTRAL_GOAL_FEATURES = PARSIMONIOUS_NEUTRAL_FEATURES
NEUTRAL_BASE_RESULT_FEATURES = PARSIMONIOUS_NEUTRAL_FEATURES
NEUTRAL_XG_RESULT_FEATURES = [
    "worldcup_fotmob_xg_matchup_team_a",
    "worldcup_fotmob_xg_matchup_team_b",
]
NEUTRAL_RESULT_FEATURES = [*NEUTRAL_BASE_RESULT_FEATURES, *NEUTRAL_XG_RESULT_FEATURES]
NEUTRAL_XG_RESULT_BLEND_WEIGHT = 0.50
NEUTRAL_MODEL_RECIPE = (
    "neutral_worldcup_v10_fifa_sum_live_no_custom_elo_depth4_fotmob_xg_probability_ensemble"
)
# Legacy callers use this list for the two goal regressors. The result
# classifier receives the separate, small xG feature list above.
NEUTRAL_FEATURES = NEUTRAL_GOAL_FEATURES
NEUTRAL_CANDIDATE_FEATURES = [
    "late85_points_swing_edge",
    "score_state_value_edge",
    "score_control_value_edge",
    "scoring_quickness_edge",
    "score_timing_edge",
    "score_control_quality_edge",
    "narrow_lead_hold_edge",
    "comfortable_lead_edge",
    "game_state_friction_edge",
    "score_control_refined_edge",
    "state_change_swing_edge",
    "early_state_change_swing_edge",
    "quality_score_control_value_edge",
    "quality_state_change_swing_edge",
    "quality_early_state_change_swing_edge",
    "quality_score_control_swing_edge",
    "late_resilience_matchup_edge",
    "low_block_sterile_matchup_edge",
    "pressure_block_matchup_edge",
    "strategic_compatibility_edge",
    "match_script_low_block_edge",
    "match_script_space_edge",
    "match_script_timing_edge",
    "match_script_state_edge",
    "match_script_data_coverage",
    "match_script_compatibility_edge",
    "match_script_quality_context_edge",
    "match_script_quality_low_block_edge",
    "match_script_quality_space_edge",
    "match_script_quality_timing_edge",
    "match_script_quality_state_edge",
    "match_script_quality_adjusted_edge",
    "recent_points_form_edge",
    "rating_drift_abs",
    "group_state_edge",
    "group_pressure_index",
    "group_pressure_edge",
    "counter_current_threat_edge",
    "clinical_low_block_matchup_edge",
    "attacking_personnel_edge",
    "star_finisher_edge",
    "attack_core_edge",
    "personnel_coverage_pair",
    "club_attack_talent_edge",
    "club_star_finisher_edge",
    "club_talent_coverage_pair",
    "worldcup_points_memory_edge",
    "worldcup_chance_quality_edge",
    "worldcup_detail_flow_edge",
    "worldcup_fotmob_xg_balance_edge",
    "worldcup_fotmob_chance_pressure_edge",
    "worldcup_fotmob_chance_coverage_pair",
    "worldcup_fotmob_interpreted_edge",
    "worldcup_fotmob_low_block_solution_edge",
    "worldcup_fotmob_transition_punch_edge",
    "worldcup_fotmob_unrewarded_pressure_edge",
    "worldcup_fotmob_finishing_discipline_edge",
    "worldcup_fotmob_current_chance_pressure_edge",
    "worldcup_fotmob_current_low_block_solution_edge",
    "worldcup_fotmob_current_transition_punch_edge",
    "worldcup_fotmob_current_unrewarded_pressure_edge",
    "worldcup_fotmob_current_controlled_dominance_edge",
    "worldcup_fotmob_current_story_edge",
    "worldcup_fotmob_xg_matchup_team_a",
    "worldcup_fotmob_xg_matchup_team_b",
]
NEUTRAL_EXPORT_FEATURES = list(
    dict.fromkeys(
        [
            *NEUTRAL_BASE_FEATURES,
            *NEUTRAL_GOAL_FEATURES,
            *NEUTRAL_RESULT_FEATURES,
            *NEUTRAL_CANDIDATE_FEATURES,
        ]
    )
)
NEUTRAL_EXPORT_COLUMNS = [
    "split",
    *NEUTRAL_EXPORT_FEATURES,
    "team_a_goals",
    "team_b_goals",
    "total_goals",
    "result",
]


def _num_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fallback: float = 0.0) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(fallback)


def blend_result_probabilities(
    base_probabilities: np.ndarray,
    xg_probabilities: np.ndarray,
    *,
    weight: float = NEUTRAL_XG_RESULT_BLEND_WEIGHT,
) -> np.ndarray:
    """Blend calibrated base and xG-aware result probabilities."""
    blend_weight = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - blend_weight) * np.asarray(base_probabilities) + blend_weight * np.asarray(
        xg_probabilities
    )


def add_neutral_treated_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    elo_edge = np.tanh(_num_series(out, "elo_diff") / 350.0)
    fifa_edge = np.tanh(_num_series(out, "fifa_points_diff") / 250.0)
    rank_edge = np.tanh(
        (_num_series(out, "team_b_fifa_rank") - _num_series(out, "team_a_fifa_rank")) / 45.0
    )
    historical_fifa_points_diff = (
        _num_series(out, "historical_fifa_points_diff")
        if "historical_fifa_points_diff" in out
        else _num_series(out, "fifa_points_diff")
    )
    team_a_historical_fifa_rank = (
        _num_series(out, "team_a_historical_fifa_rank")
        if "team_a_historical_fifa_rank" in out
        else _num_series(out, "team_a_fifa_rank")
    )
    team_b_historical_fifa_rank = (
        _num_series(out, "team_b_historical_fifa_rank")
        if "team_b_historical_fifa_rank" in out
        else _num_series(out, "team_b_fifa_rank")
    )
    team_a_historical_fifa_observed = (
        _num_series(out, "team_a_historical_fifa_observed").clip(0.0, 1.0)
        if "team_a_historical_fifa_observed" in out
        else pd.Series(1.0, index=out.index)
    )
    team_b_historical_fifa_observed = (
        _num_series(out, "team_b_historical_fifa_observed").clip(0.0, 1.0)
        if "team_b_historical_fifa_observed" in out
        else pd.Series(1.0, index=out.index)
    )
    historical_fifa_pair_observed = (
        team_a_historical_fifa_observed * team_b_historical_fifa_observed
    )
    historical_fifa_edge = historical_fifa_pair_observed * np.tanh(
        historical_fifa_points_diff / 250.0
    )
    historical_rank_edge = historical_fifa_pair_observed * np.tanh(
        (team_b_historical_fifa_rank - team_a_historical_fifa_rank) / 45.0
    )
    fifa_anchor_edge = 0.62 * np.tanh(_num_series(out, "fifa_points_diff") / 190.0) + 0.38 * np.tanh(
        (_num_series(out, "team_b_fifa_rank") - _num_series(out, "team_a_fifa_rank")) / 38.0
    )
    historical_fifa_anchor_edge = historical_fifa_pair_observed * (
        0.62 * np.tanh(historical_fifa_points_diff / 190.0)
        + 0.38
        * np.tanh(
            (team_b_historical_fifa_rank - team_a_historical_fifa_rank) / 38.0
        )
    )
    live_fifa_points_diff = (
        _num_series(out, "live_fifa_points_diff")
        if "live_fifa_points_diff" in out
        else historical_fifa_points_diff
    )
    live_fifa_anchor_edge = historical_fifa_pair_observed * (
        0.62 * np.tanh(live_fifa_points_diff / 190.0)
        + 0.38
        * np.tanh(
            (team_b_historical_fifa_rank - team_a_historical_fifa_rank) / 38.0
        )
    )
    confed_edge = np.tanh(_num_series(out, "confederation_strength_diff") / 0.8)
    out["rating_consensus_edge"] = (
        0.42 * elo_edge + 0.38 * fifa_edge + 0.12 * rank_edge + 0.08 * confed_edge
    )
    out["fifa_anchor_edge"] = fifa_anchor_edge
    out["historical_rating_consensus_edge"] = (
        0.42 * elo_edge + 0.38 * historical_fifa_edge + 0.12 * historical_rank_edge + 0.08 * confed_edge
    )
    out["historical_fifa_anchor_edge"] = historical_fifa_anchor_edge
    out["live_fifa_anchor_edge"] = live_fifa_anchor_edge

    attack_edge = _num_series(out, "team_a_attack_vs_b_defense_avg") - _num_series(
        out, "team_b_attack_vs_a_defense_avg"
    )
    recent_attack_edge = _num_series(out, "recent6_team_a_attack_vs_b_defense_avg") - _num_series(
        out, "recent6_team_b_attack_vs_a_defense_avg"
    )
    out["threat_edge"] = 0.58 * attack_edge + 0.42 * recent_attack_edge
    out["goal_balance_edge"] = 0.50 * (
        _num_series(out, "goals_for_diff") - _num_series(out, "goals_against_diff")
    ) + 0.50 * (
        _num_series(out, "recent6_goals_for_diff") - _num_series(out, "recent6_goals_against_diff")
    )
    # Plain recent points are the active form signal and also supply the
    # contextual form term used for draw parity. The legacy opponent-adjusted
    # value stays exported only for ablation diagnostics.
    out["recent_points_form_edge"] = _num_series(out, "recent6_points_diff")
    out["opponent_adjusted_quality_form_edge"] = _num_series(
        out,
        "recent6_quality_result_points_diff",
    )
    # Production uses an interpretable, provider-independent recent-points
    # signal.  The opponent-adjusted version depends on the project's custom
    # Elo and remains diagnostic only.
    out["quality_form_edge"] = out["recent_points_form_edge"]
    out["historical_rating_threat_edge"] = 0.52 * out[
        "historical_rating_consensus_edge"
    ] + 0.48 * np.tanh(out["threat_edge"] / 1.25)
    out["historical_rating_guardrail_edge"] = (
        out["historical_fifa_anchor_edge"]
        - out["historical_rating_threat_edge"]
    )
    # Keep one model-facing strength signal: FIFA SUM live points available
    # immediately before the match. Rank is determined by those same points,
    # and the project's separate Elo would duplicate another Elo-style update.
    out["rating_threat_edge"] = historical_fifa_pair_observed * np.tanh(
        live_fifa_points_diff / 190.0
    )
    out["rating_guardrail_edge"] = 0.0
    out["rating_drift_edge"] = out["live_fifa_anchor_edge"] - out[
        "historical_fifa_anchor_edge"
    ]
    out["rating_drift_abs"] = out["rating_drift_edge"].abs()
    out["rating_drift_pressure_edge"] = out["rating_drift_edge"] * (
        1.0 + out["rating_drift_abs"]
    )
    out["rating_drift_signed_magnitude_edge"] = np.sign(out["rating_drift_edge"]) * np.sqrt(
        out["rating_drift_abs"]
    )
    out["drift_tempered_guardrail_edge_025"] = out["rating_guardrail_edge"] - 0.25 * out[
        "rating_drift_edge"
    ]
    out["drift_tempered_guardrail_edge_050"] = out["rating_guardrail_edge"] - 0.50 * out[
        "rating_drift_edge"
    ]
    physical_coverage = _num_series(out, "recent6_physical_coverage").clip(lower=0.0, upper=1.0)
    fouls_volume = np.tanh((_num_series(out, "recent6_fouls_sum") - 24.0) / 10.0)
    cards_volume = np.tanh((_num_series(out, "recent6_yellow_cards_sum") - 4.0) / 2.5)
    physical_imbalance = np.tanh(
        _num_series(out, "recent6_fouls_diff").abs() / 8.0
        + _num_series(out, "recent6_yellow_cards_diff").abs() / 2.0
    )
    out["physical_disruption_index"] = physical_coverage * (
        0.55 * fouls_volume + 0.30 * cards_volume + 0.15 * physical_imbalance
    )
    control_quality_edge = (
        0.40 * out["rating_threat_edge"]
        + 0.30 * np.tanh(out["threat_edge"] / 1.25)
        + 0.20 * np.tanh(out["quality_form_edge"] / 1.8)
        + 0.10 * np.tanh(out["goal_balance_edge"] / 1.5)
    )
    parity_control_index = 1.0 - control_quality_edge.abs().clip(upper=1.0)
    tempo_index = _num_series(out, "tempo_index")
    raw_total_goals = np.expm1(tempo_index).clip(lower=0.0)
    tempo_coverage = (
        _num_series(out, "team_a_train_matches").clip(
            lower=0.0,
            upper=RECENT_FORM_WINDOW,
        )
        + _num_series(out, "team_b_train_matches").clip(
            lower=0.0,
            upper=RECENT_FORM_WINDOW,
        )
    ) / (2.0 * RECENT_FORM_WINDOW)
    reliable_total_goals = (
        tempo_coverage * raw_total_goals
        + (1.0 - tempo_coverage) * DRAW_PRESSURE_TOTAL_GOALS_PRIOR
    )
    reliable_tempo_index = np.log1p(reliable_total_goals)
    out["draw_pressure_index"] = parity_control_index * (
        1.0 / (1.0 + reliable_tempo_index)
    )

    sample_edge = np.log1p(_num_series(out, "team_a_train_matches")) - np.log1p(
        _num_series(out, "team_b_train_matches")
    )
    recent_sample_edge = np.log1p(_num_series(out, "team_a_recent6_matches")) - np.log1p(
        _num_series(out, "team_b_recent6_matches")
    )
    sample_reliability_edge = sample_edge + 0.35 * recent_sample_edge
    out["readiness_reliability_edge"] = sample_reliability_edge + 0.30 * np.tanh(
        _num_series(out, "rest_days_diff") / 5.0
    )

    a_ppm = _safe_ratio(_num_series(out, "team_a_group_points_pre"), _num_series(out, "team_a_group_matches_pre"))
    b_ppm = _safe_ratio(_num_series(out, "team_b_group_points_pre"), _num_series(out, "team_b_group_matches_pre"))
    a_group = (
        0.55 * a_ppm
        + 0.20 * _num_series(out, "team_a_group_goal_diff_pre")
        + 0.08 * _num_series(out, "team_a_group_goals_for_pre")
        - 0.08 * _num_series(out, "team_a_group_goals_against_pre")
        - 0.16 * _num_series(out, "team_a_group_position_pre")
    )
    b_group = (
        0.55 * b_ppm
        + 0.20 * _num_series(out, "team_b_group_goal_diff_pre")
        + 0.08 * _num_series(out, "team_b_group_goals_for_pre")
        - 0.08 * _num_series(out, "team_b_group_goals_against_pre")
        - 0.16 * _num_series(out, "team_b_group_position_pre")
    )
    group_weight = (
        _num_series(out, "team_a_group_matches_pre") + _num_series(out, "team_b_group_matches_pre")
    ).clip(0, 6) / 6.0
    out["group_state_edge"] = (a_group - b_group) * group_weight
    a_group_position_pressure = np.tanh((_num_series(out, "team_a_group_position_pre") - 2.0) / 2.0)
    b_group_position_pressure = np.tanh((_num_series(out, "team_b_group_position_pre") - 2.0) / 2.0)
    a_need = (
        (3.0 - a_ppm).clip(lower=0.0, upper=3.0) / 3.0
        + 0.25 * a_group_position_pressure
        - 0.12 * np.tanh(_num_series(out, "team_a_group_goal_diff_pre") / 3.0)
    )
    b_need = (
        (3.0 - b_ppm).clip(lower=0.0, upper=3.0) / 3.0
        + 0.25 * b_group_position_pressure
        - 0.12 * np.tanh(_num_series(out, "team_b_group_goal_diff_pre") / 3.0)
    )
    a_need = a_need.clip(lower=0.0, upper=1.4)
    b_need = b_need.clip(lower=0.0, upper=1.4)
    out["group_pressure_index"] = ((a_need + b_need) / 2.0) * group_weight
    out["group_pressure_edge"] = (a_need - b_need) * group_weight

    weights = np.array([1.0, 0.82, 0.67, 0.55, 0.45, 0.37])
    a_wc_score = pd.Series(0.0, index=out.index)
    b_wc_score = pd.Series(0.0, index=out.index)
    for index, weight in enumerate(weights, start=1):
        a_wc_score += weight * (
            3.0 * _num_series(out, f"team_a_worldcup_last6_{index}_win")
            + _num_series(out, f"team_a_worldcup_last6_{index}_draw")
        )
        b_wc_score += weight * (
            3.0 * _num_series(out, f"team_b_worldcup_last6_{index}_win")
            + _num_series(out, f"team_b_worldcup_last6_{index}_draw")
        )
    out["worldcup_points_memory_edge"] = (a_wc_score - b_wc_score) / (3.0 * weights.sum())
    out["late85_points_swing_edge"] = _num_series(out, "team_a_late85_points_swing") - _num_series(
        out,
        "team_b_late85_points_swing",
    )
    out["score_state_value_edge"] = _num_series(out, "team_a_score_state_value") - _num_series(
        out,
        "team_b_score_state_value",
    )
    out["score_control_value_edge"] = _num_series(out, "team_a_score_control_value") - _num_series(
        out,
        "team_b_score_control_value",
    )
    out["scoring_quickness_edge"] = _num_series(out, "team_a_scoring_quickness") - _num_series(
        out,
        "team_b_scoring_quickness",
    )
    out["score_timing_edge"] = (
        0.52 * out["score_state_value_edge"]
        + 0.28 * out["score_control_value_edge"]
        + 0.20 * out["scoring_quickness_edge"]
    )
    out["score_control_quality_edge"] = _num_series(
        out,
        "team_a_score_control_quality",
    ) - _num_series(out, "team_b_score_control_quality")
    out["narrow_lead_hold_edge"] = _num_series(out, "team_a_narrow_lead_hold") - _num_series(
        out,
        "team_b_narrow_lead_hold",
    )
    out["comfortable_lead_edge"] = _num_series(out, "team_a_comfortable_lead") - _num_series(
        out,
        "team_b_comfortable_lead",
    )
    out["game_state_friction_edge"] = _num_series(out, "team_b_game_state_friction") - _num_series(
        out,
        "team_a_game_state_friction",
    )
    out["score_control_refined_edge"] = (
        0.54 * out["score_control_quality_edge"]
        + 0.22 * out["narrow_lead_hold_edge"]
        + 0.14 * out["comfortable_lead_edge"]
        + 0.10 * out["game_state_friction_edge"]
    )
    out["state_change_swing_edge"] = _num_series(
        out,
        "team_a_state_change_swing",
    ) - _num_series(out, "team_b_state_change_swing")
    out["early_state_change_swing_edge"] = _num_series(
        out,
        "team_a_early_state_change_swing",
    ) - _num_series(out, "team_b_early_state_change_swing")
    out["quality_score_control_value_edge"] = _num_series(
        out,
        "team_a_quality_score_control_value",
    ) - _num_series(out, "team_b_quality_score_control_value")
    out["quality_state_change_swing_edge"] = _num_series(
        out,
        "team_a_quality_state_change_swing",
    ) - _num_series(out, "team_b_quality_state_change_swing")
    out["quality_early_state_change_swing_edge"] = _num_series(
        out,
        "team_a_quality_early_state_change_swing",
    ) - _num_series(out, "team_b_quality_early_state_change_swing")
    out["quality_score_control_swing_edge"] = (
        0.65 * out["quality_score_control_value_edge"]
        + 0.35 * np.tanh(out["quality_early_state_change_swing_edge"] / 1.50)
    )

    tactical_pressure_edge = (
        1.00 * _num_series(out, "recent6_shots_on_goal_diff")
        + 0.35 * _num_series(out, "recent6_shots_inside_box_diff")
        + 0.12 * _num_series(out, "recent6_total_shots_diff")
        + 0.25 * _num_series(out, "recent6_corner_kicks_diff")
    )
    tactical_directness_edge = (
        0.30 * _num_series(out, "recent6_total_shots_diff")
        + 0.45 * _num_series(out, "recent6_offsides_diff")
        + 0.25 * _num_series(out, "recent6_shots_outside_box_diff")
        - 0.0025 * _num_series(out, "recent6_total_passes_diff")
        - 0.025 * _num_series(out, "recent6_ball_possession_pct_diff")
    )
    tactical_control_edge = (
        0.10 * _num_series(out, "recent6_ball_possession_pct_diff")
        + 0.006 * _num_series(out, "recent6_passes_accurate_diff")
        + 0.05 * _num_series(out, "recent6_passes_pct_diff")
    )
    tactical_stress_edge = (
        0.80 * _num_series(out, "recent6_goalkeeper_saves_diff")
        + 0.20 * _num_series(out, "recent6_fouls_diff")
        + 0.65 * _num_series(out, "recent6_yellow_cards_diff")
    )
    out["tactical_profile_edge"] = (
        0.40 * np.tanh(tactical_pressure_edge / 4.0)
        + 0.25 * np.tanh(tactical_directness_edge / 4.0)
        + 0.20 * np.tanh(tactical_control_edge / 6.0)
        - 0.15 * np.tanh(tactical_stress_edge / 4.0)
    )
    a_late_net = _num_series(out, "team_a_late85_points_swing")
    b_late_net = _num_series(out, "team_b_late85_points_swing")
    a_late_pressure = a_late_net.clip(lower=0.0) + 0.35 * _num_series(
        out,
        "team_a_quality_early_state_change_swing",
    ).clip(lower=0.0)
    b_late_pressure = b_late_net.clip(lower=0.0) + 0.35 * _num_series(
        out,
        "team_b_quality_early_state_change_swing",
    ).clip(lower=0.0)
    a_late_vulnerability = (-a_late_net).clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_a_game_state_friction",
    ).clip(lower=0.0)
    b_late_vulnerability = (-b_late_net).clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_b_game_state_friction",
    ).clip(lower=0.0)
    out["late_resilience_matchup_edge"] = np.tanh(
        (a_late_pressure * b_late_vulnerability - b_late_pressure * a_late_vulnerability)
        / 1.15
    )

    a_coverage = _num_series(out, "team_a_tactical_detail_coverage").clip(lower=0.0, upper=1.0)
    b_coverage = _num_series(out, "team_b_tactical_detail_coverage").clip(lower=0.0, upper=1.0)
    tactical_matchup_coverage = pd.concat([a_coverage, b_coverage], axis=1).min(axis=1)

    def side_tactical(prefix: str) -> dict[str, pd.Series]:
        possession = _num_series(out, f"team_{prefix}_recent6_ball_possession_pct")
        shots_on_goal = _num_series(out, f"team_{prefix}_recent6_shots_on_goal")
        shots_off_goal = _num_series(out, f"team_{prefix}_recent6_shots_off_goal")
        shots_inside = _num_series(out, f"team_{prefix}_recent6_shots_inside_box")
        shots_outside = _num_series(out, f"team_{prefix}_recent6_shots_outside_box")
        total_shots = _num_series(out, f"team_{prefix}_recent6_total_shots")
        blocked_shots = _num_series(out, f"team_{prefix}_recent6_blocked_shots")
        corners = _num_series(out, f"team_{prefix}_recent6_corner_kicks")
        offsides = _num_series(out, f"team_{prefix}_recent6_offsides")
        saves = _num_series(out, f"team_{prefix}_recent6_goalkeeper_saves")
        total_passes = _num_series(out, f"team_{prefix}_recent6_total_passes")
        passes_pct = _num_series(out, f"team_{prefix}_recent6_passes_pct")
        goals_against = _num_series(out, f"team_{prefix}_recent6_goals_against_avg")
        shot_volume = np.tanh(total_shots / 13.0)
        sog_volume = np.tanh(shots_on_goal / 4.0)
        box_volume = np.tanh(shots_inside / 7.0)
        corner_volume = np.tanh(corners / 5.0)
        pass_volume = np.tanh(total_passes / 500.0)
        blocked_volume = np.tanh(blocked_shots / 4.0)
        pressure = np.tanh(
            (shots_on_goal + 0.45 * shots_inside + 0.12 * total_shots + 0.18 * corners)
            / 7.0
        )
        high_possession = ((possession - 50.0) / 35.0).clip(lower=0.0, upper=1.0)
        low_possession = ((50.0 - possession) / 35.0).clip(lower=0.0, upper=1.0)
        chance_creation = (
            0.34 * sog_volume
            + 0.24 * box_volume
            + 0.18 * shot_volume
            + 0.16 * corner_volume
            + 0.08 * blocked_volume
        )
        direct_attack = np.tanh(
            (
                0.85 * shots_on_goal
                + 0.38 * shots_inside
                + 0.24 * shots_outside
                + 0.18 * shots_off_goal
                + 0.25 * offsides
                - 0.0015 * total_passes
                - 0.012 * possession
            )
            / 5.0
        ).clip(lower=0.0, upper=1.0)
        finishing_signal = _num_series(
            out,
            f"team_{prefix}_clinical_finishing",
        ).clip(lower=-1.0, upper=1.0)
        finishing_positive = finishing_signal.clip(lower=0.0, upper=1.0)
        finishing_negative = (-finishing_signal).clip(lower=0.0, upper=1.0)
        save_stress = np.tanh(saves / 3.20)
        conceded_stress = np.tanh(goals_against / 1.80)
        low_block_profile = (
            0.32 * low_possession
            + 0.20 * (1.0 - shot_volume)
            + 0.18 * (1.0 - sog_volume)
            + 0.14 * (1.0 - corner_volume)
            + 0.10 * (1.0 - pass_volume)
            + 0.06 * (1.0 - np.tanh(passes_pct / 85.0))
        ).clip(lower=0.0, upper=1.0)
        attacking_posture = (
            0.26 * high_possession
            + 0.32 * chance_creation
            + 0.18 * corner_volume
            + 0.14 * pass_volume
            + 0.10 * np.tanh((passes_pct - 72.0).clip(lower=0.0) / 20.0)
        ).clip(lower=0.0, upper=1.0)
        sterile_control_risk = high_possession * (1.0 - chance_creation).clip(
            lower=0.0,
            upper=1.0,
        ) * (0.65 + 0.35 * finishing_negative)
        low_block_breaking_strength = (
            (0.68 * chance_creation + 0.32 * finishing_positive)
            * (0.45 + 0.55 * high_possession)
            - 0.45 * sterile_control_risk
        ).clip(lower=-1.0, upper=1.0)
        space_exploitation_strength = (
            0.42 * chance_creation + 0.28 * direct_attack + 0.30 * finishing_positive
        ) * (1.0 - 0.35 * high_possession)
        low_block_resistance = low_possession * (
            0.55 * (1.0 - save_stress) + 0.45 * (1.0 - conceded_stress)
        )
        low_block_vulnerability = low_possession * (
            0.55 * save_stress + 0.45 * conceded_stress
        )
        sterile_pressure = high_possession * pressure * (
            0.65 + 0.35 * finishing_negative
        ) * (1.0 - 0.35 * finishing_positive)
        lethal_pressure = pressure * (0.65 + 0.35 * finishing_positive) * (
            1.0 - 0.25 * finishing_negative
        )
        return {
            "low_block_resistance": low_block_resistance,
            "low_block_vulnerability": low_block_vulnerability,
            "sterile_pressure": sterile_pressure,
            "lethal_pressure": lethal_pressure,
            "low_block_profile": low_block_profile,
            "attacking_posture": attacking_posture,
            "low_block_breaking_strength": low_block_breaking_strength,
            "space_exploitation_strength": space_exploitation_strength,
        }

    a_tactical = side_tactical("a")
    b_tactical = side_tactical("b")
    out["low_block_sterile_matchup_edge"] = tactical_matchup_coverage * np.tanh(
        1.60
        * (
            a_tactical["low_block_resistance"] * b_tactical["sterile_pressure"]
            - b_tactical["low_block_resistance"] * a_tactical["sterile_pressure"]
        )
    )
    out["pressure_block_matchup_edge"] = tactical_matchup_coverage * np.tanh(
        1.35
        * (
            a_tactical["lethal_pressure"] * b_tactical["low_block_vulnerability"]
            - b_tactical["lethal_pressure"] * a_tactical["low_block_vulnerability"]
        )
    )
    out["strategic_compatibility_edge"] = (
        0.34 * out["late_resilience_matchup_edge"]
        + 0.38 * out["low_block_sterile_matchup_edge"]
        + 0.28 * out["pressure_block_matchup_edge"]
    )
    score_matchup_coverage = (
        _num_series(out, "team_a_score_timing_coverage").clip(
            lower=0.0,
            upper=1.0,
        )
        + _num_series(out, "team_b_score_timing_coverage").clip(
            lower=0.0,
            upper=1.0,
        )
    ) / 2.0
    out["match_script_low_block_edge"] = tactical_matchup_coverage * np.tanh(
        1.45
        * (
            a_tactical["low_block_breaking_strength"] * b_tactical["low_block_profile"]
            - b_tactical["low_block_breaking_strength"] * a_tactical["low_block_profile"]
        )
    )
    out["match_script_space_edge"] = tactical_matchup_coverage * np.tanh(
        1.35
        * (
            a_tactical["space_exploitation_strength"] * b_tactical["attacking_posture"]
            - b_tactical["space_exploitation_strength"] * a_tactical["attacking_posture"]
        )
    )
    a_early_attack = _num_series(out, "team_a_scoring_quickness").clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_a_quality_early_state_change_swing",
    ).clip(lower=0.0)
    b_early_attack = _num_series(out, "team_b_scoring_quickness").clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_b_quality_early_state_change_swing",
    ).clip(lower=0.0)
    a_early_vulnerability = (-_num_series(out, "team_a_quality_early_state_change_swing")).clip(
        lower=0.0
    ) + 0.35 * _num_series(out, "team_a_game_state_friction").clip(lower=0.0)
    b_early_vulnerability = (-_num_series(out, "team_b_quality_early_state_change_swing")).clip(
        lower=0.0
    ) + 0.35 * _num_series(out, "team_b_game_state_friction").clip(lower=0.0)
    timing_raw = 0.55 * (
        a_early_attack * b_early_vulnerability - b_early_attack * a_early_vulnerability
    ) + 0.45 * (
        a_late_pressure * b_late_vulnerability - b_late_pressure * a_late_vulnerability
    )
    out["match_script_timing_edge"] = score_matchup_coverage * np.tanh(timing_raw / 1.20)
    a_comeback = _num_series(out, "team_a_state_change_swing").clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_a_quality_state_change_swing",
    ).clip(lower=0.0)
    b_comeback = _num_series(out, "team_b_state_change_swing").clip(lower=0.0) + 0.45 * _num_series(
        out,
        "team_b_quality_state_change_swing",
    ).clip(lower=0.0)
    a_lead_leak = (-_num_series(out, "team_a_state_change_swing")).clip(lower=0.0) + 0.35 * _num_series(
        out,
        "team_a_game_state_friction",
    ).clip(lower=0.0) + 0.25 * (-_num_series(out, "team_a_narrow_lead_hold")).clip(lower=0.0)
    b_lead_leak = (-_num_series(out, "team_b_state_change_swing")).clip(lower=0.0) + 0.35 * _num_series(
        out,
        "team_b_game_state_friction",
    ).clip(lower=0.0) + 0.25 * (-_num_series(out, "team_b_narrow_lead_hold")).clip(lower=0.0)
    a_lead_protection = _num_series(out, "team_a_score_control_quality").clip(
        lower=0.0
    ) + 0.35 * _num_series(out, "team_a_narrow_lead_hold").clip(lower=0.0) + 0.20 * _num_series(
        out,
        "team_a_comfortable_lead",
    ).clip(lower=0.0)
    b_lead_protection = _num_series(out, "team_b_score_control_quality").clip(
        lower=0.0
    ) + 0.35 * _num_series(out, "team_b_narrow_lead_hold").clip(lower=0.0) + 0.20 * _num_series(
        out,
        "team_b_comfortable_lead",
    ).clip(lower=0.0)
    state_raw = (
        a_comeback * b_lead_leak
        - b_comeback * a_lead_leak
        + 0.45 * (a_lead_protection - b_lead_protection)
    )
    out["match_script_state_edge"] = score_matchup_coverage * np.tanh(state_raw / 1.35)
    out["match_script_data_coverage"] = (
        0.52 * tactical_matchup_coverage + 0.48 * score_matchup_coverage
    ).clip(lower=0.0, upper=1.0)
    a_recent_opp_elo = _num_series(out, "team_a_recent6_opponent_elo_avg")
    b_recent_opp_elo = _num_series(out, "team_b_recent6_opponent_elo_avg")
    a_elo = _num_series(out, "team_a_elo_pre")
    b_elo = _num_series(out, "team_b_elo_pre")
    a_quality_gap = np.tanh((a_recent_opp_elo - a_elo) / 220.0)
    b_quality_gap = np.tanh((b_recent_opp_elo - b_elo) / 220.0)
    a_absolute_quality = np.tanh((a_recent_opp_elo - 1500.0) / 260.0)
    b_absolute_quality = np.tanh((b_recent_opp_elo - 1500.0) / 260.0)
    a_quality_context = (0.68 * a_quality_gap + 0.32 * a_absolute_quality).clip(
        lower=-1.0,
        upper=1.0,
    )
    b_quality_context = (0.68 * b_quality_gap + 0.32 * b_absolute_quality).clip(
        lower=-1.0,
        upper=1.0,
    )
    a_quality_multiplier = (1.0 + 0.26 * a_quality_context).clip(lower=0.72, upper=1.28)
    b_quality_multiplier = (1.0 + 0.26 * b_quality_context).clip(lower=0.72, upper=1.28)
    out["match_script_quality_context_edge"] = (
        out["match_script_data_coverage"] * (a_quality_context - b_quality_context)
    )
    out["match_script_quality_low_block_edge"] = tactical_matchup_coverage * np.tanh(
        1.45
        * (
            a_quality_multiplier
            * a_tactical["low_block_breaking_strength"]
            * b_tactical["low_block_profile"]
            - b_quality_multiplier
            * b_tactical["low_block_breaking_strength"]
            * a_tactical["low_block_profile"]
        )
    )
    out["match_script_quality_space_edge"] = tactical_matchup_coverage * np.tanh(
        1.35
        * (
            a_quality_multiplier
            * a_tactical["space_exploitation_strength"]
            * b_tactical["attacking_posture"]
            - b_quality_multiplier
            * b_tactical["space_exploitation_strength"]
            * a_tactical["attacking_posture"]
        )
    )
    quality_timing_raw = 0.55 * (
        a_quality_multiplier * a_early_attack * b_early_vulnerability
        - b_quality_multiplier * b_early_attack * a_early_vulnerability
    ) + 0.45 * (
        a_quality_multiplier * a_late_pressure * b_late_vulnerability
        - b_quality_multiplier * b_late_pressure * a_late_vulnerability
    )
    out["match_script_quality_timing_edge"] = score_matchup_coverage * np.tanh(
        quality_timing_raw / 1.20
    )
    quality_state_raw = (
        a_quality_multiplier * a_comeback * b_lead_leak
        - b_quality_multiplier * b_comeback * a_lead_leak
        + 0.45 * (a_quality_multiplier * a_lead_protection - b_quality_multiplier * b_lead_protection)
    )
    out["match_script_quality_state_edge"] = score_matchup_coverage * np.tanh(
        quality_state_raw / 1.35
    )
    out["match_script_quality_adjusted_edge"] = (
        0.32 * out["match_script_quality_low_block_edge"]
        + 0.24 * out["match_script_quality_space_edge"]
        + 0.18 * out["match_script_quality_timing_edge"]
        + 0.18 * out["match_script_quality_state_edge"]
        + 0.08 * out["match_script_quality_context_edge"]
    ).clip(lower=-1.0, upper=1.0)
    # Keep the model-facing signal bilateral: adjust each team's script by the
    # quality of the opponents it proved that script against, then subtract.
    # Multiplying an already-signed edge by another signed edge can strengthen
    # the wrong team when their directions conflict.
    out["match_script_compatibility_edge"] = out[
        "match_script_quality_adjusted_edge"
    ]
    aligned_clinical_coverage = pd.concat(
        [
            _num_series(out, "team_a_clinical_coverage").clip(0.0, 1.0),
            _num_series(out, "team_b_clinical_coverage").clip(0.0, 1.0),
            _num_series(out, "team_a_low_block_coverage_aligned").clip(0.0, 1.0),
            _num_series(out, "team_b_low_block_coverage_aligned").clip(0.0, 1.0),
        ],
        axis=1,
    ).min(axis=1)
    out["clinical_low_block_matchup_edge"] = aligned_clinical_coverage * (
        _num_series(out, "team_a_clinical_finishing")
        * _num_series(out, "team_b_low_block_profile_aligned")
        - _num_series(out, "team_b_clinical_finishing")
        * _num_series(out, "team_a_low_block_profile_aligned")
    )
    a_leader_coverage = _num_series(
        out,
        "team_a_espn_leader_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    b_leader_coverage = _num_series(
        out,
        "team_b_espn_leader_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    leader_pair_coverage = pd.concat([a_leader_coverage, b_leader_coverage], axis=1).min(axis=1)
    out["espn_leader_detail_coverage_pair"] = leader_pair_coverage

    def side_worldcup_detail(prefix: str) -> dict[str, pd.Series]:
        top_xg = _num_series(out, f"team_{prefix}_recent6_espn_top_xg")
        keeper_xgc = _num_series(out, f"team_{prefix}_recent6_espn_keeper_xg_conceded")
        top_duels = _num_series(out, f"team_{prefix}_recent6_espn_top_duels_won")
        big_created = _num_series(out, f"team_{prefix}_recent6_espn_top_big_chances_created")
        big_missed = _num_series(out, f"team_{prefix}_recent6_espn_top_big_chances_missed")
        shots_on_goal = _num_series(out, f"team_{prefix}_recent6_shots_on_goal")
        passes_accurate = _num_series(out, f"team_{prefix}_recent6_passes_accurate")
        passes_pct = _num_series(out, f"team_{prefix}_recent6_passes_pct")
        possession = _num_series(out, f"team_{prefix}_recent6_ball_possession_pct")
        saves = _num_series(out, f"team_{prefix}_recent6_goalkeeper_saves")
        fouls = _num_series(out, f"team_{prefix}_recent6_fouls")

        chance_balance = (big_created - big_missed).clip(lower=0.0)
        chance_quality = (
            0.50 * np.tanh(top_xg / 1.15)
            + 0.22 * np.tanh(chance_balance / 1.40)
            + 0.28 * np.tanh(shots_on_goal / 4.20)
        )
        control = (
            0.45 * np.tanh(passes_accurate / 560.0)
            + 0.35 * np.tanh((passes_pct - 74.0).clip(lower=0.0) / 18.0)
            + 0.20 * np.tanh(possession / 58.0)
        )
        duel_grip = np.tanh(top_duels / 8.0)
        defensive_stress = (
            0.44 * np.tanh(saves / 3.75)
            + 0.36 * np.tanh(keeper_xgc / 1.45)
            + 0.20 * np.tanh(fouls / 16.0)
        )
        return {
            "chance_quality": chance_quality,
            "control": control,
            "duel_grip": duel_grip,
            "defensive_stress": defensive_stress,
        }

    a_worldcup_detail = side_worldcup_detail("a")
    b_worldcup_detail = side_worldcup_detail("b")
    out["worldcup_chance_quality_edge"] = leader_pair_coverage * (
        a_worldcup_detail["chance_quality"] - b_worldcup_detail["chance_quality"]
    )
    out["worldcup_detail_flow_edge"] = leader_pair_coverage * (
        0.58
        * (a_worldcup_detail["chance_quality"] - b_worldcup_detail["chance_quality"])
        + 0.18 * (a_worldcup_detail["control"] - b_worldcup_detail["control"])
        + 0.14 * (a_worldcup_detail["duel_grip"] - b_worldcup_detail["duel_grip"])
        + 0.10
        * (
            b_worldcup_detail["defensive_stress"]
            - a_worldcup_detail["defensive_stress"]
        )
    ).clip(lower=-1.0, upper=1.0)
    a_fotmob_coverage = _num_series(
        out,
        "team_a_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    b_fotmob_coverage = _num_series(
        out,
        "team_b_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    fotmob_pair_coverage = pd.concat(
        [a_fotmob_coverage, b_fotmob_coverage],
        axis=1,
    ).min(axis=1)
    out["worldcup_fotmob_chance_coverage_pair"] = fotmob_pair_coverage

    def side_fotmob(prefix: str) -> dict[str, pd.Series]:
        xg = _num_series(out, f"team_{prefix}_recent6_fotmob_expected_goals")
        xgc = _num_series(out, f"team_{prefix}_recent6_fotmob_expected_goals_conceded")
        xgot = _num_series(out, f"team_{prefix}_recent6_fotmob_expected_goals_on_target")
        big = _num_series(out, f"team_{prefix}_recent6_fotmob_big_chances")
        big_conceded = _num_series(out, f"team_{prefix}_recent6_fotmob_big_chances_conceded")
        big_missed = _num_series(out, f"team_{prefix}_recent6_fotmob_big_chances_missed")
        shots = _num_series(out, f"team_{prefix}_recent6_fotmob_total_shots")
        sot = _num_series(out, f"team_{prefix}_recent6_fotmob_shots_on_target")
        touches_box = _num_series(out, f"team_{prefix}_recent6_fotmob_touches_opp_box")
        creation = (
            0.46 * np.tanh(xg / 1.65)
            + 0.22 * np.tanh(big / 3.20)
            + 0.16 * np.tanh(xgot / 1.55)
            + 0.10 * np.tanh(sot / 5.20)
            + 0.06 * np.tanh(touches_box / 26.0)
        )
        waste = 0.72 * np.tanh(big_missed / 3.20) + 0.28 * np.tanh(
            (xg - xgot).clip(lower=0.0) / 1.15
        )
        resistance = (
            0.58 * (1.0 - np.tanh(xgc / 1.75))
            + 0.24 * (1.0 - np.tanh(big_conceded / 3.20))
            + 0.18 * (1.0 - np.tanh(shots / 16.0))
        )
        return {
            "xg_balance": xg - xgc,
            "creation": creation,
            "waste": waste,
            "resistance": resistance,
        }

    a_fotmob = side_fotmob("a")
    b_fotmob = side_fotmob("b")
    out["worldcup_fotmob_xg_balance_edge"] = fotmob_pair_coverage * np.tanh(
        (a_fotmob["xg_balance"] - b_fotmob["xg_balance"]) / 1.80
    )
    out["worldcup_fotmob_chance_pressure_edge"] = fotmob_pair_coverage * (
        0.50 * (a_fotmob["creation"] - b_fotmob["creation"])
        + 0.24 * (a_fotmob["resistance"] - b_fotmob["resistance"])
        + 0.26 * (b_fotmob["waste"] - a_fotmob["waste"])
    ).clip(lower=-1.0, upper=1.0)

    a_wc_fotmob_coverage = _num_series(
        out,
        "team_a_worldcup_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    b_wc_fotmob_coverage = _num_series(
        out,
        "team_b_worldcup_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    wc_fotmob_pair_coverage = pd.concat(
        [a_wc_fotmob_coverage, b_wc_fotmob_coverage],
        axis=1,
    ).min(axis=1)

    def side_worldcup_fotmob(
        prefix: str,
        history_prefix: str = "worldcup",
    ) -> dict[str, pd.Series]:
        column_prefix = f"team_{prefix}_{history_prefix}_recent6"
        xg = _num_series(out, f"{column_prefix}_fotmob_expected_goals")
        xgc = _num_series(out, f"{column_prefix}_fotmob_expected_goals_conceded")
        xgot = _num_series(out, f"{column_prefix}_fotmob_expected_goals_on_target")
        big = _num_series(out, f"{column_prefix}_fotmob_big_chances")
        big_conceded = _num_series(out, f"{column_prefix}_fotmob_big_chances_conceded")
        big_missed = _num_series(out, f"{column_prefix}_fotmob_big_chances_missed")
        shots = _num_series(out, f"{column_prefix}_fotmob_total_shots")
        shots_on_target = _num_series(out, f"{column_prefix}_fotmob_shots_on_target")
        touches_box = _num_series(out, f"{column_prefix}_fotmob_touches_opp_box")
        underlying = _num_series(out, f"{column_prefix}_fotmob_underlying_threat")
        finishing = _num_series(out, f"{column_prefix}_fotmob_finishing_signal")
        waste = _num_series(out, f"{column_prefix}_fotmob_waste_signal")
        low_possession_punch = _num_series(
            out,
            f"{column_prefix}_fotmob_low_possession_punch",
        )
        sterile_control = _num_series(
            out,
            f"{column_prefix}_fotmob_sterile_control_risk",
        )
        resistance = _num_series(
            out,
            f"{column_prefix}_fotmob_defensive_resistance",
        )
        chance_control = _num_series(
            out,
            f"{column_prefix}_fotmob_chance_control_signal",
        )
        unrewarded = _num_series(
            out,
            f"{column_prefix}_fotmob_unrewarded_pressure",
        )
        clinical = _num_series(
            out,
            f"{column_prefix}_fotmob_clinical_chance_signal",
        )
        xg_balance = xg - xgc
        shot_pressure = (
            0.34 * np.tanh(xg / 1.65)
            + 0.18 * np.tanh(xgot / 1.40)
            + 0.16 * np.tanh(big / 2.80)
            + 0.13 * np.tanh(shots / 14.0)
            + 0.11 * np.tanh(shots_on_target / 4.8)
            + 0.08 * np.tanh(touches_box / 24.0)
        )
        concession_control = (
            0.58 * (1.0 - np.tanh(xgc / 1.55))
            + 0.27 * (1.0 - np.tanh(big_conceded / 2.60))
            + 0.15 * np.tanh(xg_balance / 1.65)
        )
        controlled_dominance = (
            0.30 * chance_control
            + 0.24 * shot_pressure
            + 0.20 * concession_control
            + 0.12 * underlying
            + 0.08 * clinical
            - 0.09 * waste
            - 0.05 * sterile_control
            - 0.03 * np.tanh(big_missed / 2.70)
        ).clip(lower=-1.0, upper=1.0)
        return {
            "xg": xg,
            "xgc": xgc,
            "xgot": xgot,
            "big": big,
            "big_conceded": big_conceded,
            "big_missed": big_missed,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "touches_box": touches_box,
            "underlying": underlying,
            "finishing": finishing,
            "waste": waste,
            "low_possession_punch": low_possession_punch,
            "sterile_control": sterile_control,
            "resistance": resistance,
            "chance_control": chance_control,
            "unrewarded": unrewarded,
            "clinical": clinical,
            "xg_balance": xg_balance,
            "controlled_dominance": controlled_dominance,
        }

    a_wc_fotmob = side_worldcup_fotmob("a")
    b_wc_fotmob = side_worldcup_fotmob("b")
    low_block_pair_coverage = pd.concat(
        [
            wc_fotmob_pair_coverage,
            _num_series(out, "team_a_low_block_coverage_aligned").clip(0.0, 1.0),
            _num_series(out, "team_b_low_block_coverage_aligned").clip(0.0, 1.0),
        ],
        axis=1,
    ).min(axis=1)
    team_a_low_block = _num_series(out, "team_a_low_block_profile_aligned").clip(
        0.0,
        1.0,
    )
    team_b_low_block = _num_series(out, "team_b_low_block_profile_aligned").clip(
        0.0,
        1.0,
    )
    out["worldcup_fotmob_low_block_solution_edge"] = low_block_pair_coverage * (
        (
            a_wc_fotmob["underlying"]
            - 0.55 * a_wc_fotmob["sterile_control"]
            - 0.18 * a_wc_fotmob["waste"]
        )
        * team_b_low_block
        - (
            b_wc_fotmob["underlying"]
            - 0.55 * b_wc_fotmob["sterile_control"]
            - 0.18 * b_wc_fotmob["waste"]
        )
        * team_a_low_block
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_transition_punch_edge"] = wc_fotmob_pair_coverage * (
        a_wc_fotmob["low_possession_punch"] - b_wc_fotmob["low_possession_punch"]
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_unrewarded_pressure_edge"] = wc_fotmob_pair_coverage * (
        a_wc_fotmob["unrewarded"] - b_wc_fotmob["unrewarded"]
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_finishing_discipline_edge"] = wc_fotmob_pair_coverage * (
        a_wc_fotmob["clinical"]
        - b_wc_fotmob["clinical"]
        - 0.35 * (a_wc_fotmob["waste"] - b_wc_fotmob["waste"])
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_interpreted_edge"] = (
        wc_fotmob_pair_coverage
        * (
            0.36 * (a_wc_fotmob["chance_control"] - b_wc_fotmob["chance_control"])
            + 0.20
            * np.tanh(
                (a_wc_fotmob["xg_balance"] - b_wc_fotmob["xg_balance"]) / 1.8
            )
        )
        + 0.17 * out["worldcup_fotmob_low_block_solution_edge"]
        + 0.13 * out["worldcup_fotmob_transition_punch_edge"]
        + 0.09 * out["worldcup_fotmob_finishing_discipline_edge"]
        + 0.05 * out["worldcup_fotmob_unrewarded_pressure_edge"]
    ).clip(lower=-1.0, upper=1.0)

    a_current_wc_coverage = _num_series(
        out,
        "team_a_current_worldcup_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    b_current_wc_coverage = _num_series(
        out,
        "team_b_current_worldcup_recent6_fotmob_detail_coverage",
    ).clip(lower=0.0, upper=1.0)
    current_wc_pair_coverage = pd.concat(
        [a_current_wc_coverage, b_current_wc_coverage],
        axis=1,
    ).min(axis=1)
    a_current_wc = side_worldcup_fotmob("a", "current_worldcup")
    b_current_wc = side_worldcup_fotmob("b", "current_worldcup")
    historical_xg_matchup_a = 0.50 * (a_wc_fotmob["xg"] + b_wc_fotmob["xgc"])
    historical_xg_matchup_b = 0.50 * (b_wc_fotmob["xg"] + a_wc_fotmob["xgc"])
    current_xg_matchup_a = 0.50 * (a_current_wc["xg"] + b_current_wc["xgc"])
    current_xg_matchup_b = 0.50 * (b_current_wc["xg"] + a_current_wc["xgc"])
    xg_matchup_coverage = 0.55 * wc_fotmob_pair_coverage + 0.45 * current_wc_pair_coverage
    xg_matchup_denominator = xg_matchup_coverage.replace(0.0, np.nan)
    # Each side's xG is opponent-aware: what it creates plus what the rival
    # concedes. The wider World Cup history provides a stable baseline and the
    # current tournament supplies a controlled live update.
    out["worldcup_fotmob_xg_matchup_team_a"] = (
        (
            0.55 * wc_fotmob_pair_coverage * historical_xg_matchup_a
            + 0.45 * current_wc_pair_coverage * current_xg_matchup_a
        )
        / xg_matchup_denominator
    ).fillna(0.0)
    out["worldcup_fotmob_xg_matchup_team_b"] = (
        (
            0.55 * wc_fotmob_pair_coverage * historical_xg_matchup_b
            + 0.45 * current_wc_pair_coverage * current_xg_matchup_b
        )
        / xg_matchup_denominator
    ).fillna(0.0)
    current_low_block_pair_coverage = pd.concat(
        [
            current_wc_pair_coverage,
            _num_series(out, "team_a_low_block_coverage_aligned").clip(0.0, 1.0),
            _num_series(out, "team_b_low_block_coverage_aligned").clip(0.0, 1.0),
        ],
        axis=1,
    ).min(axis=1)
    out["worldcup_fotmob_current_chance_pressure_edge"] = current_wc_pair_coverage * (
        0.46 * (a_current_wc["chance_control"] - b_current_wc["chance_control"])
        + 0.22 * np.tanh((a_current_wc["xg_balance"] - b_current_wc["xg_balance"]) / 1.8)
        + 0.18 * (a_current_wc["clinical"] - b_current_wc["clinical"])
        + 0.14 * (b_current_wc["waste"] - a_current_wc["waste"])
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_current_low_block_solution_edge"] = current_low_block_pair_coverage * (
        (
            a_current_wc["underlying"]
            - 0.55 * a_current_wc["sterile_control"]
            - 0.18 * a_current_wc["waste"]
        )
        * team_b_low_block
        - (
            b_current_wc["underlying"]
            - 0.55 * b_current_wc["sterile_control"]
            - 0.18 * b_current_wc["waste"]
        )
        * team_a_low_block
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_current_transition_punch_edge"] = current_wc_pair_coverage * (
        a_current_wc["low_possession_punch"] - b_current_wc["low_possession_punch"]
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_current_unrewarded_pressure_edge"] = current_wc_pair_coverage * (
        a_current_wc["unrewarded"] - b_current_wc["unrewarded"]
    ).clip(lower=-1.0, upper=1.0)
    out["worldcup_fotmob_current_controlled_dominance_edge"] = current_wc_pair_coverage * (
        a_current_wc["controlled_dominance"] - b_current_wc["controlled_dominance"]
    ).clip(lower=-1.0, upper=1.0)
    current_worldcup_story = (
        0.52 * out["worldcup_fotmob_current_controlled_dominance_edge"]
        + 0.20 * out["worldcup_fotmob_current_chance_pressure_edge"]
        + 0.18 * out["worldcup_fotmob_current_low_block_solution_edge"]
        + 0.07 * out["worldcup_fotmob_current_transition_punch_edge"]
        + 0.03 * out["worldcup_fotmob_current_unrewarded_pressure_edge"]
    ).clip(lower=-1.0, upper=1.0)
    # The current tournament is most useful when interpreted through a wider
    # World Cup tactical baseline. This keeps the live story responsive while
    # supplying enough historical coverage for the conservative tree to learn.
    out["worldcup_fotmob_current_story_edge"] = (
        0.55 * out["worldcup_fotmob_interpreted_edge"]
        + 0.45 * current_worldcup_story
    ).clip(lower=-1.0, upper=1.0)
    out["club_star_finisher_edge"] = _num_series(
        out,
        "team_a_club_star_finisher_signal",
    ) - _num_series(out, "team_b_club_star_finisher_signal")
    return out


def _load_matrix(data_root: Path) -> pd.DataFrame:
    path = data_root / "processed" / "combined" / "clean_training_matrix.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    for column in NUMERIC_FEATURES + [
        "match_recency_weight",
        "home_goals",
        "away_goals",
        "total_goals",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_FEATURES:
        frame[column] = frame[column].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("unknown").astype("category")
    frame = frame.copy()
    frame["result_label"] = frame["result"].map(TARGET_RESULT_MAP)
    return frame


def _load_training_frame(data_root: Path) -> pd.DataFrame | None:
    path = data_root / "processed" / "combined" / "training_frame.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_late85_points_swing_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
    *,
    window: int = 6,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    timeline_metrics = load_late85_points_swing_metrics(
        data_root,
        training_frame.to_dict(orient="records"),
    )
    histories: dict[str, list[float]] = {}
    home_values: list[float] = []
    away_values: list[float] = []

    for _, row in training_frame.iterrows():
        home_key = normalise_team_name(str(row["home_team"]))
        away_key = normalise_team_name(str(row["away_team"]))
        home_history = histories.get(home_key, [])[-window:]
        away_history = histories.get(away_key, [])[-window:]
        home_values.append(float(np.mean(home_history)) if home_history else 0.0)
        away_values.append(float(np.mean(away_history)) if away_history else 0.0)

        match_metrics = timeline_metrics.get(str(row["match_id"]))
        if not match_metrics:
            continue
        for team_key in (home_key, away_key):
            team_metrics = match_metrics.get(team_key)
            if team_metrics is not None:
                histories.setdefault(team_key, []).append(
                    float(team_metrics.get("late85_points_swing_edge", 0.0))
                )

    clean_frame["home_recent6_late85_points_swing"] = home_values
    clean_frame["away_recent6_late85_points_swing"] = away_values
    return clean_frame


def _maybe_add_late85_points_swing(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if {
        "home_recent6_late85_points_swing",
        "away_recent6_late85_points_swing",
    }.issubset(frame.columns):
        return frame
    training_frame = training_frame if training_frame is not None else _load_training_frame(data_root)
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_late85_points_swing_to_clean_frame(data_root, frame, training_frame)


def add_score_timing_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
    *,
    window: int = 6,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    timing_metrics = load_score_timing_metrics(
        data_root,
        training_frame.to_dict(orient="records"),
    )
    histories: dict[str, list[dict[str, float]]] = {}
    home_state_values: list[float] = []
    away_state_values: list[float] = []
    home_control_values: list[float] = []
    away_control_values: list[float] = []
    home_quickness_values: list[float] = []
    away_quickness_values: list[float] = []
    home_quality_values: list[float] = []
    away_quality_values: list[float] = []
    home_narrow_hold_values: list[float] = []
    away_narrow_hold_values: list[float] = []
    home_comfortable_lead_values: list[float] = []
    away_comfortable_lead_values: list[float] = []
    home_friction_values: list[float] = []
    away_friction_values: list[float] = []
    home_state_change_values: list[float] = []
    away_state_change_values: list[float] = []
    home_early_state_change_values: list[float] = []
    away_early_state_change_values: list[float] = []
    home_quality_control_values: list[float] = []
    away_quality_control_values: list[float] = []
    home_quality_state_change_values: list[float] = []
    away_quality_state_change_values: list[float] = []
    home_quality_early_state_change_values: list[float] = []
    away_quality_early_state_change_values: list[float] = []
    home_timing_coverage_values: list[float] = []
    away_timing_coverage_values: list[float] = []

    def mean_metric(history: list[dict[str, float]], metric: str) -> float:
        values = [float(item.get(metric, 0.0)) for item in history[-window:]]
        return float(np.mean(values)) if values else 0.0

    def timing_coverage(history: list[dict[str, float]]) -> float:
        observed = sum(
            float(item.get("score_timing_observed", 0.0))
            for item in history[-window:]
        )
        return observed / max(window, 1)

    def num(value: Any, fallback: float = 0.0) -> float:
        if value in (None, ""):
            return fallback
        try:
            if pd.isna(value):
                return fallback
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def opponent_weight(row: pd.Series, prefix: str) -> float:
        return _opponent_quality_factor(num(row.get(f"{prefix}_opponent_elo_pre"), BASE_ELO))

    for _, row in training_frame.iterrows():
        home_key = normalise_team_name(str(row["home_team"]))
        away_key = normalise_team_name(str(row["away_team"]))
        home_history = histories.get(home_key, [])
        away_history = histories.get(away_key, [])
        home_timing_coverage_values.append(timing_coverage(home_history))
        away_timing_coverage_values.append(timing_coverage(away_history))
        home_state_values.append(mean_metric(home_history, "score_state_value"))
        away_state_values.append(mean_metric(away_history, "score_state_value"))
        home_control_values.append(mean_metric(home_history, "score_control_value"))
        away_control_values.append(mean_metric(away_history, "score_control_value"))
        home_quickness_values.append(mean_metric(home_history, "scoring_quickness"))
        away_quickness_values.append(mean_metric(away_history, "scoring_quickness"))
        home_quality_values.append(mean_metric(home_history, "score_control_quality"))
        away_quality_values.append(mean_metric(away_history, "score_control_quality"))
        home_narrow_hold_values.append(mean_metric(home_history, "narrow_lead_hold"))
        away_narrow_hold_values.append(mean_metric(away_history, "narrow_lead_hold"))
        home_comfortable_lead_values.append(mean_metric(home_history, "comfortable_lead"))
        away_comfortable_lead_values.append(mean_metric(away_history, "comfortable_lead"))
        home_friction_values.append(mean_metric(home_history, "game_state_friction"))
        away_friction_values.append(mean_metric(away_history, "game_state_friction"))
        home_state_change_values.append(mean_metric(home_history, "state_change_swing"))
        away_state_change_values.append(mean_metric(away_history, "state_change_swing"))
        home_early_state_change_values.append(mean_metric(home_history, "early_state_change_swing"))
        away_early_state_change_values.append(mean_metric(away_history, "early_state_change_swing"))
        home_quality_control_values.append(mean_metric(home_history, "quality_score_control_value"))
        away_quality_control_values.append(mean_metric(away_history, "quality_score_control_value"))
        home_quality_state_change_values.append(
            mean_metric(home_history, "quality_state_change_swing")
        )
        away_quality_state_change_values.append(
            mean_metric(away_history, "quality_state_change_swing")
        )
        home_quality_early_state_change_values.append(
            mean_metric(home_history, "quality_early_state_change_swing")
        )
        away_quality_early_state_change_values.append(
            mean_metric(away_history, "quality_early_state_change_swing")
        )

        match_metrics = timing_metrics.get(str(row["match_id"]), {})
        for team_key, prefix in ((home_key, "home"), (away_key, "away")):
            team_metrics = match_metrics.get(team_key)
            timing_observed = float(team_metrics is not None)
            team_metrics = team_metrics or {}
            weight = opponent_weight(row, prefix)
            score_control_value = float(team_metrics.get("score_control_value", 0.0))
            state_change_swing = float(team_metrics.get("state_change_swing", 0.0))
            early_state_change_swing = float(
                team_metrics.get("early_state_change_swing", 0.0)
            )
            histories.setdefault(team_key, []).append(
                {
                    "score_timing_observed": timing_observed,
                    "score_state_value": float(team_metrics.get("score_state_value", 0.0)),
                    "score_control_value": score_control_value,
                    "scoring_quickness": float(team_metrics.get("scoring_quickness", 0.0)),
                    "score_control_quality": float(
                        team_metrics.get("score_control_quality", 0.0)
                    ),
                    "narrow_lead_hold": float(team_metrics.get("narrow_lead_hold", 0.0)),
                    "comfortable_lead": float(team_metrics.get("comfortable_lead", 0.0)),
                    "game_state_friction": float(
                        team_metrics.get("game_state_friction", 0.0)
                    ),
                    "state_change_swing": state_change_swing,
                    "early_state_change_swing": early_state_change_swing,
                    "quality_score_control_value": score_control_value * weight,
                    "quality_state_change_swing": state_change_swing * weight,
                    "quality_early_state_change_swing": early_state_change_swing * weight,
                }
            )

    clean_frame["home_recent6_score_state_value"] = home_state_values
    clean_frame["away_recent6_score_state_value"] = away_state_values
    clean_frame["home_recent6_score_control_value"] = home_control_values
    clean_frame["away_recent6_score_control_value"] = away_control_values
    clean_frame["home_recent6_scoring_quickness"] = home_quickness_values
    clean_frame["away_recent6_scoring_quickness"] = away_quickness_values
    clean_frame["home_recent6_score_control_quality"] = home_quality_values
    clean_frame["away_recent6_score_control_quality"] = away_quality_values
    clean_frame["home_recent6_narrow_lead_hold"] = home_narrow_hold_values
    clean_frame["away_recent6_narrow_lead_hold"] = away_narrow_hold_values
    clean_frame["home_recent6_comfortable_lead"] = home_comfortable_lead_values
    clean_frame["away_recent6_comfortable_lead"] = away_comfortable_lead_values
    clean_frame["home_recent6_game_state_friction"] = home_friction_values
    clean_frame["away_recent6_game_state_friction"] = away_friction_values
    clean_frame["home_recent6_state_change_swing"] = home_state_change_values
    clean_frame["away_recent6_state_change_swing"] = away_state_change_values
    clean_frame["home_recent6_early_state_change_swing"] = home_early_state_change_values
    clean_frame["away_recent6_early_state_change_swing"] = away_early_state_change_values
    clean_frame["home_recent6_quality_score_control_value"] = home_quality_control_values
    clean_frame["away_recent6_quality_score_control_value"] = away_quality_control_values
    clean_frame["home_recent6_quality_state_change_swing"] = home_quality_state_change_values
    clean_frame["away_recent6_quality_state_change_swing"] = away_quality_state_change_values
    clean_frame["home_recent6_quality_early_state_change_swing"] = (
        home_quality_early_state_change_values
    )
    clean_frame["away_recent6_quality_early_state_change_swing"] = (
        away_quality_early_state_change_values
    )
    clean_frame["home_recent6_score_timing_coverage"] = home_timing_coverage_values
    clean_frame["away_recent6_score_timing_coverage"] = away_timing_coverage_values
    return clean_frame


def _maybe_add_score_timing(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if {
        "home_recent6_score_state_value",
        "away_recent6_score_state_value",
        "home_recent6_score_control_value",
        "away_recent6_score_control_value",
        "home_recent6_scoring_quickness",
        "away_recent6_scoring_quickness",
        "home_recent6_score_control_quality",
        "away_recent6_score_control_quality",
        "home_recent6_narrow_lead_hold",
        "away_recent6_narrow_lead_hold",
        "home_recent6_comfortable_lead",
        "away_recent6_comfortable_lead",
        "home_recent6_game_state_friction",
        "away_recent6_game_state_friction",
        "home_recent6_state_change_swing",
        "away_recent6_state_change_swing",
        "home_recent6_early_state_change_swing",
        "away_recent6_early_state_change_swing",
        "home_recent6_quality_score_control_value",
        "away_recent6_quality_score_control_value",
        "home_recent6_quality_state_change_swing",
        "away_recent6_quality_state_change_swing",
        "home_recent6_quality_early_state_change_swing",
        "away_recent6_quality_early_state_change_swing",
        "home_recent6_score_timing_coverage",
        "away_recent6_score_timing_coverage",
    }.issubset(frame.columns):
        return frame
    training_frame = training_frame if training_frame is not None else _load_training_frame(data_root)
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_score_timing_to_clean_frame(data_root, frame, training_frame)


def add_counter_efficiency_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
    *,
    window: int = COUNTER_EFFICIENCY_WINDOW,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    all_rows = _load_match_rows(data_root, combined=True)
    _infer_group_keys(all_rows)
    all_rows = _with_recent_features(all_rows, window=RECENT_FORM_WINDOW)
    exported_ids = set(training_frame["match_id"].astype(str))
    histories: dict[str, list[dict[str, Any]]] = {}
    values_by_match: dict[str, dict[str, float]] = {}

    def num(value: Any, fallback: float = 0.0) -> float:
        if value in (None, ""):
            return fallback
        try:
            if pd.isna(value):
                return fallback
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def history_item(row: dict[str, Any], side: str, opponent: str) -> dict[str, Any]:
        goals_for = int(row[f"{side}_goals"])
        goals_against = int(row[f"{opponent}_goals"])
        return {
            "goals_for": goals_for,
            "points": (
                3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            ),
            "team_elo": num(row.get(f"{side}_elo_pre"), BASE_ELO),
            "opponent_elo": num(
                row.get(f"{side}_opponent_elo_pre"),
                BASE_ELO,
            ),
            "ball_possession_pct": row.get(f"{side}_actual_ball_possession_pct"),
            "total_passes": row.get(f"{side}_actual_total_passes"),
            "passes_pct": row.get(f"{side}_actual_passes_pct"),
            "total_shots": row.get(f"{side}_actual_total_shots"),
            "shots_on_goal": row.get(f"{side}_actual_shots_on_goal"),
        }

    for row in all_rows:
        match_id = str(row["match_id"])
        if match_id in exported_ids:
            match_values: dict[str, float] = {}
            for side in ("home", "away"):
                team_key = normalise_team_name(str(row[f"{side}_team"]))
                summary = counter_history_summary(
                    histories.get(team_key, []),
                    window=window,
                )
                fit = current_underdog_fit(
                    row.get(f"{side}_elo_pre"),
                    row.get(f"{side}_opponent_elo_pre"),
                )
                match_values[side] = current_counter_threat(summary, fit)
            values_by_match[match_id] = match_values

        for side, opponent in (("home", "away"), ("away", "home")):
            team_key = normalise_team_name(str(row[f"{side}_team"]))
            histories.setdefault(team_key, []).append(history_item(row, side, opponent))

    clean_frame["home_counter_current_threat"] = [
        values_by_match.get(str(match_id), {}).get("home", 0.0)
        for match_id in training_frame["match_id"]
    ]
    clean_frame["away_counter_current_threat"] = [
        values_by_match.get(str(match_id), {}).get("away", 0.0)
        for match_id in training_frame["match_id"]
    ]
    return clean_frame


def _maybe_add_counter_efficiency(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if {
        "home_counter_current_threat",
        "away_counter_current_threat",
    }.issubset(frame.columns):
        return frame
    training_frame = training_frame if training_frame is not None else _load_training_frame(data_root)
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_counter_efficiency_to_clean_frame(data_root, frame, training_frame)


def add_clinical_finishing_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
    *,
    window: int = CLINICAL_FINISHING_WINDOW,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    all_rows = _load_match_rows(data_root, combined=True)
    exported_ids = set(training_frame["match_id"].astype(str))
    histories: dict[str, list[dict[str, Any]]] = {}
    values_by_match: dict[str, dict[str, dict[str, float]]] = {}

    def history_item(row: dict[str, Any], side: str) -> dict[str, Any]:
        return {
            "goals_for": row[f"{side}_goals"],
            "shots_on_goal": row.get(f"{side}_actual_shots_on_goal"),
            "total_shots": row.get(f"{side}_actual_total_shots"),
            "ball_possession_pct": row.get(
                f"{side}_actual_ball_possession_pct"
            ),
            "total_passes": row.get(f"{side}_actual_total_passes"),
            "passes_pct": row.get(f"{side}_actual_passes_pct"),
            "corner_kicks": row.get(f"{side}_actual_corner_kicks"),
        }

    for row in all_rows:
        match_id = str(row["match_id"])
        if match_id in exported_ids:
            values_by_match[match_id] = {}
            for side in ("home", "away"):
                team_key = normalise_team_name(str(row[f"{side}_team"]))
                values_by_match[match_id][side] = clinical_finishing_summary(
                    histories.get(team_key, []),
                    window=window,
                )
        for side in ("home", "away"):
            team_key = normalise_team_name(str(row[f"{side}_team"]))
            histories.setdefault(team_key, []).append(history_item(row, side))

    output_fields = (
        "clinical_signal",
        "clinical_coverage",
        "low_block_profile",
        "low_block_coverage",
    )
    for side in ("home", "away"):
        for field in output_fields:
            clean_frame[f"{side}_{field}_aligned"] = [
                values_by_match.get(str(match_id), {})
                .get(side, {})
                .get(field, 0.0)
                for match_id in training_frame["match_id"]
            ]
    return clean_frame


def _maybe_add_clinical_finishing(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {
        f"{side}_{field}_aligned"
        for side in ("home", "away")
        for field in (
            "clinical_signal",
            "clinical_coverage",
            "low_block_profile",
            "low_block_coverage",
        )
    }
    if required.issubset(frame.columns):
        return frame
    training_frame = (
        training_frame
        if training_frame is not None
        else _load_training_frame(data_root)
    )
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_clinical_finishing_to_clean_frame(
        data_root,
        frame,
        training_frame,
    )


def add_player_attacking_personnel_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    snapshots, _, timelines = build_personnel_state(data_root)
    fields = (
        "attacking_personnel_signal",
        "star_finisher_signal",
        "attack_core_signal",
        "personnel_coverage",
    )
    values = {
        f"{side}_{field}": []
        for side in ("home", "away")
        for field in fields
    }
    for row in training_frame.to_dict(orient="records"):
        try:
            match_date = pd.Timestamp(row["date"]).date()
        except (TypeError, ValueError):
            match_date = None
        home_key = normalise_team_name(str(row.get("home_team") or ""))
        away_key = normalise_team_name(str(row.get("away_team") or ""))
        key = (
            (match_date, tuple(sorted((home_key, away_key))))
            if match_date is not None and home_key and away_key
            else None
        )
        match_values = snapshots.get(key, {}) if key is not None else {}
        for side, team_key in (("home", home_key), ("away", away_key)):
            summary = match_values.get(team_key)
            if summary is None and match_date is not None:
                summary = personnel_summary_before(
                    timelines,
                    team_key,
                    match_date,
                )
            summary = summary or {}
            for field in fields:
                values[f"{side}_{field}"].append(
                    float(summary.get(field, 0.0))
                )
    for column, column_values in values.items():
        clean_frame[column] = column_values
    return clean_frame


def _maybe_add_player_attacking_personnel(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {
        f"{side}_{field}"
        for side in ("home", "away")
        for field in (
            "attacking_personnel_signal",
            "star_finisher_signal",
            "attack_core_signal",
            "personnel_coverage",
        )
    }
    if required.issubset(frame.columns):
        return frame
    training_frame = (
        training_frame
        if training_frame is not None
        else _load_training_frame(data_root)
    )
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_player_attacking_personnel_to_clean_frame(
        data_root,
        frame,
        training_frame,
    )


def add_club_attacking_talent_to_clean_frame(
    data_root: Path,
    clean_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    clean_frame = clean_frame.copy()
    snapshots, timelines = build_club_talent_state(data_root)
    fields = (
        "club_attack_talent_signal",
        "club_star_finisher_signal",
        "club_attack_coverage",
    )
    values = {
        f"{side}_{field}": []
        for side in ("home", "away")
        for field in fields
    }
    for row in training_frame.to_dict(orient="records"):
        try:
            match_date = pd.Timestamp(row["date"]).date()
        except (TypeError, ValueError):
            match_date = None
        home_key = normalise_team_name(str(row.get("home_team") or ""))
        away_key = normalise_team_name(str(row.get("away_team") or ""))
        key = (
            (match_date, tuple(sorted((home_key, away_key))))
            if match_date is not None and home_key and away_key
            else None
        )
        match_values = snapshots.get(key, {}) if key is not None else {}
        for side, team_key in (("home", home_key), ("away", away_key)):
            summary = match_values.get(team_key)
            if summary is None and match_date is not None:
                summary = club_talent_summary_before(
                    timelines,
                    team_key,
                    match_date,
                )
            summary = summary or {}
            for field in fields:
                values[f"{side}_{field}"].append(
                    float(summary.get(field, 0.0))
                )
    for column, column_values in values.items():
        clean_frame[column] = column_values
    return clean_frame


def _maybe_add_club_attacking_talent(
    data_root: Path,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {
        f"{side}_{field}"
        for side in ("home", "away")
        for field in (
            "club_attack_talent_signal",
            "club_star_finisher_signal",
            "club_attack_coverage",
        )
    }
    if required.issubset(frame.columns):
        return frame
    training_frame = (
        training_frame
        if training_frame is not None
        else _load_training_frame(data_root)
    )
    if training_frame is None or len(training_frame) != len(frame):
        return frame
    return add_club_attacking_talent_to_clean_frame(
        data_root,
        frame,
        training_frame,
    )


def _build_neutral_frame(frame: pd.DataFrame, *, augment: bool) -> pd.DataFrame:
    def num(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            if pd.isna(value):
                return 0.0
        except TypeError:
            pass
        return float(value)

    def has_value(value: Any) -> bool:
        if value in (None, ""):
            return False
        try:
            return not bool(pd.isna(value))
        except TypeError:
            return True

    def side(prefix: str, row: pd.Series) -> dict[str, Any]:
        other = "away" if prefix == "home" else "home"
        return {
            "team_confederation": row[f"{prefix}_confederation"],
            "team_train_matches": row[f"{prefix}_team_train_matches"],
            "team_goals_for_avg": row[f"{prefix}_team_goals_for_avg"],
            "team_goals_against_avg": row[f"{prefix}_team_goals_against_avg"],
            "team_confederation_strength": row[f"{prefix}_confederation_strength"],
            "team_cross_confederation_matches": row[f"{prefix}_cross_confederation_matches"],
            "team_cross_confederation_strength": row[f"{prefix}_cross_confederation_strength"],
            "team_recent6_matches": row[f"{prefix}_recent6_matches"],
            "team_recent6_goals_for_avg": row[f"{prefix}_recent6_goals_for_avg"],
            "team_recent6_goals_against_avg": row[f"{prefix}_recent6_goals_against_avg"],
            "team_recent6_points_avg": row[f"{prefix}_recent6_points_avg"],
            "team_recent6_win_rate": row[f"{prefix}_recent6_win_rate"],
            "team_recent6_goal_diff_avg": row[f"{prefix}_recent6_goal_diff_avg"],
            "team_recent6_opponent_elo_avg": row[f"{prefix}_recent6_opponent_elo_avg"],
            "team_recent6_quality_result_points_avg": row[
                f"{prefix}_recent6_quality_result_points_avg"
            ],
            "team_recent6_quality_goal_balance_avg": row[
                f"{prefix}_recent6_quality_goal_balance_avg"
            ],
            "team_rest_days": row[f"{prefix}_rest_days"],
            "team_elo_pre": row[f"{prefix}_elo_pre"],
            "opponent_elo_pre": row[f"{prefix}_opponent_elo_pre"],
            "team_worldcup_recent6_win_rate": row[f"{prefix}_worldcup_recent6_win_rate"],
            "team_fifa_rank": row[f"{prefix}_fifa_rank"],
            "team_fifa_points": row[f"{prefix}_fifa_points"],
            "team_historical_fifa_rank": row[f"{prefix}_historical_fifa_rank"],
            "team_historical_fifa_points": row[f"{prefix}_historical_fifa_points"],
            "team_historical_fifa_observed": row.get(
                f"{prefix}_historical_fifa_observed",
                1.0,
            ),
            "team_live_fifa_points": row.get(
                f"{prefix}_live_fifa_points",
                row[f"{prefix}_historical_fifa_points"],
            ),
            "team_late85_points_swing": row.get(f"{prefix}_recent6_late85_points_swing", 0.0),
            "team_score_state_value": row.get(f"{prefix}_recent6_score_state_value", 0.0),
            "team_score_control_value": row.get(f"{prefix}_recent6_score_control_value", 0.0),
            "team_scoring_quickness": row.get(f"{prefix}_recent6_scoring_quickness", 0.0),
            "team_score_control_quality": row.get(f"{prefix}_recent6_score_control_quality", 0.0),
            "team_narrow_lead_hold": row.get(f"{prefix}_recent6_narrow_lead_hold", 0.0),
            "team_comfortable_lead": row.get(f"{prefix}_recent6_comfortable_lead", 0.0),
            "team_game_state_friction": row.get(f"{prefix}_recent6_game_state_friction", 0.0),
            "team_state_change_swing": row.get(f"{prefix}_recent6_state_change_swing", 0.0),
            "team_early_state_change_swing": row.get(
                f"{prefix}_recent6_early_state_change_swing",
                0.0,
            ),
            "team_quality_score_control_value": row.get(
                f"{prefix}_recent6_quality_score_control_value",
                0.0,
            ),
            "team_quality_state_change_swing": row.get(
                f"{prefix}_recent6_quality_state_change_swing",
                0.0,
            ),
            "team_quality_early_state_change_swing": row.get(
                f"{prefix}_recent6_quality_early_state_change_swing",
                0.0,
            ),
            "team_score_timing_coverage": row.get(
                f"{prefix}_recent6_score_timing_coverage",
                0.0,
            ),
            "team_counter_current_threat": row.get(
                f"{prefix}_counter_current_threat",
                0.0,
            ),
            "team_clinical_finishing": row.get(
                f"{prefix}_clinical_signal_aligned",
                0.0,
            ),
            "team_clinical_coverage": row.get(
                f"{prefix}_clinical_coverage_aligned",
                0.0,
            ),
            "team_low_block_profile_aligned": row.get(
                f"{prefix}_low_block_profile_aligned",
                0.0,
            ),
            "team_low_block_coverage_aligned": row.get(
                f"{prefix}_low_block_coverage_aligned",
                0.0,
            ),
            "team_attacking_personnel_signal": row.get(
                f"{prefix}_attacking_personnel_signal",
                0.0,
            ),
            "team_star_finisher_signal": row.get(
                f"{prefix}_star_finisher_signal",
                0.0,
            ),
            "team_attack_core_signal": row.get(
                f"{prefix}_attack_core_signal",
                0.0,
            ),
            "team_personnel_coverage": row.get(
                f"{prefix}_personnel_coverage",
                0.0,
            ),
            "team_club_attack_talent_signal": row.get(
                f"{prefix}_club_attack_talent_signal",
                0.0,
            ),
            "team_club_star_finisher_signal": row.get(
                f"{prefix}_club_star_finisher_signal",
                0.0,
            ),
            "team_club_attack_coverage": row.get(
                f"{prefix}_club_attack_coverage",
                0.0,
            ),
            "is_tournament_host": row[f"{prefix}_is_tournament_host"],
            "group_matches_pre": row[f"{prefix}_group_matches_pre"],
            "group_points_pre": row[f"{prefix}_group_points_pre"],
            "group_goal_diff_pre": row[f"{prefix}_group_goal_diff_pre"],
            "group_goals_for_pre": row[f"{prefix}_group_goals_for_pre"],
            "group_goals_against_pre": row[f"{prefix}_group_goals_against_pre"],
            "group_position_pre": row[f"{prefix}_group_position_pre"],
            "team_goals": row[f"{prefix}_goals"],
            "opponent_goals": row[f"{other}_goals"],
            "h2h_goals_avg": row[f"h2h_recent_2y_{prefix}_goals_avg"],
            "h2h_opponent_goals_avg": row[f"h2h_recent_2y_{other}_goals_avg"],
            "h2h_points_avg": row[f"h2h_recent_2y_{prefix}_points_avg"],
            "h2h_opponent_points_avg": row[f"h2h_recent_2y_{other}_points_avg"],
            "h2h_penalty_wins": row[f"h2h_recent_2y_{prefix}_penalty_wins"],
            "h2h_opponent_penalty_wins": row[f"h2h_recent_2y_{other}_penalty_wins"],
            "detail_stats": {
                feature: row[f"{prefix}_recent6_{feature}_avg"] for feature in DETAIL_STAT_FEATURES
            },
            "worldcup_detail_stats": {
                feature: row[f"{prefix}_worldcup_recent6_{feature}_avg"]
                for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
            },
            "current_worldcup_detail_stats": {
                feature: row[f"{prefix}_current_worldcup_recent6_{feature}_avg"]
                for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
            },
            "detail_coverage": sum(
                int(has_value(row.get(f"{prefix}_recent6_{feature}_avg")))
                for feature in CORE_DETAIL_STAT_FEATURES
            )
            / len(CORE_DETAIL_STAT_FEATURES),
            "espn_leader_detail_coverage": sum(
                int(has_value(row.get(f"{prefix}_recent6_{feature}_avg")))
                for feature in ESPN_LEADER_DETAIL_FEATURES
            )
            / len(ESPN_LEADER_DETAIL_FEATURES),
        }

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        pairs = [("home", "away")]
        if augment and row["split"] == "train":
            pairs.append(("away", "home"))
        for a_prefix, b_prefix in pairs:
            a = side(a_prefix, row)
            b = side(b_prefix, row)
            result = "draw"
            if a["team_goals"] > b["team_goals"]:
                result = "team_a"
            elif b["team_goals"] > a["team_goals"]:
                result = "team_b"
            a_fouls = a["detail_stats"].get("fouls")
            b_fouls = b["detail_stats"].get("fouls")
            a_yellows = a["detail_stats"].get("yellow_cards")
            b_yellows = b["detail_stats"].get("yellow_cards")
            physical_coverage = (
                int(has_value(a_fouls))
                + int(has_value(b_fouls))
                + int(has_value(a_yellows))
                + int(has_value(b_yellows))
            ) / 4.0
            records.append(
                {
                    "split": row["split"],
                    "match_recency_weight": row.get("match_recency_weight", 1.0),
                    "competition_name": row["competition_name"],
                    "competition_family": row["competition_family"],
                    "competition_type": row["competition_type"],
                    "stage_or_round": row["stage_or_round"],
                    "is_friendly": row["is_friendly"],
                    "is_qualifier": row["is_qualifier"],
                    "team_a_confederation": a["team_confederation"],
                    "team_b_confederation": b["team_confederation"],
                    "confederation_matchup": (
                        f"{a['team_confederation']}_vs_{b['team_confederation']}"
                    ),
                    "confederation_matchup_unordered": "_vs_".join(
                        sorted([str(a["team_confederation"]), str(b["team_confederation"])])
                    ),
                    "same_confederation": row["same_confederation"],
                    "is_knockout": row["is_knockout"],
                    "host_home_advantage_diff": (
                        num(a["is_tournament_host"]) if a_prefix == "home" else 0.0
                    )
                    - (num(b["is_tournament_host"]) if b_prefix == "home" else 0.0),
                    "competition_train_matches": row["competition_train_matches"],
                    "competition_goals_avg": (
                        row["competition_home_goals_avg"] + row["competition_away_goals_avg"]
                    )
                    / 2,
                    "phase_train_matches": row["phase_train_matches"],
                    "phase_goals_avg": (row["phase_home_goals_avg"] + row["phase_away_goals_avg"])
                    / 2,
                    "team_a_group_matches_pre": a["group_matches_pre"],
                    "team_a_group_points_pre": a["group_points_pre"],
                    "team_a_group_goal_diff_pre": a["group_goal_diff_pre"],
                    "team_a_group_goals_for_pre": a["group_goals_for_pre"],
                    "team_a_group_goals_against_pre": a["group_goals_against_pre"],
                    "team_a_group_position_pre": a["group_position_pre"],
                    "team_b_group_matches_pre": b["group_matches_pre"],
                    "team_b_group_points_pre": b["group_points_pre"],
                    "team_b_group_goal_diff_pre": b["group_goal_diff_pre"],
                    "team_b_group_goals_for_pre": b["group_goals_for_pre"],
                    "team_b_group_goals_against_pre": b["group_goals_against_pre"],
                    "team_b_group_position_pre": b["group_position_pre"],
                    "group_points_diff_pre": num(a["group_points_pre"])
                    - num(b["group_points_pre"]),
                    "group_goal_diff_diff_pre": num(a["group_goal_diff_pre"])
                    - num(b["group_goal_diff_pre"]),
                    "group_position_diff_pre": num(a["group_position_pre"])
                    - num(b["group_position_pre"]),
                    "team_a_train_matches": a["team_train_matches"],
                    "team_a_goals_for_avg": a["team_goals_for_avg"],
                    "team_a_goals_against_avg": a["team_goals_against_avg"],
                    "team_a_confederation_strength": a["team_confederation_strength"],
                    "team_a_cross_confederation_matches": a["team_cross_confederation_matches"],
                    "team_a_cross_confederation_strength": a["team_cross_confederation_strength"],
                    "team_b_train_matches": b["team_train_matches"],
                    "team_b_goals_for_avg": b["team_goals_for_avg"],
                    "team_b_goals_against_avg": b["team_goals_against_avg"],
                    "teams_goals_for_sum": num(a["team_goals_for_avg"])
                    + num(b["team_goals_for_avg"]),
                    "teams_goals_against_sum": num(a["team_goals_against_avg"])
                    + num(b["team_goals_against_avg"]),
                    "team_a_attack_vs_b_defense_avg": (
                        num(a["team_goals_for_avg"]) + num(b["team_goals_against_avg"])
                    )
                    / 2,
                    "team_b_attack_vs_a_defense_avg": (
                        num(b["team_goals_for_avg"]) + num(a["team_goals_against_avg"])
                    )
                    / 2,
                    "match_attack_defense_volume": (
                        num(a["team_goals_for_avg"])
                        + num(b["team_goals_for_avg"])
                        + num(a["team_goals_against_avg"])
                        + num(b["team_goals_against_avg"])
                    )
                    / 2,
                    "team_b_confederation_strength": b["team_confederation_strength"],
                    "team_b_cross_confederation_matches": b["team_cross_confederation_matches"],
                    "team_b_cross_confederation_strength": b["team_cross_confederation_strength"],
                    "team_a_recent6_matches": a["team_recent6_matches"],
                    "team_a_recent6_goals_for_avg": a["team_recent6_goals_for_avg"],
                    "team_a_recent6_goals_against_avg": a["team_recent6_goals_against_avg"],
                    "team_a_recent6_points_avg": a["team_recent6_points_avg"],
                    "team_a_recent6_win_rate": a["team_recent6_win_rate"],
                    "team_b_recent6_matches": b["team_recent6_matches"],
                    "team_b_recent6_goals_for_avg": b["team_recent6_goals_for_avg"],
                    "team_b_recent6_goals_against_avg": b["team_recent6_goals_against_avg"],
                    "recent6_goals_for_sum": num(a["team_recent6_goals_for_avg"])
                    + num(b["team_recent6_goals_for_avg"]),
                    "recent6_goals_against_sum": num(a["team_recent6_goals_against_avg"])
                    + num(b["team_recent6_goals_against_avg"]),
                    "recent6_team_a_attack_vs_b_defense_avg": (
                        num(a["team_recent6_goals_for_avg"])
                        + num(b["team_recent6_goals_against_avg"])
                    )
                    / 2,
                    "recent6_team_b_attack_vs_a_defense_avg": (
                        num(b["team_recent6_goals_for_avg"])
                        + num(a["team_recent6_goals_against_avg"])
                    )
                    / 2,
                    "recent6_match_attack_defense_volume": (
                        num(a["team_recent6_goals_for_avg"])
                        + num(b["team_recent6_goals_for_avg"])
                        + num(a["team_recent6_goals_against_avg"])
                        + num(b["team_recent6_goals_against_avg"])
                    )
                    / 2,
                    "tempo_index": np.log1p(
                        max(
                            0.0,
                            0.5
                            * (
                                num(a["team_goals_for_avg"])
                                + num(b["team_goals_for_avg"])
                                + num(a["team_goals_against_avg"])
                                + num(b["team_goals_against_avg"])
                            )
                            / 2
                            + 0.5
                            * (
                                num(a["team_recent6_goals_for_avg"])
                                + num(b["team_recent6_goals_for_avg"])
                                + num(a["team_recent6_goals_against_avg"])
                                + num(b["team_recent6_goals_against_avg"])
                            )
                            / 2,
                        )
                    ),
                    "team_b_recent6_points_avg": b["team_recent6_points_avg"],
                    "team_b_recent6_win_rate": b["team_recent6_win_rate"],
                    "recent6_quality_result_points_diff": num(
                        a["team_recent6_quality_result_points_avg"]
                    )
                    - num(b["team_recent6_quality_result_points_avg"]),
                    "recent6_quality_goal_balance_diff": num(
                        a["team_recent6_quality_goal_balance_avg"]
                    )
                    - num(b["team_recent6_quality_goal_balance_avg"]),
                    "team_a_rest_days": a["team_rest_days"],
                    "team_b_rest_days": b["team_rest_days"],
                    "team_a_late85_points_swing": a["team_late85_points_swing"],
                    "team_b_late85_points_swing": b["team_late85_points_swing"],
                    "team_a_score_state_value": a["team_score_state_value"],
                    "team_b_score_state_value": b["team_score_state_value"],
                    "team_a_score_control_value": a["team_score_control_value"],
                    "team_b_score_control_value": b["team_score_control_value"],
                    "team_a_scoring_quickness": a["team_scoring_quickness"],
                    "team_b_scoring_quickness": b["team_scoring_quickness"],
                    "team_a_score_control_quality": a["team_score_control_quality"],
                    "team_b_score_control_quality": b["team_score_control_quality"],
                    "team_a_narrow_lead_hold": a["team_narrow_lead_hold"],
                    "team_b_narrow_lead_hold": b["team_narrow_lead_hold"],
                    "team_a_comfortable_lead": a["team_comfortable_lead"],
                    "team_b_comfortable_lead": b["team_comfortable_lead"],
                    "team_a_game_state_friction": a["team_game_state_friction"],
                    "team_b_game_state_friction": b["team_game_state_friction"],
                    "team_a_state_change_swing": a["team_state_change_swing"],
                    "team_b_state_change_swing": b["team_state_change_swing"],
                    "team_a_early_state_change_swing": a["team_early_state_change_swing"],
                    "team_b_early_state_change_swing": b["team_early_state_change_swing"],
                    "team_a_quality_score_control_value": a[
                        "team_quality_score_control_value"
                    ],
                    "team_b_quality_score_control_value": b[
                        "team_quality_score_control_value"
                    ],
                    "team_a_quality_state_change_swing": a["team_quality_state_change_swing"],
                    "team_b_quality_state_change_swing": b["team_quality_state_change_swing"],
                    "team_a_quality_early_state_change_swing": a[
                        "team_quality_early_state_change_swing"
                    ],
                    "team_b_quality_early_state_change_swing": b[
                        "team_quality_early_state_change_swing"
                    ],
                    "team_a_score_timing_coverage": a[
                        "team_score_timing_coverage"
                    ],
                    "team_b_score_timing_coverage": b[
                        "team_score_timing_coverage"
                    ],
                    "team_a_counter_current_threat": a[
                        "team_counter_current_threat"
                    ],
                    "team_b_counter_current_threat": b[
                        "team_counter_current_threat"
                    ],
                    "team_a_clinical_finishing": a[
                        "team_clinical_finishing"
                    ],
                    "team_b_clinical_finishing": b[
                        "team_clinical_finishing"
                    ],
                    "team_a_clinical_coverage": a[
                        "team_clinical_coverage"
                    ],
                    "team_b_clinical_coverage": b[
                        "team_clinical_coverage"
                    ],
                    "team_a_low_block_profile_aligned": a[
                        "team_low_block_profile_aligned"
                    ],
                    "team_b_low_block_profile_aligned": b[
                        "team_low_block_profile_aligned"
                    ],
                    "team_a_low_block_coverage_aligned": a[
                        "team_low_block_coverage_aligned"
                    ],
                    "team_b_low_block_coverage_aligned": b[
                        "team_low_block_coverage_aligned"
                    ],
                    "team_a_attacking_personnel_signal": a[
                        "team_attacking_personnel_signal"
                    ],
                    "team_b_attacking_personnel_signal": b[
                        "team_attacking_personnel_signal"
                    ],
                    "team_a_star_finisher_signal": a[
                        "team_star_finisher_signal"
                    ],
                    "team_b_star_finisher_signal": b[
                        "team_star_finisher_signal"
                    ],
                    "team_a_attack_core_signal": a[
                        "team_attack_core_signal"
                    ],
                    "team_b_attack_core_signal": b[
                        "team_attack_core_signal"
                    ],
                    "team_a_personnel_coverage": a[
                        "team_personnel_coverage"
                    ],
                    "team_b_personnel_coverage": b[
                        "team_personnel_coverage"
                    ],
                    "attacking_personnel_edge": num(
                        a["team_attacking_personnel_signal"]
                    )
                    - num(b["team_attacking_personnel_signal"]),
                    "star_finisher_edge": num(
                        a["team_star_finisher_signal"]
                    )
                    - num(b["team_star_finisher_signal"]),
                    "attack_core_edge": num(
                        a["team_attack_core_signal"]
                    )
                    - num(b["team_attack_core_signal"]),
                    "personnel_coverage_pair": min(
                        num(a["team_personnel_coverage"]),
                        num(b["team_personnel_coverage"]),
                    ),
                    "team_a_club_attack_talent_signal": a[
                        "team_club_attack_talent_signal"
                    ],
                    "team_b_club_attack_talent_signal": b[
                        "team_club_attack_talent_signal"
                    ],
                    "team_a_club_star_finisher_signal": a[
                        "team_club_star_finisher_signal"
                    ],
                    "team_b_club_star_finisher_signal": b[
                        "team_club_star_finisher_signal"
                    ],
                    "team_a_club_attack_coverage": a[
                        "team_club_attack_coverage"
                    ],
                    "team_b_club_attack_coverage": b[
                        "team_club_attack_coverage"
                    ],
                    "club_attack_talent_edge": num(
                        a["team_club_attack_talent_signal"]
                    )
                    - num(b["team_club_attack_talent_signal"]),
                    "club_talent_coverage_pair": min(
                        num(a["team_club_attack_coverage"]),
                        num(b["team_club_attack_coverage"]),
                    ),
                    "counter_current_threat_edge": num(
                        a["team_counter_current_threat"]
                    )
                    - num(b["team_counter_current_threat"]),
                    "goals_for_diff": num(a["team_goals_for_avg"]) - num(b["team_goals_for_avg"]),
                    "goals_against_diff": num(a["team_goals_against_avg"])
                    - num(b["team_goals_against_avg"]),
                    "confederation_strength_diff": num(a["team_confederation_strength"])
                    - num(b["team_confederation_strength"]),
                    "cross_confederation_strength_diff": num(a["team_cross_confederation_strength"])
                    - num(b["team_cross_confederation_strength"]),
                    "cross_confederation_matches_diff": num(a["team_cross_confederation_matches"])
                    - num(b["team_cross_confederation_matches"]),
                    "recent6_goals_for_diff": num(a["team_recent6_goals_for_avg"])
                    - num(b["team_recent6_goals_for_avg"]),
                    "recent6_goals_against_diff": num(a["team_recent6_goals_against_avg"])
                    - num(b["team_recent6_goals_against_avg"]),
                    "recent6_points_diff": num(a["team_recent6_points_avg"])
                    - num(b["team_recent6_points_avg"]),
                    "recent6_win_rate_diff": num(a["team_recent6_win_rate"])
                    - num(b["team_recent6_win_rate"]),
                    "recent6_goal_diff_diff": num(a["team_recent6_goal_diff_avg"])
                    - num(b["team_recent6_goal_diff_avg"]),
                    "recent6_opponent_elo_diff": num(a["team_recent6_opponent_elo_avg"])
                    - num(b["team_recent6_opponent_elo_avg"]),
                    "rest_days_diff": num(a["team_rest_days"]) - num(b["team_rest_days"]),
                    "team_a_elo_pre": a["team_elo_pre"],
                    "team_b_elo_pre": b["team_elo_pre"],
                    "elo_diff": num(a["team_elo_pre"]) - num(b["team_elo_pre"]),
                    "team_a_recent6_opponent_elo_avg": a["team_recent6_opponent_elo_avg"],
                    "team_b_recent6_opponent_elo_avg": b["team_recent6_opponent_elo_avg"],
                    "team_a_worldcup_recent6_win_rate": a["team_worldcup_recent6_win_rate"],
                    "team_b_worldcup_recent6_win_rate": b["team_worldcup_recent6_win_rate"],
                    "worldcup_recent6_win_rate_diff": num(a["team_worldcup_recent6_win_rate"])
                    - num(b["team_worldcup_recent6_win_rate"]),
                    "worldcup_memory_edge": 1.2
                    * (
                        num(a["team_worldcup_recent6_win_rate"])
                        - num(b["team_worldcup_recent6_win_rate"])
                    ),
                    "recent6_fouls_sum": num(a_fouls) + num(b_fouls),
                    "recent6_yellow_cards_sum": num(a_yellows) + num(b_yellows),
                    "recent6_fouls_diff": num(a_fouls) - num(b_fouls),
                    "recent6_yellow_cards_diff": num(a_yellows) - num(b_yellows),
                    "recent6_physical_coverage": physical_coverage,
                    "team_a_tactical_detail_coverage": a["detail_coverage"],
                    "team_b_tactical_detail_coverage": b["detail_coverage"],
                    "team_a_espn_leader_detail_coverage": a["espn_leader_detail_coverage"],
                    "team_b_espn_leader_detail_coverage": b["espn_leader_detail_coverage"],
                    "espn_leader_detail_coverage_pair": min(
                        num(a["espn_leader_detail_coverage"]),
                        num(b["espn_leader_detail_coverage"]),
                    ),
                    "team_a_fifa_rank": a["team_fifa_rank"],
                    "team_b_fifa_rank": b["team_fifa_rank"],
                    "fifa_rank_diff": num(a["team_fifa_rank"]) - num(b["team_fifa_rank"]),
                    "team_a_fifa_points": a["team_fifa_points"],
                    "team_b_fifa_points": b["team_fifa_points"],
                    "fifa_points_diff": num(a["team_fifa_points"]) - num(b["team_fifa_points"]),
                    "team_a_historical_fifa_rank": a["team_historical_fifa_rank"],
                    "team_b_historical_fifa_rank": b["team_historical_fifa_rank"],
                    "historical_fifa_rank_diff": num(a["team_historical_fifa_rank"])
                    - num(b["team_historical_fifa_rank"]),
                    "team_a_historical_fifa_points": a["team_historical_fifa_points"],
                    "team_b_historical_fifa_points": b["team_historical_fifa_points"],
                    "historical_fifa_points_diff": num(a["team_historical_fifa_points"])
                    - num(b["team_historical_fifa_points"]),
                    "team_a_historical_fifa_observed": num(
                        a["team_historical_fifa_observed"]
                    ),
                    "team_b_historical_fifa_observed": num(
                        b["team_historical_fifa_observed"]
                    ),
                    "team_a_live_fifa_points": a["team_live_fifa_points"],
                    "team_b_live_fifa_points": b["team_live_fifa_points"],
                    "live_fifa_points_diff": num(a["team_live_fifa_points"])
                    - num(b["team_live_fifa_points"]),
                    "h2h_recent_2y_matches": row["h2h_recent_2y_matches"],
                    "h2h_recent_2y_days_since_last": row["h2h_recent_2y_days_since_last"],
                    "h2h_recent_2y_team_a_goals_avg": a["h2h_goals_avg"],
                    "h2h_recent_2y_team_b_goals_avg": b["h2h_goals_avg"],
                    "h2h_recent_2y_goal_diff_avg": num(a["h2h_goals_avg"])
                    - num(b["h2h_goals_avg"]),
                    "h2h_recent_2y_team_a_points_avg": a["h2h_points_avg"],
                    "h2h_recent_2y_team_b_points_avg": b["h2h_points_avg"],
                    "h2h_recent_2y_points_diff": num(a["h2h_points_avg"])
                    - num(b["h2h_points_avg"]),
                    "h2h_recent_2y_draw_rate": row["h2h_recent_2y_draw_rate"],
                    "h2h_recent_2y_penalty_shootout_matches": row[
                        "h2h_recent_2y_penalty_shootout_matches"
                    ],
                    "h2h_recent_2y_team_a_penalty_wins": a["h2h_penalty_wins"],
                    "h2h_recent_2y_team_b_penalty_wins": b["h2h_penalty_wins"],
                    "h2h_recent_2y_penalty_wins_diff": num(a["h2h_penalty_wins"])
                    - num(b["h2h_penalty_wins"]),
                    "team_a_goals": a["team_goals"],
                    "team_b_goals": b["team_goals"],
                    "total_goals": a["team_goals"] + b["team_goals"],
                    "result_label": {"team_a": 0, "draw": 1, "team_b": 2}[result],
                    "result": result,
                }
            )
            for index in range(1, 7):
                records[-1][f"team_a_worldcup_last6_{index}_win"] = row[
                    f"{a_prefix}_worldcup_last6_{index}_win"
                ]
                records[-1][f"team_a_worldcup_last6_{index}_draw"] = row[
                    f"{a_prefix}_worldcup_last6_{index}_draw"
                ]
                records[-1][f"team_a_worldcup_last6_{index}_loss"] = row[
                    f"{a_prefix}_worldcup_last6_{index}_loss"
                ]
                records[-1][f"team_b_worldcup_last6_{index}_win"] = row[
                    f"{b_prefix}_worldcup_last6_{index}_win"
                ]
                records[-1][f"team_b_worldcup_last6_{index}_draw"] = row[
                    f"{b_prefix}_worldcup_last6_{index}_draw"
                ]
                records[-1][f"team_b_worldcup_last6_{index}_loss"] = row[
                    f"{b_prefix}_worldcup_last6_{index}_loss"
                ]
            for feature in DETAIL_STAT_FEATURES:
                records[-1][f"team_a_recent6_{feature}"] = a["detail_stats"][feature]
                records[-1][f"team_b_recent6_{feature}"] = b["detail_stats"][feature]
                if a["detail_stats"][feature] is None or b["detail_stats"][feature] is None:
                    records[-1][f"recent6_{feature}_diff"] = None
                else:
                    records[-1][f"recent6_{feature}_diff"] = (
                        a["detail_stats"][feature] - b["detail_stats"][feature]
                    )
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                records[-1][f"team_a_worldcup_recent6_{feature}"] = a[
                    "worldcup_detail_stats"
                ][feature]
                records[-1][f"team_b_worldcup_recent6_{feature}"] = b[
                    "worldcup_detail_stats"
                ][feature]
                records[-1][f"team_a_current_worldcup_recent6_{feature}"] = a[
                    "current_worldcup_detail_stats"
                ][feature]
                records[-1][f"team_b_current_worldcup_recent6_{feature}"] = b[
                    "current_worldcup_detail_stats"
                ][feature]
    neutral = pd.DataFrame.from_records(records)
    for column in [*NEUTRAL_NUMERIC_FEATURES, *NEUTRAL_HELPER_NUMERIC_FEATURES]:
        neutral[column] = pd.to_numeric(neutral[column], errors="coerce")
    neutral = neutral.fillna(0)
    for column in CATEGORICAL_FEATURES:
        neutral[column] = neutral[column].fillna("unknown").astype("category")
    return add_neutral_treated_features(neutral)


def export_neutral_training_matrix(data_root: Path) -> list[dict[str, Any]]:
    frame = _load_matrix(data_root)
    frame = _maybe_add_late85_points_swing(data_root, frame)
    frame = _maybe_add_score_timing(data_root, frame)
    frame = _maybe_add_counter_efficiency(data_root, frame)
    frame = _maybe_add_clinical_finishing(data_root, frame)
    frame = _maybe_add_player_attacking_personnel(data_root, frame)
    frame = _maybe_add_club_attacking_talent(data_root, frame)
    neutral = _build_neutral_frame(frame, augment=False)
    return neutral[NEUTRAL_EXPORT_COLUMNS].to_dict(orient="records")


def _calibrated_classifier_importances(
    model: CalibratedClassifierCV,
    features: list[str],
) -> list[dict[str, float | str]]:
    importances: list[np.ndarray] = []
    for calibrated_classifier in getattr(model, "calibrated_classifiers_", []):
        estimator = getattr(calibrated_classifier, "estimator", None)
        if estimator is None:
            continue
        raw_importances = getattr(estimator, "feature_importances_", None)
        if raw_importances is None:
            continue
        importances.append(np.asarray(raw_importances, dtype=float))
    if not importances:
        return []
    mean_importances = np.mean(importances, axis=0)
    return sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(features, mean_importances, strict=True)
        ],
        key=lambda item: float(item["importance"]),
        reverse=True,
    )


def train_lightgbm(data_root: Path) -> dict[str, Any]:
    return train_lightgbm_neutral(data_root)


def train_lightgbm_neutral(data_root: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = frame if frame is not None else _load_matrix(data_root)
    frame = _maybe_add_late85_points_swing(data_root, frame)
    frame = _maybe_add_score_timing(data_root, frame)
    frame = _maybe_add_counter_efficiency(data_root, frame)
    frame = _maybe_add_clinical_finishing(data_root, frame)
    frame = _maybe_add_club_attacking_talent(data_root, frame)
    neutral_frame = _build_neutral_frame(frame, augment=True)
    train = neutral_frame[neutral_frame["split"] == "train"].copy()
    test = neutral_frame[neutral_frame["split"] == "test"].copy()
    goal_features = list(NEUTRAL_GOAL_FEATURES)
    base_result_features = list(NEUTRAL_BASE_RESULT_FEATURES)
    result_features = list(NEUTRAL_RESULT_FEATURES)
    x_train = train[goal_features]
    x_test = test[goal_features]
    base_result_x_train = train[base_result_features]
    base_result_x_test = test[base_result_features]
    result_x_train = train[result_features]
    result_x_test = test[result_features]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in goal_features]
    base_result_categorical = [
        feature for feature in CATEGORICAL_FEATURES if feature in base_result_features
    ]
    result_categorical = [feature for feature in CATEGORICAL_FEATURES if feature in result_features]
    train_weights = train["match_recency_weight"].astype(float).to_numpy()

    reg_params = {
        "objective": "regression",
        "n_estimators": 350,
        "learning_rate": 0.035,
        "num_leaves": 12,
        "max_depth": 4,
        "min_child_samples": 60,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "reg_alpha": 0.10,
        "reg_lambda": 1.0,
        "min_split_gain": 0.005,
        "random_state": 42,
        "verbosity": -1,
    }
    team_a_model = lgb.LGBMRegressor(**reg_params)
    team_b_model = lgb.LGBMRegressor(**reg_params)
    team_a_model.fit(
        x_train,
        train["team_a_goals"],
        sample_weight=train_weights,
        categorical_feature=categorical,
    )
    team_b_model.fit(
        x_train,
        train["team_b_goals"],
        sample_weight=train_weights,
        categorical_feature=categorical,
    )
    team_a_pred = np.clip(team_a_model.predict(x_test), 0, None)
    team_b_pred = np.clip(team_b_model.predict(x_test), 0, None)

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=12,
        max_depth=4,
        min_child_samples=60,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.10,
        reg_lambda=1.0,
        min_split_gain=0.005,
        random_state=42,
        verbosity=-1,
    )
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calibrated.fit(
        base_result_x_train,
        train["result_label"],
        sample_weight=train_weights,
        categorical_feature=base_result_categorical,
    )
    xg_classifier = CalibratedClassifierCV(
        lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=12,
            max_depth=4,
            min_child_samples=60,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.10,
            reg_lambda=1.0,
            min_split_gain=0.005,
            random_state=42,
            verbosity=-1,
        ),
        method="sigmoid",
        cv=3,
    )
    xg_classifier.fit(
        result_x_train,
        train["result_label"],
        sample_weight=train_weights,
        categorical_feature=result_categorical,
    )
    probabilities = blend_result_probabilities(
        calibrated.predict_proba(base_result_x_test),
        xg_classifier.predict_proba(result_x_test),
    )
    predicted_labels = probabilities.argmax(axis=1)
    actual_labels = test["result_label"].to_numpy()

    metrics = {
        "model": "lightgbm_neutral",
        "model_id": NEUTRAL_MODEL_RECIPE,
        "matches": int(len(frame)),
        "train_rows_after_augmentation": int(len(train)),
        "test_matches": int(len(test)),
        "features": len(goal_features),
        "goal_features": len(goal_features),
        "result_features": len(result_features),
        "xg_result_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
        "base_features_available": len(NEUTRAL_BASE_FEATURES),
        "recency_weight_min": round(float(train_weights.min()), 4),
        "recency_weight_max": round(float(train_weights.max()), 4),
        "recency_weight_mean": round(float(train_weights.mean()), 4),
        "mae_team_a_goals": round(float(mean_absolute_error(test["team_a_goals"], team_a_pred)), 4),
        "mae_team_b_goals": round(float(mean_absolute_error(test["team_b_goals"], team_b_pred)), 4),
        "result_accuracy": round(float(accuracy_score(actual_labels, predicted_labels)), 4),
        "log_loss": round(float(log_loss(actual_labels, probabilities, labels=[0, 1, 2])), 4),
    }
    output_dir = data_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "team_a_goals_model": team_a_model,
            "team_b_goals_model": team_b_model,
            "result_model": calibrated,
            "xg_result_model": xg_classifier,
            "features": goal_features,
            "team_a_goal_features": goal_features,
        "team_b_goal_features": goal_features,
        "result_features": base_result_features,
            "xg_result_features": result_features,
            "result_probability_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
            "model_id": NEUTRAL_MODEL_RECIPE,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
        },
        output_dir / "lightgbm_neutral_model.joblib",
    )
    payload = {
        "metrics": metrics,
        "features_list": goal_features,
        "goal_features": goal_features,
        "result_features": result_features,
        "team_a_goal_importances": sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(
                    goal_features, team_a_model.feature_importances_, strict=True
                )
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
        "team_b_goal_importances": sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(
                    goal_features, team_b_model.feature_importances_, strict=True
                )
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
        "result_classifier_importances": _calibrated_classifier_importances(
            calibrated,
            goal_features,
        ),
        "xg_result_classifier_importances": _calibrated_classifier_importances(
            xg_classifier,
            result_features,
        ),
    }
    (output_dir / "lightgbm_neutral_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions = test[
        [
            "competition_name",
            "competition_family",
            "competition_type",
            "stage_or_round",
            "team_a_goals",
            "team_b_goals",
            "result",
        ]
    ].copy()
    predictions["expected_team_a_goals"] = np.round(team_a_pred, 4)
    predictions["expected_team_b_goals"] = np.round(team_b_pred, 4)
    predictions["team_a_win_probability"] = np.round(probabilities[:, 0], 6)
    predictions["draw_probability"] = np.round(probabilities[:, 1], 6)
    predictions["team_b_win_probability"] = np.round(probabilities[:, 2], 6)
    predictions["predicted_result"] = [NEUTRAL_RESULT_LABELS[index] for index in predicted_labels]
    predictions.to_csv(
        data_root / "processed" / "combined" / "lightgbm_neutral_predictions.csv",
        index=False,
    )
    return metrics
