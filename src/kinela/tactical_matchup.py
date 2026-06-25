from __future__ import annotations

import math
from typing import Any

from kinela.counter_efficiency import (
    _clip01,
    _optional_float,
    _weighted_component,
    counter_match_profile,
    current_underdog_fit,
)


TACTICAL_PROFILE_WINDOW = 20
PRIOR_LOW_BLOCK_NON_LOSS_RATE = 0.42
PRIOR_LOW_BLOCK_CONTAINMENT_RATE = 0.58
PRIOR_POSSESSION_WIN_RATE = 0.52
PRIOR_POSSESSION_GOALS = 1.45


def _sample_strength(effective_matches: float, window: int) -> float:
    if effective_matches <= 0.0:
        return 0.0
    return math.log1p(effective_matches) / math.log1p(max(window, 1))


def _high_possession_profile(item: dict[str, Any]) -> dict[str, float]:
    possession = _optional_float(item.get("ball_possession_pct"))
    total_passes = _optional_float(item.get("total_passes"))
    total_shots = _optional_float(item.get("total_shots"))
    shots_on_goal = _optional_float(item.get("shots_on_goal"))
    team_elo = _optional_float(item.get("team_elo")) or 1500.0
    opponent_elo = _optional_float(item.get("opponent_elo")) or 1500.0

    high_possession = (
        _clip01((possession - 52.0) / 18.0) if possession is not None else None
    )
    high_pass_volume = (
        _clip01((total_passes - 420.0) / 260.0) if total_passes is not None else None
    )
    application_score, style_coverage = _weighted_component(
        [(0.75, high_possession), (0.25, high_pass_volume)]
    )
    attack_coverage = sum(value is not None for value in (total_shots, shots_on_goal)) / 2.0
    coverage = 0.65 * style_coverage + 0.35 * attack_coverage
    favorite_quality_weight = max(
        0.75,
        min(1.25, 1.0 + (team_elo - opponent_elo) / 600.0),
    )
    applied = float(
        coverage >= 0.60
        and possession is not None
        and possession >= 53.0
        and application_score >= 0.15
    )
    return {
        "possession_application_score": application_score,
        "possession_profile_coverage": coverage,
        "possession_applied": applied,
        "possession_effective_weight": applied
        * application_score
        * favorite_quality_weight,
    }


def tactical_history_summary(
    history: list[dict[str, Any]],
    *,
    window: int = TACTICAL_PROFILE_WINDOW,
) -> dict[str, float]:
    recent = history[-window:]
    low_block_covered = 0
    low_block_applied = 0
    low_block_weight = 0.0
    low_block_points = 0.0
    low_block_non_losses = 0.0
    low_block_conceded = 0.0
    low_block_contained = 0.0
    low_block_clean_sheets = 0.0

    possession_covered = 0
    possession_applied = 0
    possession_weight = 0.0
    possession_goals = 0.0
    possession_points = 0.0
    possession_wins = 0.0
    possession_shots_on_goal = 0.0
    possession_conceded = 0.0
    possession_failed_wins = 0.0

    for item in recent:
        goals_for = _optional_float(item.get("goals_for")) or 0.0
        goals_against = _optional_float(item.get("goals_against")) or 0.0
        points = _optional_float(item.get("points")) or 0.0

        low_block = counter_match_profile(item)
        if low_block["counter_profile_coverage"] >= 0.60:
            low_block_covered += 1
        if low_block["counter_applied"]:
            low_block_applied += 1
            weight = low_block["counter_effective_weight"]
            low_block_weight += weight
            low_block_points += weight * points
            low_block_non_losses += weight * float(points >= 1.0)
            low_block_conceded += weight * goals_against
            low_block_contained += weight * float(goals_against <= 1.0)
            low_block_clean_sheets += weight * float(goals_against == 0.0)

        possession = _high_possession_profile(item)
        if possession["possession_profile_coverage"] >= 0.60:
            possession_covered += 1
        if possession["possession_applied"]:
            possession_applied += 1
            weight = possession["possession_effective_weight"]
            possession_weight += weight
            possession_goals += weight * goals_for
            possession_points += weight * points
            possession_wins += weight * float(points == 3.0)
            possession_conceded += weight * goals_against
            possession_failed_wins += weight * float(points < 3.0)
            shots_on_goal = _optional_float(item.get("shots_on_goal"))
            if shots_on_goal is not None:
                possession_shots_on_goal += weight * shots_on_goal

    low_block_application_rate = (
        low_block_applied / low_block_covered if low_block_covered else 0.0
    )
    low_block_sample = _sample_strength(low_block_weight, window)
    low_block_non_loss_rate = (
        low_block_non_losses + 2.0 * PRIOR_LOW_BLOCK_NON_LOSS_RATE
    ) / (low_block_weight + 2.0)
    low_block_containment_rate = (
        low_block_contained + 2.0 * PRIOR_LOW_BLOCK_CONTAINMENT_RATE
    ) / (low_block_weight + 2.0)
    low_block_clean_sheet_rate = (
        low_block_clean_sheets + 2.0 * 0.22
    ) / (low_block_weight + 2.0)
    low_block_goals_against = (
        low_block_conceded + 2.0 * 1.20
    ) / (low_block_weight + 2.0)
    low_block_result_rate = (
        low_block_points + 2.0 * 3.0 * 0.38
    ) / (3.0 * (low_block_weight + 2.0))
    low_block_survival_quality = _clip01(
        0.36 * low_block_non_loss_rate
        + 0.32 * low_block_containment_rate
        + 0.16 * low_block_clean_sheet_rate
        + 0.16 * _clip01((1.8 - low_block_goals_against) / 1.8)
    )
    low_block_habit = low_block_application_rate * low_block_sample
    low_block_reliable_survival = low_block_survival_quality * low_block_habit

    possession_application_rate = (
        possession_applied / possession_covered if possession_covered else 0.0
    )
    possession_sample = _sample_strength(possession_weight, window)
    possession_goals_rate = (
        possession_goals + 2.0 * PRIOR_POSSESSION_GOALS
    ) / (possession_weight + 2.0)
    possession_win_rate = (
        possession_wins + 2.0 * PRIOR_POSSESSION_WIN_RATE
    ) / (possession_weight + 2.0)
    possession_points_rate = (
        possession_points + 2.0 * 3.0 * 0.52
    ) / (3.0 * (possession_weight + 2.0))
    possession_shot_creation = (
        possession_shots_on_goal + 2.0 * 4.2
    ) / (possession_weight + 2.0)
    possession_goals_against = (
        possession_conceded + 2.0 * 1.05
    ) / (possession_weight + 2.0)
    possession_failed_win_rate = (
        possession_failed_wins + 2.0 * (1.0 - PRIOR_POSSESSION_WIN_RATE)
    ) / (possession_weight + 2.0)
    possession_productivity_quality = _clip01(
        0.36 * _clip01(possession_goals_rate / 2.4)
        + 0.28 * possession_win_rate
        + 0.20 * possession_points_rate
        + 0.16 * _clip01(possession_shot_creation / 6.5)
    )
    possession_vulnerability_quality = _clip01(
        0.58 * possession_failed_win_rate
        + 0.42 * _clip01(possession_goals_against / 2.0)
    )
    possession_habit = possession_application_rate * possession_sample
    possession_reliable_productivity = possession_productivity_quality * possession_habit
    possession_reliable_vulnerability = (
        possession_vulnerability_quality * possession_habit
    )

    return {
        "low_block_covered_matches": float(low_block_covered),
        "low_block_applied_matches": float(low_block_applied),
        "low_block_application_rate": low_block_application_rate,
        "low_block_sample_strength": low_block_sample,
        "low_block_non_loss_rate": low_block_non_loss_rate,
        "low_block_result_rate": low_block_result_rate,
        "low_block_survival_quality": low_block_survival_quality,
        "low_block_habit_strength": low_block_habit,
        "low_block_reliable_survival": low_block_reliable_survival,
        "possession_covered_matches": float(possession_covered),
        "possession_applied_matches": float(possession_applied),
        "possession_application_rate": possession_application_rate,
        "possession_sample_strength": possession_sample,
        "possession_goals_rate": possession_goals_rate,
        "possession_win_rate": possession_win_rate,
        "possession_points_rate": possession_points_rate,
        "possession_productivity_quality": possession_productivity_quality,
        "possession_vulnerability_quality": possession_vulnerability_quality,
        "possession_habit_strength": possession_habit,
        "possession_reliable_productivity": possession_reliable_productivity,
        "possession_reliable_vulnerability": possession_reliable_vulnerability,
    }


def current_favorite_fit(team_elo: Any, opponent_elo: Any) -> float:
    return current_underdog_fit(opponent_elo, team_elo)


def tactical_matchup_features(
    team_a: dict[str, float],
    team_b: dict[str, float],
    *,
    team_a_elo: Any,
    team_b_elo: Any,
    team_a_counter_threat: float,
    team_b_counter_threat: float,
) -> dict[str, float]:
    a_low_block_likelihood = current_underdog_fit(team_a_elo, team_b_elo) * (
        0.55 + 0.45 * team_a["low_block_application_rate"]
    )
    b_low_block_likelihood = current_underdog_fit(team_b_elo, team_a_elo) * (
        0.55 + 0.45 * team_b["low_block_application_rate"]
    )
    a_possession_likelihood = current_favorite_fit(team_a_elo, team_b_elo) * (
        0.55 + 0.45 * team_a["possession_application_rate"]
    )
    b_possession_likelihood = current_favorite_fit(team_b_elo, team_a_elo) * (
        0.55 + 0.45 * team_b["possession_application_rate"]
    )

    a_current_survival = (
        a_low_block_likelihood * team_a["low_block_reliable_survival"]
    )
    b_current_survival = (
        b_low_block_likelihood * team_b["low_block_reliable_survival"]
    )
    a_current_productivity = (
        a_possession_likelihood * team_a["possession_reliable_productivity"]
    )
    b_current_productivity = (
        b_possession_likelihood * team_b["possession_reliable_productivity"]
    )
    a_possession_goal_productivity = (
        a_possession_likelihood
        * team_a["possession_habit_strength"]
        * _clip01(team_a["possession_goals_rate"] / 2.4)
    )
    b_possession_goal_productivity = (
        b_possession_likelihood
        * team_b["possession_habit_strength"]
        * _clip01(team_b["possession_goals_rate"] / 2.4)
    )
    a_possession_win_productivity = (
        a_possession_likelihood
        * team_a["possession_habit_strength"]
        * team_a["possession_win_rate"]
    )
    b_possession_win_productivity = (
        b_possession_likelihood
        * team_b["possession_habit_strength"]
        * team_b["possession_win_rate"]
    )
    a_low_block_result_survival = (
        a_low_block_likelihood
        * team_a["low_block_habit_strength"]
        * team_a["low_block_result_rate"]
    )
    b_low_block_result_survival = (
        b_low_block_likelihood
        * team_b["low_block_habit_strength"]
        * team_b["low_block_result_rate"]
    )

    a_breaks_b_block = (
        a_current_productivity
        * b_low_block_likelihood
        * (1.0 - team_b["low_block_survival_quality"])
    )
    b_breaks_a_block = (
        b_current_productivity
        * a_low_block_likelihood
        * (1.0 - team_a["low_block_survival_quality"])
    )
    a_counter_exploits_b = (
        team_a_counter_threat
        * b_possession_likelihood
        * team_b["possession_reliable_vulnerability"]
    )
    b_counter_exploits_a = (
        team_b_counter_threat
        * a_possession_likelihood
        * team_a["possession_reliable_vulnerability"]
    )
    a_stalemate = (
        a_current_survival
        * b_possession_likelihood
        * (1.0 - team_b["possession_productivity_quality"])
    )
    b_stalemate = (
        b_current_survival
        * a_possession_likelihood
        * (1.0 - team_a["possession_productivity_quality"])
    )

    block_breaking_edge = a_breaks_b_block - b_breaks_a_block
    counter_dominance_edge = a_counter_exploits_b - b_counter_exploits_a
    compatibility_v2_edge = (
        0.56 * block_breaking_edge + 0.44 * counter_dominance_edge
    )
    return {
        "team_a_low_block_likelihood": a_low_block_likelihood,
        "team_b_low_block_likelihood": b_low_block_likelihood,
        "team_a_possession_likelihood": a_possession_likelihood,
        "team_b_possession_likelihood": b_possession_likelihood,
        "low_block_survival_edge": a_current_survival - b_current_survival,
        "low_block_result_survival_edge": a_low_block_result_survival
        - b_low_block_result_survival,
        "possession_productivity_edge": a_current_productivity
        - b_current_productivity,
        "possession_goal_productivity_edge": a_possession_goal_productivity
        - b_possession_goal_productivity,
        "possession_win_productivity_edge": a_possession_win_productivity
        - b_possession_win_productivity,
        "block_breaking_matchup_edge": block_breaking_edge,
        "counter_dominance_matchup_edge": counter_dominance_edge,
        "compatibility_v2_edge": compatibility_v2_edge,
        "tactical_stalemate_pressure": _clip01(a_stalemate + b_stalemate),
        "tactical_matchup_intensity": _clip01(
            a_breaks_b_block
            + b_breaks_a_block
            + a_counter_exploits_b
            + b_counter_exploits_a
        ),
    }
