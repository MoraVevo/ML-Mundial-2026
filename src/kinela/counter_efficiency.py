from __future__ import annotations

import math
from typing import Any


PRIOR_GOALS_PER_POSSESSION_UNIT = 1.0
PRIOR_GOALS_PER_SHOT_ON_GOAL = 0.30
PRIOR_GOALS_PER_SHOT = 0.10
PRIOR_POINTS_RATE = 0.40
COUNTER_EFFICIENCY_WINDOW = 20


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _weighted_component(values: list[tuple[float, float | None]]) -> tuple[float, float]:
    usable = [(weight, value) for weight, value in values if value is not None]
    if not usable:
        return 0.0, 0.0
    total_weight = sum(weight for weight, _ in usable)
    return (
        sum(weight * float(value) for weight, value in usable) / total_weight,
        total_weight / sum(weight for weight, _ in values),
    )


def _relative_signal(rate: float, prior: float) -> float:
    return math.tanh(math.log(max(rate, 1e-6) / prior))


def counter_match_profile(item: dict[str, Any]) -> dict[str, float]:
    possession = _optional_float(item.get("ball_possession_pct"))
    total_passes = _optional_float(item.get("total_passes"))
    total_shots = _optional_float(item.get("total_shots"))
    shots_on_goal = _optional_float(item.get("shots_on_goal"))
    passes_pct = _optional_float(item.get("passes_pct"))
    team_elo = _optional_float(item.get("team_elo")) or 1500.0
    opponent_elo = _optional_float(item.get("opponent_elo")) or 1500.0

    low_possession = (
        _clip01((48.0 - possession) / 18.0) if possession is not None else None
    )
    low_pass_volume = (
        _clip01((420.0 - total_passes) / 220.0) if total_passes is not None else None
    )
    application_score, style_coverage = _weighted_component(
        [(0.75, low_possession), (0.25, low_pass_volume)]
    )
    attack_coverage = sum(value is not None for value in (total_shots, shots_on_goal)) / 2.0
    coverage = 0.65 * style_coverage + 0.35 * attack_coverage
    opponent_quality_weight = max(
        0.70,
        min(1.30, 1.0 + (opponent_elo - team_elo) / 500.0),
    )
    applied = float(
        coverage >= 0.60
        and possession is not None
        and possession <= 47.0
        and application_score >= 0.20
    )
    effective_weight = applied * application_score * opponent_quality_weight
    retention_quality = (
        math.tanh((passes_pct - 78.0) / 12.0) if passes_pct is not None else 0.0
    )
    return {
        "counter_application_score": application_score,
        "counter_profile_coverage": coverage,
        "counter_applied": applied,
        "counter_effective_weight": effective_weight,
        "counter_opponent_quality_weight": opponent_quality_weight,
        "counter_retention_quality": retention_quality,
    }


def counter_history_summary(
    history: list[dict[str, Any]],
    *,
    window: int,
) -> dict[str, float]:
    recent = history[-window:]
    covered_count = 0
    applied_count = 0
    effective_matches = 0.0
    weighted_goals = 0.0
    weighted_points = 0.0
    weighted_possession_units = 0.0
    weighted_shots_on_goal = 0.0
    weighted_shots = 0.0
    weighted_retention = 0.0
    retention_weight = 0.0

    for item in recent:
        profile = counter_match_profile(item)
        if profile["counter_profile_coverage"] >= 0.60:
            covered_count += 1
        if not profile["counter_applied"]:
            continue

        applied_count += 1
        weight = profile["counter_effective_weight"]
        effective_matches += weight
        goals = _optional_float(item.get("goals_for")) or 0.0
        points = _optional_float(item.get("points")) or 0.0
        possession = _optional_float(item.get("ball_possession_pct")) or 50.0
        shots_on_goal = _optional_float(item.get("shots_on_goal"))
        total_shots = _optional_float(item.get("total_shots"))
        passes_pct = _optional_float(item.get("passes_pct"))

        weighted_goals += weight * goals
        weighted_points += weight * points
        weighted_possession_units += weight * max(possession / 50.0, 0.25)
        if shots_on_goal is not None:
            weighted_shots_on_goal += weight * max(shots_on_goal, 0.0)
        if total_shots is not None:
            weighted_shots += weight * max(total_shots, 0.0)
        if passes_pct is not None:
            weighted_retention += weight * profile["counter_retention_quality"]
            retention_weight += weight

    application_rate = applied_count / covered_count if covered_count else 0.0
    sample_strength = (
        math.log1p(effective_matches) / math.log1p(max(window, 1))
        if effective_matches
        else 0.0
    )
    goals_per_possession = (
        weighted_goals + 2.0 * PRIOR_GOALS_PER_POSSESSION_UNIT
    ) / (weighted_possession_units + 2.0)
    goals_per_shot_on_goal = (
        weighted_goals + 3.0 * PRIOR_GOALS_PER_SHOT_ON_GOAL
    ) / (weighted_shots_on_goal + 3.0)
    goals_per_shot = (
        weighted_goals + 8.0 * PRIOR_GOALS_PER_SHOT
    ) / (weighted_shots + 8.0)
    productive_possession_signal = _relative_signal(
        goals_per_possession,
        PRIOR_GOALS_PER_POSSESSION_UNIT,
    )
    shot_on_goal_conversion_signal = _relative_signal(
        goals_per_shot_on_goal,
        PRIOR_GOALS_PER_SHOT_ON_GOAL,
    )
    shot_conversion_signal = _relative_signal(
        goals_per_shot,
        PRIOR_GOALS_PER_SHOT,
    )
    scoring_efficiency = (
        0.45 * productive_possession_signal
        + 0.35 * shot_on_goal_conversion_signal
        + 0.20 * shot_conversion_signal
    )
    retention_signal = (
        weighted_retention / retention_weight if retention_weight else 0.0
    )
    productive_retention = 0.85 * scoring_efficiency + 0.15 * retention_signal
    points_rate = (
        weighted_points + 2.0 * 3.0 * PRIOR_POINTS_RATE
    ) / (3.0 * (effective_matches + 2.0))
    result_efficiency = math.tanh((points_rate - PRIOR_POINTS_RATE) / 0.25)
    habit_strength = application_rate * sample_strength
    reliable_scoring = productive_retention * habit_strength
    reliable_success = (
        0.75 * productive_retention + 0.25 * result_efficiency
    ) * habit_strength

    return {
        "counter_covered_matches": float(covered_count),
        "counter_applied_matches": float(applied_count),
        "counter_effective_matches": effective_matches,
        "counter_application_rate": application_rate,
        "counter_sample_strength": sample_strength,
        "counter_goals_per_possession": goals_per_possession,
        "counter_goals_per_shot_on_goal": goals_per_shot_on_goal,
        "counter_goals_per_shot": goals_per_shot,
        "counter_productive_possession": productive_possession_signal,
        "counter_shot_on_goal_conversion": shot_on_goal_conversion_signal,
        "counter_shot_conversion": shot_conversion_signal,
        "counter_scoring_efficiency": scoring_efficiency,
        "counter_retention_signal": retention_signal,
        "counter_productive_retention": productive_retention,
        "counter_points_rate": points_rate,
        "counter_result_efficiency": result_efficiency,
        "counter_habit_strength": habit_strength,
        "counter_reliable_scoring": reliable_scoring,
        "counter_reliable_success": reliable_success,
    }


def current_underdog_fit(team_elo: Any, opponent_elo: Any) -> float:
    team = _optional_float(team_elo) or 1500.0
    opponent = _optional_float(opponent_elo) or 1500.0
    return _clip01((opponent - team + 25.0) / 225.0)


def current_counter_threat(summary: dict[str, float], underdog_fit: float) -> float:
    positive_efficiency = _clip01(
        0.5 + 0.5 * summary["counter_productive_retention"]
    )
    return (
        underdog_fit
        * summary["counter_habit_strength"]
        * positive_efficiency
    )
