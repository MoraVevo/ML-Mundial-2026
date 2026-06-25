from __future__ import annotations

import math
from typing import Any


CLINICAL_FINISHING_WINDOW = 12


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _relative_signal(rate: float, prior: float) -> float:
    return math.tanh(math.log(max(rate, 1e-6) / prior))


def _weighted_available(
    values: list[tuple[float, float | None]],
) -> tuple[float, float]:
    observed = [(weight, value) for weight, value in values if value is not None]
    if not observed:
        return 0.0, 0.0
    observed_weight = sum(weight for weight, _ in observed)
    return (
        sum(weight * float(value) for weight, value in observed) / observed_weight,
        observed_weight / sum(weight for weight, _ in values),
    )


def _match_low_block(item: dict[str, Any]) -> tuple[float, float]:
    possession = optional_float(item.get("ball_possession_pct"))
    total_passes = optional_float(item.get("total_passes"))
    passes_pct = optional_float(item.get("passes_pct"))
    total_shots = optional_float(item.get("total_shots"))
    shots_on_goal = optional_float(item.get("shots_on_goal"))
    corners = optional_float(item.get("corner_kicks"))
    return _weighted_available(
        [
            (
                0.32,
                _clip01((50.0 - possession) / 35.0)
                if possession is not None
                else None,
            ),
            (
                0.20,
                1.0 - math.tanh(total_shots / 13.0)
                if total_shots is not None
                else None,
            ),
            (
                0.18,
                1.0 - math.tanh(shots_on_goal / 4.0)
                if shots_on_goal is not None
                else None,
            ),
            (
                0.14,
                1.0 - math.tanh(corners / 5.0)
                if corners is not None
                else None,
            ),
            (
                0.10,
                1.0 - math.tanh(total_passes / 500.0)
                if total_passes is not None
                else None,
            ),
            (
                0.06,
                1.0 - math.tanh(passes_pct / 85.0)
                if passes_pct is not None
                else None,
            ),
        ]
    )


def clinical_finishing_summary(
    history: list[dict[str, Any]],
    *,
    window: int = CLINICAL_FINISHING_WINDOW,
) -> dict[str, float]:
    recent = history[-window:]
    covered_sot = 0
    covered_shots = 0
    sot_total = 0.0
    shots_total = 0.0
    goals_with_sot = 0.0
    goals_with_shots = 0.0
    low_block_weight = 0.0
    low_block_total = 0.0

    for item in recent:
        goals = max(0.0, optional_float(item.get("goals_for")) or 0.0)
        shots_on_goal = optional_float(item.get("shots_on_goal"))
        total_shots = optional_float(item.get("total_shots"))
        if shots_on_goal is not None:
            covered_sot += 1
            sot = max(0.0, shots_on_goal)
            sot_total += sot
            goals_with_sot += min(goals, sot)
        if total_shots is not None:
            covered_shots += 1
            shots = max(0.0, total_shots)
            shots_total += shots
            goals_with_shots += min(
                goals,
                shots,
                max(0.0, shots_on_goal)
                if shots_on_goal is not None
                else goals,
            )
        low_block, coverage = _match_low_block(item)
        low_block_total += low_block * coverage
        low_block_weight += coverage

    sot_conversion = (goals_with_sot + 8.0 * 0.30) / (sot_total + 8.0)
    shot_accuracy = (sot_total + 20.0 * 0.33) / (shots_total + 20.0)
    goal_per_shot = (goals_with_shots + 20.0 * 0.10) / (shots_total + 20.0)
    sot_coverage = covered_sot / max(window, 1)
    shot_coverage = covered_shots / max(window, 1)
    sample_strength = min(
        1.0,
        math.log1p(sot_total + 0.25 * shots_total) / math.log1p(30.0),
    ) * math.sqrt(min(sot_coverage, shot_coverage))
    conversion_signal = _relative_signal(sot_conversion, 0.30)
    accuracy_signal = _relative_signal(shot_accuracy, 0.33)
    goal_shot_signal = _relative_signal(goal_per_shot, 0.10)
    clinical_signal = sample_strength * (
        0.60 * conversion_signal
        + 0.25 * accuracy_signal
        + 0.15 * goal_shot_signal
    )
    return {
        "clinical_signal": clinical_signal,
        "clinical_coverage": min(sot_coverage, shot_coverage),
        "low_block_profile": (
            low_block_total / low_block_weight if low_block_weight else 0.0
        ),
        "low_block_coverage": min(1.0, low_block_weight / max(window, 1)),
    }
