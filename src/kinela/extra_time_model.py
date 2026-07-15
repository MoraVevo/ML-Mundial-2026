from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtraTimeOutcomeProbabilities:
    team_a_win: float
    draw: float
    team_b_win: float


class ExtraTimePredictor:
    """Small, neutral extra-time goal model loaded from a portable artifact.

    The model is deliberately separate from the 90-minute model.  Its output is
    the number of *additional* goals scored in periods 3 and 4 only.
    """

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.total_goal_lambda = max(0.0, float(artifact["total_goal_lambda"]))
        self.allocation_exponent = max(0.0, float(artifact.get("allocation_exponent", 0.0)))

    def expected_goals(
        self,
        team_a_90_xg: float | None = None,
        team_b_90_xg: float | None = None,
    ) -> tuple[float, float]:
        """Return extra-time-only xG, optionally using a validated strength split."""

        if (
            self.allocation_exponent <= 0.0
            or team_a_90_xg is None
            or team_b_90_xg is None
        ):
            return self.total_goal_lambda / 2.0, self.total_goal_lambda / 2.0

        strength_a = max(float(team_a_90_xg), 0.05) ** self.allocation_exponent
        strength_b = max(float(team_b_90_xg), 0.05) ** self.allocation_exponent
        share_a = strength_a / (strength_a + strength_b)
        return self.total_goal_lambda * share_a, self.total_goal_lambda * (1.0 - share_a)

    def outcome_probabilities(
        self,
        team_a_90_xg: float | None = None,
        team_b_90_xg: float | None = None,
        *,
        max_goals: int = 12,
    ) -> ExtraTimeOutcomeProbabilities:
        lambda_a, lambda_b = self.expected_goals(team_a_90_xg, team_b_90_xg)
        probabilities_a = [_poisson_probability(goals, lambda_a) for goals in range(max_goals + 1)]
        probabilities_b = [_poisson_probability(goals, lambda_b) for goals in range(max_goals + 1)]
        mass = sum(probabilities_a) * sum(probabilities_b)
        team_a_win = sum(
            probabilities_a[a_goals] * probabilities_b[b_goals]
            for a_goals in range(max_goals + 1)
            for b_goals in range(max_goals + 1)
            if a_goals > b_goals
        )
        draw = sum(
            probabilities_a[goals] * probabilities_b[goals]
            for goals in range(max_goals + 1)
        )
        if mass <= 0.0:
            return ExtraTimeOutcomeProbabilities(0.0, 1.0, 0.0)
        team_a_win /= mass
        draw /= mass
        team_b_win = max(0.0, 1.0 - team_a_win - draw)
        return ExtraTimeOutcomeProbabilities(team_a_win, draw, team_b_win)

    def team_a_advance_probability(
        self,
        penalty_probability_a: float,
        team_a_90_xg: float | None = None,
        team_b_90_xg: float | None = None,
    ) -> float:
        outcomes = self.outcome_probabilities(team_a_90_xg, team_b_90_xg)
        return outcomes.team_a_win + outcomes.draw * min(
            1.0,
            max(0.0, float(penalty_probability_a)),
        )


def _poisson_probability(goals: int, expected_goals: float) -> float:
    if expected_goals == 0.0:
        return 1.0 if goals == 0 else 0.0
    return math.exp(-expected_goals + goals * math.log(expected_goals) - math.lgamma(goals + 1))
