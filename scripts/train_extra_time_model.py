from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.extra_time_model import ExtraTimePredictor  # noqa: E402
from kinela.providers.statsbomb import MEN_EXTRA_TIME_TOURNAMENTS  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_rows(data_root: Path) -> list[dict[str, Any]]:
    raw = data_root / "raw" / "statsbomb"
    rows: list[dict[str, Any]] = []
    for competition_id, season_id, competition, season in MEN_EXTRA_TIME_TOURNAMENTS:
        matches = _read_json(raw / "matches" / str(competition_id) / f"{season_id}.json")
        for match in matches:
            stage = str((match.get("competition_stage") or {}).get("name") or "")
            if "group" in stage.casefold():
                continue
            match_id = int(match["match_id"])
            events = _read_json(raw / "events" / f"{match_id}.json")
            periods = {int(event.get("period") or 0) for event in events}
            if not periods.intersection({3, 4}):
                continue

            team_a = str(match["home_team"]["home_team_name"])
            team_b = str(match["away_team"]["away_team_name"])
            xg_90 = {team_a: 0.0, team_b: 0.0}
            extra_time_goals = {team_a: 0, team_b: 0}
            for event in events:
                period = int(event.get("period") or 0)
                event_type = str((event.get("type") or {}).get("name") or "")
                event_team = str((event.get("team") or {}).get("name") or "")
                shot = event.get("shot") or {}
                if period in {1, 2} and event_type == "Shot" and event_team in xg_90:
                    xg_90[event_team] += float(shot.get("statsbomb_xg") or 0.0)
                if period not in {3, 4}:
                    continue
                if (
                    event_type == "Shot"
                    and str((shot.get("outcome") or {}).get("name") or "") == "Goal"
                    and event_team in extra_time_goals
                ):
                    extra_time_goals[event_team] += 1
                elif event_type == "Own Goal Against" and event_team in extra_time_goals:
                    opponent = team_b if event_team == team_a else team_a
                    extra_time_goals[opponent] += 1

            rows.append(
                {
                    "match_id": match_id,
                    "match_date": match["match_date"],
                    "competition": competition,
                    "season": season,
                    "tournament": f"{competition} {season}",
                    "stage": stage,
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_xg_90": xg_90[team_a],
                    "team_b_xg_90": xg_90[team_b],
                    "team_a_extra_time_goals": extra_time_goals[team_a],
                    "team_b_extra_time_goals": extra_time_goals[team_b],
                }
            )
    return sorted(rows, key=lambda row: (row["match_date"], row["match_id"]))


def _poisson_nll(observed: int, expected: float) -> float:
    expected = max(expected, 1e-12)
    return expected - observed * math.log(expected) + math.lgamma(observed + 1)


def _split(total_lambda: float, xg_a: float, xg_b: float, exponent: float) -> tuple[float, float]:
    if exponent <= 0.0:
        return total_lambda / 2.0, total_lambda / 2.0
    strength_a = max(xg_a, 0.05) ** exponent
    strength_b = max(xg_b, 0.05) ** exponent
    share_a = strength_a / (strength_a + strength_b)
    return total_lambda * share_a, total_lambda * (1.0 - share_a)


def _cross_validate(rows: list[dict[str, Any]], exponent: float) -> dict[str, float]:
    total_nll = 0.0
    absolute_error = 0.0
    correct_outcomes = 0
    for held_out in sorted({str(row["tournament"]) for row in rows}):
        train = [row for row in rows if row["tournament"] != held_out]
        test = [row for row in rows if row["tournament"] == held_out]
        total_lambda = sum(
            int(row["team_a_extra_time_goals"]) + int(row["team_b_extra_time_goals"])
            for row in train
        ) / len(train)
        predictor = ExtraTimePredictor(
            {"total_goal_lambda": total_lambda, "allocation_exponent": exponent}
        )
        for row in test:
            expected_a, expected_b = _split(
                total_lambda,
                float(row["team_a_xg_90"]),
                float(row["team_b_xg_90"]),
                exponent,
            )
            observed_a = int(row["team_a_extra_time_goals"])
            observed_b = int(row["team_b_extra_time_goals"])
            total_nll += _poisson_nll(observed_a, expected_a)
            total_nll += _poisson_nll(observed_b, expected_b)
            absolute_error += abs(observed_a - expected_a) + abs(observed_b - expected_b)
            probabilities = predictor.outcome_probabilities(
                float(row["team_a_xg_90"]), float(row["team_b_xg_90"])
            )
            predicted = max(
                ("team_a", probabilities.team_a_win),
                ("draw", probabilities.draw),
                ("team_b", probabilities.team_b_win),
                key=lambda item: item[1],
            )[0]
            observed = (
                "team_a"
                if observed_a > observed_b
                else "team_b"
                if observed_b > observed_a
                else "draw"
            )
            correct_outcomes += int(predicted == observed)
    team_observations = 2 * len(rows)
    return {
        "poisson_nll_per_team": total_nll / team_observations,
        "goal_mae_per_team": absolute_error / team_observations,
        "three_way_outcome_accuracy": correct_outcomes / len(rows),
    }


def _bootstrap_delta(
    rows: list[dict[str, Any]], candidate_exponent: float, *, samples: int = 20_000
) -> dict[str, float]:
    total_lambda = sum(
        int(row["team_a_extra_time_goals"]) + int(row["team_b_extra_time_goals"])
        for row in rows
    ) / len(rows)
    deltas: list[float] = []
    for row in rows:
        observed = (
            int(row["team_a_extra_time_goals"]),
            int(row["team_b_extra_time_goals"]),
        )
        equal = (total_lambda / 2.0, total_lambda / 2.0)
        candidate = _split(
            total_lambda,
            float(row["team_a_xg_90"]),
            float(row["team_b_xg_90"]),
            candidate_exponent,
        )
        deltas.append(
            sum(_poisson_nll(goal, expected) for goal, expected in zip(observed, candidate))
            - sum(_poisson_nll(goal, expected) for goal, expected in zip(observed, equal))
        )
    rng = random.Random(42)
    bootstrap_means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples)
    )
    return {
        "mean_nll_delta_per_match_vs_equal": sum(deltas) / len(deltas),
        "ci95_low": bootstrap_means[int(samples * 0.025)],
        "ci95_high": bootstrap_means[int(samples * 0.975) - 1],
        "probability_candidate_better": sum(value < 0.0 for value in bootstrap_means)
        / samples,
    }


def train(data_root: Path) -> dict[str, Any]:
    rows = _extract_rows(data_root)
    if not rows:
        raise RuntimeError("No verified StatsBomb matches with periods 3/4 were found")

    processed = data_root / "processed" / "extra_time"
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / "statsbomb_extra_time_matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    candidates = {
        "equal_split": 0.0,
        "quarter_strength_split": 0.25,
        "half_strength_split": 0.5,
        "proportional_strength_split": 1.0,
    }
    backtest = {name: _cross_validate(rows, exponent) for name, exponent in candidates.items()}
    best_name = min(backtest, key=lambda name: backtest[name]["poisson_nll_per_team"])
    equal_nll = backtest["equal_split"]["poisson_nll_per_team"]
    best_improvement = (equal_nll - backtest[best_name]["poisson_nll_per_team"]) / equal_nll

    # A tiny win in 29 matches is not enough to create a team-strength effect.
    # Require a material out-of-tournament improvement and bootstrap evidence.
    bootstrap = _bootstrap_delta(rows, candidates[best_name]) if best_name != "equal_split" else {}
    selected_name = best_name
    if (
        best_name != "equal_split"
        and (best_improvement < 0.01 or bootstrap.get("ci95_high", 1.0) >= 0.0)
    ):
        selected_name = "equal_split"

    total_goals = sum(
        int(row["team_a_extra_time_goals"]) + int(row["team_b_extra_time_goals"])
        for row in rows
    )
    outcome_counts = Counter(
        "team_a"
        if int(row["team_a_extra_time_goals"]) > int(row["team_b_extra_time_goals"])
        else "team_b"
        if int(row["team_b_extra_time_goals"]) > int(row["team_a_extra_time_goals"])
        else "draw"
        for row in rows
    )
    artifact = {
        "schema_version": 1,
        "model": "independent_poisson_extra_time_goals",
        "selected_candidate": selected_name,
        "total_goal_lambda": total_goals / len(rows),
        "allocation_exponent": candidates[selected_name],
        "training_matches": len(rows),
        "training_extra_time_goals": total_goals,
        "training_tournaments": sorted({str(row["tournament"]) for row in rows}),
        "observed_outcomes": dict(outcome_counts),
        "validation": {
            "method": "leave-one-tournament-out",
            "candidates": backtest,
            "raw_best_candidate": best_name,
            "raw_best_relative_nll_improvement_vs_equal": best_improvement,
            "raw_best_bootstrap": bootstrap,
            "selection_rule": (
                "Use a strength split only with >=1% NLL improvement and a bootstrap 95% "
                "interval wholly below zero; otherwise retain the neutral equal split."
            ),
        },
        "data_policy": {
            "regulation_periods": [1, 2],
            "extra_time_periods": [3, 4],
            "shootout_period": 5,
            "source": "StatsBomb Open Data",
            "attribution": "Data provided by StatsBomb",
            "api_football_rows_used": 0,
            "api_football_exclusion_reason": (
                "Fixture-level status is not sufficient to distinguish true extra time from "
                "competitions that go directly to penalties; event periods are required."
            ),
        },
    }
    artifact_path = data_root / "static" / "extra_time_model.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neutral extra-time model")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(train(args.data_root), indent=2))


if __name__ == "__main__":
    main()
