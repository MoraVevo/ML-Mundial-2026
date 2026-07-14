from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.worldcup_2026 import WorldCup2026Simulator  # noqa: E402


def _manual_worldcup_shootouts(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if not row.get("team_a_penalty_goals") or not row.get("team_b_penalty_goals"):
            continue
        rows.append(
            {
                "date": date.fromisoformat(row["date"]),
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "team_a_penalty_goals": float(row["team_a_penalty_goals"]),
                "team_b_penalty_goals": float(row["team_b_penalty_goals"]),
                "penalty_winner": row.get("penalty_winner", ""),
            }
        )
    return rows


def _loaded_shootouts(simulator: WorldCup2026Simulator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in simulator.penalty_shootouts:
        rows.append(
            {
                "date": row["date"],
                "source": row["source"],
                "competition_name": row["competition_name"],
                "team_a": row["home_team"],
                "team_b": row["away_team"],
                "team_a_penalty_goals": float(row["home_penalty_goals"]),
                "team_b_penalty_goals": float(row["away_penalty_goals"]),
                "penalty_winner": (
                    row["home_team"]
                    if float(row["home_penalty_goals"]) > float(row["away_penalty_goals"])
                    else row["away_team"]
                ),
            }
        )
    return rows


def _evaluate_rows(
    simulator: WorldCup2026Simulator,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    decisive_correct = 0
    ties = 0
    for row in rows:
        probability_a = simulator.penalty_model.team_a_probability(
            row["team_a"],
            row["team_b"],
            row["date"],
        )
        tied = math.isclose(probability_a, 0.5, abs_tol=1e-12)
        predicted_winner = (
            None if tied else row["team_a"] if probability_a > 0.5 else row["team_b"]
        )
        actual_winner = row["penalty_winner"]
        is_correct = None if tied else predicted_winner == actual_winner
        ties += int(tied)
        decisive_correct += int(bool(is_correct))
        details.append(
            {
                "date": row["date"].isoformat(),
                "source": row["source"],
                "competition_name": row["competition_name"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "team_a_penalty_goals": row["team_a_penalty_goals"],
                "team_b_penalty_goals": row["team_b_penalty_goals"],
                "actual_winner": actual_winner,
                "predicted_winner": predicted_winner,
                "correct": int(is_correct) if is_correct is not None else None,
                "prob_team_a": round(float(probability_a), 6),
                "prob_team_b": round(float(1.0 - probability_a), 6),
            }
        )
    matches = len(rows)
    expected_correct = decisive_correct + 0.5 * ties
    return (
        {
            "matches": matches,
            "decisive_predictions": matches - ties,
            "unresolved_50_50_predictions": ties,
            "expected_correct": expected_correct,
            "accuracy": round(expected_correct / matches, 4) if matches else None,
            "avg_favorite_probability": round(
                sum(max(item["prob_team_a"], item["prob_team_b"]) for item in details) / matches,
                4,
            )
            if matches
            else None,
        },
        details,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the production penalty shootout model.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/worldcup2026_penalty_shootout_model_audit.json"),
    )
    args = parser.parse_args()

    simulator = WorldCup2026Simulator(args.data_root, seed=42, engine="lightgbm")
    artifact_path = args.data_root / "static" / "penalty_shootout_model.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    temporal_report_path = Path("outputs/penalty_shootout_model_evaluation.json")
    temporal_report = (
        json.loads(temporal_report_path.read_text(encoding="utf-8"))
        if temporal_report_path.exists()
        else None
    )
    historical_rows = _loaded_shootouts(simulator)
    manual_rows = _manual_worldcup_shootouts(args.data_root)
    combined_rows = sorted([*historical_rows, *manual_rows], key=lambda row: row["date"])

    historical_summary, historical_details = _evaluate_rows(simulator, historical_rows)
    manual_summary, manual_details = _evaluate_rows(simulator, manual_rows)
    combined_summary, combined_details = _evaluate_rows(simulator, combined_rows)

    payload = {
        "model": artifact["model_id"],
        "implementation": (
            "Neutral and symmetric model selected against an explicit 50/50 candidate on "
            "pre-2018 temporal folds. The production artifact keeps 50/50 because every "
            "fitted candidate had worse out-of-sample probabilistic scores."
        ),
        "active_features": artifact["features"],
        "primary_temporal_evaluation": temporal_report,
        "historical_loaded_shootouts_in_sample_diagnostic": historical_summary,
        "worldcup_2026_manual_shootouts_diagnostic": manual_summary,
        "combined_in_sample_diagnostic": combined_summary,
        "details": combined_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_name(f"{args.output.stem}_matches.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_details[0].keys()))
        writer.writeheader()
        writer.writerows(combined_details)
    print(json.dumps({"summary_path": str(args.output), "matches_path": str(csv_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
