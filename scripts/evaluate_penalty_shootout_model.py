from __future__ import annotations

import argparse
import csv
import json
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
    correct = 0
    for row in rows:
        probability_a = simulator.penalty_model.team_a_probability(
            row["team_a"],
            row["team_b"],
            row["date"],
        )
        predicted_winner = row["team_a"] if probability_a >= 0.5 else row["team_b"]
        actual_winner = row["penalty_winner"]
        is_correct = predicted_winner == actual_winner
        correct += int(is_correct)
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
                "correct": int(is_correct),
                "prob_team_a": round(float(probability_a), 6),
                "prob_team_b": round(float(1.0 - probability_a), 6),
            }
        )
    matches = len(rows)
    return (
        {
            "matches": matches,
            "correct": correct,
            "accuracy": round(correct / matches, 4) if matches else None,
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
    parser = argparse.ArgumentParser(description="Audit the World Cup penalty shootout fallback model.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/worldcup2026_penalty_shootout_model_audit.json"),
    )
    args = parser.parse_args()

    simulator = WorldCup2026Simulator(args.data_root, seed=42, engine="lightgbm")
    historical_rows = _loaded_shootouts(simulator)
    manual_rows = _manual_worldcup_shootouts(args.data_root)
    combined_rows = sorted([*historical_rows, *manual_rows], key=lambda row: row["date"])

    historical_summary, historical_details = _evaluate_rows(simulator, historical_rows)
    manual_summary, manual_details = _evaluate_rows(simulator, manual_rows)
    combined_summary, combined_details = _evaluate_rows(simulator, combined_rows)

    payload = {
        "model": "PenaltyShootoutModel fallback",
        "implementation": (
            "Uses only squad-quality difference from football-data squad profiles, passed "
            "through a logistic function and clipped to 38%-62%. Loaded shootout history is "
            "available in the simulator but is not currently used by team_a_probability."
        ),
        "active_features": [
            "squad_top11_competition_strength",
            "squad_depth_competition_strength",
            "squad_top5_competition_strength",
            "known_strength_share",
            "squad_elite_competition_share",
            "player_availability_penalty",
        ],
        "inactive_loaded_context": [
            "historical_penalty_shootout_scored_minus_conceded_last3",
            "head_to_head_penalty_wins",
        ],
        "historical_loaded_shootouts": historical_summary,
        "worldcup_2026_manual_shootouts": manual_summary,
        "combined_diagnostic": combined_summary,
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
