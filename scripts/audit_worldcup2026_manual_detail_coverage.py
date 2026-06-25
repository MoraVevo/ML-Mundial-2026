from __future__ import annotations

import csv
import json
from pathlib import Path


DETAIL_FIELDS = [
    "ball_possession_pct",
    "total_passes",
    "passes_accurate",
    "passes_pct",
    "total_shots",
    "shots_on_goal",
    "expected_goals",
    "goalkeeper_saves",
    "fouls",
    "corner_kicks",
    "yellow_cards",
    "red_cards",
]


def _has_value(value: object) -> bool:
    return value not in (None, "")


def main() -> None:
    data_root = Path("data")
    manual_path = data_root / "static" / "worldcup_2026_manual_results.csv"
    training_path = data_root / "processed" / "combined" / "training_frame.csv"
    output = Path("outputs/worldcup2026_manual_detail_coverage_audit.json")

    manual_rows = list(csv.DictReader(manual_path.open(encoding="utf-8")))
    training = {
        row["match_id"]: row
        for row in csv.DictReader(training_path.open(encoding="utf-8"))
    }

    matches = []
    missing_all_detail = []
    partial_detail = []
    complete_core_detail = []
    core_fields = ["ball_possession_pct", "total_shots", "shots_on_goal", "fouls", "corner_kicks"]

    for manual in manual_rows:
        match_id = f"fd:{manual['match_id']}"
        row = training.get(match_id, {})
        side_payloads = {}
        match_has_any = False
        match_core_complete = True
        for side in ("home", "away"):
            present = [
                field
                for field in DETAIL_FIELDS
                if _has_value(row.get(f"{side}_actual_{field}"))
            ]
            missing_core = [
                field
                for field in core_fields
                if not _has_value(row.get(f"{side}_actual_{field}"))
            ]
            match_has_any = match_has_any or bool(present)
            match_core_complete = match_core_complete and not missing_core
            side_payloads[side] = {
                "team": row.get(f"{side}_team"),
                "present_detail_fields": present,
                "missing_core_fields": missing_core,
                "coverage": round(len(present) / len(DETAIL_FIELDS), 4),
            }
        item = {
            "match_id": manual["match_id"],
            "date": manual["date"],
            "stage": manual["stage"],
            "team_a": manual["team_a"],
            "team_b": manual["team_b"],
            "score": f"{manual['team_a_goals']}-{manual['team_b_goals']}",
            "has_any_detail": match_has_any,
            "core_detail_complete": match_core_complete,
            "home": side_payloads["home"],
            "away": side_payloads["away"],
        }
        matches.append(item)
        if not match_has_any:
            missing_all_detail.append(item)
        elif not match_core_complete:
            partial_detail.append(item)
        else:
            complete_core_detail.append(item)

    payload = {
        "manual_results": len(manual_rows),
        "matches_with_any_detail": len(manual_rows) - len(missing_all_detail),
        "matches_missing_all_detail": len(missing_all_detail),
        "matches_with_complete_core_detail": len(complete_core_detail),
        "matches_with_partial_detail": len(partial_detail),
        "core_fields": core_fields,
        "matches": matches,
        "missing_all_detail": missing_all_detail,
        "partial_detail": partial_detail,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
