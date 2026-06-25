from scripts.audit_tactical_coverage_bias import audit


def _row(*, competition: str, complete: bool) -> dict[str, str]:
    row = {
        "competition_family": "test",
        "competition_name": competition,
        "source": "fixture",
    }
    fields = [
        "shots_on_goal",
        "shots_off_goal",
        "total_shots",
        "blocked_shots",
        "shots_inside_box",
        "shots_outside_box",
        "fouls",
        "corner_kicks",
        "offsides",
        "ball_possession_pct",
        "yellow_cards",
        "red_cards",
        "goalkeeper_saves",
        "total_passes",
        "passes_accurate",
        "passes_pct",
        "expected_goals",
        "goals_prevented",
    ]
    for side in ("home", "away"):
        for field in fields:
            row[f"{side}_actual_{field}"] = "1" if complete else ""
            row[f"{side}_recent6_{field}_avg"] = "1" if complete else ""
    return row


def test_audit_separates_complete_and_sparse_competitions() -> None:
    rows = [_row(competition="Complete Cup", complete=True) for _ in range(20)]
    rows += [_row(competition="Sparse Cup", complete=False) for _ in range(20)]

    report = {item["competition_name"]: item for item in audit(rows)}

    assert report["Complete Cup"]["admission_class"] == "broad_enough_for_ablation"
    assert report["Complete Cup"]["actual_core_match_rate"] == 1.0
    assert report["Sparse Cup"]["admission_class"] == "too_sparse"
    assert report["Sparse Cup"]["recent_matchup_zero_rate"] == 1.0
