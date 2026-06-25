from __future__ import annotations

import json
import sys
import argparse
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from kinela.fifa_ranking import normalise_team_name  # noqa: E402
from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NEUTRAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    _build_neutral_frame,
    _maybe_add_clinical_finishing,
    _maybe_add_club_attacking_talent,
    _maybe_add_counter_efficiency,
    _maybe_add_late85_points_swing,
    _maybe_add_score_timing,
)
from kinela.worldcup_2026 import (  # noqa: E402
    RECENT_FORM_WINDOW,
    WorldCup2026Simulator,
    _normalise_name,
    current_counter_threat,
    current_underdog_fit,
)


DEFAULT_OUTPUT = Path("outputs/next4_all_played_worldcup_2026-06-23.json")
MODEL_OUTPUT = Path("data/models/lightgbm_neutral_all_played_wc2026.joblib")


def _canonical_match_key(row: pd.Series) -> tuple[str, tuple[str, str]]:
    teams = sorted(
        (
            normalise_team_name(str(row["home_team"])),
            normalise_team_name(str(row["away_team"])),
        )
    )
    return str(row["date"]), (teams[0], teams[1])


def _deduplicated_training_indices(metadata: pd.DataFrame) -> list[int]:
    keep = (
        metadata.sort_values(["canonical_key", "priority"])
        .drop_duplicates("canonical_key", keep="last")
        .index
    )
    return sorted(int(index) for index in keep)


def _all_played_training_frame(data_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    training = pd.read_csv(
        data_root / "processed" / "combined" / "training_frame_national.csv",
        low_memory=False,
    )
    clean = pd.read_csv(
        data_root / "processed" / "combined" / "clean_training_matrix_national.csv",
        low_memory=False,
    )
    if len(training) != len(clean):
        raise RuntimeError("Training and clean matrices are not aligned")

    metadata = training[
        ["match_id", "date", "source", "home_team", "away_team"]
    ].copy()
    metadata["canonical_key"] = metadata.apply(_canonical_match_key, axis=1)
    metadata["priority"] = np.select(
        [
            metadata["source"].eq("manual-worldcup-2026"),
            metadata["match_id"].astype(str).str.startswith("fd:"),
        ],
        [2, 1],
        default=0,
    )
    keep = _deduplicated_training_indices(metadata)
    duplicate_rows_removed = len(clean) - len(keep)
    clean = clean.loc[keep].reset_index(drop=True)
    clean["split"] = "train"
    clean = _maybe_add_late85_points_swing(data_root, clean, training.loc[keep])
    clean = _maybe_add_score_timing(data_root, clean, training.loc[keep])
    clean = _maybe_add_counter_efficiency(data_root, clean, training.loc[keep])
    clean = _maybe_add_clinical_finishing(data_root, clean, training.loc[keep])
    clean = _maybe_add_club_attacking_talent(data_root, clean, training.loc[keep])
    neutral = _build_neutral_frame(clean, augment=True)
    played_worldcup = metadata[
        metadata["source"].eq("manual-worldcup-2026")
    ]["canonical_key"].nunique()
    return neutral, {
        "source_rows": int(len(training)),
        "unique_matches": int(len(clean)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "played_worldcup_matches_in_train": int(played_worldcup),
        "training_rows_after_neutral_augmentation": int(len(neutral)),
    }


def _train_model(data_root: Path) -> tuple[dict, dict[str, int]]:
    train, counts = _all_played_training_frame(data_root)
    features = list(NEUTRAL_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    weights = train["match_recency_weight"].astype(float).to_numpy(copy=True)

    reg_params = {
        "objective": "regression",
        "n_estimators": 350,
        "learning_rate": 0.035,
        "num_leaves": 23,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbosity": -1,
    }
    clf_params = {
        "objective": "multiclass",
        "n_estimators": 300,
        "learning_rate": 0.035,
        "num_leaves": 23,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbosity": -1,
    }
    team_a_model = lgb.LGBMRegressor(**reg_params)
    team_b_model = lgb.LGBMRegressor(**reg_params)
    team_a_model.fit(
        train[features],
        train["team_a_goals"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    team_b_model.fit(
        train[features],
        train["team_b_goals"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    result_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**clf_params),
        method="sigmoid",
        cv=3,
    )
    result_model.fit(
        train[features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    model = {
        "team_a_goals_model": team_a_model,
        "team_b_goals_model": team_b_model,
        "result_model": result_model,
        "features": features,
        "feature_recipe": f"{NEUTRAL_MODEL_RECIPE}_all_played_wc2026",
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "training_policy": (
            "All available completed national-team matches, including the played "
            "World Cup 2026 matches, are used for future-match prediction after "
            "deduplication."
        ),
    }
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    return model, counts


def _deduplicate_simulator_history(simulator: WorldCup2026Simulator) -> int:
    selected: dict[tuple[str, tuple[str, str]], dict] = {}
    priorities: dict[tuple[str, tuple[str, str]], int] = {}
    for row in simulator.history:
        key = (
            str(row["date"]),
            tuple(
                sorted(
                    (
                        normalise_team_name(str(row["home_team"])),
                        normalise_team_name(str(row["away_team"])),
                    )
                )
            ),
        )
        source = str(row.get("source") or "")
        priority = 2 if source == "manual-worldcup-2026" else 1 if str(
            row.get("match_id") or ""
        ).startswith("fd:") else 0
        if key not in selected or priority >= priorities[key]:
            selected[key] = row
            priorities[key] = priority
    removed = len(simulator.history) - len(selected)
    simulator.history = sorted(selected.values(), key=lambda row: row["date_obj"])
    simulator.team_histories = simulator._index_team_histories()
    simulator.head_to_head_histories = simulator._index_head_to_head_histories()
    simulator.worldcup_histories = simulator._index_worldcup_histories()
    simulator.elo_ratings = simulator._build_elo_ratings()
    (
        simulator.confederation_stats,
        simulator.team_cross_confederation_stats,
    ) = simulator._build_confederation_contexts()
    simulator.fifa_point_overrides = simulator._build_fifa_point_overrides()
    contexts = simulator._goal_contexts()
    simulator.global_goal_avg = contexts["global_avg"]
    simulator.major_goal_avg = contexts["major_avg"]
    simulator.group_goal_avg = contexts["group_avg"]
    simulator.knockout_goal_avg = contexts["knockout_avg"]
    simulator.major_match_count = contexts["major_matches"]
    simulator.group_match_count = contexts["group_matches"]
    simulator.knockout_match_count = contexts["knockout_matches"]
    simulator.prediction_cache.clear()
    return removed


def _group_tables(simulator: WorldCup2026Simulator) -> dict[str, dict[str, dict]]:
    tables: dict[str, dict[str, dict]] = {}
    fixture_groups = {
        str(match["id"]): str(match.get("group") or "")
        for match in simulator.group_matches
    }
    for group, teams in simulator.groups.items():
        tables[group] = {
            team: {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0}
            for team in teams
    }
    for match_id, result in simulator.manual_results.items():
        group = str(result.get("group") or fixture_groups.get(match_id) or "")
        if group not in tables:
            continue
        team_a = str(result["team_a"])
        team_b = str(result["team_b"])
        goals_a = int(result["team_a_goals"])
        goals_b = int(result["team_b_goals"])
        for team in (team_a, team_b):
            tables[group].setdefault(
                team,
                {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0},
            )
            tables[group][team]["played"] += 1
        tables[group][team_a]["gf"] += goals_a
        tables[group][team_a]["ga"] += goals_b
        tables[group][team_b]["gf"] += goals_b
        tables[group][team_b]["ga"] += goals_a
        if goals_a > goals_b:
            tables[group][team_a]["points"] += 3
        elif goals_b > goals_a:
            tables[group][team_b]["points"] += 3
        else:
            tables[group][team_a]["points"] += 1
            tables[group][team_b]["points"] += 1
    return tables


def _recent_record(simulator: WorldCup2026Simulator, team: str, before: date) -> dict:
    history = simulator._team_history(team, before, include_simulated=False)
    recent = history[-RECENT_FORM_WINDOW:]
    wins = sum(item["points"] == 3 for item in recent)
    draws = sum(item["points"] == 1 for item in recent)
    losses = sum(item["points"] == 0 for item in recent)
    return {
        "matches": len(recent),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": int(sum(item["gf"] for item in recent)),
        "goals_against": int(sum(item["ga"] for item in recent)),
    }


def _worldcup_record(simulator: WorldCup2026Simulator, team: str) -> dict:
    key = _normalise_name(team)
    rows = [
        item
        for item in simulator.team_histories.get(key, [])
        if item["date"] >= date(2026, 6, 11)
    ]
    return {
        "matches": len(rows),
        "wins": sum(item["points"] == 3 for item in rows),
        "draws": sum(item["points"] == 1 for item in rows),
        "losses": sum(item["points"] == 0 for item in rows),
        "goals_for": int(sum(item["gf"] for item in rows)),
        "goals_against": int(sum(item["ga"] for item in rows)),
    }


def _team_cv(
    simulator: WorldCup2026Simulator,
    team: str,
    opponent: str,
    match_date: date,
) -> dict:
    profile = simulator._team_features(team, match_date, include_simulated=False)
    detail = profile["detail_stats"]
    elo = float(simulator._elo(team))
    opponent_elo = float(simulator._elo(opponent))
    counter_threat = current_counter_threat(
        profile["counter_summary"],
        current_underdog_fit(elo, opponent_elo),
    )
    fifa = simulator._fifa_ranking_features(team)
    return {
        "team": team,
        "recent_record": _recent_record(simulator, team, match_date),
        "worldcup_2026_record": _worldcup_record(simulator, team),
        "recent_points_per_match": round(float(profile["recent6_points_avg"]), 3),
        "recent_win_rate": round(float(profile["recent6_win_rate"]), 3),
        "recent_goals_for_avg": round(float(profile["recent6_goals_for_avg"]), 3),
        "recent_goals_against_avg": round(
            float(profile["recent6_goals_against_avg"]), 3
        ),
        "recent_opponent_elo_avg": round(
            float(profile["recent6_opponent_elo_avg"]), 1
        ),
        "current_elo": round(elo, 1),
        "fifa_rank": int(fifa["fifa_rank"]),
        "live_fifa_points": round(float(fifa["fifa_points"]), 2),
        "tactical_detail_coverage": round(float(profile["detail_coverage"]), 3),
        "recent_possession_pct": round(float(detail["ball_possession_pct"]), 1),
        "recent_total_shots": round(float(detail["total_shots"]), 1),
        "recent_shots_on_goal": round(float(detail["shots_on_goal"]), 1),
        "recent_total_passes": round(float(detail["total_passes"]), 1),
        "recent_pass_accuracy_pct": round(float(detail["passes_pct"]), 1),
        "counter_current_threat": round(float(counter_threat), 4),
        "club_star_finisher_signal": round(
            float(profile["club_star_finisher_signal"]),
            4,
        ),
        "club_attack_coverage": round(float(profile["club_attack_coverage"]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_OUTPUT
        if args.limit == 4
        else Path(f"outputs/next{args.limit}_all_played_worldcup_2026-06-24.json")
    )
    data_root = Path("data")
    model, training_counts = _train_model(data_root)
    simulator = WorldCup2026Simulator(data_root, seed=42, engine="lightgbm")
    simulator.lightgbm_model = model
    training_counts["simulator_history_duplicates_removed"] = (
        _deduplicate_simulator_history(simulator)
    )
    group_tables = _group_tables(simulator)
    played_ids = set(simulator.manual_results)
    rows = []
    cvs: dict[str, dict] = {}

    for fixture in sorted(simulator.group_matches, key=lambda item: item["utcDate"]):
        match_id = str(fixture["id"])
        if match_id in played_ids:
            continue
        team_a = fixture["homeTeam"]["name"]
        team_b = fixture["awayTeam"]["name"]
        if not team_a or not team_b:
            continue
        match_date = date.fromisoformat(fixture["utcDate"][:10])
        group = str(fixture.get("group") or "")
        prediction = simulator.lightgbm_prediction(
            team_a,
            team_b,
            match_date,
            "GROUP_STAGE",
            group_tables.get(group),
        )
        probabilities = prediction["probabilities"]
        labels = [team_a, "Empate", team_b]
        rows.append(
            {
                "fixture_id": int(match_id),
                "utc_date": fixture["utcDate"],
                "group": group.replace("GROUP_", ""),
                "team_a": team_a,
                "team_b": team_b,
                "expected_goals_a": round(float(prediction["team_a_goals"]), 3),
                "expected_goals_b": round(float(prediction["team_b_goals"]), 3),
                "prob_team_a": round(float(probabilities[0]), 4),
                "prob_draw": round(float(probabilities[1]), 4),
                "prob_team_b": round(float(probabilities[2]), 4),
                "most_likely": labels[int(np.argmax(probabilities))],
            }
        )
        cvs[team_a] = _team_cv(simulator, team_a, team_b, match_date)
        cvs[team_b] = _team_cv(simulator, team_b, team_a, match_date)
        if len(rows) == args.limit:
            break

    coverage_path = Path("outputs/worldcup2026_manual_detail_coverage_audit.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    payload = {
        "model": model["feature_recipe"],
        "features": model["features"],
        "training": training_counts,
        "worldcup_detail_coverage": {
            "matches": coverage["manual_results"],
            "complete_core_detail": coverage["matches_with_complete_core_detail"],
            "missing_all_detail": coverage["matches_missing_all_detail"],
            "partial_detail": coverage["matches_with_partial_detail"],
        },
        "predictions": rows,
        "team_cvs": cvs,
        "notes": [
            "CV means the recent team profile used to contextualize the prediction.",
            "Unsupported provider fields remain blank and are not fabricated.",
            "This all-played model is for future predictions; official World Cup accuracy "
            "continues to use the held-out 2026 matches.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
