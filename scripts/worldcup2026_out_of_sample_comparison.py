from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_model_evaluation_report import (  # noqa: E402
    CLF_PARAMS,
    REG_PARAMS,
    _is_worldcup_2026,
    _load_frames,
    _prepare_neutral,
    _split_worldcup_2026,
)
from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NEUTRAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    NEUTRAL_RESULT_LABELS,
)


LABEL_TO_SOURCE_RESULT = {0: "home", 1: "draw", 2: "away"}
LABEL_TO_NEUTRAL_RESULT = dict(enumerate(NEUTRAL_RESULT_LABELS))
PROBABILITY_COLUMNS = {
    0: "team_a_win_probability",
    1: "draw_probability",
    2: "team_b_win_probability",
}


@dataclass
class ScenarioResult:
    name: str
    title: str
    policy: str
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def _latest_manual_result_date(data_root: Path) -> str:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return date.today().isoformat()
    with path.open(encoding="utf-8") as handle:
        dates = [row["date"] for row in csv.DictReader(handle) if row.get("date")]
    return max(dates, default=date.today().isoformat())


def split_worldcup_2026_walkforward(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    target_index: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_index not in training.index:
        raise IndexError(f"Target index {target_index} is not present in training frame")
    worldcup_mask = _is_worldcup_2026(training)
    if not bool(worldcup_mask.loc[target_index]):
        raise ValueError("Walk-forward target must be a played World Cup 2026 match")

    dated = training.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="raise")
    order = dated.sort_values(["_date", "row_index"], kind="stable").index.tolist()
    target_position = order.index(target_index)
    prior_indexes = set(order[:target_position])

    split = clean.copy()
    split["split"] = "excluded"
    split.loc[split.index.isin(prior_indexes), "split"] = "train"
    split.loc[target_index, "split"] = "test"

    target_row = training.loc[target_index]
    return split, {
        "target_index": int(target_index),
        "target_date": str(target_row["date"]),
        "policy": (
            "Walk-forward Mundial 2026: para cada partido se entrena con "
            "partidos de selecciones ubicados antes de ese partido en orden "
            "cronologico y row_index, incluyendo partidos previos del Mundial "
            "2026; el partido predicho se fuerza como unico test y no entra al "
            "entrenamiento."
        ),
    }


def _fit_predict(
    data_root: Path,
    training: pd.DataFrame,
    split_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    neutral = _prepare_neutral(data_root, training, split_clean)
    train = neutral[neutral["split"].eq("train")].copy()
    test = neutral[neutral["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("Evaluation split produced an empty train or test set")

    features = list(NEUTRAL_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    weights = train["match_recency_weight"].astype(float).to_numpy(copy=True)

    team_a_model = lgb.LGBMRegressor(**REG_PARAMS)
    team_b_model = lgb.LGBMRegressor(**REG_PARAMS)
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

    classifier = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS),
        method="sigmoid",
        cv=3,
    )
    classifier.fit(
        train[features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=categorical,
    )

    probabilities = np.zeros((len(test), 3), dtype=float)
    raw_probabilities = classifier.predict_proba(test[features])
    for class_position, label in enumerate(classifier.classes_):
        probabilities[:, int(label)] = raw_probabilities[:, class_position]

    pred_a = np.clip(team_a_model.predict(test[features]), 0.0, None)
    pred_b = np.clip(team_b_model.predict(test[features]), 0.0, None)
    labels = test["result_label"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)

    result = test[
        [
            "team_a_goals",
            "team_b_goals",
            "result_label",
        ]
    ].copy()
    result["predicted_label"] = predicted
    result["expected_team_a_goals"] = pred_a
    result["expected_team_b_goals"] = pred_b
    for label, column in PROBABILITY_COLUMNS.items():
        result[column] = probabilities[:, label]
    result["correct"] = labels == predicted

    metrics = {
        "train_rows_after_augmentation": int(len(train)),
        "train_matches": int(split_clean["split"].eq("train").sum()),
        "test_matches": int(len(test)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "correct": int((labels == predicted).sum()),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
        "mae_team_a_goals": round(
            float(mean_absolute_error(test["team_a_goals"], pred_a)),
            4,
        ),
        "mae_team_b_goals": round(
            float(mean_absolute_error(test["team_b_goals"], pred_b)),
            4,
        ),
    }
    metrics["mae_goals_avg"] = round(
        (metrics["mae_team_a_goals"] + metrics["mae_team_b_goals"]) / 2.0,
        4,
    )
    return result.reset_index(drop=True), metrics


def _attach_match_context(
    training_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    scenario: str,
    train_matches: int | list[int],
) -> pd.DataFrame:
    details = training_rows[
        [
            "row_index",
            "date",
            "competition_name",
            "competition_family",
            "competition_type",
            "stage_or_round",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ]
    ].reset_index(drop=True)
    out = pd.concat([details, predictions.reset_index(drop=True)], axis=1)
    out.insert(0, "scenario", scenario)
    out["actual_result"] = out["result_label"].map(LABEL_TO_NEUTRAL_RESULT)
    out["predicted_result"] = out["predicted_label"].map(LABEL_TO_NEUTRAL_RESULT)
    out["actual_source_result"] = out["result_label"].map(LABEL_TO_SOURCE_RESULT)
    out["predicted_source_result"] = out["predicted_label"].map(LABEL_TO_SOURCE_RESULT)
    out["predicted_score"] = (
        out["expected_team_a_goals"].round(2).astype(str)
        + "-"
        + out["expected_team_b_goals"].round(2).astype(str)
    )
    out["correct"] = out["correct"].astype(bool)
    out["train_matches_available"] = train_matches
    ordered_columns = [
        "scenario",
        "row_index",
        "date",
        "competition_name",
        "competition_family",
        "competition_type",
        "stage_or_round",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "actual_source_result",
        "predicted_source_result",
        "actual_result",
        "predicted_result",
        "correct",
        "team_a_win_probability",
        "draw_probability",
        "team_b_win_probability",
        "expected_team_a_goals",
        "expected_team_b_goals",
        "predicted_score",
        "train_matches_available",
    ]
    return out[ordered_columns]


def evaluate_full_worldcup_holdout(
    data_root: Path,
    training: pd.DataFrame,
    clean: pd.DataFrame,
) -> ScenarioResult:
    split_clean, split_info = _split_worldcup_2026(training, clean)
    predictions, metrics = _fit_predict(data_root, training, split_clean)
    test_rows = training.loc[split_clean["split"].eq("test")]
    log = _attach_match_context(
        test_rows,
        predictions,
        "full_worldcup_holdout",
        int(split_clean["split"].eq("train").sum()),
    )
    return ScenarioResult(
        name="full_worldcup_holdout",
        title="Out of sample Mundial completo",
        policy=split_info["policy"],
        metrics=metrics,
        predictions=log,
    )


def evaluate_worldcup_walkforward(
    data_root: Path,
    training: pd.DataFrame,
    clean: pd.DataFrame,
) -> ScenarioResult:
    target_indexes = (
        training.loc[_is_worldcup_2026(training)]
        .assign(_date=lambda frame: pd.to_datetime(frame["date"], errors="raise"))
        .sort_values(["_date", "row_index"], kind="stable")
        .index.tolist()
    )
    prediction_logs: list[pd.DataFrame] = []
    policy = ""
    for position, target_index in enumerate(target_indexes, start=1):
        split_clean, split_info = split_worldcup_2026_walkforward(
            training,
            clean,
            int(target_index),
        )
        policy = split_info["policy"]
        predictions, _ = _fit_predict(data_root, training, split_clean)
        test_rows = training.loc[[target_index]]
        log = _attach_match_context(
            test_rows,
            predictions,
            "walkforward_prior_matches",
            int(split_clean["split"].eq("train").sum()),
        )
        log["walkforward_step"] = position
        prediction_logs.append(log)

    details = pd.concat(prediction_logs, ignore_index=True)
    labels = details["actual_source_result"].map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    predicted = details["predicted_source_result"].map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    probabilities = details[
        ["team_a_win_probability", "draw_probability", "team_b_win_probability"]
    ].to_numpy()
    mae_a = float(mean_absolute_error(details["home_goals"], details["expected_team_a_goals"]))
    mae_b = float(mean_absolute_error(details["away_goals"], details["expected_team_b_goals"]))
    metrics = {
        "test_matches": int(len(details)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "correct": int((labels == predicted).sum()),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
        "mae_team_a_goals": round(mae_a, 4),
        "mae_team_b_goals": round(mae_b, 4),
        "mae_goals_avg": round((mae_a + mae_b) / 2.0, 4),
        "first_train_matches": int(details["train_matches_available"].min()),
        "last_train_matches": int(details["train_matches_available"].max()),
    }
    return ScenarioResult(
        name="walkforward_prior_matches",
        title="Out of sample solo partido predicho",
        policy=policy,
        metrics=metrics,
        predictions=details,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full World Cup 2026 holdout vs walk-forward evaluation.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    training, clean = _load_frames(args.data_root)
    holdout = evaluate_full_worldcup_holdout(args.data_root, training, clean)
    walkforward = evaluate_worldcup_walkforward(args.data_root, training, clean)

    suffix = _latest_manual_result_date(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = args.output_dir / f"worldcup2026_full_holdout_match_log_{suffix}.csv"
    walkforward_path = args.output_dir / f"worldcup2026_walkforward_match_log_{suffix}.csv"
    summary_path = args.output_dir / f"worldcup2026_oos_comparison_{suffix}.json"
    holdout.predictions.to_csv(holdout_path, index=False)
    walkforward.predictions.to_csv(walkforward_path, index=False)

    payload = {
        "model": "lightgbm_neutral",
        "model_id": NEUTRAL_MODEL_RECIPE,
        "features": list(NEUTRAL_FEATURES),
        "evaluations": [
            {
                "name": holdout.name,
                "title": holdout.title,
                "policy": holdout.policy,
                "metrics": holdout.metrics,
                "match_log": str(holdout_path),
            },
            {
                "name": walkforward.name,
                "title": walkforward.title,
                "policy": walkforward.policy,
                "metrics": walkforward.metrics,
                "match_log": str(walkforward_path),
            },
        ],
        "winner": (
            walkforward.name
            if walkforward.metrics["accuracy"] > holdout.metrics["accuracy"]
            else holdout.name
            if holdout.metrics["accuracy"] > walkforward.metrics["accuracy"]
            else "tie"
        ),
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**payload, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
