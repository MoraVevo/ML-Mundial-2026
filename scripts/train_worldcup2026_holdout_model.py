from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.lightgbm_model import (
    CATEGORICAL_FEATURES,
    NEUTRAL_GOAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    NEUTRAL_RESULT_FEATURES,
    NEUTRAL_XG_RESULT_BLEND_WEIGHT,
    blend_result_probabilities,
)

from generate_model_evaluation_report import (  # noqa: E402
    CLF_PARAMS,
    REG_PARAMS,
    _prepare_neutral,
    _split_worldcup_2026,
)


def _first_existing(data_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = data_root / candidate
        if path.exists():
            return path
    joined = ", ".join(str(data_root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing required processed file. Tried: {joined}")


def train_holdout_model(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_path = _first_existing(
        data_root,
        [
            "processed/combined/training_frame_national.csv",
            "processed/combined/training_frame.csv",
            "processed/training_frame_national.csv",
            "processed/training_frame.csv",
        ],
    )
    clean_path = _first_existing(
        data_root,
        [
            "processed/combined/clean_training_matrix_national.csv",
            "processed/combined/clean_training_matrix.csv",
            "processed/clean_training_matrix_national.csv",
            "processed/clean_training_matrix.csv",
        ],
    )
    training = pd.read_csv(training_path)
    clean = pd.read_csv(clean_path)
    split_clean, split_meta = _split_worldcup_2026(training, clean)
    neutral = _prepare_neutral(data_root, training, split_clean)

    train = neutral[neutral["split"].eq("train")].copy()
    test = neutral[neutral["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("World Cup holdout split produced an empty train or test set")

    goal_features = list(NEUTRAL_GOAL_FEATURES)
    result_features = list(NEUTRAL_RESULT_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in goal_features]
    result_categorical = [feature for feature in CATEGORICAL_FEATURES if feature in result_features]
    weights = train["match_recency_weight"].astype(float).to_numpy(copy=True)

    team_a_model = lgb.LGBMRegressor(**REG_PARAMS)
    team_b_model = lgb.LGBMRegressor(**REG_PARAMS)
    team_a_model.fit(
        train[goal_features],
        train["team_a_goals"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    team_b_model.fit(
        train[goal_features],
        train["team_b_goals"],
        sample_weight=weights,
        categorical_feature=categorical,
    )

    result_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS),
        method="sigmoid",
        cv=3,
    )
    result_model.fit(
        train[goal_features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    xg_result_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS),
        method="sigmoid",
        cv=3,
    )
    xg_result_model.fit(
        train[result_features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=result_categorical,
    )

    probabilities = blend_result_probabilities(
        result_model.predict_proba(test[goal_features]),
        xg_result_model.predict_proba(test[result_features]),
    )
    labels = test["result_label"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    pred_a = np.clip(team_a_model.predict(test[goal_features]), 0.0, None)
    pred_b = np.clip(team_b_model.predict(test[goal_features]), 0.0, None)
    mae_a = float(mean_absolute_error(test["team_a_goals"], pred_a))
    mae_b = float(mean_absolute_error(test["team_b_goals"], pred_b))

    model = {
        "team_a_goals_model": team_a_model,
        "team_b_goals_model": team_b_model,
        "result_model": result_model,
        "xg_result_model": xg_result_model,
        "features": goal_features,
        "team_a_goal_features": goal_features,
        "team_b_goal_features": goal_features,
        "result_features": goal_features,
        "xg_result_features": result_features,
        "result_probability_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
        "model_id": f"{NEUTRAL_MODEL_RECIPE}_worldcup_2026_holdout",
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "training_policy": split_meta["policy"],
    }
    metadata = {
        "model_id": model["model_id"],
        "features": goal_features,
        "goal_features": goal_features,
        "result_features": result_features,
        "xg_result_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
        "training_policy": split_meta["policy"],
        "test_start": split_meta["test_start"],
        "test_end": split_meta["test_end"],
        "train_matches": int(train["split"].eq("train").sum() / 2),
        "test_matches": int(len(test)),
        "metrics": {
            "accuracy": round(float(accuracy_score(labels, predicted)), 4),
            "correct": int((labels == predicted).sum()),
            "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
            "mae_team_a_goals": round(mae_a, 4),
            "mae_team_b_goals": round(mae_b, 4),
            "mae_goals_avg": round((mae_a + mae_b) / 2.0, 4),
        },
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the public World Cup 2026 holdout model.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/lightgbm_neutral_worldcup_holdout.joblib"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("outputs/worldcup2026_holdout_model_metadata.json"),
    )
    args = parser.parse_args()

    model, metadata = train_holdout_model(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(args.output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
