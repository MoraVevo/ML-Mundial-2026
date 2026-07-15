from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from generate_model_evaluation_report import CLF_PARAMS, REG_PARAMS, _prepare_neutral  # noqa: E402
from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NEUTRAL_BASE_RESULT_FEATURES,
    NEUTRAL_GOAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    NEUTRAL_RESULT_FEATURES,
    NEUTRAL_XG_RESULT_BLEND_WEIGHT,
)


def _first_existing(data_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = data_root / candidate
        if path.exists():
            return path
    joined = ", ".join(str(data_root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing required processed file. Tried: {joined}")


def train_all_played_model(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_path = _first_existing(
        data_root,
        [
            "processed/combined/training_frame_national.csv",
            "processed/combined/training_frame.csv",
        ],
    )
    clean_path = _first_existing(
        data_root,
        [
            "processed/combined/clean_training_matrix_national.csv",
            "processed/combined/clean_training_matrix.csv",
        ],
    )
    training = pd.read_csv(training_path, low_memory=False)
    clean = pd.read_csv(clean_path, low_memory=False).copy()
    clean["split"] = "train"
    neutral = _prepare_neutral(data_root, training, clean)
    train = neutral[neutral["split"].eq("train")].copy()
    if train.empty:
        raise RuntimeError("All-played training split produced no rows")

    goal_features = list(NEUTRAL_GOAL_FEATURES)
    base_result_features = list(NEUTRAL_BASE_RESULT_FEATURES)
    result_features = list(NEUTRAL_RESULT_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in goal_features]
    result_categorical = [feature for feature in CATEGORICAL_FEATURES if feature in result_features]
    base_result_categorical = [feature for feature in CATEGORICAL_FEATURES if feature in base_result_features]
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
        train[base_result_features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=base_result_categorical,
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

    worldcup_rows = training[
        training["source"].astype(str).eq("manual-worldcup-2026")
    ].copy()
    latest_played = str(worldcup_rows["date"].max()) if not worldcup_rows.empty else ""
    model_id = f"{NEUTRAL_MODEL_RECIPE}_all_played_wc2026"
    policy = (
        "All available completed national-team matches are used for future-match "
        "prediction, including played World Cup 2026 matches through "
        f"{latest_played}. No future or unplayed World Cup 2026 rows are included."
    )
    model = {
        "team_a_goals_model": team_a_model,
        "team_b_goals_model": team_b_model,
        "result_model": result_model,
        "xg_result_model": xg_result_model,
        "features": goal_features,
        "team_a_goal_features": goal_features,
        "team_b_goal_features": goal_features,
        "result_features": base_result_features,
        "xg_result_features": result_features,
        "result_probability_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
        "model_id": model_id,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "training_policy": policy,
    }
    metadata = {
        "model_id": model_id,
        "features": goal_features,
        "goal_features": goal_features,
        "result_features": result_features,
        "xg_result_blend_weight": NEUTRAL_XG_RESULT_BLEND_WEIGHT,
        "training_policy": policy,
        "train_matches": int(len(training)),
        "train_rows_after_augmentation": int(len(train)),
        "worldcup_2026_played_matches": int(len(worldcup_rows)),
        "latest_worldcup_2026_played_date": latest_played,
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the World Cup 2026 model with every currently played match.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/lightgbm_neutral_all_played_wc2026.joblib"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("outputs/worldcup2026_all_played_model_metadata.json"),
    )
    args = parser.parse_args()

    model, metadata = train_all_played_model(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"model_path": str(args.output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
