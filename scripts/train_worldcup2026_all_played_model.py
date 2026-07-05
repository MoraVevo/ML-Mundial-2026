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
from kinela.lightgbm_model import CATEGORICAL_FEATURES, NEUTRAL_FEATURES, NEUTRAL_MODEL_RECIPE  # noqa: E402


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

    result_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS),
        method="sigmoid",
        cv=3,
    )
    result_model.fit(
        train[features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=categorical,
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
        "features": features,
        "model_id": model_id,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "training_policy": policy,
    }
    metadata = {
        "model_id": model_id,
        "features": features,
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
