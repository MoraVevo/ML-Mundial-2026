from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

from generate_model_evaluation_report import (  # noqa: E402
    CLF_PARAMS,
    _is_worldcup_2026,
    _load_frames,
    _prepare_neutral,
)
from kinela.lightgbm_model import CATEGORICAL_FEATURES, NEUTRAL_FEATURES  # noqa: E402


BLOCK_SIZE = 6
MIN_WC_TRAIN_MATCHES = 36
RANDOM_SEED = 42

FEATURE_SETS = {
    "active_12": list(NEUTRAL_FEATURES),
    "active_12_plus_chance_quality": [
        *NEUTRAL_FEATURES,
        "worldcup_chance_quality_edge",
    ],
    "active_12_plus_detail_flow": [
        *NEUTRAL_FEATURES,
        "worldcup_detail_flow_edge",
    ],
    "active_12_plus_both_detail_edges": [
        *NEUTRAL_FEATURES,
        "worldcup_chance_quality_edge",
        "worldcup_detail_flow_edge",
    ],
}


def _sort_worldcup(training: pd.DataFrame) -> pd.DataFrame:
    worldcup = training.loc[_is_worldcup_2026(training)].copy()
    worldcup["_date"] = pd.to_datetime(worldcup["date"], errors="raise")
    if "row_index" not in worldcup:
        worldcup["row_index"] = worldcup.index
    return worldcup.sort_values(["_date", "row_index"], kind="stable")


def _split_before_test(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    test_indices: list[int],
) -> pd.DataFrame:
    dated = training.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="raise")
    if "row_index" not in dated:
        dated["row_index"] = dated.index

    first = dated.loc[test_indices].sort_values(["_date", "row_index"], kind="stable").iloc[0]
    before_test = (
        dated["_date"].lt(first["_date"])
        | (dated["_date"].eq(first["_date"]) & dated["row_index"].lt(first["row_index"]))
    )
    test_mask = dated.index.isin(test_indices)

    split = clean.copy()
    split["split"] = "excluded"
    split.loc[before_test & ~test_mask, "split"] = "train"
    split.loc[test_mask, "split"] = "test"
    return split


def _evaluate_feature_set(
    neutral: pd.DataFrame,
    split_clean: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    train = neutral.loc[neutral["split"].eq("train")].copy()
    test = neutral.loc[neutral["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("Empty train or test split")

    for feature in features:
        if feature not in train:
            train[feature] = 0.0
            test[feature] = 0.0

    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    weights = train["match_recency_weight"].astype(float).to_numpy()
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
    probabilities = classifier.predict_proba(test[features])
    labels = test["result_label"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    return {
        "train_matches": int(split_clean["split"].eq("train").sum()),
        "train_augmented_rows": int(len(train)),
        "test_matches": int(len(test)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "correct": int((labels == predicted).sum()),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
        "predicted_labels": predicted.astype(int).tolist(),
        "actual_labels": labels.astype(int).tolist(),
    }


def _evaluate_split(
    name: str,
    data_root: Path,
    training: pd.DataFrame,
    clean: pd.DataFrame,
    test_indices: list[int],
) -> dict[str, Any]:
    split_clean = _split_before_test(training, clean, test_indices)
    neutral = _prepare_neutral(data_root, training, split_clean)
    test_training = training.loc[test_indices].copy()
    test_training["_date"] = pd.to_datetime(test_training["date"], errors="raise")
    test_training = test_training.sort_values(["_date", "row_index"], kind="stable")

    results = {
        feature_set: _evaluate_feature_set(neutral, split_clean, features)
        for feature_set, features in FEATURE_SETS.items()
    }
    return {
        "name": name,
        "test_match_ids": test_training["match_id"].astype(str).tolist(),
        "test_dates": test_training["date"].astype(str).tolist(),
        "test_matches": [
            {
                "date": str(row["date"]),
                "match_id": str(row["match_id"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "result": str(row["result"]),
            }
            for _, row in test_training.iterrows()
        ],
        "feature_sets": results,
    }


def _feature_diagnostics(data_root: Path, training: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    split = clean.copy()
    split["split"] = np.where(_is_worldcup_2026(training), "test", "train")
    neutral = _prepare_neutral(data_root, training, split)
    worldcup = neutral.loc[neutral["competition_family"].astype(str).eq("national_world_cup")].copy()
    candidate_cols = [
        "worldcup_chance_quality_edge",
        "worldcup_detail_flow_edge",
        "espn_leader_detail_coverage_pair",
    ]
    coverage = {
        column: {
            "non_zero_rows": int((worldcup[column].fillna(0) != 0).sum()),
            "mean": round(float(worldcup[column].fillna(0).mean()), 4),
            "std": round(float(worldcup[column].fillna(0).std(ddof=0)), 4),
        }
        for column in candidate_cols
    }
    correlations: dict[str, list[dict[str, Any]]] = {}
    corr_frame = worldcup[[*NEUTRAL_FEATURES, *candidate_cols]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    for candidate in candidate_cols[:2]:
        series = corr_frame.corr(numeric_only=True)[candidate].drop(labels=[candidate], errors="ignore")
        correlations[candidate] = [
            {"feature": str(feature), "correlation": round(float(value), 4)}
            for feature, value in series.abs().sort_values(ascending=False).head(8).items()
            if not pd.isna(value)
        ]
    return {
        "worldcup_rows_after_neutral_augmentation": int(len(worldcup)),
        "coverage": coverage,
        "top_absolute_correlations": correlations,
    }


def main() -> None:
    data_root = Path("data")
    training, clean = _load_frames(data_root)
    worldcup = _sort_worldcup(training)
    if len(worldcup) < MIN_WC_TRAIN_MATCHES + BLOCK_SIZE:
        raise RuntimeError("Not enough World Cup 2026 matches for expanding walk-forward blocks")

    last6_indices = worldcup.tail(BLOCK_SIZE).index.astype(int).tolist()
    last6 = _evaluate_split(
        "last_6_played_worldcup_matches",
        data_root,
        training,
        clean,
        last6_indices,
    )

    folds = []
    for start in range(MIN_WC_TRAIN_MATCHES, len(worldcup), BLOCK_SIZE):
        block = worldcup.iloc[start : start + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE:
            break
        folds.append(
            _evaluate_split(
                f"expanding_wc_train_{start}_test_{start + 1}_{start + len(block)}",
                data_root,
                training,
                clean,
                block.index.astype(int).tolist(),
            )
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for feature_set in FEATURE_SETS:
        total_correct = sum(fold["feature_sets"][feature_set]["correct"] for fold in folds)
        total_matches = sum(fold["feature_sets"][feature_set]["test_matches"] for fold in folds)
        aggregate[feature_set] = {
            "folds": len(folds),
            "test_matches": total_matches,
            "correct": total_correct,
            "accuracy": round(total_correct / total_matches, 4) if total_matches else None,
            "mean_log_loss": round(
                float(np.mean([fold["feature_sets"][feature_set]["log_loss"] for fold in folds])),
                4,
            )
            if folds
            else None,
        }

    payload = {
        "experiment": "worldcup_2026_espn_detail_walkforward",
        "notes": [
            "ESPN leader stats are treated as top-player signals, not team totals.",
            "Each split trains only on rows ordered before the first test match.",
            "World Cup detail features are tested as candidates and do not replace the active recipe.",
        ],
        "feature_sets": {key: value for key, value in FEATURE_SETS.items()},
        "last6": last6,
        "walkforward_aggregate": aggregate,
        "walkforward_folds": folds,
        "feature_diagnostics": _feature_diagnostics(data_root, training, clean),
    }
    output = Path("outputs/worldcup2026_espn_detail_walkforward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
