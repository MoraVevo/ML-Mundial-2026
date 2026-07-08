from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from generate_model_evaluation_report import (  # noqa: E402
    CLF_PARAMS,
    REG_PARAMS,
    _is_worldcup_2026,
    _prepare_neutral,
)
from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NEUTRAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
)


TARGET_LABELS = {0: "team_a", 1: "draw", 2: "team_b"}


@dataclass(frozen=True)
class TrainedModels:
    team_a_goals: lgb.LGBMRegressor
    team_b_goals: lgb.LGBMRegressor
    result: CalibratedClassifierCV


def _first_existing(data_root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = data_root / candidate
        if path.exists():
            return path
    joined = ", ".join(str(data_root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing required processed file. Tried: {joined}")


def _load_frames(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return (
        pd.read_csv(training_path, low_memory=False),
        pd.read_csv(clean_path, low_memory=False).copy(),
    )


def _ordered_worldcup_matches(training: pd.DataFrame) -> pd.DataFrame:
    matches = training.loc[_is_worldcup_2026(training)].copy()
    matches["_date"] = pd.to_datetime(matches["date"], errors="raise")
    matches["_row_order"] = pd.to_numeric(matches["row_index"], errors="coerce")
    return matches.sort_values(["_date", "_row_order"], kind="stable")


def _split_for_target(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    target_index: int,
) -> pd.DataFrame:
    dated = training.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="raise")
    dated["_row_order"] = pd.to_numeric(dated["row_index"], errors="coerce")
    target = dated.loc[target_index]
    target_date = target["_date"]
    target_order = target["_row_order"]
    worldcup = _is_worldcup_2026(dated)
    before_target = dated["_date"].lt(target_date) | (
        dated["_date"].eq(target_date) & dated["_row_order"].lt(target_order)
    )

    split = clean.copy()
    split["split"] = "excluded"
    split.loc[~worldcup | (worldcup & before_target), "split"] = "train"
    split.loc[target_index, "split"] = "test"
    return split


def _split_frozen_pre_worldcup(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    target_indices: set[int],
) -> pd.DataFrame:
    dated = training.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="raise")
    worldcup = _is_worldcup_2026(dated)
    first_worldcup_date = dated.loc[worldcup, "_date"].min()

    split = clean.copy()
    split["split"] = "excluded"
    split.loc[dated["_date"].lt(first_worldcup_date) & ~worldcup, "split"] = "train"
    split.loc[list(target_indices), "split"] = "test"
    return split


def _train_models(train: pd.DataFrame, features: list[str], categorical: list[str]) -> TrainedModels:
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
    return TrainedModels(
        team_a_goals=team_a_model,
        team_b_goals=team_b_model,
        result=result_model,
    )


def _predict_one(
    models: TrainedModels,
    row: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float, float]:
    probabilities = models.result.predict_proba(row[features])[0]
    pred_a = float(np.clip(models.team_a_goals.predict(row[features])[0], 0.0, None))
    pred_b = float(np.clip(models.team_b_goals.predict(row[features])[0], 0.0, None))
    return probabilities, pred_a, pred_b


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    labels = np.array([row["actual_label"] for row in rows], dtype=int)
    probabilities = np.array(
        [[row[f"{prefix}_p_a"], row[f"{prefix}_p_draw"], row[f"{prefix}_p_b"]] for row in rows],
        dtype=float,
    )
    predicted = np.array([row[f"{prefix}_predicted_label"] for row in rows], dtype=int)
    pred_a = np.array([row[f"{prefix}_pred_team_a_goals"] for row in rows], dtype=float)
    pred_b = np.array([row[f"{prefix}_pred_team_b_goals"] for row in rows], dtype=float)
    actual_a = np.array([row["team_a_goals"] for row in rows], dtype=float)
    actual_b = np.array([row["team_b_goals"] for row in rows], dtype=float)
    return {
        "matches": int(len(rows)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "correct": int((labels == predicted).sum()),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
        "mae_team_a_goals": round(float(mean_absolute_error(actual_a, pred_a)), 4),
        "mae_team_b_goals": round(float(mean_absolute_error(actual_b, pred_b)), 4),
        "mae_goals_avg": round(
            float((mean_absolute_error(actual_a, pred_a) + mean_absolute_error(actual_b, pred_b)) / 2.0),
            4,
        ),
    }


def _running_accuracy(rows: list[dict[str, Any]], prefix: str) -> None:
    correct = 0
    for index, row in enumerate(rows, start=1):
        correct += int(row["actual_label"] == row[f"{prefix}_predicted_label"])
        row[f"{prefix}_running_correct"] = correct
        row[f"{prefix}_running_accuracy"] = round(correct / index, 4)


def evaluate_walk_forward(
    data_root: Path,
    *,
    limit: int | None = None,
    start: int = 1,
    progress_every: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training, clean = _load_frames(data_root)
    worldcup_matches = _ordered_worldcup_matches(training)
    if start < 1:
        raise ValueError("--start must be 1 or greater")
    selected = worldcup_matches.iloc[start - 1 :]
    if limit is not None:
        selected = selected.head(limit)
    if selected.empty:
        raise RuntimeError("No World Cup 2026 matches selected for evaluation")

    features = list(NEUTRAL_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    target_indices = set(int(index) for index in selected.index)

    frozen_split = _split_frozen_pre_worldcup(training, clean, target_indices)
    frozen_neutral = _prepare_neutral(data_root, training, frozen_split)
    frozen_train = frozen_neutral[frozen_neutral["split"].eq("train")].copy()
    frozen_tests = frozen_neutral[frozen_neutral["split"].eq("test")].copy()
    frozen_models = _train_models(frozen_train, features, categorical)

    rows: list[dict[str, Any]] = []
    for cursor, (target_index, target) in enumerate(selected.iterrows()):
        sequence = start + cursor
        split = _split_for_target(training, clean, int(target_index))
        neutral = _prepare_neutral(data_root, training, split)
        train = neutral[neutral["split"].eq("train")].copy()
        test = neutral[neutral["split"].eq("test")].copy()
        if len(test) != 1:
            raise RuntimeError(f"Expected one test row for index {target_index}, got {len(test)}")

        online_models = _train_models(train, features, categorical)
        online_probs, online_pred_a, online_pred_b = _predict_one(online_models, test, features)

        frozen_test = frozen_tests.iloc[[cursor]]
        frozen_probs, frozen_pred_a, frozen_pred_b = _predict_one(frozen_models, frozen_test, features)

        actual_label = int(test["result_label"].iloc[0])
        row = {
            "sequence": int(sequence),
            "date": str(target["date"]),
            "match_id": str(target.get("match_id", "")),
            "stage_or_round": str(target.get("stage_or_round", "")),
            "team_a": str(target["home_team"]),
            "team_b": str(target["away_team"]),
            "team_a_goals": int(test["team_a_goals"].iloc[0]),
            "team_b_goals": int(test["team_b_goals"].iloc[0]),
            "actual_label": actual_label,
            "actual_result": TARGET_LABELS[actual_label],
            "online_train_matches": int(train["split"].eq("train").sum() / 2),
            "previous_worldcup_matches_in_train": int(sequence - 1),
            "online_p_a": float(online_probs[0]),
            "online_p_draw": float(online_probs[1]),
            "online_p_b": float(online_probs[2]),
            "online_predicted_label": int(np.argmax(online_probs)),
            "online_prediction": TARGET_LABELS[int(np.argmax(online_probs))],
            "online_pred_team_a_goals": online_pred_a,
            "online_pred_team_b_goals": online_pred_b,
            "frozen_p_a": float(frozen_probs[0]),
            "frozen_p_draw": float(frozen_probs[1]),
            "frozen_p_b": float(frozen_probs[2]),
            "frozen_predicted_label": int(np.argmax(frozen_probs)),
            "frozen_prediction": TARGET_LABELS[int(np.argmax(frozen_probs))],
            "frozen_pred_team_a_goals": frozen_pred_a,
            "frozen_pred_team_b_goals": frozen_pred_b,
        }
        row["online_correct"] = int(row["online_predicted_label"] == actual_label)
        row["frozen_correct"] = int(row["frozen_predicted_label"] == actual_label)
        rows.append(row)

        if progress_every and (len(rows) == 1 or len(rows) % progress_every == 0 or len(rows) == len(selected)):
            online_metrics = _metrics(rows, "online")
            frozen_metrics = _metrics(rows, "frozen")
            print(
                "walk-forward "
                f"{len(rows)}/{len(selected)} | "
                f"online {online_metrics['correct']}/{online_metrics['matches']} "
                f"({online_metrics['accuracy']:.2%}) | "
                f"frozen {frozen_metrics['correct']}/{frozen_metrics['matches']} "
                f"({frozen_metrics['accuracy']:.2%})",
                flush=True,
            )

    _running_accuracy(rows, "online")
    _running_accuracy(rows, "frozen")

    summary = {
        "model_id": NEUTRAL_MODEL_RECIPE,
        "policy": (
            "Walk-forward World Cup 2026 evaluation. For each selected played match, "
            "the online model trains on all national-team matches available before that "
            "match, including earlier World Cup 2026 matches, predicts the next match, "
            "then the next step includes that played match. The frozen baseline trains "
            "once on pre-World-Cup national-team matches and predicts the same selected "
            "matches with the same pre-match row features."
        ),
        "features": features,
        "first_worldcup_match_sequence": int(start),
        "first_training_row_index": int(selected.iloc[0]["row_index"]),
        "selected_matches": int(len(selected)),
        "selected_start_date": str(selected["_date"].min().date()),
        "selected_end_date": str(selected["_date"].max().date()),
        "online_retrained": _metrics(rows, "online"),
        "frozen_pre_worldcup": _metrics(rows, "frozen"),
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether sequentially retraining with played World Cup 2026 matches helps.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of WC matches to evaluate.")
    parser.add_argument("--start", type=int, default=1, help="1-based WC match sequence to start from.")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/worldcup2026_walk_forward_retrain_evaluation.json"),
    )
    args = parser.parse_args()

    summary, details = evaluate_walk_forward(
        args.data_root,
        limit=args.limit,
        start=args.start,
        progress_every=args.progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path = args.output.with_name(f"{args.output.stem}_matches.csv")
    details.to_csv(details_path, index=False)
    print(
        json.dumps(
            {
                "summary_path": str(args.output),
                "matches_path": str(details_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
