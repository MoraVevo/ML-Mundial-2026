from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.penalty_model import (  # noqa: E402
    PENALTY_MODEL_FEATURES,
    build_penalty_feature_frame,
    filter_fifa_affiliated_data,
)

COIN_PROBABILITY = 0.5
SELECTION_WINDOWS = (2002, 2006, 2010, 2014)


def _candidate_specs() -> list[dict[str, Any]]:
    history = PENALTY_MODEL_FEATURES[:-1]
    feature_sets = {
        "history": history,
        "history_plus_elo": PENALTY_MODEL_FEATURES,
        "elo_only": ["elo_edge"],
    }
    specs: list[dict[str, Any]] = [
        {
            "name": "honest_coin_flip",
            "model_type": "coin_flip",
            "feature_set": "none",
            "features": [],
            "complexity": 0,
        }
    ]
    for feature_set, features in feature_sets.items():
        for c_value in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            specs.append(
                {
                    "name": f"{feature_set}_logistic_c{c_value:g}",
                    "model_type": "symmetric_logistic",
                    "feature_set": feature_set,
                    "features": features,
                    "c": c_value,
                    "complexity": len(features),
                }
            )
    specs.extend(
        [
            {
                "name": "shallow_random_forest",
                "model_type": "random_forest",
                "feature_set": "history_plus_elo",
                "features": PENALTY_MODEL_FEATURES,
                "complexity": 20,
            },
            {
                "name": "regularized_hist_gradient_boosting",
                "model_type": "hist_gradient_boosting",
                "feature_set": "history_plus_elo",
                "features": PENALTY_MODEL_FEATURES,
                "complexity": 30,
            },
        ]
    )
    return specs


def _fit(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    if spec["model_type"] == "coin_flip":
        return None
    features = list(spec["features"])
    x = frame[features].astype(float).to_numpy()
    y = frame["target"].astype(int).to_numpy()
    x_augmented = np.vstack((x, -x))
    y_augmented = np.concatenate((y, 1 - y))
    if spec["model_type"] == "symmetric_logistic":
        model: Any = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec["c"]),
                        fit_intercept=False,
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif spec["model_type"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=2,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        )
    elif spec["model_type"] == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=2,
            min_samples_leaf=20,
            l2_regularization=5.0,
            learning_rate=0.03,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {spec['model_type']}")
    model.fit(x_augmented, y_augmented)
    return model


def _predict_symmetric(
    model: Any,
    frame: pd.DataFrame,
    spec: dict[str, Any],
) -> np.ndarray:
    if spec["model_type"] == "coin_flip":
        return np.full(len(frame), COIN_PROBABILITY)
    x = frame[list(spec["features"])].astype(float).to_numpy()
    forward = model.predict_proba(x)[:, 1]
    reverse = model.predict_proba(-x)[:, 1]
    return 0.5 * (forward + 1.0 - reverse)


def _wilson_interval(correct: float, matches: int) -> list[float] | None:
    if not matches:
        return None
    z = 1.959963984540054
    p = correct / matches
    denominator = 1.0 + z * z / matches
    center = (p + z * z / (2.0 * matches)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * matches)) / matches) / denominator
    return [round(center - radius, 4), round(center + radius, 4)]


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    y = frame["target"].astype(int).to_numpy()
    ties = np.isclose(probabilities, 0.5, atol=1e-12)
    decisive = ~ties
    predicted = probabilities > 0.5
    decisive_correct = int((predicted[decisive] == y[decisive]).sum())
    expected_correct = decisive_correct + 0.5 * int(ties.sum())
    return {
        "matches": int(len(frame)),
        "decisive_predictions": int(decisive.sum()),
        "unresolved_50_50_predictions": int(ties.sum()),
        "expected_correct": expected_correct,
        "accuracy": round(expected_correct / len(frame), 4),
        "accuracy_wilson_95": _wilson_interval(expected_correct, len(frame)),
        "decisive_accuracy": (
            round(decisive_correct / int(decisive.sum()), 4) if decisive.any() else None
        ),
        "log_loss": round(float(log_loss(y, probabilities, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y, probabilities)), 6),
        "roc_auc": (
            round(float(roc_auc_score(y, probabilities)), 6)
            if len(set(y)) > 1 and not ties.all()
            else None
        ),
        "mean_confidence": round(float(np.mean(np.maximum(probabilities, 1 - probabilities))), 6),
    }


def _candidate_selection(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        fold_rows: list[pd.DataFrame] = []
        fold_probabilities: list[np.ndarray] = []
        for start_year in SELECTION_WINDOWS:
            train = frame[frame["date"] < pd.Timestamp(start_year, 1, 1).date()]
            test = frame[
                (frame["date"] >= pd.Timestamp(start_year, 1, 1).date())
                & (frame["date"] < pd.Timestamp(start_year + 4, 1, 1).date())
            ]
            if len(train) < 50 or test.empty:
                continue
            model = _fit(train, spec)
            fold_rows.append(test)
            fold_probabilities.append(_predict_symmetric(model, test, spec))
        evaluated = pd.concat(fold_rows, ignore_index=True)
        candidate = dict(spec)
        candidate["selection_metrics_pre_2018"] = _metrics(
            evaluated,
            np.concatenate(fold_probabilities),
        )
        candidates.append(candidate)
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["selection_metrics_pre_2018"]["log_loss"],
            item["selection_metrics_pre_2018"]["brier"],
            item["complexity"],
            -item["selection_metrics_pre_2018"]["accuracy"],
        ),
    )
    return ranked[0], ranked


def _prediction_detail(row: dict[str, Any], probability: float) -> dict[str, Any]:
    tied = math.isclose(probability, 0.5, abs_tol=1e-12)
    predicted_winner = None
    correct = None
    if not tied:
        predicted_winner = row["team_a"] if probability > 0.5 else row["team_b"]
        correct = int((probability > 0.5) == bool(row["target"]))
    return {
        "date": row["date"].isoformat(),
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "winner": row["winner"],
        "probability_team_a": round(float(probability), 6),
        "predicted_winner": predicted_winner,
        "correct": correct,
    }


def _expanding_backtest(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    *,
    start_year: int,
    end_year: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated: list[pd.DataFrame] = []
    predictions: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        train = frame[frame["date"] < pd.Timestamp(year, 1, 1).date()]
        test = frame[
            (frame["date"] >= pd.Timestamp(year, 1, 1).date())
            & (frame["date"] < pd.Timestamp(year + 1, 1, 1).date())
        ]
        if test.empty:
            continue
        model = _fit(train, spec)
        probabilities = _predict_symmetric(model, test, spec)
        evaluated.append(test)
        predictions.append(probabilities)
        details.extend(
            _prediction_detail(row, probability)
            for row, probability in zip(test.to_dict("records"), probabilities, strict=True)
        )
    combined = pd.concat(evaluated, ignore_index=True)
    return _metrics(combined, np.concatenate(predictions)), details


def _fixed_holdout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = _fit(train, spec)
    probabilities = _predict_symmetric(model, test, spec)
    details = [
        _prediction_detail(row, probability)
        for row, probability in zip(test.to_dict("records"), probabilities, strict=True)
    ]
    return _metrics(test, probabilities), details


def _first_shooter_audit(frame: pd.DataFrame) -> dict[str, Any]:
    known = frame[frame["first_shooter_edge"].ne(0)].copy()
    first_won = np.where(
        known["first_shooter_edge"].to_numpy() > 0,
        known["target"].to_numpy(),
        1 - known["target"].to_numpy(),
    )
    known["first_shooter_won"] = first_won

    def summary(test: pd.DataFrame) -> dict[str, Any]:
        wins = int(test["first_shooter_won"].sum())
        return {
            "known_first_shooter_matches": int(len(test)),
            "first_shooter_wins": wins,
            "first_shooter_win_rate": round(wins / len(test), 4) if len(test) else None,
        }

    def period(start: int, end: int) -> dict[str, Any]:
        return summary(
            known[
            (known["date"] >= pd.Timestamp(start, 1, 1).date())
            & (known["date"] < pd.Timestamp(end + 1, 1, 1).date())
            ]
        )

    temporal_rows: list[pd.DataFrame] = []
    temporal_probabilities: list[np.ndarray] = []
    for year in range(2018, 2026):
        train = known[known["date"] < pd.Timestamp(year, 1, 1).date()]
        test = known[
            (known["date"] >= pd.Timestamp(year, 1, 1).date())
            & (known["date"] < pd.Timestamp(year + 1, 1, 1).date())
        ]
        if test.empty:
            continue
        first_probability = (float(train["first_shooter_won"].sum()) + 2.0) / (
            len(train) + 4.0
        )
        team_a_probability = np.where(
            test["first_shooter_edge"].to_numpy() > 0,
            first_probability,
            1.0 - first_probability,
        )
        temporal_rows.append(test)
        temporal_probabilities.append(team_a_probability)
    modern = pd.concat(temporal_rows, ignore_index=True)
    modern_probabilities = np.concatenate(temporal_probabilities)
    return {
        "availability": (
            "The identity of the first shooter is known only after the coin toss, "
            "so it cannot be a pre-match feature."
        ),
        "all_known": period(1976, 2026),
        "pre_2018": period(1976, 2017),
        "modern_2018_2025": period(2018, 2025),
        "worldcup_2026": summary(
            known[
                (known["date"] >= pd.Timestamp(2026, 6, 11).date())
                & known["is_world_cup"].eq(1)
            ]
        ),
        "causal_beta_model_modern_2018_2025": _metrics(modern, modern_probabilities),
        "decision": (
            "Rejected from the pre-match model: unavailable before the shootout and "
            "its modern out-of-sample advantage is not stable."
        ),
    }


def _artifact(
    frame: pd.DataFrame,
    selected: dict[str, Any],
    team_records: dict[str, list[dict[str, Any]]],
    h2h_records: dict[str, list[dict[str, Any]]],
    elo_ratings: dict[str, float],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 2,
        "model_id": (
            "neutral_penalty_v2_honest_coin_baseline"
            if selected["model_type"] == "coin_flip"
            else "neutral_penalty_v2_causal_symmetric_logistic"
        ),
        "model_type": selected["model_type"],
        "trained_through": max(frame["date"]).isoformat(),
        "source": "martj42/international_results CC0-1.0",
        "training_matches": int(len(frame)),
        "features": list(selected["features"]),
        "selection_metrics_pre_2018": selected["selection_metrics_pre_2018"],
        "probability_policy": (
            "Neutral and symmetric. Exact 50/50 is retained when no candidate "
            "improves proper scoring rules out of sample."
        ),
    }
    if selected["model_type"] == "coin_flip":
        base["probability"] = COIN_PROBABILITY
        return base
    if selected["model_type"] != "symmetric_logistic":
        raise RuntimeError(
            "The selected tree model needs an explicit portable serializer before deployment."
        )
    model = _fit(frame, selected)
    scaler = model.named_steps["scale"]
    estimator = model.named_steps["model"]
    features = list(selected["features"])
    base.update(
        {
            "regularization_c": float(selected["c"]),
            "scaler_mean": dict(zip(features, scaler.mean_.tolist(), strict=True)),
            "scaler_scale": dict(zip(features, scaler.scale_.tolist(), strict=True)),
            "coefficients": dict(zip(features, estimator.coef_[0].tolist(), strict=True)),
            "team_records": team_records,
            "h2h_records": h2h_records,
            "elo_ratings": elo_ratings,
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neutral national-team shootout model.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/open_international_results"),
    )
    parser.add_argument(
        "--fifa-ranking",
        type=Path,
        default=Path("data/processed/fifa/mens_ranking_latest.csv"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/static/penalty_shootout_model.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/penalty_shootout_model_evaluation.json"),
    )
    args = parser.parse_args()

    shootouts = pd.read_csv(args.source_dir / "shootouts.csv")
    results = pd.read_csv(args.source_dir / "results.csv")
    fifa_ranking = pd.read_csv(args.fifa_ranking)
    shootouts, results, affiliation_audit = filter_fifa_affiliated_data(
        shootouts,
        results,
        set(fifa_ranking["team_key"].astype(str)),
    )
    frame, team_records, h2h_records, elo_ratings = build_penalty_feature_frame(
        shootouts,
        results,
    )
    frame = frame[frame["date"] >= pd.Timestamp(1976, 1, 1).date()].reset_index(drop=True)
    selected, candidates = _candidate_selection(frame)
    best_non_coin = next(
        candidate for candidate in candidates if candidate["model_type"] != "coin_flip"
    )

    modern_metrics, modern_details = _expanding_backtest(
        frame,
        selected,
        start_year=2018,
        end_year=2025,
    )
    worldcup_train = frame[frame["date"] < pd.Timestamp(2026, 6, 11).date()]
    worldcup_test = frame[
        (frame["date"] >= pd.Timestamp(2026, 6, 11).date())
        & frame["is_world_cup"].eq(1)
    ]
    worldcup_metrics, worldcup_details = _fixed_holdout(
        worldcup_train,
        worldcup_test,
        selected,
    )
    non_coin_modern_metrics, _ = _expanding_backtest(
        frame,
        best_non_coin,
        start_year=2018,
        end_year=2025,
    )
    non_coin_worldcup_metrics, non_coin_worldcup_details = _fixed_holdout(
        worldcup_train,
        worldcup_test,
        best_non_coin,
    )

    artifact = _artifact(frame, selected, team_records, h2h_records, elo_ratings)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "model_id": artifact["model_id"],
        "selection_policy": (
            "The recipe is selected only on temporal folds ending before 2018, using "
            "log loss then Brier score. Accuracy is secondary. The 2018-2025 and "
            "World Cup 2026 samples are untouched evaluations, never tuning data."
        ),
        "affiliation_filter": affiliation_audit,
        "eligible_rows_since_1976": int(len(frame)),
        "selected_candidate": selected,
        "candidate_ranking_pre_2018": candidates,
        "best_non_coin_candidate_diagnostic": {
            "candidate": best_non_coin,
            "modern_expanding_backtest_2018_2025": non_coin_modern_metrics,
            "worldcup_2026_holdout": non_coin_worldcup_metrics,
            "worldcup_2026_details": non_coin_worldcup_details,
            "deployment_decision": (
                "Rejected because it did not beat 50/50 on the pre-2018 proper-score "
                "selection sample and its later performance is not stable across periods."
            ),
        },
        "modern_expanding_backtest_2018_2025": modern_metrics,
        "worldcup_2026_holdout": worldcup_metrics,
        "worldcup_2026_details": worldcup_details,
        "first_shooter_audit": _first_shooter_audit(frame),
        "modern_details": modern_details,
        "coin_flip_reference": {
            "expected_accuracy": 0.5,
            "log_loss": round(float(-math.log(0.5)), 6),
            "brier": 0.25,
        },
        "artifact": str(args.artifact),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
