from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kinela.lightgbm_model import (
    CATEGORICAL_FEATURES,
    _build_neutral_frame,
    _load_matrix,
)
from kinela.model import DETAIL_STAT_FEATURES
from kinela.worldcup_2026 import WorldCup2026Simulator, _normalise_name


DETAIL_DIFF_FEATURES = [f"recent6_{feature}_diff" for feature in DETAIL_STAT_FEATURES]
MODEL_PATH = Path("models") / "lightgbm_neutral_fouls_model.joblib"
METRICS_PATH = Path("models") / "lightgbm_neutral_fouls_metrics.json"
POISSON_MODEL_PATH = Path("models") / "poisson_neutral_fouls_model.joblib"
POISSON_METRICS_PATH = Path("models") / "poisson_neutral_fouls_metrics.json"
PARSIMONIOUS_FOULS_FEATURES = [
    "stage_or_round",
    "same_confederation",
    "team_a_rest_days",
    "team_b_rest_days",
    "elo_diff",
    "team_a_fifa_rank",
    "team_b_fifa_rank",
    "team_a_cross_confederation_strength",
    "team_b_cross_confederation_strength",
    "match_attack_defense_volume",
    "team_a_recent6_fouls_avg",
    "team_b_recent6_fouls_avg",
    "recent6_yellow_cards_diff",
]
FOULS_FEATURES = PARSIMONIOUS_FOULS_FEATURES


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_foul_targets(data_root: Path) -> dict[tuple[str, str], float]:
    targets: dict[tuple[str, str], float] = {}
    api_path = data_root / "processed" / "api_football" / "team_detailed_stats.csv"
    if api_path.exists():
        for row in csv.DictReader(api_path.open(encoding="utf-8")):
            fouls = _to_float(row.get("fouls"))
            if fouls is None:
                continue
            targets[(f"api:{row['fixture_id']}", _normalise_name(row.get("team")))] = fouls

    statsbomb_path = data_root / "processed" / "statsbomb_world_cup_2022" / "team_match_stats.csv"
    if statsbomb_path.exists():
        for row in csv.DictReader(statsbomb_path.open(encoding="utf-8")):
            fouls = _to_float(row.get("fouls"))
            if fouls is None:
                continue
            targets[(f"sb:{row['match_id']}", _normalise_name(row.get("team")))] = fouls
    return targets


def _load_training_metadata(data_root: Path) -> pd.DataFrame:
    path = data_root / "processed" / "combined" / "training_frame.csv"
    return pd.DataFrame(list(csv.DictReader(path.open(encoding="utf-8"))))


def _has_complete_foul_targets(
    targets: dict[tuple[str, str], float],
    match_id: str,
    home_team: str,
    away_team: str,
) -> bool:
    return (
        (match_id, _normalise_name(home_team)) in targets
        and (match_id, _normalise_name(away_team)) in targets
    )


def _apply_fouls_temporal_split(
    data_root: Path,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    targets = _load_foul_targets(data_root)
    valid_indices = [
        index
        for index, meta in metadata.iterrows()
        if _has_complete_foul_targets(
            targets,
            meta["match_id"],
            meta["home_team"],
            meta["away_team"],
        )
    ]
    if not valid_indices:
        raise ValueError("No matches with complete foul targets found")

    ordered = sorted(valid_indices, key=lambda index: metadata.iloc[index]["date"])
    test_count = max(1, round(len(ordered) * test_fraction))
    train_indices = set(ordered[:-test_count])
    test_indices = set(ordered[-test_count:])

    result = frame.copy()
    result["split"] = "unused"
    for index in train_indices:
        result.at[index, "split"] = "train"
    for index in test_indices:
        result.at[index, "split"] = "test"
    return result


def _target_rows(
    data_root: Path,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    augment: bool,
) -> list[dict[str, Any]]:
    targets = _load_foul_targets(data_root)
    rows: list[dict[str, Any]] = []
    for frame_index, row in frame.iterrows():
        meta = metadata.iloc[frame_index]
        match_id = meta["match_id"]
        home_key = _normalise_name(meta["home_team"])
        away_key = _normalise_name(meta["away_team"])
        home_fouls = targets.get((match_id, home_key))
        away_fouls = targets.get((match_id, away_key))
        pairs = [("home", "away", home_fouls, away_fouls)]
        if augment and row["split"] == "train":
            pairs.append(("away", "home", away_fouls, home_fouls))
        for a_side, b_side, a_fouls, b_fouls in pairs:
            rows.append(
                {
                    "match_id": match_id,
                    "date": meta["date"],
                    "team_a": meta[f"{a_side}_team"],
                    "team_b": meta[f"{b_side}_team"],
                    "team_a_fouls": a_fouls,
                    "team_b_fouls": b_fouls,
                    "total_fouls": None
                    if a_fouls is None or b_fouls is None
                    else a_fouls + b_fouls,
                }
            )
    return rows


def _detail_absolute_rows(frame: pd.DataFrame, *, augment: bool) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        pairs = [("home", "away")]
        if augment and row["split"] == "train":
            pairs.append(("away", "home"))
        for a_side, b_side in pairs:
            rows.append(
                {
                    "team_a_recent6_fouls_avg": _to_float(
                        row.get(f"{a_side}_recent6_fouls_avg")
                    )
                    or 0.0,
                    "team_b_recent6_fouls_avg": _to_float(
                        row.get(f"{b_side}_recent6_fouls_avg")
                    )
                    or 0.0,
                }
            )
    return rows


def _fouls_frame(
    data_root: Path,
    *,
    augment: bool,
    fouls_temporal_split: bool = True,
) -> pd.DataFrame:
    frame = _load_matrix(data_root)
    metadata = _load_training_metadata(data_root)
    if fouls_temporal_split:
        frame = _apply_fouls_temporal_split(data_root, frame, metadata)
    neutral = _build_neutral_frame(frame, augment=augment)
    targets = pd.DataFrame(_target_rows(data_root, frame, metadata, augment=augment))
    detail_absolutes = pd.DataFrame(_detail_absolute_rows(frame, augment=augment))
    if len(neutral) != len(targets):
        raise ValueError(f"Feature/target row mismatch: {len(neutral)} != {len(targets)}")
    if len(neutral) != len(detail_absolutes):
        raise ValueError(
            f"Feature/detail row mismatch: {len(neutral)} != {len(detail_absolutes)}"
        )
    combined = pd.concat(
        [neutral.reset_index(drop=True), detail_absolutes, targets],
        axis=1,
    )
    combined = combined.dropna(subset=["team_a_fouls", "team_b_fouls", "total_fouls"]).copy()
    for column in DETAIL_DIFF_FEATURES:
        if column not in combined:
            combined[column] = 0.0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
    return combined


def export_neutral_fouls_matrix(data_root: Path) -> list[dict[str, Any]]:
    frame = _fouls_frame(data_root, augment=False)
    columns = [
        "split",
        "match_id",
        "date",
        "team_a",
        "team_b",
        *FOULS_FEATURES,
        "team_a_fouls",
        "team_b_fouls",
        "total_fouls",
    ]
    return frame[columns].to_dict(orient="records")


def train_lightgbm_neutral_fouls(data_root: Path) -> dict[str, Any]:
    frame = _fouls_frame(data_root, augment=True)
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    for column in CATEGORICAL_FEATURES:
        train[column] = train[column].astype("category")
        test[column] = test[column].astype("category")

    x_train = train[FOULS_FEATURES]
    x_test = test[FOULS_FEATURES]
    weights = train["match_recency_weight"].astype(float).to_numpy()
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in FOULS_FEATURES]
    params = {
        "objective": "regression",
        "n_estimators": 160,
        "learning_rate": 0.04,
        "num_leaves": 7,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "verbosity": -1,
    }
    models = {
        "team_a_fouls": lgb.LGBMRegressor(**params),
        "team_b_fouls": lgb.LGBMRegressor(**params),
        "total_fouls": lgb.LGBMRegressor(**params),
    }
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "model": "lightgbm_neutral_fouls",
        "training_rows_after_augmentation": int(len(train)),
        "test_matches": int(len(test)),
        "features": len(FOULS_FEATURES),
        "feature_budget_rule": "parsimonious: 13 pre-one-hot features for 181 unique train matches",
        "target_coverage_rows": int(len(frame)),
    }
    for target, model in models.items():
        model.fit(
            x_train,
            train[target],
            sample_weight=weights,
            categorical_feature=categorical,
        )
        pred = np.clip(model.predict(x_test), 0, None)
        predictions[target] = pred
        metrics[f"mae_{target}"] = round(float(mean_absolute_error(test[target], pred)), 4)
        metrics[f"rmse_{target}"] = round(
            float(np.sqrt(mean_squared_error(test[target], pred))),
            4,
        )
        metrics[f"r2_{target}"] = round(float(r2_score(test[target], pred)), 4)

    output_dir = data_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": FOULS_FEATURES}, output_dir / MODEL_PATH.name)

    payload = {
        "metrics": metrics,
        "feature_importances": {
            target: sorted(
                [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(
                        FOULS_FEATURES,
                        model.feature_importances_,
                        strict=True,
                    )
                ],
                key=lambda item: item["importance"],
                reverse=True,
            )
            for target, model in models.items()
        },
    }
    (output_dir / METRICS_PATH.name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = test[["match_id", "date", "team_a", "team_b", "team_a_fouls", "team_b_fouls"]].copy()
    out["expected_team_a_fouls"] = np.round(predictions["team_a_fouls"], 2)
    out["expected_team_b_fouls"] = np.round(predictions["team_b_fouls"], 2)
    out["expected_total_fouls"] = np.round(predictions["total_fouls"], 2)
    out.to_csv(
        data_root / "processed" / "combined" / "lightgbm_neutral_fouls_predictions.csv",
        index=False,
    )
    return metrics


def _poisson_pipeline() -> Pipeline:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in PARSIMONIOUS_FOULS_FEATURES]
    numeric = [feature for feature in PARSIMONIOUS_FOULS_FEATURES if feature not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", PoissonRegressor(alpha=1.0, max_iter=1000)),
        ]
    )


def train_poisson_neutral_fouls(data_root: Path) -> dict[str, Any]:
    frame = _fouls_frame(data_root, augment=True)
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    x_train = train[PARSIMONIOUS_FOULS_FEATURES]
    x_test = test[PARSIMONIOUS_FOULS_FEATURES]
    weights = train["match_recency_weight"].astype(float).to_numpy()
    models = {
        "team_a_fouls": _poisson_pipeline(),
        "team_b_fouls": _poisson_pipeline(),
        "total_fouls": _poisson_pipeline(),
    }
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "model": "poisson_neutral_fouls",
        "training_rows_after_augmentation": int(len(train)),
        "test_matches": int(len(test)),
        "features": len(PARSIMONIOUS_FOULS_FEATURES),
        "feature_budget_rule": "parsimonious: 13 pre-one-hot features for 181 unique train matches",
        "target_coverage_rows": int(len(frame)),
    }
    for target, model in models.items():
        model.fit(x_train, train[target], model__sample_weight=weights)
        pred = np.clip(model.predict(x_test), 0, None)
        predictions[target] = pred
        metrics[f"mae_{target}"] = round(float(mean_absolute_error(test[target], pred)), 4)
        metrics[f"rmse_{target}"] = round(
            float(np.sqrt(mean_squared_error(test[target], pred))),
            4,
        )
        metrics[f"r2_{target}"] = round(float(r2_score(test[target], pred)), 4)

    output_dir = data_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models, "features": PARSIMONIOUS_FOULS_FEATURES},
        output_dir / POISSON_MODEL_PATH.name,
    )
    (output_dir / POISSON_METRICS_PATH.name).write_text(
        json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = test[["match_id", "date", "team_a", "team_b", "team_a_fouls", "team_b_fouls"]].copy()
    out["expected_team_a_fouls"] = np.round(predictions["team_a_fouls"], 2)
    out["expected_team_b_fouls"] = np.round(predictions["team_b_fouls"], 2)
    out["expected_total_fouls"] = np.round(predictions["total_fouls"], 2)
    out.to_csv(
        data_root / "processed" / "combined" / "poisson_neutral_fouls_predictions.csv",
        index=False,
    )
    return metrics


def _predict_match(
    simulator: WorldCup2026Simulator,
    model_bundle: dict[str, Any],
    team_a: str,
    team_b: str,
    match_date: date,
    stage: str,
) -> dict[str, float]:
    row = simulator._lightgbm_features(team_a, team_b, match_date, stage)
    frame = pd.DataFrame([{feature: row.get(feature, 0) for feature in model_bundle["features"]}])
    for column in CATEGORICAL_FEATURES:
        if column in frame:
            frame[column] = frame[column].fillna("unknown").astype("category")
    result = {}
    for target, model in model_bundle["models"].items():
        result[target] = round(float(max(0.0, model.predict(frame)[0])), 2)
    return result


def predict_next_team_fouls(
    data_root: Path,
    *,
    teams: list[str],
    from_date: date,
    limit_per_team: int = 1,
    model_name: str = "lightgbm",
) -> dict[str, Any]:
    if model_name == "poisson":
        model_path = data_root / POISSON_MODEL_PATH
        label = "poisson_neutral_fouls"
    else:
        model_path = data_root / MODEL_PATH
        label = "lightgbm_neutral_fouls"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing fouls model: {model_path}. Run train first.")
    model_bundle = joblib.load(model_path)
    simulator = WorldCup2026Simulator(data_root, engine="poisson")
    rows: list[dict[str, Any]] = []
    for team in teams:
        team_key = _normalise_name(team)
        matches = []
        for match in simulator.fixtures:
            match_date = date.fromisoformat(match["utcDate"][:10])
            if match_date < from_date:
                continue
            home = (match.get("homeTeam") or {}).get("name") or ""
            away = (match.get("awayTeam") or {}).get("name") or ""
            if team_key not in {_normalise_name(home), _normalise_name(away)}:
                continue
            matches.append((match_date, match))
        for match_date, match in sorted(matches, key=lambda item: item[0])[:limit_per_team]:
            team_a = match["homeTeam"]["name"]
            team_b = match["awayTeam"]["name"]
            prediction = _predict_match(
                simulator,
                model_bundle,
                team_a,
                team_b,
                match_date,
                match.get("stage") or "GROUP_STAGE",
            )
            rows.append(
                {
                    "requested_team": team,
                    "date": match_date.isoformat(),
                    "group": str(match.get("group", "")).replace("GROUP_", ""),
                    "match_id": match.get("id"),
                    "team_a": team_a,
                    "team_b": team_b,
                    "expected_team_a_fouls": prediction["team_a_fouls"],
                    "expected_team_b_fouls": prediction["team_b_fouls"],
                    "expected_total_fouls": prediction["total_fouls"],
                }
            )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    slug = "_".join(_normalise_name(team).replace(" ", "_") for team in teams)
    csv_path = output_dir / f"{model_name}_fouls_next_matches_{slug}.csv"
    json_path = output_dir / f"{model_name}_fouls_next_matches_{slug}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model": label,
        "from_date": from_date.isoformat(),
        "teams": teams,
        "matches": rows,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
