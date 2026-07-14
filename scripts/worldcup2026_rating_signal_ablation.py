"""Causal ablation of official FIFA, reconstructed live FIFA and Elo signals.

The World Cup 2026 rows are always held out.  Every rating attached to a match
is the value available immediately before that match; completed-match updates
are applied only after the snapshot is recorded.
"""

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
    _prepare_neutral,
    _split_worldcup_2026,
)
from kinela.fifa_ranking import (  # noqa: E402
    fifa_sum_match_importance,
    normalise_team_name,
    update_fifa_sum_points,
)
from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DRAW_PRESSURE_TOTAL_GOALS_PRIOR,
    NEUTRAL_BASE_RESULT_FEATURES,
    NEUTRAL_GOAL_FEATURES,
    NEUTRAL_RESULT_FEATURES,
    NEUTRAL_XG_RESULT_BLEND_WEIGHT,
    blend_result_probabilities,
)
from kinela.model import RECENT_FORM_WINDOW  # noqa: E402


@dataclass(frozen=True)
class EloRecipe:
    name: str
    k: float
    scale: float = 400.0
    margin: float = 0.15
    friendly_weight: float = 1.0
    worldcup_weight: float = 1.0
    seed_from_fifa: bool = False


def _categorical(features: list[str]) -> list[str]:
    return [item for item in CATEGORICAL_FEATURES if item in features]


def _metrics(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    labels = test["result_label"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    mae_a = float(mean_absolute_error(test["team_a_goals"], pred_a))
    mae_b = float(mean_absolute_error(test["team_b_goals"], pred_b))
    return {
        "matches": int(len(test)),
        "correct": int((predicted == labels).sum()),
        "accuracy": round(float(accuracy_score(labels, predicted)), 6),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 6),
        "mae_team_a_goals": round(mae_a, 6),
        "mae_team_b_goals": round(mae_b, 6),
        "mae_goals_avg": round((mae_a + mae_b) / 2.0, 6),
    }


def _fit_variant(
    frame: pd.DataFrame,
    *,
    extra_features: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    train = frame[frame["split"].eq("train")].copy()
    test = frame[frame["split"].eq("test")].copy()
    goal_features = [*NEUTRAL_GOAL_FEATURES, *extra_features]
    base_result_features = [*NEUTRAL_BASE_RESULT_FEATURES, *extra_features]
    result_features = [*NEUTRAL_RESULT_FEATURES, *extra_features]
    weights = train["match_recency_weight"].astype(float).to_numpy(copy=True)

    goal_a = lgb.LGBMRegressor(**REG_PARAMS).fit(
        train[goal_features],
        train["team_a_goals"],
        sample_weight=weights,
        categorical_feature=_categorical(goal_features),
    )
    goal_b = lgb.LGBMRegressor(**REG_PARAMS).fit(
        train[goal_features],
        train["team_b_goals"],
        sample_weight=weights,
        categorical_feature=_categorical(goal_features),
    )
    base = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS), method="sigmoid", cv=3
    ).fit(
        train[base_result_features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=_categorical(base_result_features),
    )
    xg = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS), method="sigmoid", cv=3
    ).fit(
        train[result_features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=_categorical(result_features),
    )
    probabilities = blend_result_probabilities(
        base.predict_proba(test[base_result_features]),
        xg.predict_proba(test[result_features]),
        weight=NEUTRAL_XG_RESULT_BLEND_WEIGHT,
    )
    pred_a = np.clip(goal_a.predict(test[goal_features]), 0.0, None)
    pred_b = np.clip(goal_b.predict(test[goal_features]), 0.0, None)
    full = _metrics(test, probabilities, pred_a, pred_b)
    windows: dict[str, dict[str, Any]] = {}
    for size in (25, 50, 75, len(test)):
        if size > len(test):
            continue
        windows[str(size)] = _metrics(
            test.iloc[:size], probabilities[:size], pred_a[:size], pred_b[:size]
        )
    return full, windows


def _rating_components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    observed = (
        frame["team_a_historical_fifa_observed"].astype(float).clip(0.0, 1.0)
        * frame["team_b_historical_fifa_observed"].astype(float).clip(0.0, 1.0)
    )
    elo = np.tanh(frame["elo_diff"].astype(float) / 350.0)
    official_points = observed * np.tanh(
        frame["historical_fifa_points_diff"].astype(float) / 250.0
    )
    live_points = observed * np.tanh(frame["live_fifa_points_diff"].astype(float) / 250.0)
    rank = observed * np.tanh(
        (
            frame["team_b_historical_fifa_rank"].astype(float)
            - frame["team_a_historical_fifa_rank"].astype(float)
        )
        / 45.0
    )
    official_anchor = observed * (
        0.62
        * np.tanh(frame["historical_fifa_points_diff"].astype(float) / 190.0)
        + 0.38
        * np.tanh(
            (
                frame["team_b_historical_fifa_rank"].astype(float)
                - frame["team_a_historical_fifa_rank"].astype(float)
            )
            / 38.0
        )
    )
    live_anchor = observed * (
        0.62 * np.tanh(frame["live_fifa_points_diff"].astype(float) / 190.0)
        + 0.38
        * np.tanh(
            (
                frame["team_b_historical_fifa_rank"].astype(float)
                - frame["team_a_historical_fifa_rank"].astype(float)
            )
            / 38.0
        )
    )
    return {
        "observed": observed,
        "elo": elo,
        "official_points": official_points,
        "live_points": live_points,
        "rank": rank,
        "official_anchor": official_anchor,
        "live_anchor": live_anchor,
        "confed": np.tanh(frame["confederation_strength_diff"].astype(float) / 0.8),
        "threat": np.tanh(frame["threat_edge"].astype(float) / 1.25),
    }


def _apply_rating_variant(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = frame.copy()
    c = _rating_components(out)
    if name == "production_fifa_sum_live_no_custom_elo":
        return out, ()
    if name == "no_rating_form_only":
        out["rating_threat_edge"] = c["threat"]
        out["rating_guardrail_edge"] = 0.0
        return out, ()
    if name == "official_fifa_only":
        consensus = 0.76 * c["official_points"] + 0.24 * c["rank"]
        out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
        out["rating_guardrail_edge"] = c["official_anchor"] - out["rating_threat_edge"]
        return out, ()
    if name == "elo_only":
        consensus = 0.84 * c["elo"] + 0.16 * c["confed"]
        out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
        out["rating_guardrail_edge"] = 0.0
        return out, ()
    if name == "live_fifa_only":
        consensus = 0.76 * c["live_points"] + 0.24 * c["rank"]
        out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
        out["rating_guardrail_edge"] = c["live_anchor"] - out["rating_threat_edge"]
        return out, ()
    if name == "live_fifa_plus_elo":
        consensus = (
            0.42 * c["elo"]
            + 0.38 * c["live_points"]
            + 0.12 * c["rank"]
            + 0.08 * c["confed"]
        )
        out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
        out["rating_guardrail_edge"] = c["live_anchor"] - out["rating_threat_edge"]
        return out, ()
    if name == "production_plus_legacy_live_drift":
        out["rating_live_drift_edge"] = c["live_anchor"] - c["official_anchor"]
        return out, ("rating_live_drift_edge",)
    raise KeyError(name)


def _elo_diffs(training: pd.DataFrame, recipe: EloRecipe) -> np.ndarray:
    ratings: dict[str, float] = {}
    diffs: list[float] = []
    for _, row in training.iterrows():
        a = normalise_team_name(str(row["home_team"]))
        b = normalise_team_name(str(row["away_team"]))
        if a not in ratings:
            ratings[a] = (
                float(row["home_historical_fifa_points"])
                if recipe.seed_from_fifa and float(row["home_historical_fifa_observed"])
                else 1500.0
            )
        if b not in ratings:
            ratings[b] = (
                float(row["away_historical_fifa_points"])
                if recipe.seed_from_fifa and float(row["away_historical_fifa_observed"])
                else 1500.0
            )
        ra, rb = ratings[a], ratings[b]
        diffs.append(ra - rb)
        expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / recipe.scale))
        ga, gb = int(row["home_goals"]), int(row["away_goals"])
        actual = 1.0 if ga > gb else 0.5 if ga == gb else 0.0
        margin = min(abs(ga - gb), 3)
        weight = 1.0
        if str(row.get("competition_type", "")).casefold() == "friendly":
            weight *= recipe.friendly_weight
        if str(row.get("competition_name", "")).casefold() == "fifa world cup":
            weight *= recipe.worldcup_weight
        change = recipe.k * weight * (1.0 + recipe.margin * margin) * (actual - expected)
        ratings[a], ratings[b] = ra + change, rb - change
    return np.asarray(diffs, dtype=float)


def _expand_match_values(values: np.ndarray, split_clean: pd.DataFrame) -> np.ndarray:
    expanded: list[float] = []
    for value, split in zip(values, split_clean["split"], strict=True):
        expanded.append(float(value))
        if split == "train":
            expanded.append(float(-value))
    return np.asarray(expanded, dtype=float)


def _fifa_importance(row: pd.Series) -> float:
    return fifa_sum_match_importance(
        str(row.get("competition_name", "")),
        str(row.get("competition_type", "")),
        str(row.get("stage_or_round", "")),
    )


def _fifa_sum_live_diffs(
    training: pd.DataFrame,
    history_path: Path,
    manual_results_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(history_path, low_memory=False)
    schedule_points: dict[str, dict[str, float]] = {}
    for schedule_id, rows in history.groupby("schedule_id", sort=False):
        schedule_points[str(schedule_id)] = {
            normalise_team_name(str(row.team_name)): float(row.total_points)
            for row in rows.itertuples()
        }
    manual = pd.read_csv(manual_results_path, dtype={"match_id": str})
    extra_winners = {
        str(row.match_id).split(":")[-1]: str(row.extra_time_winner)
        for row in manual.itertuples()
        if pd.notna(row.extra_time_winner) and str(row.extra_time_winner).strip()
    }
    points: dict[str, float] = {}
    active_schedule = ""
    diffs: list[float] = []
    snapshots: list[dict[str, Any]] = []
    for _, row in training.iterrows():
        schedule_id = str(row.get("home_historical_fifa_schedule_id", ""))
        if schedule_id and schedule_id != "nan" and schedule_id != active_schedule:
            points = dict(schedule_points.get(schedule_id, {}))
            active_schedule = schedule_id
        a = normalise_team_name(str(row["home_team"]))
        b = normalise_team_name(str(row["away_team"]))
        pa = points.get(a, float(row["home_historical_fifa_points"]))
        pb = points.get(b, float(row["away_historical_fifa_points"]))
        points.setdefault(a, pa)
        points.setdefault(b, pb)
        ordered = sorted(points.items(), key=lambda item: (-item[1], item[0]))
        ranks = {team: position for position, (team, _) in enumerate(ordered, start=1)}
        diffs.append(pa - pb)
        snapshots.append(
            {
                "match_id": str(row["match_id"]),
                "date": str(row["date"]),
                "stage": str(row["stage_or_round"]),
                "team_a": str(row["home_team"]),
                "team_b": str(row["away_team"]),
                "official_schedule_id": active_schedule,
                "team_a_official_rank": int(row["home_historical_fifa_rank"]),
                "team_b_official_rank": int(row["away_historical_fifa_rank"]),
                "team_a_official_points": float(row["home_historical_fifa_points"]),
                "team_b_official_points": float(row["away_historical_fifa_points"]),
                "team_a_live_rank_pre_match": ranks[a],
                "team_b_live_rank_pre_match": ranks[b],
                "team_a_live_points_pre_match": round(pa, 6),
                "team_b_live_points_pre_match": round(pb, 6),
            }
        )
        penalty_a = pd.to_numeric(pd.Series([row.get("home_penalty_goals")]), errors="coerce").iloc[0]
        penalty_b = pd.to_numeric(pd.Series([row.get("away_penalty_goals")]), errors="coerce").iloc[0]
        if pd.notna(penalty_a) and pd.notna(penalty_b):
            actual_a, actual_b = ((0.75, 0.5) if penalty_a > penalty_b else (0.5, 0.75))
        else:
            extra_winner = extra_winners.get(str(row["match_id"]).split(":")[-1], "")
            if extra_winner:
                actual_a, actual_b = (
                    (1.0, 0.0)
                    if normalise_team_name(extra_winner) == a
                    else (0.0, 1.0)
                )
            else:
                ga, gb = int(row["home_goals"]), int(row["away_goals"])
                actual_a, actual_b = (
                    (1.0, 0.0) if ga > gb else (0.0, 1.0) if gb > ga else (0.5, 0.5)
                )
        is_final_competition_knockout = (
            str(row.get("competition_name", "")).casefold() == "fifa world cup"
            and str(row.get("stage_or_round", "")).upper() != "GROUP_STAGE"
        )
        points[a], points[b] = update_fifa_sum_points(
            pa,
            pb,
            team_a_result=actual_a,
            team_b_result=actual_b,
            importance=_fifa_importance(row),
            protect_negative=is_final_competition_knockout,
        )
    latest = pd.DataFrame(
        [
            {"rank": position, "team_key": team, "live_points": round(value, 6)}
            for position, (team, value) in enumerate(
                sorted(points.items(), key=lambda item: (-item[1], item[0])), start=1
            )
        ]
    )
    return np.asarray(diffs, dtype=float), pd.DataFrame(snapshots), latest


def _apply_fifa_sum_variant(
    neutral: pd.DataFrame,
    expanded_live_diff: np.ndarray,
    *,
    use_elo: bool,
    add_drift: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = neutral.copy()
    out["fifa_sum_live_points_diff"] = expanded_live_diff
    c = _rating_components(out)
    live_points = c["observed"] * np.tanh(out["fifa_sum_live_points_diff"] / 250.0)
    live_anchor = c["official_anchor"] + c["observed"] * 0.62 * (
        np.tanh(out["fifa_sum_live_points_diff"] / 190.0)
        - np.tanh(out["historical_fifa_points_diff"] / 190.0)
    )
    if use_elo:
        consensus = 0.42 * c["elo"] + 0.38 * live_points + 0.12 * c["rank"] + 0.08 * c["confed"]
    else:
        consensus = 0.76 * live_points + 0.24 * c["rank"]
    out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
    out["rating_guardrail_edge"] = live_anchor - out["rating_threat_edge"]
    if add_drift:
        out["rating_fifa_sum_live_drift_edge"] = live_anchor - c["official_anchor"]
        return out, ("rating_fifa_sum_live_drift_edge",)
    return out, ()


def _apply_single_points_rating(
    neutral: pd.DataFrame,
    expanded_points_diff: np.ndarray,
    *,
    plain_form: bool,
) -> pd.DataFrame:
    """Use one causal rating signal without a second Elo or rank channel."""
    out = neutral.copy()
    out["rating_threat_edge"] = np.tanh(expanded_points_diff / 190.0)
    out["rating_guardrail_edge"] = 0.0
    if plain_form:
        out["quality_form_edge"] = out["recent_points_form_edge"]
        control_quality = (
            0.40 * out["rating_threat_edge"]
            + 0.30 * np.tanh(out["threat_edge"].astype(float) / 1.25)
            + 0.20 * np.tanh(out["quality_form_edge"].astype(float) / 1.8)
            + 0.10 * np.tanh(out["goal_balance_edge"].astype(float) / 1.5)
        )
        parity = 1.0 - control_quality.abs().clip(upper=1.0)
        raw_total_goals = np.expm1(out["tempo_index"].astype(float)).clip(lower=0.0)
        coverage = (
            out["team_a_train_matches"].astype(float).clip(0.0, RECENT_FORM_WINDOW)
            + out["team_b_train_matches"].astype(float).clip(0.0, RECENT_FORM_WINDOW)
        ) / (2.0 * RECENT_FORM_WINDOW)
        reliable_total = coverage * raw_total_goals + (
            1.0 - coverage
        ) * DRAW_PRESSURE_TOTAL_GOALS_PRIOR
        out["draw_pressure_index"] = parity * (1.0 / (1.0 + np.log1p(reliable_total)))
    return out


def _apply_elo_recipe(
    neutral: pd.DataFrame,
    training: pd.DataFrame,
    split_clean: pd.DataFrame,
    recipe: EloRecipe,
) -> pd.DataFrame:
    out = neutral.copy()
    out["elo_diff"] = _expand_match_values(_elo_diffs(training, recipe), split_clean)
    c = _rating_components(out)
    consensus = (
        0.42 * c["elo"]
        + 0.38 * c["official_points"]
        + 0.12 * c["rank"]
        + 0.08 * c["confed"]
    )
    out["rating_threat_edge"] = 0.52 * consensus + 0.48 * c["threat"]
    out["rating_guardrail_edge"] = c["official_anchor"] - out["rating_threat_edge"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/worldcup2026_rating_signal_ablation_2026-07-13.json"),
    )
    args = parser.parse_args()
    training = pd.read_csv(
        args.data_root / "processed/combined/training_frame_national.csv", low_memory=False
    )
    clean = pd.read_csv(
        args.data_root / "processed/combined/clean_training_matrix_national.csv", low_memory=False
    )
    split_clean, split_meta = _split_worldcup_2026(training, clean)
    neutral = _prepare_neutral(args.data_root, training, split_clean)
    fifa_sum_diffs, fifa_sum_snapshots, fifa_sum_latest = _fifa_sum_live_diffs(
        training,
        args.data_root / "processed/fifa/mens_ranking_history.csv",
        args.data_root / "static/worldcup_2026_manual_results.csv",
    )
    expanded_fifa_sum_diffs = _expand_match_values(fifa_sum_diffs, split_clean)

    variants: list[dict[str, Any]] = []
    for name in (
        "production_fifa_sum_live_no_custom_elo",
        "no_rating_form_only",
        "official_fifa_only",
        "elo_only",
        "live_fifa_only",
        "live_fifa_plus_elo",
        "production_plus_legacy_live_drift",
    ):
        frame, extra = _apply_rating_variant(neutral, name)
        full, windows = _fit_variant(frame, extra_features=extra)
        item = {"variant": name, "extra_features": list(extra), "metrics": full, "windows": windows}
        variants.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    for name, use_elo, add_drift in (
        ("fifa_sum_live_only", False, False),
        ("fifa_sum_live_plus_elo", True, False),
        ("official_plus_elo_plus_fifa_sum_drift", True, True),
    ):
        frame, extra = _apply_fifa_sum_variant(
            neutral,
            expanded_fifa_sum_diffs,
            use_elo=use_elo,
            add_drift=add_drift,
        )
        full, windows = _fit_variant(frame, extra_features=extra)
        item = {"variant": name, "extra_features": list(extra), "metrics": full, "windows": windows}
        variants.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    for name, source_diff, plain_form in (
        ("fifa_sum_points_only_quality_form", expanded_fifa_sum_diffs, False),
        ("fifa_sum_points_only_plain_form_no_custom_elo", expanded_fifa_sum_diffs, True),
        (
            "official_fifa_points_only_plain_form_no_custom_elo",
            neutral["historical_fifa_points_diff"].astype(float).to_numpy(),
            True,
        ),
    ):
        frame = _apply_single_points_rating(
            neutral,
            source_diff,
            plain_form=plain_form,
        )
        full, windows = _fit_variant(frame)
        item = {"variant": name, "extra_features": [], "metrics": full, "windows": windows}
        variants.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    elo_recipes = (
        EloRecipe("elo_current_rebuilt_k24", 24.0),
        EloRecipe("elo_k16_friendlies060", 16.0, friendly_weight=0.6),
        EloRecipe("elo_k20_friendlies060", 20.0, friendly_weight=0.6),
        EloRecipe("elo_k16_friendlies060_wc125", 16.0, friendly_weight=0.6, worldcup_weight=1.25),
        EloRecipe("elo_k16_scale600_friendlies060", 16.0, scale=600.0, friendly_weight=0.6),
        EloRecipe("elo_fifa_seed_k16_friendlies060", 16.0, friendly_weight=0.6, seed_from_fifa=True),
    )
    elo_results: list[dict[str, Any]] = []
    for recipe in elo_recipes:
        frame = _apply_elo_recipe(neutral, training, split_clean, recipe)
        full, windows = _fit_variant(frame)
        item = {"variant": recipe.name, "recipe": recipe.__dict__, "metrics": full, "windows": windows}
        elo_results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    test = neutral[neutral["split"].eq("test")]
    correlation_columns = [
        "elo_diff",
        "historical_fifa_points_diff",
        "live_fifa_points_diff",
        "rating_threat_edge",
        "rating_guardrail_edge",
        "rating_drift_edge",
    ]
    correlations = test[correlation_columns].astype(float).corr().round(6).to_dict()
    ranked = sorted(
        [*variants, *elo_results],
        key=lambda item: (
            -item["metrics"]["accuracy"],
            item["metrics"]["log_loss"],
            item["metrics"]["mae_goals_avg"],
        ),
    )
    payload = {
        "evaluation_policy": split_meta["policy"],
        "ranking_history_policy": "Latest official FIFA publication strictly before each match; reconstructed live updates are applied only after each completed match.",
        "rows": {"matches": int(len(training)), "neutral_rows": int(len(neutral)), "worldcup_test_matches": int(len(test))},
        "correlations_worldcup_test": correlations,
        "rating_variants": variants,
        "elo_variants": elo_results,
        "holdout_ranking_not_selection_basis": [item["variant"] for item in ranked],
        "best_holdout_diagnostic": ranked[0],
        "recommended_variant": "production_fifa_sum_live_no_custom_elo",
        "selection_policy": (
            "Choose the single causal FIFA SUM Live signal for conceptual validity and "
            "parsimony; do not tune or select a recipe on this same World Cup holdout."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output.with_name("worldcup2026_fifa_sum_live_pre_match_history.csv")
    fifa_sum_snapshots.to_csv(snapshot_path, index=False)
    latest_path = args.output.with_name("worldcup2026_fifa_sum_live_latest.csv")
    fifa_sum_latest.to_csv(latest_path, index=False)
    payload["fifa_sum_live_history_path"] = str(snapshot_path)
    payload["fifa_sum_live_latest_path"] = str(latest_path)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommended_variant": payload["recommended_variant"],
                "best_holdout_diagnostic": ranked[0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
