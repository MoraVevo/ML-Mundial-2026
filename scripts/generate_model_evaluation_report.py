from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
)

from kinela.lightgbm_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NEUTRAL_FEATURES,
    NEUTRAL_MODEL_RECIPE,
    _build_neutral_frame,
    _calibrated_classifier_importances,
    _maybe_add_clinical_finishing,
    _maybe_add_club_attacking_talent,
    _maybe_add_counter_efficiency,
    _maybe_add_late85_points_swing,
    _maybe_add_score_timing,
)


RESULT_LABELS = ["team_a", "draw", "team_b"]
RANDOM_SEED = 42
EXTERNAL_TEST_MATCHES = 104

REG_PARAMS = {
    "objective": "regression",
    "n_estimators": 350,
    "learning_rate": 0.035,
    "num_leaves": 23,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}
CLF_PARAMS = {
    "objective": "multiclass",
    "n_estimators": 300,
    "learning_rate": 0.035,
    "num_leaves": 23,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}


@dataclass
class EvaluationResult:
    name: str
    title: str
    policy: str
    train_matches: int
    test_matches: int
    test_start: str
    test_end: str
    metrics: dict[str, Any]
    confusion: list[list[int]]
    feature_importances: dict[str, list[dict[str, Any]]]
    test_competitions: dict[str, int]
    result_distribution: dict[str, int]


def _load_frames(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = data_root / "processed" / "combined" / "training_frame_national.csv"
    clean_path = data_root / "processed" / "combined" / "clean_training_matrix_national.csv"
    if not training_path.exists() or not clean_path.exists():
        raise FileNotFoundError(
            "Run `kinela export training-frame-national` and "
            "`kinela export clean-training-matrix-national` first."
        )
    training = pd.read_csv(training_path, low_memory=False)
    clean = pd.read_csv(clean_path, low_memory=False)
    if len(training) != len(clean):
        raise RuntimeError("National training and clean matrices are not aligned")
    return training, clean


def _is_worldcup_2026(training: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(training["date"], errors="coerce")
    names = training["competition_name"].astype(str).str.casefold()
    sources = training["source"].astype(str)
    return (
        sources.eq("manual-worldcup-2026")
        | (
            names.eq("fifa world cup")
            & dates.ge(pd.Timestamp("2026-06-11"))
        )
    )


def _is_friendly(training: pd.DataFrame) -> pd.Series:
    return training["is_friendly"].astype(str).str.casefold().isin({"true", "1", "yes"})


def _split_worldcup_2026(training: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    split = clean.copy()
    test_mask = _is_worldcup_2026(training)
    test_dates = pd.to_datetime(training.loc[test_mask, "date"], errors="raise")
    cutoff = test_dates.min()
    all_dates = pd.to_datetime(training["date"], errors="raise")
    split["split"] = np.select(
        [test_mask, all_dates.lt(cutoff)],
        ["test", "train"],
        default="excluded",
    )
    return split, {
        "test_start": str(test_dates.min().date()),
        "test_end": str(test_dates.max().date()),
        "policy": (
            "Played World Cup 2026 matches are forced to test. Training uses only "
            "national-team matches before the first World Cup 2026 match; same-date "
            "and later rows are excluded."
        ),
    }


def _split_external_random_temporal(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    test_matches: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dated = training.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="raise")
    eligible = dated.loc[
        ~_is_friendly(dated)
        & ~_is_worldcup_2026(dated)
        & dated["competition_name"].notna()
    ].copy()
    eligible = eligible.sort_values(["_date", "row_index"], kind="stable")
    if len(eligible) < test_matches:
        raise RuntimeError(f"Only {len(eligible)} external official matches are available")

    recent_pool_start = eligible["_date"].quantile(0.65, interpolation="nearest")
    pool = eligible.loc[eligible["_date"].ge(recent_pool_start)].copy()
    if len(pool) < test_matches:
        pool = eligible.tail(test_matches * 2).copy()

    selected = pool.sample(n=test_matches, random_state=seed).sort_values(
        ["_date", "row_index"],
        kind="stable",
    )
    selected_index = set(selected.index.tolist())
    cutoff = selected["_date"].min()
    split = clean.copy()
    all_dates = dated["_date"]
    split["split"] = np.select(
        [training.index.isin(selected_index), all_dates.lt(cutoff)],
        ["test", "train"],
        default="excluded",
    )
    return split, {
        "test_start": str(selected["_date"].min().date()),
        "test_end": str(selected["_date"].max().date()),
        "policy": (
            f"Random holdout of {test_matches} non-friendly, non-World-Cup-2026 "
            f"national matches from the recent official-match pool, seed={seed}. "
            "Training uses only matches before the earliest selected test date; "
            "same-date and later non-selected rows are excluded."
        ),
    }


def _prepare_neutral(data_root: Path, training: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    enriched = _maybe_add_late85_points_swing(data_root, clean, training)
    enriched = _maybe_add_score_timing(data_root, enriched, training)
    enriched = _maybe_add_counter_efficiency(data_root, enriched, training)
    enriched = _maybe_add_clinical_finishing(data_root, enriched, training)
    enriched = _maybe_add_club_attacking_talent(data_root, enriched, training)
    return _build_neutral_frame(enriched, augment=True)


def _importances(model: Any, features: list[str]) -> list[dict[str, Any]]:
    return sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(features, model.feature_importances_, strict=True)
        ],
        key=lambda item: float(item["importance"]),
        reverse=True,
    )


def _evaluate(
    name: str,
    title: str,
    policy: str,
    data_root: Path,
    training: pd.DataFrame,
    split_clean: pd.DataFrame,
) -> EvaluationResult:
    neutral = _prepare_neutral(data_root, training, split_clean)
    train = neutral[neutral["split"].eq("train")].copy()
    test = neutral[neutral["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"{name} produced an empty train or test split")

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
    pred_a = np.clip(team_a_model.predict(test[features]), 0.0, None)
    pred_b = np.clip(team_b_model.predict(test[features]), 0.0, None)

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
    mae_a = float(mean_absolute_error(test["team_a_goals"], pred_a))
    mae_b = float(mean_absolute_error(test["team_b_goals"], pred_b))
    test_dates = pd.to_datetime(training.loc[split_clean["split"].eq("test"), "date"])
    test_training_rows = training.loc[split_clean["split"].eq("test")]

    metrics = {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "correct": int((labels == predicted).sum()),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1, 2])), 4),
        "mae_team_a_goals": round(mae_a, 4),
        "mae_team_b_goals": round(mae_b, 4),
        "mae_goals_avg": round((mae_a + mae_b) / 2.0, 4),
    }
    return EvaluationResult(
        name=name,
        title=title,
        policy=policy,
        train_matches=int(train["split"].eq("train").sum() / 2),
        test_matches=int(len(test)),
        test_start=str(test_dates.min().date()),
        test_end=str(test_dates.max().date()),
        metrics=metrics,
        confusion=confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist(),
        feature_importances={
            "result_classifier": _calibrated_classifier_importances(classifier, features),
            "team_a_goals": _importances(team_a_model, features),
            "team_b_goals": _importances(team_b_model, features),
        },
        test_competitions={
            str(key): int(value)
            for key, value in test_training_rows["competition_name"].value_counts().items()
        },
        result_distribution={
            str(key): int(value)
            for key, value in test_training_rows["result"].value_counts().items()
        },
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbf8",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#30343b",
            "axes.labelcolor": "#30343b",
            "xtick.color": "#30343b",
            "ytick.color": "#30343b",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "savefig.bbox": "tight",
            "savefig.dpi": 170,
        }
    )


def _bar_label(ax: plt.Axes, values: list[float], fmt: str = "{:.2f}") -> None:
    for index, value in enumerate(values):
        ax.text(index, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def _plot_metrics(results: list[EvaluationResult], asset_dir: Path) -> None:
    labels = ["WC 2026", "External"]
    colors = ["#2f6f73", "#b06d3b"]
    accuracy = [result.metrics["accuracy"] for result in results]
    logloss = [result.metrics["log_loss"] for result in results]
    mae = [result.metrics["mae_goals_avg"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title, ylabel in [
        (axes[0], accuracy, "Accuracy", "higher is better"),
        (axes[1], logloss, "Log loss", "lower is better"),
        (axes[2], mae, "Goal MAE", "lower is better"),
    ]:
        ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=0)
        _bar_label(ax, values)
    fig.suptitle("Model evaluation summary", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(asset_dir / "metrics_summary.png")
    plt.close(fig)


def _plot_confusion(result: EvaluationResult, asset_dir: Path) -> None:
    matrix = np.asarray(result.confusion)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_title(f"Confusion matrix: {result.title}")
    ax.set_xticks(range(3), RESULT_LABELS)
    ax.set_yticks(range(3), RESULT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for y in range(3):
        for x in range(3):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="#202124")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(asset_dir / f"confusion_{result.name}.png")
    plt.close(fig)


def _plot_importance(result: EvaluationResult, asset_dir: Path) -> None:
    rows = result.feature_importances["result_classifier"][:12]
    features = [row["feature"] for row in rows][::-1]
    values = [row["importance"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(features, values, color="#4f7cac")
    ax.set_title(f"Result-model feature importance: {result.title}")
    ax.set_xlabel("Mean split importance")
    ax.grid(axis="x", alpha=0.22)
    fig.savefig(asset_dir / f"feature_importance_{result.name}.png")
    plt.close(fig)


def _write_markdown(results: list[EvaluationResult], asset_dir: Path, output: Path) -> None:
    draw_notes = []
    for result in results:
        matrix = np.asarray(result.confusion)
        predicted_draws = int(matrix[:, 1].sum())
        actual_draws = int(matrix[1, :].sum())
        draw_notes.append((result.title, actual_draws, predicted_draws))

    lines = [
        "# Model evaluation",
        "",
        "This report evaluates the current neutral parsimonious recipe with temporal "
        "guards designed to avoid training on future information.",
        "",
        f"Feature recipe: `{NEUTRAL_MODEL_RECIPE}`",
        "",
        "## Test policies",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result.title}",
                "",
                result.policy,
                "",
                f"- Train matches: {result.train_matches}",
                f"- Test matches: {result.test_matches}",
                f"- Test window: {result.test_start} to {result.test_end}",
                "",
            ]
        )

    lines.extend(
        [
            "## Metrics",
            "",
            "| Evaluation | Accuracy | Correct | Log loss | MAE team A | MAE team B | MAE avg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        metric = result.metrics
        lines.append(
            f"| {result.title} | {metric['accuracy']:.4f} | "
            f"{metric['correct']}/{result.test_matches} | {metric['log_loss']:.4f} | "
            f"{metric['mae_team_a_goals']:.4f} | {metric['mae_team_b_goals']:.4f} | "
            f"{metric['mae_goals_avg']:.4f} |"
        )
    lines.extend(
        [
            "",
            "![Metrics summary](assets/model_evaluation/metrics_summary.png)",
            "",
            "## Technical interpretation",
            "",
            "The model is directionally useful on winners, but the current decision "
            "threshold is conservative around draws. In these holdouts it assigns "
            "draw probability for log-loss calibration, yet the top class rarely "
            "becomes `draw`.",
            "",
            "| Evaluation | Actual draws | Predicted draws as top class |",
            "|---|---:|---:|",
        ]
    )
    for title, actual_draws, predicted_draws in draw_notes:
        lines.append(f"| {title} | {actual_draws} | {predicted_draws} |")
    lines.extend(
        [
            "",
            "This is why log loss is shown next to accuracy: accuracy alone hides "
            "whether the model is placing useful probability mass on draws and "
            "close matches. The MAE values are reported separately because the goal "
            "regressors can be directionally acceptable even when the 1X2 classifier "
            "chooses the wrong class.",
            "",
            "## Confusion matrices",
            "",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"### {result.title}",
                "",
                f"![Confusion matrix](assets/model_evaluation/confusion_{result.name}.png)",
                "",
            ]
        )

    lines.extend(["## Feature importance", ""])
    for result in results:
        top = result.feature_importances["result_classifier"][:8]
        lines.extend(
            [
                f"### {result.title}",
                "",
                f"![Feature importance](assets/model_evaluation/feature_importance_{result.name}.png)",
                "",
                "| Feature | Importance |",
                "|---|---:|",
            ]
        )
        for row in top:
            lines.append(f"| `{row['feature']}` | {float(row['importance']):.2f} |")
        lines.append("")

    lines.extend(
        [
            "## Feature construction summary",
            "",
            "The active model uses only pre-match features. The 12 production features are:",
            "",
        ]
    )
    for feature in NEUTRAL_FEATURES:
        lines.append(f"- `{feature}`")
    lines.extend(
        [
            "",
            "High-level groups:",
            "",
            "- Rating strength: FIFA/ranking, Elo-style team strength, rating guardrail and drift.",
            "- Recent form: opponent-adjusted recent points and goal-balance signals.",
            "- Match context: competition family, stage/round and draw-pressure context.",
            "- Tactical/attacking profile: match-script compatibility, clinical low-block matchup and club star-finisher signal.",
            "",
            "Excluded from model features: target goals/results, raw identifiers, raw dates, source names and post-match statistics from the evaluated match.",
            "",
            "## Leakage controls",
            "",
            "The World Cup holdout is the primary model accuracy because it matches the target domain. "
            "The external random temporal holdout is a robustness diagnostic: it samples non-World-Cup-2026 "
            "official national-team matches but still trains only on matches before the first selected test date.",
            "",
            "Both evaluations rebuild features before fitting and keep the following columns out of the model: "
            "`match_id`, `source`, raw `date`, teams, goals, final result and provider identifiers. "
            "Date is used only to define chronological splits and pre-match rolling context.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--asset-dir", type=Path, default=Path("docs/assets/model_evaluation"))
    parser.add_argument("--report", type=Path, default=Path("docs/model_evaluation.md"))
    args = parser.parse_args()

    _apply_style()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    training, clean = _load_frames(args.data_root)

    wc_clean, wc_info = _split_worldcup_2026(training, clean)
    external_clean, external_info = _split_external_random_temporal(
        training,
        clean,
        EXTERNAL_TEST_MATCHES,
        RANDOM_SEED,
    )
    results = [
        _evaluate(
            "worldcup_2026",
            "World Cup 2026 holdout",
            wc_info["policy"],
            args.data_root,
            training,
            wc_clean,
        ),
        _evaluate(
            "external_random_temporal",
            "External random temporal holdout",
            external_info["policy"],
            args.data_root,
            training,
            external_clean,
        ),
    ]

    _plot_metrics(results, args.asset_dir)
    for result in results:
        _plot_confusion(result, args.asset_dir)
        _plot_importance(result, args.asset_dir)
    _write_markdown(results, args.asset_dir, args.report)

    payload = {
        "model": "lightgbm_neutral_parsimonious",
        "feature_recipe": NEUTRAL_MODEL_RECIPE,
        "features": list(NEUTRAL_FEATURES),
        "evaluations": [result.__dict__ for result in results],
    }
    summary_path = args.asset_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
