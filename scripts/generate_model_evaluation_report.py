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


RESULT_LABELS = ["Equipo A", "Empate", "Equipo B"]
RANDOM_SEED = 42
EXTERNAL_TEST_MATCHES = 104

REG_PARAMS = {
    "objective": "regression",
    "n_estimators": 350,
    "learning_rate": 0.035,
    "num_leaves": 12,
    "max_depth": 4,
    "min_child_samples": 60,
    "subsample": 0.82,
    "colsample_bytree": 0.82,
    "reg_alpha": 0.10,
    "reg_lambda": 1.0,
    "min_split_gain": 0.005,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}
CLF_PARAMS = {
    "objective": "multiclass",
    "n_estimators": 300,
    "learning_rate": 0.035,
    "num_leaves": 12,
    "max_depth": 4,
    "min_child_samples": 60,
    "subsample": 0.82,
    "colsample_bytree": 0.82,
    "reg_alpha": 0.10,
    "reg_lambda": 1.0,
    "min_split_gain": 0.005,
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
    error_analysis: dict[str, list[dict[str, Any]]]


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
            f"Test aleatorio de {test_matches} partidos nacionales no amistosos "
            "y fuera del Mundial 2026, tomados del pool reciente de partidos "
            f"oficiales, seed={seed}. El entrenamiento usa solo partidos "
            "anteriores a la primera fecha seleccionada de test; los partidos "
            "seleccionados como test no se usan para entrenar."
        ),
    }


def _split_combined_objective(
    training: pd.DataFrame,
    clean: pd.DataFrame,
    worldcup_split: pd.DataFrame,
    external_split: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split = clean.copy()
    test_mask = worldcup_split["split"].eq("test") | external_split["split"].eq("test")
    split["split"] = "excluded"
    split.loc[test_mask, "split"] = "test"
    test_dates = pd.to_datetime(training.loc[test_mask, "date"], errors="raise")
    cutoff = test_dates.min()
    all_dates = pd.to_datetime(training["date"], errors="raise")
    split.loc[all_dates.lt(cutoff) & ~test_mask, "split"] = "train"
    return split, {
        "test_start": str(test_dates.min().date()),
        "test_end": str(test_dates.max().date()),
        "policy": (
            "Diagnostico combinado del objetivo: todos los partidos jugados del "
            "Mundial 2026 mas el test externo temporal de partidos oficiales no "
            "amistosos. El entrenamiento usa solo partidos anteriores a la primera "
            "fecha seleccionada de test y ningun partido de test entra al entrenamiento."
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


def _group_errors(frame: pd.DataFrame, group_col: str, min_matches: int = 1) -> list[dict[str, Any]]:
    label_map = {"home": "Equipo A", "draw": "Empate", "away": "Equipo B"}
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_col, dropna=False):
        if len(group) < min_matches:
            continue
        rows.append(
            {
                "group": label_map.get(str(value), str(value)),
                "matches": int(len(group)),
                "accuracy": round(float((group["actual_label"] == group["predicted_label"]).mean()), 4),
                "log_loss": round(
                    float(log_loss(group["actual_label"], group[["p_a", "p_draw", "p_b"]], labels=[0, 1, 2])),
                    4,
                )
                if len(set(group["actual_label"])) > 1
                else None,
                "mae_team_a_goals": round(float(group["abs_error_a"].mean()), 4),
                "mae_team_b_goals": round(float(group["abs_error_b"].mean()), 4),
                "mae_goals_avg": round(float(group["abs_error_avg"].mean()), 4),
            }
        )
    return sorted(rows, key=lambda item: (-float(item["mae_goals_avg"]), -int(item["matches"])))


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
    details = test_training_rows[
        ["date", "competition_name", "competition_family", "competition_type", "stage_or_round", "result"]
    ].reset_index(drop=True)
    details["actual_label"] = labels
    details["predicted_label"] = predicted
    details["p_a"] = probabilities[:, 0]
    details["p_draw"] = probabilities[:, 1]
    details["p_b"] = probabilities[:, 2]
    details["team_a_goals"] = test["team_a_goals"].to_numpy()
    details["team_b_goals"] = test["team_b_goals"].to_numpy()
    details["pred_team_a_goals"] = pred_a
    details["pred_team_b_goals"] = pred_b
    details["abs_error_a"] = np.abs(details["team_a_goals"] - details["pred_team_a_goals"])
    details["abs_error_b"] = np.abs(details["team_b_goals"] - details["pred_team_b_goals"])
    details["abs_error_avg"] = (details["abs_error_a"] + details["abs_error_b"]) / 2.0

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
        error_analysis={
            "by_competition": _group_errors(details, "competition_name"),
            "by_stage": _group_errors(details, "stage_or_round"),
            "by_result": _group_errors(details, "result"),
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
    labels = ["WC 2026", "External", "Combinado"][: len(results)]
    colors = ["#2f6f73", "#b06d3b", "#4f7cac"][: len(results)]
    accuracy = [result.metrics["accuracy"] for result in results]
    logloss = [result.metrics["log_loss"] for result in results]
    mae = [result.metrics["mae_goals_avg"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title, ylabel in [
        (axes[0], accuracy, "Accuracy", "mas alto es mejor"),
        (axes[1], logloss, "Log loss", "mas bajo es mejor"),
        (axes[2], mae, "MAE de goles", "mas bajo es mejor"),
    ]:
        ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=0)
        _bar_label(ax, values)
    fig.suptitle("Resumen de evaluacion del modelo", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(asset_dir / "metrics_summary.png")
    plt.close(fig)


def _plot_confusion(result: EvaluationResult, asset_dir: Path) -> None:
    matrix = np.asarray(result.confusion)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    ax.set_title(f"Matriz de confusion: {result.title}")
    ax.set_xticks(range(3), RESULT_LABELS)
    ax.set_yticks(range(3), RESULT_LABELS)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    threshold = matrix.max() * 0.45
    for y in range(3):
        for x in range(3):
            color = "white" if matrix[y, x] > threshold else "#111827"
            ax.text(
                x,
                y,
                str(matrix[y, x]),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(asset_dir / f"confusion_{result.name}.png")
    plt.close(fig)


def _plot_importance(result: EvaluationResult, asset_dir: Path) -> None:
    rows = result.feature_importances["result_classifier"][:12]
    features = [row["feature"] for row in rows][::-1]
    values = [row["importance"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(features, values, color="#4f7cac")
    ax.set_title(f"Importancia de features: {result.title}")
    ax.set_xlabel("Importancia media por splits")
    ax.grid(axis="x", alpha=0.22)
    fig.savefig(asset_dir / f"feature_importance_{result.name}.png")
    plt.close(fig)


def _write_markdown(results: list[EvaluationResult], asset_dir: Path, output: Path) -> None:
    feature_descriptions = {
        "competition_family": "Familia de competicion: Mundial, eliminatoria, torneo continental, Nations League u otra categoria nacional.",
        "stage_or_round": "Fase o ronda del partido. Da contexto competitivo: grupo, knockout, jornada, final, etc.",
        "rating_threat_edge": "Ventaja combinada de fuerza/ranking y amenaza ofensiva esperada entre Equipo A y Equipo B.",
        "quality_form_edge": "Forma reciente ajustada por calidad del rival, no solo puntos crudos.",
        "goal_balance_edge": "Diferencia de balance goleador reciente e historico: goles a favor menos goles recibidos.",
        "draw_pressure_index": "Indice de paridad y baja separacion esperada; ayuda a calibrar partidos cerrados.",
        "score_timing_edge": "Ventaja validada de score timing: cuanto controlo el marcador, que tan temprano golpeo, si rescato o perdio puntos tarde, y cuanto tiempo paso persiguiendo el partido.",
        "score_control_value_edge": "Ventaja en control de marcador: capacidad reciente de sostener o transformar estados de partido.",
        "quality_score_control_swing_edge": "Control de marcador ajustado por calidad del rival y cambios tempranos de estado; prioriza senales de resultado sobre marcador exacto.",
        "rating_guardrail_edge": "Correccion de seguridad cuando las senales de amenaza se alejan demasiado del rating base.",
        "rating_drift_abs": "Magnitud del cambio reciente entre rating historico y rating vivo; captura incertidumbre/volatilidad.",
        "match_script_compatibility_edge": "Compatibilidad tactica estimada entre estilos de partido de ambos equipos.",
        "clinical_low_block_matchup_edge": "Cruce entre definicion ofensiva y capacidad/riesgo contra bloques bajos.",
        "club_attack_talent_edge": "Ventaja de talento ofensivo de plantel/club cuando hay cobertura previa suficiente; faltantes reducen cobertura en vez de inventar valor.",
        "club_star_finisher_edge": "Ventaja del mejor finalizador reciente de club dentro del nucleo usado por la seleccion; prioriza techo goleador sobre promedio de talento.",
        "worldcup_points_memory_edge": "Memoria ponderada de puntos en los ultimos partidos mundialistas disponibles antes del partido.",
        "worldcup_fotmob_xg_balance_edge": "Balance reciente de xG creado menos xG concedido en Mundiales desde FotMob, con cobertura bilateral.",
        "worldcup_fotmob_chance_pressure_edge": "Presion reciente de ocasiones mundialistas desde FotMob: xG, big chances, desperdicio y resistencia defensiva con compuerta de cobertura.",
        "worldcup_fotmob_chance_coverage_pair": "Cobertura bilateral de datos FotMob World Cup antes del partido; evita que senales parciales entren como rendimiento real.",
        "worldcup_fotmob_interpreted_edge": "Sintesis interpretada de dominio de ocasiones mundialistas: control de chance, balance xG, solucion contra bloque bajo, transicion y disciplina de definicion.",
        "worldcup_fotmob_low_block_solution_edge": "Capacidad mundialista reciente de convertir amenaza en ocasiones utiles contra un rival con perfil de bloque bajo, penalizando posesion esteril y desperdicio.",
        "worldcup_fotmob_transition_punch_edge": "Peligro reciente con baja posesion: senal de equipos que producen xG, big chances y tiros claros sin necesitar dominar la pelota.",
        "worldcup_fotmob_unrewarded_pressure_edge": "Presion ofensiva no premiada en el marcador: buenos volumenes de amenaza con empates/derrotas, util para no castigar demasiado un resultado adverso.",
        "worldcup_fotmob_finishing_discipline_edge": "Calidad de definicion ajustada por desperdicio: diferencia entre convertir lo generado y dejar big chances sin premio.",
        "worldcup_fotmob_current_chance_pressure_edge": "Lectura del Mundial actual antes del partido: quien viene controlando mejor las ocasiones, xG, definicion y desperdicio dentro del mismo torneo.",
        "worldcup_fotmob_current_low_block_solution_edge": "Respuesta mostrada en el Mundial actual frente a perfiles de bloque bajo, usando amenaza real y riesgo de control esteril.",
        "worldcup_fotmob_current_transition_punch_edge": "Peligro de transicion observado en el Mundial actual: generar sin mucha posesion, especialmente util ante rivales que conceden espacio.",
        "worldcup_fotmob_current_unrewarded_pressure_edge": "Partidos recientes del Mundial actual donde el equipo genero suficiente amenaza aunque el marcador no lo reflejara.",
        "worldcup_fotmob_current_controlled_dominance_edge": "Dominio controlado en el Mundial actual: amenaza y control de ocasiones con defensa estable, penalizando desperdicio y posesion esteril.",
        "worldcup_fotmob_current_story_edge": "Lectura conservadora del Mundial actual: dominio controlado, presion de ocasiones, soluciones ante bloque bajo, transicion y presion no premiada, siempre con cobertura bilateral.",
    }
    draw_notes = []
    for result in results:
        matrix = np.asarray(result.confusion)
        predicted_draws = int(matrix[:, 1].sum())
        actual_draws = int(matrix[1, :].sum())
        draw_notes.append((result.title, actual_draws, predicted_draws))

    lines = [
        "# Evaluacion del modelo",
        "",
        "Este reporte evalua el modelo neutral con cortes temporales disenados "
        "para que los partidos de test no entren al entrenamiento.",
        "",
        f"Identificador del modelo: `{NEUTRAL_MODEL_RECIPE}`",
        "",
        "## Politicas de test",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result.title}",
                "",
                result.policy,
                "",
                f"- Partidos de entrenamiento: {result.train_matches}",
                f"- Partidos de test: {result.test_matches}",
                f"- Ventana de test: {result.test_start} a {result.test_end}",
                "",
            ]
        )

    lines.extend(
        [
            "## Metricas",
            "",
            "| Evaluacion | Accuracy | Correctos | Log loss | MAE equipo A | MAE equipo B | MAE prom. |",
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
            "![Resumen de metricas](assets/model_evaluation/metrics_summary.png)",
            "",
            "## Interpretacion tecnica",
            "",
            "El modelo es util para direccionar ganadores, pero el umbral actual es "
            "conservador con los empates. En estos tests asigna probabilidad al "
            "empate para calibracion via log loss, pero la clase con mayor probabilidad "
            "casi nunca termina siendo `empate`.",
            "",
            "La metrica principal sigue siendo el test Mundial 2026. El test combinado "
            "se incluye como diagnostico del objetivo mixto, no como reemplazo de la "
            "lectura mundialista.",
            "",
            "| Evaluacion | Empates reales | Empates predichos como clase principal |",
            "|---|---:|---:|",
        ]
    )
    for title, actual_draws, predicted_draws in draw_notes:
        lines.append(f"| {title} | {actual_draws} | {predicted_draws} |")
    lines.extend(
        [
            "",
            "Por eso se muestra log loss junto a accuracy: accuracy sola oculta si "
            "el modelo esta asignando probabilidad util a empates y partidos cerrados. "
            "El MAE se reporta aparte porque los regresores de goles pueden estar "
            "razonablemente calibrados aunque el clasificador 1X2 elija otra clase.",
            "",
            "## Matrices de confusion",
            "",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"### {result.title}",
                "",
                f"![Matriz de confusion](assets/model_evaluation/confusion_{result.name}.png)",
                "",
            ]
        )

    lines.extend(["## Importancia de features", ""])
    for result in results:
        top = result.feature_importances["result_classifier"][:8]
        lines.extend(
            [
                f"### {result.title}",
                "",
                f"![Importancia de features](assets/model_evaluation/feature_importance_{result.name}.png)",
                "",
                "| Feature | Importancia |",
                "|---|---:|",
            ]
        )
        for row in top:
            lines.append(f"| `{row['feature']}` | {float(row['importance']):.2f} |")
        lines.append("")

    lines.extend(
        [
            "## Analisis de error",
            "",
            "Las siguientes tablas ordenan los grupos por mayor MAE promedio de goles. "
            "Sirven para ver donde el modelo sufre mas, no como ranking definitivo: "
            "algunos grupos tienen pocas observaciones.",
            "",
        ]
    )
    for result in results:
        lines.extend([f"### {result.title}", ""])
        for key, title in [
            ("by_competition", "Por competicion"),
            ("by_stage", "Por fase/ronda"),
            ("by_result", "Por resultado real"),
        ]:
            lines.extend(
                [
                    f"#### {title}",
                    "",
                    "| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in result.error_analysis[key][:8]:
                log_loss_value = "n/a" if row["log_loss"] is None else f"{float(row['log_loss']):.4f}"
                lines.append(
                    f"| {row['group']} | {row['matches']} | {row['accuracy']:.4f} | "
                    f"{log_loss_value} | {row['mae_team_a_goals']:.4f} | "
                    f"{row['mae_team_b_goals']:.4f} | {row['mae_goals_avg']:.4f} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Construccion de features",
            "",
            f"El modelo activo usa {len(NEUTRAL_FEATURES)} features prepartido:",
            "",
            "| Feature | Significado |",
            "|---|---|",
        ]
    )
    for feature in NEUTRAL_FEATURES:
        lines.append(f"| `{feature}` | {feature_descriptions.get(feature, '')} |")
    lines.extend(
        [
            "",
            "Grupos conceptuales:",
            "",
            "- Fuerza/rating: ranking FIFA, fuerza tipo Elo y guardrails de ranking.",
            "- Forma reciente: puntos ajustados por rival y balance de goles.",
            "- Contexto del partido: tipo de competicion, fase/ronda y presion de empate.",
            "- Perfil ofensivo del Mundial actual: score timing, control de ocasiones, dominio controlado y finalizador diferencial.",
            "",
            "Quedan fuera de los features: goles objetivo, resultado final, ids crudos, fecha cruda, equipos, fuente y estadisticas postpartido del encuentro evaluado.",
            "",
            "## Controles anti-leakage",
            "",
            "El test Mundial 2026 es la metrica principal porque evalua el mismo tipo de partido "
            "que se quiere predecir. Esos partidos no se usan para entrenar: se separan como test "
            "y el modelo se ajusta solo con partidos anteriores al inicio del Mundial 2026.",
            "",
            "El test externo temporal revisa si el modelo tambien se sostiene fuera del Mundial. "
            "Selecciona partidos oficiales nacionales fuera del Mundial 2026 y entrena solo con partidos "
            "anteriores al primer partido seleccionado como test. En otras palabras: el accuracy de cada "
            "test se calcula sobre partidos que el modelo no vio durante entrenamiento.",
            "",
            "Ambas evaluaciones reconstruyen features antes de entrenar. La fecha se usa para cortes "
            "cronologicos y contexto rolling prepartido; no entra como feature directa.",
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
    combined_clean, combined_info = _split_combined_objective(
        training,
        clean,
        wc_clean,
        external_clean,
    )
    results = [
        _evaluate(
            "worldcup_2026",
            "Test Mundial 2026",
            wc_info["policy"]
            .replace("Played World Cup 2026 matches are forced to test.", "Los partidos jugados del Mundial 2026 se fuerzan como test.")
            .replace("Training uses only national-team matches before the first World Cup 2026 match; same-date and later rows are excluded.", "El entrenamiento usa solo partidos de selecciones anteriores al primer partido del Mundial 2026; esos partidos de test no se usan para entrenar."),
            args.data_root,
            training,
            wc_clean,
        ),
        _evaluate(
            "external_random_temporal",
            "Test externo temporal",
            external_info["policy"],
            args.data_root,
            training,
            external_clean,
        ),
        _evaluate(
            "combined_objective",
            "Test combinado objetivo",
            combined_info["policy"],
            args.data_root,
            training,
            combined_clean,
        ),
    ]

    _plot_metrics(results, args.asset_dir)
    for result in results:
        _plot_confusion(result, args.asset_dir)
        _plot_importance(result, args.asset_dir)
    _write_markdown(results, args.asset_dir, args.report)

    payload = {
        "model": "lightgbm_neutral",
        "model_id": NEUTRAL_MODEL_RECIPE,
        "features": list(NEUTRAL_FEATURES),
        "evaluations": [result.__dict__ for result in results],
    }
    summary_path = args.asset_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
