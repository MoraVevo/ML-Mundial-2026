from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kinela.etl import FINISHED_STATUSES, _competition_stage, _competition_type
from kinela.fifa_ranking import load_fifa_ranking, normalise_team_name


HISTORY_WINDOW = 6
POSITION_ORDER = ("G", "D", "M", "F")
CATEGORICAL_FEATURES = [
    "competition_type",
    "competition_stage",
    "formation",
]
NUMERIC_FEATURES = [
    "formation_forwards",
    "xi_avg_prior_goals_for",
    "xi_avg_prior_goals_against",
    "xi_prior_goal_balance",
    "xi_known_player_score_count",
    "xi_minutes_available_count",
    "team_recent_offsides_won_avg",
    "opponent_recent_offsides_won_avg",
    "round_number",
    "group_progress",
    "is_late_group_match",
    "table_position_pre",
    "opponent_table_position_pre",
    "table_position_diff",
    "table_points_pre",
    "opponent_table_points_pre",
    "table_points_diff",
    "recent_matches",
    "recent_goals_for_avg",
    "recent_goals_against_avg",
    "recent_points_avg",
    "opponent_recent_matches",
    "opponent_recent_goals_for_avg",
    "opponent_recent_goals_against_avg",
    "opponent_recent_points_avg",
    "fifa_rank",
    "opponent_fifa_rank",
    "fifa_rank_diff",
    "fifa_points",
    "opponent_fifa_points",
    "fifa_points_diff",
    "recent_attack_vs_opponent_defense",
    "lineup_attack_vs_opponent_recent_defense",
]
TRAINING_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _num(value: Any, fallback: float = 0.0) -> float:
    if value in (None, ""):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _formation_lines(formation: str | None) -> tuple[int, int, int]:
    parts = [int(part) for part in (formation or "").split("-") if part.isdigit()]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], sum(parts[2:])


def _read_finished_details(data_root: Path) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for path in (data_root / "raw" / "api_football" / "fixtures").glob("details-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("response") or []:
            fixture = item.get("fixture") or {}
            fixture_id = fixture.get("id")
            if fixture_id is not None:
                by_id[int(fixture_id)] = item
    rows = [
        item
        for item in by_id.values()
        if (item.get("fixture") or {}).get("status", {}).get("short") in FINISHED_STATUSES
        and (item.get("goals") or {}).get("home") is not None
        and (item.get("goals") or {}).get("away") is not None
        and item.get("lineups")
    ]
    return sorted(rows, key=lambda item: (item["fixture"]["date"], int(item["fixture"]["id"])))


def _lineup_by_team(item: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lineups: dict[int, dict[str, Any]] = {}
    for lineup in item.get("lineups") or []:
        team = lineup.get("team") or {}
        if team.get("id") is not None:
            lineups[int(team["id"])] = lineup
    return lineups


def _player_minutes(item: dict[str, Any]) -> dict[tuple[int, int], float]:
    minutes: dict[tuple[int, int], float] = {}
    for team_block in item.get("players") or []:
        team_id = (team_block.get("team") or {}).get("id")
        if team_id is None:
            continue
        for player_block in team_block.get("players") or []:
            player_id = (player_block.get("player") or {}).get("id")
            if player_id is None:
                continue
            stats = (player_block.get("statistics") or [{}])[0]
            minute_value = ((stats.get("games") or {}).get("minutes"))
            minutes[(int(team_id), int(player_id))] = _num(minute_value)
    return minutes


def _lineup_features(
    lineup: dict[str, Any],
    *,
    team_id: int,
    player_history: dict[int, deque[dict[str, float]]],
    minutes_by_player: dict[tuple[int, int], float],
) -> dict[str, Any]:
    starters = [
        entry.get("player") or {}
        for entry in lineup.get("startXI") or []
        if (entry.get("player") or {}).get("id") is not None
    ]
    substitutes = [
        entry.get("player") or {}
        for entry in lineup.get("substitutes") or []
        if (entry.get("player") or {}).get("id") is not None
    ]
    position_counts = Counter(player.get("pos") or "UNK" for player in starters)
    defender_line, midfield_line, forward_line = _formation_lines(lineup.get("formation"))

    offensive_scores: list[float] = []
    defensive_scores: list[float] = []
    sample_sizes: list[float] = []
    minutes_available = 0
    for player in starters:
        player_id = int(player["id"])
        history = list(player_history[player_id])
        sample_sizes.append(float(len(history)))
        if (team_id, player_id) in minutes_by_player:
            minutes_available += 1
        if history:
            offensive_scores.append(sum(row["gf"] for row in history) / len(history))
            defensive_scores.append(sum(row["ga"] for row in history) / len(history))

    sample_players = len(offensive_scores)
    xi_offense = sum(offensive_scores) / sample_players if sample_players else 0.0
    xi_defense = sum(defensive_scores) / sample_players if sample_players else 0.0
    return {
        "formation": lineup.get("formation") or "unknown",
        "formation_defenders": defender_line,
        "formation_midfielders": midfield_line,
        "formation_forwards": forward_line,
        "xi_starters": len(starters),
        "bench_players": len(substitutes),
        "xi_goalkeepers": position_counts.get("G", 0),
        "xi_defenders": position_counts.get("D", 0),
        "xi_midfielders": position_counts.get("M", 0),
        "xi_forwards": position_counts.get("F", 0),
        "xi_avg_prior_goals_for": round(xi_offense, 4),
        "xi_avg_prior_goals_against": round(xi_defense, 4),
        "xi_prior_goal_balance": round(xi_offense - xi_defense, 4),
        "xi_known_player_score_count": sample_players,
        "xi_avg_player_history_matches": round(sum(sample_sizes) / len(starters), 4)
        if starters
        else 0.0,
        "xi_minutes_available_count": minutes_available,
    }


def _team_recent(history: deque[dict[str, float]]) -> dict[str, float]:
    rows = list(history)[-HISTORY_WINDOW:]
    if not rows:
        return {
            "recent_matches": 0,
            "recent_goals_for_avg": 0.0,
            "recent_goals_against_avg": 0.0,
            "recent_points_avg": 0.0,
            "recent_offsides_won_avg": 0.0,
        }
    return {
        "recent_matches": len(rows),
        "recent_goals_for_avg": round(sum(row["gf"] for row in rows) / len(rows), 4),
        "recent_goals_against_avg": round(sum(row["ga"] for row in rows) / len(rows), 4),
        "recent_points_avg": round(sum(row["points"] for row in rows) / len(rows), 4),
        "recent_offsides_won_avg": round(sum(row.get("offsides_won", 0.0) for row in rows) / len(rows), 4),
    }


def _team_detail_stats(item: dict[str, Any]) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for team_stats in item.get("statistics") or []:
        team = team_stats.get("team") or {}
        team_id = team.get("id")
        if team_id is None:
            continue
        values: dict[str, float] = {}
        for stat in team_stats.get("statistics") or []:
            if stat.get("type") == "Offsides":
                values["offsides"] = _num(stat.get("value"))
        stats[int(team_id)] = values
    return stats


def _round_context(round_name: str, competition_stage: str) -> dict[str, Any]:
    group_match = re.search(r"\bGroup\s+([A-Z])\s*-\s*(\d+)\b", round_name or "", re.IGNORECASE)
    stage_match = re.search(r"\bGroup Stage\s*-\s*(\d+)\b", round_name or "", re.IGNORECASE)
    round_number = 0
    group_key = ""
    if group_match:
        group_key = group_match.group(1).upper()
        round_number = int(group_match.group(2))
    elif stage_match:
        round_number = int(stage_match.group(1))
    is_group_stage = (competition_stage or "").lower() == "group_stage"
    group_progress = min(round_number / 10.0, 1.0) if is_group_stage and round_number else 0.0
    return {
        "group_key": group_key,
        "round_number": round_number,
        "group_progress": round(group_progress, 4),
        "is_late_group_match": int(is_group_stage and round_number >= 5),
    }


def _competition_table_key(league_id: int | None, season: int | str | None, context: dict[str, Any]) -> tuple[Any, ...] | None:
    if context["group_key"]:
        return (league_id, season, context["group_key"])
    # South American qualifiers are a single table despite not having a group letter.
    if league_id == 34 and context["round_number"]:
        return (league_id, season, "single_table")
    return None


def _standing_snapshot(
    standings: dict[int, dict[str, float]],
    team_id: int,
) -> dict[str, float]:
    ordered = sorted(
        standings.items(),
        key=lambda item: (
            -item[1].get("points", 0.0),
            -(item[1].get("gf", 0.0) - item[1].get("ga", 0.0)),
            -item[1].get("gf", 0.0),
            item[0],
        ),
    )
    positions = {current_team_id: index + 1 for index, (current_team_id, _) in enumerate(ordered)}
    team_stats = standings.get(team_id, {"points": 0.0, "gf": 0.0, "ga": 0.0, "played": 0.0})
    return {
        "position": float(positions.get(team_id, 0)),
        "points": float(team_stats.get("points", 0.0)),
        "played": float(team_stats.get("played", 0.0)),
    }


def _fifa_features(team: str, opponent: str, rankings: dict[str, dict[str, float | int | str]]) -> dict[str, float]:
    team_rank = rankings.get(normalise_team_name(team), {})
    opponent_rank = rankings.get(normalise_team_name(opponent), {})
    rank = _num(team_rank.get("rank"), 100.0)
    opponent_rank_value = _num(opponent_rank.get("rank"), 100.0)
    points = _num(team_rank.get("points"))
    opponent_points = _num(opponent_rank.get("points"))
    return {
        "fifa_rank": rank,
        "opponent_fifa_rank": opponent_rank_value,
        "fifa_rank_diff": rank - opponent_rank_value,
        "fifa_points": points,
        "opponent_fifa_points": opponent_points,
        "fifa_points_diff": points - opponent_points,
    }


def export_post_lineup_goals_matrix(data_root: Path) -> list[dict[str, Any]]:
    rankings = load_fifa_ranking(data_root)
    fixtures = _read_finished_details(data_root)
    team_history: dict[int, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=HISTORY_WINDOW))
    player_history: dict[int, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=HISTORY_WINDOW))
    table_history: dict[tuple[Any, ...], dict[int, dict[str, float]]] = defaultdict(dict)
    records: list[dict[str, Any]] = []

    for item in fixtures:
        fixture = item["fixture"]
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        fixture_id = int(fixture["id"])
        played_at = _parse_dt(fixture["date"])
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home["id"])
        away_id = int(away["id"])
        home_goals = int(goals["home"])
        away_goals = int(goals["away"])
        lineup_map = _lineup_by_team(item)
        minutes = _player_minutes(item)
        detail_stats = _team_detail_stats(item)
        round_name = league.get("round") or ""
        league_id = league.get("id")
        round_values = _round_context(round_name, _competition_stage(round_name))
        table_key = _competition_table_key(league_id, league.get("season"), round_values)
        shared = {
            "fixture_id": fixture_id,
            "date": played_at.date().isoformat(),
            "timestamp": int(fixture.get("timestamp") or played_at.timestamp()),
            "league_id": league_id,
            "league_name": league.get("name") or "",
            "league_season": league.get("season") or "",
            "competition_type": _competition_type(league_id, league.get("name") or "", round_name),
            "competition_stage": _competition_stage(round_name),
            "round": round_name,
            "round_number": round_values["round_number"],
            "group_progress": round_values["group_progress"],
            "is_late_group_match": round_values["is_late_group_match"],
        }
        sides = [
            (home_id, home.get("name") or "", away_id, away.get("name") or "", home_goals, away_goals),
            (away_id, away.get("name") or "", home_id, home.get("name") or "", away_goals, home_goals),
        ]
        pending_history_updates: list[tuple[int, int, float, float, float]] = []
        for team_id, team, opponent_id, opponent, gf, ga in sides:
            lineup = lineup_map.get(team_id)
            if not lineup:
                continue
            recent = _team_recent(team_history[team_id])
            opponent_recent = _team_recent(team_history[opponent_id])
            table_snapshot = (
                _standing_snapshot(table_history[table_key], team_id)
                if table_key is not None
                else {"position": 0.0, "points": 0.0, "played": 0.0}
            )
            opponent_table_snapshot = (
                _standing_snapshot(table_history[table_key], opponent_id)
                if table_key is not None
                else {"position": 0.0, "points": 0.0, "played": 0.0}
            )
            lineup_values = _lineup_features(
                lineup,
                team_id=team_id,
                player_history=player_history,
                minutes_by_player=minutes,
            )
            record = {
                **shared,
                "team": team,
                "opponent": opponent,
                **lineup_values,
                **recent,
                "opponent_recent_matches": opponent_recent["recent_matches"],
                "opponent_recent_goals_for_avg": opponent_recent["recent_goals_for_avg"],
                "opponent_recent_goals_against_avg": opponent_recent["recent_goals_against_avg"],
                "opponent_recent_points_avg": opponent_recent["recent_points_avg"],
                "team_recent_offsides_won_avg": recent["recent_offsides_won_avg"],
                "opponent_recent_offsides_won_avg": opponent_recent["recent_offsides_won_avg"],
                "table_position_pre": table_snapshot["position"],
                "opponent_table_position_pre": opponent_table_snapshot["position"],
                "table_position_diff": table_snapshot["position"] - opponent_table_snapshot["position"],
                "table_points_pre": table_snapshot["points"],
                "opponent_table_points_pre": opponent_table_snapshot["points"],
                "table_points_diff": table_snapshot["points"] - opponent_table_snapshot["points"],
                **_fifa_features(team, opponent, rankings),
                "goals_for": gf,
                "goals_against": ga,
                "total_goals": gf + ga,
            }
            record["recent_attack_vs_opponent_defense"] = round(
                (record["recent_goals_for_avg"] + record["opponent_recent_goals_against_avg"])
                / 2,
                4,
            )
            record["lineup_attack_vs_opponent_recent_defense"] = round(
                (record["xi_avg_prior_goals_for"] + record["opponent_recent_goals_against_avg"])
                / 2,
                4,
            )
            records.append(record)
            points = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
            for player in (lineup.get("startXI") or []):
                player_id = ((player.get("player") or {}).get("id"))
                if player_id is not None:
                    pending_history_updates.append((int(player_id), team_id, float(gf), float(ga), points))
        for team_id, _, _, _, gf, ga in sides:
            points = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
            opponent_id = next(side[2] for side in sides if side[0] == team_id)
            offsides_won = detail_stats.get(opponent_id, {}).get("offsides", 0.0)
            team_history[team_id].append(
                {
                    "gf": float(gf),
                    "ga": float(ga),
                    "points": points,
                    "offsides_won": float(offsides_won),
                }
            )
        if table_key is not None:
            standings = table_history[table_key]
            for team_id, _, _, _, gf, ga in sides:
                points = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
                current = standings.setdefault(
                    team_id,
                    {"played": 0.0, "points": 0.0, "gf": 0.0, "ga": 0.0},
                )
                current["played"] += 1.0
                current["points"] += points
                current["gf"] += float(gf)
                current["ga"] += float(ga)
        for player_id, _, gf, ga, points in pending_history_updates:
            player_history[player_id].append({"gf": gf, "ga": ga, "points": points})
    return records


def write_post_lineup_goals_matrix(data_root: Path) -> dict[str, Any]:
    rows = export_post_lineup_goals_matrix(data_root)
    output = data_root / "processed" / "api_football" / "post_lineup_goals_matrix.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": len(rows),
        "matches": len({row["fixture_id"] for row in rows}),
        "path": str(output),
    }


def _baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    train_mean = float(train["goals_for"].mean())
    recent = test["recent_attack_vs_opponent_defense"].astype(float).to_numpy()
    fifa_adjustment = np.clip(-test["fifa_rank_diff"].astype(float).to_numpy() / 120.0, -0.45, 0.45)
    return {
        "global_mean": np.full(len(test), train_mean),
        "recent_attack_defense": np.clip(recent, 0.05, 5.0),
        "recent_plus_fifa": np.clip(recent + fifa_adjustment, 0.05, 5.0),
    }


def train_post_lineup_goals(data_root: Path) -> dict[str, Any]:
    matrix_path = data_root / "processed" / "api_football" / "post_lineup_goals_matrix.csv"
    if not matrix_path.exists():
        write_post_lineup_goals_matrix(data_root)
    frame = pd.read_csv(matrix_path)
    frame = frame.sort_values(["timestamp", "fixture_id", "team"]).reset_index(drop=True)
    split_index = max(1, int(len(frame) * 0.8))
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.15,
        random_state=42,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train[TRAINING_FEATURES], train["goals_for"])
    predictions = np.clip(pipeline.predict(test[TRAINING_FEATURES]), 0.0, None)

    metrics: dict[str, Any] = {
        "model": "post_lineup_goals_hist_gradient_boosting",
        "rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": len(TRAINING_FEATURES),
        "mae": round(float(mean_absolute_error(test["goals_for"], predictions)), 4),
        "rmse": round(float(mean_squared_error(test["goals_for"], predictions) ** 0.5), 4),
    }
    for name, baseline in _baseline_predictions(train, test).items():
        metrics[f"{name}_mae"] = round(float(mean_absolute_error(test["goals_for"], baseline)), 4)
        metrics[f"{name}_rmse"] = round(float(mean_squared_error(test["goals_for"], baseline) ** 0.5), 4)

    output_dir = data_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "features": TRAINING_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target": "goals_for",
        },
        output_dir / "post_lineup_goals_model.joblib",
    )
    payload = {"metrics": metrics}
    (output_dir / "post_lineup_goals_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_frame = test[
        [
            "fixture_id",
            "date",
            "league_name",
            "team",
            "opponent",
            "formation",
            "goals_for",
            "goals_against",
        ]
    ].copy()
    predictions_frame["expected_goals_for"] = np.round(predictions, 4)
    predictions_frame.to_csv(
        data_root / "processed" / "api_football" / "post_lineup_goals_predictions.csv",
        index=False,
    )
    return metrics
