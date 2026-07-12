from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
import bisect
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import lightgbm as lgb

from kinela.counter_efficiency import (
    COUNTER_EFFICIENCY_WINDOW,
    counter_history_summary,
    current_counter_threat,
    current_underdog_fit,
)
from kinela.clinical_finishing import (
    CLINICAL_FINISHING_WINDOW,
    clinical_finishing_summary,
)
from kinela.club_attacking_talent import (
    build_club_talent_state,
    club_talent_summary_before,
)
from kinela.fifa_ranking import (
    load_fifa_ranking,
    normalise_team_name,
    update_live_fifa_points,
)
from kinela.lightgbm_model import (
    CATEGORICAL_FEATURES,
    NEUTRAL_FEATURES,
    add_neutral_treated_features,
    blend_result_probabilities,
)
from kinela.model import (
    BASE_ELO,
    DETAIL_STAT_FEATURES,
    FOTMOB_WORLD_CUP_DETAIL_FEATURES,
    H2H_LOOKBACK_DAYS,
    RECENT_FORM_WINDOW,
    _cross_confederation_adjusted_points,
    _opponent_quality_factor,
    _quality_goal_balance,
    _team_confederation,
    _normalise_stage_or_round,
    load_late85_points_swing_metrics,
    load_score_timing_metrics,
)


@dataclass(frozen=True)
class TeamStanding:
    team: str
    group: str
    points: int
    goals_for: int
    goals_against: int

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


class PenaltyShootoutModel:
    """Fallback shootout model, ready to be replaced by a trained model."""

    def __init__(self, simulator: WorldCup2026Simulator) -> None:
        self.simulator = simulator

    def _team_penalty_net(self, team: str, before: date) -> float:
        team_key = _normalise_name(team)
        recent: list[dict[str, float]] = []
        for row in self.simulator.penalty_shootouts:
            if row["date"] >= before:
                continue
            if row["home_key"] == team_key:
                recent.append(
                    {
                        "scored": row["home_penalty_goals"],
                        "conceded": row["away_penalty_goals"],
                    }
                )
            elif row["away_key"] == team_key:
                recent.append(
                    {
                        "scored": row["away_penalty_goals"],
                        "conceded": row["home_penalty_goals"],
                    }
                )
        recent = recent[-3:]
        scored = 0.0
        conceded = 0.0
        for row in recent:
            scored += row["scored"]
            conceded += row["conceded"]
        return (scored - conceded) / max(1.0, scored + conceded)

    def team_a_probability(self, team_a: str, team_b: str, match_date: date) -> float:
        squad_diff = (
            self.simulator._squad_quality_score(team_a) - self.simulator._squad_quality_score(team_b)
        ) / 5
        score = 0.10 * squad_diff
        probability = 1 / (1 + math.exp(-score))
        return max(0.38, min(0.62, probability))


R32_SLOTS = [
    (73, ("2", "GROUP_A"), ("2", "GROUP_B")),
    (74, ("1", "GROUP_E"), ("3", ("GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D", "GROUP_F"))),
    (75, ("1", "GROUP_F"), ("2", "GROUP_C")),
    (76, ("1", "GROUP_C"), ("2", "GROUP_F")),
    (77, ("1", "GROUP_I"), ("3", ("GROUP_C", "GROUP_D", "GROUP_F", "GROUP_G", "GROUP_H"))),
    (78, ("2", "GROUP_E"), ("2", "GROUP_I")),
    (79, ("1", "GROUP_A"), ("3", ("GROUP_C", "GROUP_E", "GROUP_F", "GROUP_H", "GROUP_I"))),
    (80, ("1", "GROUP_L"), ("3", ("GROUP_E", "GROUP_H", "GROUP_I", "GROUP_J", "GROUP_K"))),
    (81, ("1", "GROUP_D"), ("3", ("GROUP_B", "GROUP_E", "GROUP_F", "GROUP_I", "GROUP_J"))),
    (82, ("1", "GROUP_G"), ("3", ("GROUP_A", "GROUP_E", "GROUP_H", "GROUP_I", "GROUP_J"))),
    (83, ("2", "GROUP_K"), ("2", "GROUP_L")),
    (84, ("1", "GROUP_H"), ("2", "GROUP_J")),
    (85, ("1", "GROUP_B"), ("3", ("GROUP_E", "GROUP_F", "GROUP_G", "GROUP_I", "GROUP_J"))),
    (86, ("1", "GROUP_J"), ("2", "GROUP_H")),
    (87, ("1", "GROUP_K"), ("3", ("GROUP_D", "GROUP_E", "GROUP_I", "GROUP_J", "GROUP_L"))),
    (88, ("2", "GROUP_D"), ("2", "GROUP_G")),
]

THIRD_PLACE_ASSIGNMENT_BY_MATCH = {
    74: "vs_1E",
    77: "vs_1I",
    79: "vs_1A",
    80: "vs_1L",
    81: "vs_1D",
    82: "vs_1G",
    85: "vs_1B",
    87: "vs_1K",
}

ROUND_OF_16 = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80)]
ROUND_OF_16 += [(93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
QUARTER_FINALS = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SEMI_FINALS = [(101, 97, 98), (102, 99, 100)]


def _normalise_name(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    text = " ".join(text.replace(".", "").split())
    aliases = {
        "korea republic": "south korea",
        "czechia": "czech republic",
        "turkiye": "turkey",
        "bosnia-herzegovina": "bosnia and herzegovina",
        "cote divoire": "ivory coast",
    }
    return aliases.get(text, text)


def _poisson_sample(lam: float, rng: random.Random) -> int:
    threshold = math.exp(-lam)
    k = 0
    product = 1.0
    while product > threshold:
        k += 1
        product *= rng.random()
    return k - 1


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _mean(values: list[float], fallback: float) -> float:
    return sum(values) / len(values) if values else fallback


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quality_result_points(points: int, goal_diff: int, opponent_elo: float) -> float:
    if points == 0:
        base = 0.35 if goal_diff == -1 else 0.15 if goal_diff == -2 else 0.0
    else:
        base = float(points)
    margin_bonus = 0.25 * max(0, min(2, goal_diff))
    opponent_factor = max(0.75, min(1.25, opponent_elo / BASE_ELO))
    return (base + margin_bonus) * opponent_factor


def _is_2026_host(team: str) -> bool:
    return _normalise_name(team) in {"canada", "mexico", "united states", "usa"}


def _warn_if_model_runtime_mismatch(model: dict[str, Any], path: Path) -> None:
    expected = {
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
    }
    mismatches = [
        f"{key} artifact={model[key]} runtime={runtime_version}"
        for key, runtime_version in expected.items()
        if model.get(key) and str(model[key]) != str(runtime_version)
    ]
    if mismatches:
        warnings.warn(
            "WARNING: LightGBM model runtime mismatch for "
            f"{path}: {'; '.join(mismatches)}. Use the project .venv or retrain.",
            RuntimeWarning,
            stacklevel=2,
        )


class WorldCup2026Simulator:
    def __init__(self, data_root: Path, seed: int = 42, engine: str = "lightgbm") -> None:
        self.data_root = data_root
        self.rng = random.Random(seed)
        self.engine = engine
        self.lightgbm_model = self._load_lightgbm_model() if engine == "lightgbm" else None
        self.penalty_model = PenaltyShootoutModel(self)
        self.simulated_histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.current_match_records: list[dict[str, Any]] = []
        self.prediction_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.history = self._load_history()
        self.late85_points_swing_metrics = load_late85_points_swing_metrics(
            self.data_root,
            self.history,
        )
        self.score_timing_metrics = load_score_timing_metrics(
            self.data_root,
            self.history,
        )
        self.club_talent_snapshots, self.club_talent_timelines = (
            build_club_talent_state(self.data_root)
        )
        self.penalty_shootouts = self._load_penalty_shootouts()
        self.team_histories = self._index_team_histories()
        self.head_to_head_histories = self._index_head_to_head_histories()
        self.elo_ratings = self._build_elo_ratings()
        self.confederation_stats, self.team_cross_confederation_stats = (
            self._build_confederation_contexts()
        )
        self.worldcup_histories = self._index_worldcup_histories()
        self.fifa_rankings = load_fifa_ranking(data_root)
        self.fallback_fifa_rank = (
            max((int(item["rank"]) for item in self.fifa_rankings.values()), default=211) + 25
        )
        self.fallback_fifa_points = min(
            (float(item["points"]) for item in self.fifa_rankings.values()),
            default=800.0,
        )
        self.tournament_start_fifa_point_overrides = self._build_fifa_point_overrides(
            include_manual_worldcup=False,
        )
        self.base_fifa_point_overrides = self._build_fifa_point_overrides(
            include_manual_worldcup=True,
        )
        self.fifa_point_overrides = dict(self.base_fifa_point_overrides)
        self.squad_quality = self._load_squad_quality()
        self.squad_players = self._load_squad_players()
        self.player_availability = self._load_player_availability()
        self.third_place_assignment_table = self._load_third_place_assignment_table()
        self.fixtures = self._load_fixtures()
        self.manual_results = self._load_manual_results()
        self.group_matches = [match for match in self.fixtures if match["stage"] == "GROUP_STAGE"]
        self.groups = self._build_groups()
        contexts = self._goal_contexts()
        self.global_goal_avg = contexts["global_avg"]
        self.major_goal_avg = contexts["major_avg"]
        self.group_goal_avg = contexts["group_avg"]
        self.knockout_goal_avg = contexts["knockout_avg"]
        self.major_match_count = contexts["major_matches"]
        self.group_match_count = contexts["group_matches"]
        self.knockout_match_count = contexts["knockout_matches"]
        if self.engine == "lightgbm" and self.lightgbm_model is not None:
            self._precompute_lightgbm_cache()

    def _load_lightgbm_model(self) -> dict[str, Any] | None:
        models_dir = self.data_root / "models"
        for path in (
            models_dir / "lightgbm_neutral_all_played_wc2026.joblib",
            models_dir / "lightgbm_neutral_worldcup_holdout.joblib",
            models_dir / "lightgbm_neutral_model.joblib",
        ):
            if path.exists():
                model = joblib.load(path)
                _warn_if_model_runtime_mismatch(model, path)
                return model
        return None

    def _refresh_derived_state(self) -> None:
        self.simulated_histories = defaultdict(list)
        self.current_match_records = []
        self.prediction_cache.clear()
        self.team_histories = self._index_team_histories()
        self.head_to_head_histories = self._index_head_to_head_histories()
        self.worldcup_histories = self._index_worldcup_histories()
        self.elo_ratings = self._build_elo_ratings()
        self.confederation_stats, self.team_cross_confederation_stats = (
            self._build_confederation_contexts()
        )
        self.tournament_start_fifa_point_overrides = self._build_fifa_point_overrides(
            include_manual_worldcup=False,
        )
        self.base_fifa_point_overrides = self._build_fifa_point_overrides(
            include_manual_worldcup=True,
        )
        self.fifa_point_overrides = dict(self.base_fifa_point_overrides)
        contexts = self._goal_contexts()
        self.global_goal_avg = contexts["global_avg"]
        self.major_goal_avg = contexts["major_avg"]
        self.group_goal_avg = contexts["group_avg"]
        self.knockout_goal_avg = contexts["knockout_avg"]
        self.major_match_count = contexts["major_matches"]
        self.group_match_count = contexts["group_matches"]
        self.knockout_match_count = contexts["knockout_matches"]

    def _load_third_place_assignment_table(self) -> dict[str, dict[int, str]]:
        path = self.data_root / "static" / "worldcup_2026_third_place_assignments.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Missing World Cup 2026 third-place assignment table: "
                f"{path}. Run with the local data/static table present."
            )
        table: dict[str, dict[int, str]] = {}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            qualified_groups = "".join(sorted(row["qualified_groups"]))
            table[qualified_groups] = {
                match_id: row[column]
                for match_id, column in THIRD_PLACE_ASSIGNMENT_BY_MATCH.items()
            }
        if len(table) != 495:
            raise ValueError(f"Expected 495 third-place combinations, found {len(table)}")
        return table

    def _load_history(self) -> list[dict[str, Any]]:
        path = self.data_root / "processed" / "combined" / "training_frame.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for row in rows:
            row["date_obj"] = date.fromisoformat(row["date"])
            row["home_goals"] = int(row["home_goals"])
            row["away_goals"] = int(row["away_goals"])
        return sorted(rows, key=lambda row: row["date_obj"])

    def _load_statsbomb_shootout_scores(self) -> dict[str, dict[str, float]]:
        path = self.data_root / "processed" / "statsbomb_world_cup_2022" / "goals.csv"
        if not path.exists():
            return {}
        scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row.get("is_shootout") != "True":
                continue
            scores[row["match_id"]][_normalise_name(row["team"])] += 1
        return {match_id: dict(team_scores) for match_id, team_scores in scores.items()}

    def _load_penalty_shootouts(self) -> list[dict[str, Any]]:
        min_date = date(2018, 1, 1)
        shootouts: dict[tuple[date, str, str], dict[str, Any]] = {}
        api_path = self.data_root / "processed" / "api_football" / "matches.csv"
        if api_path.exists():
            for row in csv.DictReader(api_path.open(encoding="utf-8")):
                match_date = date.fromisoformat(row["date"])
                if match_date < min_date:
                    continue
                if row.get("competition_type") == "friendly":
                    continue
                home_penalties = _float(row.get("home_penalty_goals"))
                away_penalties = _float(row.get("away_penalty_goals"))
                if home_penalties is None or away_penalties is None:
                    continue
                home_key = _normalise_name(row["home_team"])
                away_key = _normalise_name(row["away_team"])
                key = (match_date, *sorted((home_key, away_key)))
                shootouts[key] = {
                    "date": match_date,
                    "source": "api-football",
                    "competition_name": row.get("league_name", ""),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_penalty_goals": home_penalties,
                    "away_penalty_goals": away_penalties,
                }
        statsbomb_matches_path = self.data_root / "processed" / "statsbomb_world_cup_2022" / "matches.csv"
        statsbomb_scores = self._load_statsbomb_shootout_scores()
        if statsbomb_matches_path.exists():
            for row in csv.DictReader(statsbomb_matches_path.open(encoding="utf-8")):
                shootout = statsbomb_scores.get(row["match_id"], {})
                if not shootout:
                    continue
                match_date = date.fromisoformat(row["date"])
                if match_date < min_date:
                    continue
                home_key = _normalise_name(row["home_team"])
                away_key = _normalise_name(row["away_team"])
                key = (match_date, *sorted((home_key, away_key)))
                shootouts[key] = {
                    "date": match_date,
                    "source": "statsbomb-open-data",
                    "competition_name": "FIFA World Cup",
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_penalty_goals": shootout.get(home_key, 0.0),
                    "away_penalty_goals": shootout.get(away_key, 0.0),
                }
        return sorted(shootouts.values(), key=lambda row: row["date"])

    def _load_statsbomb_worldcup(self) -> list[dict[str, Any]]:
        path = self.data_root / "processed" / "statsbomb_world_cup_2022" / "matches.csv"
        if not path.exists():
            return []
        rows = []
        for row in csv.DictReader(path.open(encoding="utf-8")):
            rows.append(
                {
                    "date": date.fromisoformat(row["date"]),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_goals": int(row["home_score"]),
                    "away_goals": int(row["away_score"]),
                }
            )
        return sorted(rows, key=lambda row: row["date"])

    def _index_team_histories(self) -> dict[str, list[dict[str, Any]]]:
        indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.history:
            home = _normalise_name(row["home_team"])
            away = _normalise_name(row["away_team"])
            match_late85 = self.late85_points_swing_metrics.get(str(row["match_id"]), {})
            match_score_timing = self.score_timing_metrics.get(str(row["match_id"]), {})
            home_late85 = float(
                (match_late85.get(normalise_team_name(row["home_team"])) or {}).get(
                    "late85_points_swing_edge",
                    0.0,
                )
            )
            away_late85 = float(
                (match_late85.get(normalise_team_name(row["away_team"])) or {}).get(
                    "late85_points_swing_edge",
                    0.0,
                )
            )
            indexed[home].append(
                {
                    "date": row["date_obj"],
                    "source": row.get("source", ""),
                    "competition_name": row.get("competition_name", ""),
                    "gf": row["home_goals"],
                    "ga": row["away_goals"],
                    "points": _points(row["home_goals"], row["away_goals"]),
                    "opponent_elo": _float(row.get("home_opponent_elo_pre")) or BASE_ELO,
                    "quality_result_points": _quality_result_points(
                        _points(row["home_goals"], row["away_goals"]),
                        row["home_goals"] - row["away_goals"],
                        _float(row.get("home_opponent_elo_pre")) or BASE_ELO,
                    ),
                    "quality_goal_balance": _quality_goal_balance(
                        row["home_goals"],
                        row["away_goals"],
                        _float(row.get("home_opponent_elo_pre")) or BASE_ELO,
                    ),
                    "late85_points_swing": home_late85,
                    "score_timing": match_score_timing.get(
                        normalise_team_name(row["home_team"]),
                        {},
                    ),
                    "counter_item": {
                        "goals_for": row["home_goals"],
                        "points": _points(row["home_goals"], row["away_goals"]),
                        "team_elo": _float(row.get("home_elo_pre")) or BASE_ELO,
                        "opponent_elo": _float(row.get("home_opponent_elo_pre"))
                        or BASE_ELO,
                        "ball_possession_pct": _float(
                            row.get("home_actual_ball_possession_pct")
                        ),
                        "total_passes": _float(row.get("home_actual_total_passes")),
                        "passes_pct": _float(row.get("home_actual_passes_pct")),
                        "total_shots": _float(row.get("home_actual_total_shots")),
                        "shots_on_goal": _float(row.get("home_actual_shots_on_goal")),
                    },
                    "detail_stats": {
                        feature: (
                            _float(row.get(f"home_actual_{feature}"))
                            if feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
                            or _float(row.get(f"home_actual_{feature}")) is not None
                            else _float(row.get(f"home_recent6_{feature}_avg"))
                        )
                        for feature in DETAIL_STAT_FEATURES
                    },
                }
            )
            indexed[away].append(
                {
                    "date": row["date_obj"],
                    "source": row.get("source", ""),
                    "competition_name": row.get("competition_name", ""),
                    "gf": row["away_goals"],
                    "ga": row["home_goals"],
                    "points": _points(row["away_goals"], row["home_goals"]),
                    "opponent_elo": _float(row.get("away_opponent_elo_pre")) or BASE_ELO,
                    "quality_result_points": _quality_result_points(
                        _points(row["away_goals"], row["home_goals"]),
                        row["away_goals"] - row["home_goals"],
                        _float(row.get("away_opponent_elo_pre")) or BASE_ELO,
                    ),
                    "quality_goal_balance": _quality_goal_balance(
                        row["away_goals"],
                        row["home_goals"],
                        _float(row.get("away_opponent_elo_pre")) or BASE_ELO,
                    ),
                    "late85_points_swing": away_late85,
                    "score_timing": match_score_timing.get(
                        normalise_team_name(row["away_team"]),
                        {},
                    ),
                    "counter_item": {
                        "goals_for": row["away_goals"],
                        "points": _points(row["away_goals"], row["home_goals"]),
                        "team_elo": _float(row.get("away_elo_pre")) or BASE_ELO,
                        "opponent_elo": _float(row.get("away_opponent_elo_pre"))
                        or BASE_ELO,
                        "ball_possession_pct": _float(
                            row.get("away_actual_ball_possession_pct")
                        ),
                        "total_passes": _float(row.get("away_actual_total_passes")),
                        "passes_pct": _float(row.get("away_actual_passes_pct")),
                        "total_shots": _float(row.get("away_actual_total_shots")),
                        "shots_on_goal": _float(row.get("away_actual_shots_on_goal")),
                    },
                    "detail_stats": {
                        feature: (
                            _float(row.get(f"away_actual_{feature}"))
                            if feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
                            or _float(row.get(f"away_actual_{feature}")) is not None
                            else _float(row.get(f"away_recent6_{feature}_avg"))
                        )
                        for feature in DETAIL_STAT_FEATURES
                    },
                }
            )
        return {team: sorted(matches, key=lambda item: item["date"]) for team, matches in indexed.items()}

    def _index_head_to_head_histories(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.history:
            home_key = _normalise_name(row["home_team"])
            away_key = _normalise_name(row["away_team"])
            key = tuple(sorted((home_key, away_key)))
            indexed[key].append(
                {
                    "date": row["date_obj"],
                    "home_key": home_key,
                    "away_key": away_key,
                    "home_goals": int(row["home_goals"]),
                    "away_goals": int(row["away_goals"]),
                    "home_penalty_goals": _float(row.get("home_penalty_goals")),
                    "away_penalty_goals": _float(row.get("away_penalty_goals")),
                }
            )
        return {key: sorted(matches, key=lambda item: item["date"]) for key, matches in indexed.items()}

    def _build_elo_ratings(self) -> dict[str, float]:
        ratings: dict[str, float] = defaultdict(lambda: BASE_ELO)
        for row in self.history:
            home = _normalise_name(row["home_team"])
            away = _normalise_name(row["away_team"])
            home_elo = ratings[home]
            away_elo = ratings[away]
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            actual_home = (
                1.0
                if row["home_goals"] > row["away_goals"]
                else 0.5
                if row["home_goals"] == row["away_goals"]
                else 0.0
            )
            goal_margin = min(abs(row["home_goals"] - row["away_goals"]), 3)
            k_factor = 24 * (1 + 0.15 * goal_margin)
            change = k_factor * (actual_home - expected_home)
            ratings[home] = home_elo + change
            ratings[away] = away_elo - change
        return dict(ratings)

    def _build_confederation_contexts(self) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        confederations: dict[str, dict[str, float]] = defaultdict(
            lambda: {"matches": 0, "points": 0.0, "adjusted_points": 0.0, "gf": 0.0, "ga": 0.0}
        )
        cross: dict[str, dict[str, float]] = defaultdict(
            lambda: {"matches": 0, "points": 0.0, "adjusted_points": 0.0, "gf": 0.0, "ga": 0.0}
        )
        for row in self.history:
            for side, opp in (("home", "away"), ("away", "home")):
                team = row[f"{side}_team"]
                opponent = row[f"{opp}_team"]
                team_confederation = _team_confederation(team)
                opponent_confederation = _team_confederation(opponent)
                gf = int(row[f"{side}_goals"])
                ga = int(row[f"{opp}_goals"])
                points = _points(gf, ga)
                adjusted_points = float(points)
                confed_stats = confederations[team_confederation]
                confed_stats["matches"] += 1
                confed_stats["points"] += points
                confed_stats["adjusted_points"] += adjusted_points
                confed_stats["gf"] += gf
                confed_stats["ga"] += ga
                if (
                    team_confederation != "unknown"
                    and opponent_confederation != "unknown"
                    and team_confederation != opponent_confederation
                ):
                    adjusted_points = _cross_confederation_adjusted_points(
                        points,
                        gf - ga,
                        _float(row.get(f"{side}_elo_pre")) or BASE_ELO,
                        _float(row.get(f"{opp}_elo_pre")) or BASE_ELO,
                    )
                    confed_stats["adjusted_points"] += adjusted_points - points
                    team_stats = cross[_normalise_name(team)]
                    team_stats["matches"] += 1
                    team_stats["points"] += points
                    team_stats["adjusted_points"] += adjusted_points
                    team_stats["gf"] += gf
                    team_stats["ga"] += ga
        return dict(confederations), dict(cross)

    def _strength(self, stats: dict[str, float] | None, fallback: float = 1.0) -> float:
        if not stats or not stats.get("matches"):
            return fallback
        matches = stats["matches"]
        points_per_match = stats.get("adjusted_points", stats["points"]) / matches
        goal_diff_per_match = (stats["gf"] - stats["ga"]) / matches
        return points_per_match + 0.35 * goal_diff_per_match

    def _index_worldcup_histories(self) -> dict[str, list[dict[str, Any]]]:
        indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._load_statsbomb_worldcup():
            home = _normalise_name(row["home_team"])
            away = _normalise_name(row["away_team"])
            indexed[home].append(
                {
                    "date": row["date"],
                    "outcome": 1
                    if row["home_goals"] > row["away_goals"]
                    else -1
                    if row["away_goals"] > row["home_goals"]
                    else 0,
                }
            )
            indexed[away].append(
                {
                    "date": row["date"],
                    "outcome": 1
                    if row["away_goals"] > row["home_goals"]
                    else -1
                    if row["home_goals"] > row["away_goals"]
                    else 0,
                }
            )
        return {team: sorted(matches, key=lambda item: item["date"]) for team, matches in indexed.items()}

    def _load_fixtures(self) -> list[dict[str, Any]]:
        path = self.data_root / "raw" / "football_data" / "competitions" / "WC" / "matches.json"
        return json.loads(path.read_text(encoding="utf-8"))["matches"]

    def _load_manual_results(self) -> dict[str, dict[str, Any]]:
        path = self.data_root / "static" / "worldcup_2026_manual_results.csv"
        if not path.exists():
            return {}
        results: dict[str, dict[str, Any]] = {}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            match_id = str(row.get("match_id") or "").strip()
            if not match_id:
                continue
            team_a_goals = int(row["team_a_goals"])
            team_b_goals = int(row["team_b_goals"])
            winner = row.get("winner") or None
            team_a_penalty_goals = row.get("team_a_penalty_goals") or None
            team_b_penalty_goals = row.get("team_b_penalty_goals") or None
            results[match_id] = {
                **row,
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "winner": winner,
                "team_a_penalty_goals": (
                    int(team_a_penalty_goals) if team_a_penalty_goals is not None else None
                ),
                "team_b_penalty_goals": (
                    int(team_b_penalty_goals) if team_b_penalty_goals is not None else None
                ),
                "penalty_winner": row.get("penalty_winner") or None,
            }
        return results
    def _load_squad_quality(self) -> dict[str, dict[str, float]]:
        path = self.data_root / "processed" / "football_data" / "squad_quality.csv"
        if not path.exists():
            return {}
        quality: dict[str, dict[str, float]] = {}
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row.get("competition_code") != "WC":
                continue
            key = _normalise_name(row.get("team"))
            if not key:
                continue
            values: dict[str, float] = {}
            for column in (
                "squad_size",
                "avg_age",
                "known_strength_share",
                "squad_avg_competition_strength",
                "squad_top11_competition_strength",
                "squad_top5_competition_strength",
                "squad_depth_competition_strength",
                "squad_top_competition_share",
                "squad_elite_competition_share",
            ):
                values[column] = float(row.get(column) or 0)
            quality[key] = values
        return quality

    def _load_squad_players(self) -> dict[str, dict[str, dict[str, float | str]]]:
        path = self.data_root / "processed" / "football_data" / "squad_players.csv"
        if not path.exists():
            return {}
        players: dict[str, dict[str, dict[str, float | str]]] = defaultdict(dict)
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row.get("competition_code") != "WC":
                continue
            team_key = _normalise_name(row.get("team"))
            player_key = _normalise_name(row.get("player"))
            if not team_key or not player_key:
                continue
            players[team_key][player_key] = {
                "position": row.get("position") or "Unknown",
                "avg_competition_strength": float(row.get("avg_competition_strength") or 0),
                "max_competition_strength": float(row.get("max_competition_strength") or 0),
                "top_competition_share": float(row.get("top_competition_share") or 0),
                "elite_competition_share": float(row.get("elite_competition_share") or 0),
            }
        return {team: dict(team_players) for team, team_players in players.items()}

    def _load_player_availability(self) -> dict[str, list[dict[str, Any]]]:
        path = self.data_root / "static" / "worldcup_2026_player_availability.csv"
        availability: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if path.exists():
            for row in csv.DictReader(path.open(encoding="utf-8")):
                team_key = _normalise_name(row.get("team"))
                player_key = _normalise_name(row.get("player"))
                if not team_key or not player_key:
                    continue
                availability[team_key].append(
                    {
                        **row,
                        "player_key": player_key,
                        "unavailable_from": date.fromisoformat(row["unavailable_from"]),
                        "unavailable_until": date.fromisoformat(row["unavailable_until"]),
                        "importance": float(row.get("importance") or 0),
                    }
                )
        for row in self._yellow_card_availability_from_manual_results():
            availability[row["team_key"]].append(row)
        return {team: rows for team, rows in availability.items()}

    def _yellow_card_block(self, stage: str) -> str | None:
        stage_key = (stage or "").upper()
        if stage_key == "GROUP_STAGE":
            return "group_stage"
        if stage_key in {"ROUND_OF_32", "LAST_16", "QUARTER_FINALS"}:
            return "knockout_to_quarterfinal"
        return None

    def _next_fixture_date_for_team(self, team: str, after: date) -> date | None:
        fixtures_path = self.data_root / "raw" / "football_data" / "competitions" / "WC" / "matches.json"
        if not fixtures_path.exists():
            return None
        team_key = _normalise_name(team)
        payload = json.loads(fixtures_path.read_text(encoding="utf-8"))
        dates: list[date] = []
        for match in payload.get("matches", []):
            home = _normalise_name((match.get("homeTeam") or {}).get("name"))
            away = _normalise_name((match.get("awayTeam") or {}).get("name"))
            if team_key not in {home, away}:
                continue
            match_date = date.fromisoformat(match["utcDate"][:10])
            if match_date > after:
                dates.append(match_date)
        return min(dates) if dates else None

    def _yellow_cards_from_notes(self, notes: str) -> list[tuple[str, str]]:
        match = re.search(r"yellow cards?:\s*(.*?)(?:\.\s|$)", notes or "", flags=re.IGNORECASE)
        if not match:
            return []
        cards: list[tuple[str, str]] = []
        for segment in match.group(1).split(";"):
            team_match = re.search(r"(.+?)\s+for\s+([^;]+)$", segment.strip(), flags=re.IGNORECASE)
            if not team_match:
                continue
            players_text = team_match.group(1)
            team = team_match.group(2).strip()
            for raw_player in re.findall(r"([A-Za-zÀ-ÿ' .-]+?)\s+\d+(?:\+\d+)?'", players_text):
                player = re.sub(r"^(and|,)\s+", "", raw_player.strip(), flags=re.IGNORECASE)
                if player:
                    cards.append((team, player))
        return cards

    def _yellow_card_availability_from_manual_results(self) -> list[dict[str, Any]]:
        manual_path = self.data_root / "static" / "worldcup_2026_manual_results.csv"
        if not manual_path.exists():
            return []
        by_player_block: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in csv.DictReader(manual_path.open(encoding="utf-8")):
            block = self._yellow_card_block(row.get("stage", ""))
            if block is None:
                continue
            match_date = date.fromisoformat(row["date"])
            for team, player in self._yellow_cards_from_notes(row.get("notes", "")):
                by_player_block[(_normalise_name(team), _normalise_name(player), block)].append(
                    {
                        "team": team,
                        "player": player,
                        "date": match_date,
                        "source": row.get("source", "manual-worldcup-2026"),
                    }
                )

        suspensions: list[dict[str, Any]] = []
        for (team_key, player_key, block), cards in by_player_block.items():
            if len(cards) < 2:
                continue
            second_card = sorted(cards, key=lambda item: item["date"])[1]
            next_match_date = self._next_fixture_date_for_team(second_card["team"], second_card["date"])
            if next_match_date is None:
                continue
            suspensions.append(
                {
                    "team": second_card["team"],
                    "team_key": team_key,
                    "player": second_card["player"],
                    "player_key": player_key,
                    "position": "Unknown",
                    "unavailable_from": next_match_date,
                    "unavailable_until": next_match_date,
                    "reason": "yellow_card_accumulation",
                    "importance": 0.0,
                    "source": second_card["source"],
                    "notes": f"Automatic one-match suspension after two yellow cards in {block}.",
                }
            )
        return suspensions

    def _player_availability_penalty(self, team: str, match_date: date | None) -> float:
        if match_date is None:
            return 0.0
        team_key = _normalise_name(team)
        players = self.squad_players.get(team_key, {})
        penalty = 0.0
        position_weights = {
            "goalkeeper": 1.10,
            "defence": 0.95,
            "defender": 0.95,
            "midfield": 1.00,
            "offence": 1.05,
            "forward": 1.05,
        }
        for item in self.player_availability.get(team_key, []):
            if not (item["unavailable_from"] <= match_date <= item["unavailable_until"]):
                continue
            player = players.get(item["player_key"], {})
            position = str(player.get("position") or item.get("position") or "Unknown").lower()
            position_weight = next(
                (weight for marker, weight in position_weights.items() if marker in position),
                1.0,
            )
            local_score = (
                0.60 * float(player.get("max_competition_strength") or 0)
                + 0.25 * float(player.get("avg_competition_strength") or 0)
                + 1.00 * float(player.get("top_competition_share") or 0)
                + 1.00 * float(player.get("elite_competition_share") or 0)
            )
            local_importance = min(1.0, local_score / 10.0)
            importance = max(float(item["importance"]), local_importance)
            if item.get("reason") == "yellow_card_accumulation":
                # Accumulated yellows only matter through the player actually lost.
                penalty += position_weight * importance
            else:
                penalty += position_weight * (0.35 + 0.65 * importance)
        return min(3.0, penalty)

    def _squad_quality_score(self, team: str, match_date: date | None = None) -> float:
        quality = self.squad_quality.get(_normalise_name(team), {})
        if not quality:
            return 0.0
        base_score = (
            0.45 * quality.get("squad_top11_competition_strength", 0.0)
            + 0.25 * quality.get("squad_depth_competition_strength", 0.0)
            + 0.20 * quality.get("squad_top5_competition_strength", 0.0)
            + 1.50 * quality.get("known_strength_share", 0.0)
            + 1.00 * quality.get("squad_elite_competition_share", 0.0)
        )
        return max(0.0, base_score - self._player_availability_penalty(team, match_date))

    def _squad_strength_multipliers(
        self,
        team_a: str,
        team_b: str,
        match_date: date | None = None,
    ) -> tuple[float, float]:
        team_a_penalty = self._player_availability_penalty(team_a, match_date)
        team_b_penalty = self._player_availability_penalty(team_b, match_date)
        diff = self._squad_quality_score(team_a, match_date) - self._squad_quality_score(team_b, match_date)
        diff = max(-4.0, min(4.0, diff))
        team_a_multiplier = 1 + 0.025 * diff
        team_b_multiplier = 1 - 0.025 * diff
        team_a_multiplier *= max(0.90, 1 - 0.025 * team_a_penalty)
        team_b_multiplier *= max(0.90, 1 - 0.025 * team_b_penalty)
        team_a_multiplier *= min(1.06, 1 + 0.0125 * team_b_penalty)
        team_b_multiplier *= min(1.06, 1 + 0.0125 * team_a_penalty)
        return (
            max(0.88, min(1.12, team_a_multiplier)),
            max(0.88, min(1.12, team_b_multiplier)),
        )

    def _build_groups(self) -> dict[str, list[str]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for match in self.group_matches:
            groups[match["group"]].add(match["homeTeam"]["name"])
            groups[match["group"]].add(match["awayTeam"]["name"])
        return {group: sorted(teams) for group, teams in groups.items()}

    def _goal_contexts(self) -> dict[str, float | int]:
        all_goals = []
        major_goals = []
        group_goals = []
        knockout_goals = []
        for row in self.history:
            pair = [row["home_goals"], row["away_goals"]]
            all_goals.extend(pair)
            if row["competition_type"] == "major_tournament":
                major_goals.extend(pair)
                stage = (row["stage_or_round"] or "").lower()
                if "group" in stage:
                    group_goals.extend(pair)
                if any(marker in stage for marker in ("final", "last", "quarter", "semi")):
                    knockout_goals.extend(pair)
        global_avg = _mean(all_goals, 1.35)
        major_avg = _mean(major_goals, global_avg)
        return {
            "global_avg": global_avg,
            "major_avg": major_avg,
            "group_avg": _mean(group_goals, major_avg),
            "knockout_avg": _mean(knockout_goals, major_avg),
            "major_matches": len(major_goals) // 2,
            "group_matches": len(group_goals) // 2,
            "knockout_matches": len(knockout_goals) // 2,
        }

    def _team_history(
        self,
        team: str,
        before: date,
        *,
        include_simulated: bool = True,
    ) -> list[dict[str, Any]]:
        key = _normalise_name(team)
        matches = self.team_histories.get(key, [])
        if include_simulated:
            matches = matches + self.simulated_histories.get(key, [])
        matches = sorted(matches, key=lambda item: item["date"])
        index = bisect.bisect_left([item["date"] for item in matches], before)
        return matches[:index]

    def _team_features(
        self,
        team: str,
        match_date: date,
        *,
        include_simulated: bool = True,
    ) -> dict[str, float | int | None]:
        history = self._team_history(team, match_date, include_simulated=include_simulated)
        recent = history[-RECENT_FORM_WINDOW:]
        goals_for_avg = _mean([item["gf"] for item in history], self.global_goal_avg)
        goals_against_avg = _mean([item["ga"] for item in history], self.global_goal_avg)
        detail_stats = {}
        detail_coverage_values = []
        for feature in DETAIL_STAT_FEATURES:
            values = [
                item["detail_stats"][feature]
                for item in recent
                if item.get("detail_stats", {}).get(feature) is not None
            ]
            detail_stats[feature] = _mean(values, 0.0)
            detail_coverage_values.append(1.0 if values else 0.0)

        def fotmob_worldcup_detail_stats(matches: list[dict[str, Any]]) -> dict[str, float]:
            stats: dict[str, float] = {}
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                values = [
                    item["detail_stats"][feature]
                    for item in matches
                    if item.get("detail_stats", {}).get(feature) is not None
                ]
                stats[feature] = _mean(values, 0.0)
            return stats

        worldcup_recent = [
            item
            for item in history
            if item.get("competition_name") == "FIFA World Cup"
            and item.get("detail_stats", {}).get("fotmob_detail_coverage") is not None
        ][-RECENT_FORM_WINDOW:]
        current_worldcup_recent = [
            item
            for item in worldcup_recent
            if item["date"].year == match_date.year
        ][-RECENT_FORM_WINDOW:]
        score_metric_names = [
            "score_state_value",
            "score_control_value",
            "scoring_quickness",
            "score_control_quality",
            "narrow_lead_hold",
            "comfortable_lead",
            "game_state_friction",
            "state_change_swing",
            "early_state_change_swing",
        ]
        score_timing_coverage = (
            sum(bool(item.get("score_timing")) for item in recent)
            / max(RECENT_FORM_WINDOW, 1)
        )
        score_timing = {
            metric: _mean(
                [
                    float(item.get("score_timing", {}).get(metric, 0.0))
                    for item in recent
                ],
                0.0,
            )
            for metric in score_metric_names
        }
        quality_score_timing = {
            metric: _mean(
                [
                    float(item.get("score_timing", {}).get(metric, 0.0))
                    * _opponent_quality_factor(float(item.get("opponent_elo", BASE_ELO)))
                    for item in recent
                ],
                0.0,
            )
            for metric in [
                "score_control_value",
                "state_change_swing",
                "early_state_change_swing",
            ]
        }
        counter_summary = counter_history_summary(
            [
                item["counter_item"]
                for item in history
                if item.get("counter_item") is not None
            ],
            window=COUNTER_EFFICIENCY_WINDOW,
        )
        clinical_summary = clinical_finishing_summary(
            [
                {
                    "goals_for": item["gf"],
                    "shots_on_goal": item.get("detail_stats", {}).get(
                        "shots_on_goal"
                    ),
                    "total_shots": item.get("detail_stats", {}).get(
                        "total_shots"
                    ),
                    "ball_possession_pct": item.get("detail_stats", {}).get(
                        "ball_possession_pct"
                    ),
                    "total_passes": item.get("detail_stats", {}).get(
                        "total_passes"
                    ),
                    "passes_pct": item.get("detail_stats", {}).get(
                        "passes_pct"
                    ),
                    "corner_kicks": item.get("detail_stats", {}).get(
                        "corner_kicks"
                    ),
                }
                for item in history
            ],
            window=CLINICAL_FINISHING_WINDOW,
        )
        club_talent_summary = club_talent_summary_before(
            self.club_talent_timelines,
            team,
            match_date,
        )
        return {
            "matches": len(history),
            "goals_for_avg": goals_for_avg,
            "goals_against_avg": goals_against_avg,
            "recent6_goals_for_avg": _mean([item["gf"] for item in recent], goals_for_avg),
            "recent6_goals_against_avg": _mean([item["ga"] for item in recent], goals_against_avg),
            "recent6_points_avg": _mean([item["points"] for item in recent], 1.0),
            "recent6_win_rate": _mean(
                [1.0 if item["points"] == 3 else 0.0 for item in recent],
                1 / 3,
            ),
            "recent6_goal_diff_avg": _mean([item["gf"] - item["ga"] for item in recent], 0.0),
            "recent6_opponent_elo_avg": _mean(
                [item["opponent_elo"] for item in recent],
                BASE_ELO,
            ),
            "recent6_quality_result_points_avg": _mean(
                [item["quality_result_points"] for item in recent],
                1.0,
            ),
            "recent6_quality_goal_balance_avg": _mean(
                [item["quality_goal_balance"] for item in recent],
                0.0,
            ),
            "recent6_late85_points_swing": _mean(
                [float(item.get("late85_points_swing", 0.0)) for item in recent],
                0.0,
            ),
            "recent6_score_state_value": score_timing["score_state_value"],
            "recent6_score_control_value": score_timing["score_control_value"],
            "recent6_scoring_quickness": score_timing["scoring_quickness"],
            "recent6_score_control_quality": score_timing["score_control_quality"],
            "recent6_narrow_lead_hold": score_timing["narrow_lead_hold"],
            "recent6_comfortable_lead": score_timing["comfortable_lead"],
            "recent6_game_state_friction": score_timing["game_state_friction"],
            "recent6_state_change_swing": score_timing["state_change_swing"],
            "recent6_early_state_change_swing": score_timing["early_state_change_swing"],
            "recent6_quality_score_control_value": quality_score_timing[
                "score_control_value"
            ],
            "recent6_quality_state_change_swing": quality_score_timing[
                "state_change_swing"
            ],
            "recent6_quality_early_state_change_swing": quality_score_timing[
                "early_state_change_swing"
            ],
            "recent6_score_timing_coverage": score_timing_coverage,
            "counter_summary": counter_summary,
            "clinical_summary": clinical_summary,
            "club_attack_talent_signal": club_talent_summary[
                "club_attack_talent_signal"
            ],
            "club_star_finisher_signal": club_talent_summary[
                "club_star_finisher_signal"
            ],
            "club_attack_coverage": club_talent_summary["club_attack_coverage"],
            "worldcup_recent6_win_rate": self._worldcup_recent6_win_rate(team, match_date),
            "rest_days": (match_date - history[-1]["date"]).days if history else None,
            "detail_stats": detail_stats,
            "worldcup_detail_stats": fotmob_worldcup_detail_stats(worldcup_recent),
            "current_worldcup_detail_stats": fotmob_worldcup_detail_stats(
                current_worldcup_recent,
            ),
            "detail_coverage": _mean(detail_coverage_values, 0.0),
        }

    def _worldcup_recent6_win_rate(self, team: str, before: date) -> float:
        matches = self.worldcup_histories.get(_normalise_name(team), [])
        index = bisect.bisect_left([item["date"] for item in matches], before)
        recent = matches[:index][-6:]
        return _mean([1.0 if item["outcome"] == 1 else 0.0 for item in recent], 1 / 3)

    def _fifa_ranking_features(self, team: str) -> dict[str, float]:
        key = normalise_team_name(team)
        ranking = self.fifa_rankings.get(key, {})
        return {
            "observed": float(bool(ranking)),
            "fifa_rank": float(ranking.get("rank", self.fallback_fifa_rank)),
            "official_fifa_points": float(
                ranking.get("points", self.fallback_fifa_points)
            ),
            "fifa_points": float(
                self.fifa_point_overrides.get(
                    key,
                    float(ranking.get("points", self.fallback_fifa_points)),
                )
            ),
        }

    def _elo(self, team: str) -> float:
        return self.elo_ratings.get(_normalise_name(team), BASE_ELO)

    def _build_fifa_point_overrides(
        self,
        *,
        include_manual_worldcup: bool = True,
    ) -> dict[str, float]:
        points: dict[str, float] = {
            key: float(value.get("points", self.fallback_fifa_points))
            for key, value in self.fifa_rankings.items()
        }
        if not include_manual_worldcup:
            return points
        for row in self.history:
            if row.get("source") != "manual-worldcup-2026":
                continue
            home = normalise_team_name(row["home_team"])
            away = normalise_team_name(row["away_team"])
            home_points = points.get(home, self.fallback_fifa_points)
            away_points = points.get(away, self.fallback_fifa_points)
            points[home], points[away] = update_live_fifa_points(
                home_points,
                away_points,
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
        return points

    def _reset_tournament_state(self) -> None:
        self.simulated_histories = defaultdict(list)
        self.current_match_records = []
        self.fifa_point_overrides = dict(self.tournament_start_fifa_point_overrides)
        self.prediction_cache.clear()

    def _worldcup_last6_flags(self, team: str, before: date) -> dict[str, int]:
        matches = self.worldcup_histories.get(_normalise_name(team), [])
        index = bisect.bisect_left([item["date"] for item in matches], before)
        recent = matches[:index][-6:]
        flags = {}
        for item_index in range(6):
            outcome = recent[-(item_index + 1)]["outcome"] if item_index < len(recent) else None
            flags[f"worldcup_last6_{item_index + 1}_win"] = int(outcome == 1)
            flags[f"worldcup_last6_{item_index + 1}_draw"] = int(outcome == 0)
            flags[f"worldcup_last6_{item_index + 1}_loss"] = int(outcome == -1)
        return flags

    def _head_to_head_features(self, team_a: str, team_b: str, before: date) -> dict[str, float | int]:
        team_a_key = _normalise_name(team_a)
        team_b_key = _normalise_name(team_b)
        key = tuple(sorted((team_a_key, team_b_key)))
        recent = [
            match
            for match in self.head_to_head_histories.get(key, [])
            if 0 < (before - match["date"]).days <= H2H_LOOKBACK_DAYS
        ]
        if not recent:
            return {
                "matches": 0,
                "days_since_last": 0,
                "team_a_goals_avg": 0.0,
                "team_b_goals_avg": 0.0,
                "goal_diff_avg": 0.0,
                "team_a_points_avg": 0.0,
                "team_b_points_avg": 0.0,
                "points_diff": 0.0,
                "draw_rate": 0.0,
                "penalty_shootout_matches": 0,
                "team_a_penalty_wins": 0,
                "team_b_penalty_wins": 0,
                "penalty_wins_diff": 0,
            }
        a_goals = []
        b_goals = []
        a_points = []
        b_points = []
        draws = 0
        penalty_matches = 0
        a_penalty_wins = 0
        b_penalty_wins = 0
        for match in recent:
            if match["home_key"] == team_a_key:
                goals_a = match["home_goals"]
                goals_b = match["away_goals"]
                penalties_a = match.get("home_penalty_goals")
                penalties_b = match.get("away_penalty_goals")
            else:
                goals_a = match["away_goals"]
                goals_b = match["home_goals"]
                penalties_a = match.get("away_penalty_goals")
                penalties_b = match.get("home_penalty_goals")
            a_goals.append(goals_a)
            b_goals.append(goals_b)
            a_points.append(3 if goals_a > goals_b else 1 if goals_a == goals_b else 0)
            b_points.append(3 if goals_b > goals_a else 1 if goals_a == goals_b else 0)
            draws += int(goals_a == goals_b)
            if penalties_a is not None and penalties_b is not None:
                penalty_matches += 1
                a_penalty_wins += int(penalties_a > penalties_b)
                b_penalty_wins += int(penalties_b > penalties_a)
        a_goals_avg = _mean(a_goals, 0.0)
        b_goals_avg = _mean(b_goals, 0.0)
        a_points_avg = _mean(a_points, 0.0)
        b_points_avg = _mean(b_points, 0.0)
        return {
            "matches": len(recent),
            "days_since_last": (before - recent[-1]["date"]).days,
            "team_a_goals_avg": a_goals_avg,
            "team_b_goals_avg": b_goals_avg,
            "goal_diff_avg": a_goals_avg - b_goals_avg,
            "team_a_points_avg": a_points_avg,
            "team_b_points_avg": b_points_avg,
            "points_diff": a_points_avg - b_points_avg,
            "draw_rate": draws / len(recent),
            "penalty_shootout_matches": penalty_matches,
            "team_a_penalty_wins": a_penalty_wins,
            "team_b_penalty_wins": b_penalty_wins,
            "penalty_wins_diff": a_penalty_wins - b_penalty_wins,
        }

    def _group_table_context(
        self,
        team_a: str,
        team_b: str,
        group_table: dict[str, dict[str, Any]] | None,
    ) -> dict[str, float | int]:
        def empty(team: str) -> dict[str, Any]:
            return {"team": team, "points": 0, "gf": 0, "ga": 0}

        table = group_table or {}
        rankings = sorted(
            table.values(),
            key=lambda item: (
                int(item.get("points", 0)),
                int(item.get("gf", 0)) - int(item.get("ga", 0)),
                int(item.get("gf", 0)),
                str(item.get("team", "")),
            ),
            reverse=True,
        )
        positions = {item["team"]: index + 1 for index, item in enumerate(rankings)}
        a = {**empty(team_a), **table.get(team_a, {})}
        b = {**empty(team_b), **table.get(team_b, {})}
        a_gd = int(a.get("gf", 0)) - int(a.get("ga", 0))
        b_gd = int(b.get("gf", 0)) - int(b.get("ga", 0))
        return {
            "team_a_group_matches_pre": int(a.get("played", 0)),
            "team_a_group_points_pre": int(a.get("points", 0)),
            "team_a_group_goal_diff_pre": a_gd,
            "team_a_group_goals_for_pre": int(a.get("gf", 0)),
            "team_a_group_goals_against_pre": int(a.get("ga", 0)),
            "team_a_group_position_pre": positions.get(team_a, 0),
            "team_b_group_matches_pre": int(b.get("played", 0)),
            "team_b_group_points_pre": int(b.get("points", 0)),
            "team_b_group_goal_diff_pre": b_gd,
            "team_b_group_goals_for_pre": int(b.get("gf", 0)),
            "team_b_group_goals_against_pre": int(b.get("ga", 0)),
            "team_b_group_position_pre": positions.get(team_b, 0),
            "group_points_diff_pre": int(a.get("points", 0)) - int(b.get("points", 0)),
            "group_goal_diff_diff_pre": a_gd - b_gd,
            "group_position_diff_pre": positions.get(team_a, 0) - positions.get(team_b, 0),
        }

    def _stage_context(self, stage: str) -> dict[str, float | int | str]:
        stage_text = stage.lower()
        if stage_text == "group_stage":
            return {
                "stage_or_round": _normalise_stage_or_round(stage),
                "phase_train_matches": self.group_match_count,
                "phase_goals_avg": self.group_goal_avg,
            }
        return {
            "stage_or_round": _normalise_stage_or_round(stage),
            "phase_train_matches": self.knockout_match_count,
            "phase_goals_avg": self.knockout_goal_avg,
        }

    def _lightgbm_features(
        self,
        team_a: str,
        team_b: str,
        match_date: date,
        stage: str,
        group_table: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Keep recent-form features anchored to real completed matches; simulated
        # results only update live FIFA overrides and explicit group context.
        a = self._team_features(team_a, match_date, include_simulated=False)
        b = self._team_features(team_b, match_date, include_simulated=False)
        fifa_a = self._fifa_ranking_features(team_a)
        fifa_b = self._fifa_ranking_features(team_b)
        stage_context = self._stage_context(stage)
        a_elo = self._elo(team_a)
        b_elo = self._elo(team_b)
        a_counter_threat = current_counter_threat(
            a["counter_summary"],
            current_underdog_fit(a_elo, b_elo),
        )
        b_counter_threat = current_counter_threat(
            b["counter_summary"],
            current_underdog_fit(b_elo, a_elo),
        )
        a_confederation = _team_confederation(team_a)
        b_confederation = _team_confederation(team_b)
        a_confederation_strength = self._strength(self.confederation_stats.get(a_confederation))
        b_confederation_strength = self._strength(self.confederation_stats.get(b_confederation))
        a_cross_stats = self.team_cross_confederation_stats.get(_normalise_name(team_a), {})
        b_cross_stats = self.team_cross_confederation_stats.get(_normalise_name(team_b), {})
        a_cross_strength = self._strength(a_cross_stats)
        b_cross_strength = self._strength(b_cross_stats)
        h2h = self._head_to_head_features(team_a, team_b, match_date)
        group_context = self._group_table_context(team_a, team_b, group_table)
        row: dict[str, Any] = {
            "competition_family": "national_world_cup",
            "stage_or_round": stage_context["stage_or_round"],
            "is_friendly": 0,
            "is_qualifier": 0,
            "team_a_confederation": a_confederation,
            "team_b_confederation": b_confederation,
            "confederation_matchup": f"{a_confederation}_vs_{b_confederation}",
            "confederation_matchup_unordered": "_vs_".join(
                sorted([a_confederation, b_confederation])
            ),
            "same_confederation": int(a_confederation == b_confederation),
            "competition_train_matches": self.major_match_count,
            "competition_goals_avg": self.major_goal_avg,
            "phase_train_matches": stage_context["phase_train_matches"],
            "phase_goals_avg": stage_context["phase_goals_avg"],
            "host_home_advantage_diff": int(stage == "GROUP_STAGE" and _is_2026_host(team_a)),
            **group_context,
            "team_a_train_matches": a["matches"],
            "team_a_goals_for_avg": a["goals_for_avg"],
            "team_a_goals_against_avg": a["goals_against_avg"],
            "team_a_confederation_strength": a_confederation_strength,
            "team_a_cross_confederation_matches": a_cross_stats.get("matches", 0),
            "team_a_cross_confederation_strength": a_cross_strength,
            "team_b_train_matches": b["matches"],
            "team_b_goals_for_avg": b["goals_for_avg"],
            "team_b_goals_against_avg": b["goals_against_avg"],
            "teams_goals_for_sum": float(a["goals_for_avg"]) + float(b["goals_for_avg"]),
            "teams_goals_against_sum": float(a["goals_against_avg"]) + float(b["goals_against_avg"]),
            "team_a_attack_vs_b_defense_avg": (
                float(a["goals_for_avg"]) + float(b["goals_against_avg"])
            )
            / 2,
            "team_b_attack_vs_a_defense_avg": (
                float(b["goals_for_avg"]) + float(a["goals_against_avg"])
            )
            / 2,
            "match_attack_defense_volume": (
                float(a["goals_for_avg"])
                + float(b["goals_for_avg"])
                + float(a["goals_against_avg"])
                + float(b["goals_against_avg"])
            )
            / 2,
            "team_b_confederation_strength": b_confederation_strength,
            "team_b_cross_confederation_matches": b_cross_stats.get("matches", 0),
            "team_b_cross_confederation_strength": b_cross_strength,
            "team_a_recent6_matches": min(int(a["matches"]), RECENT_FORM_WINDOW),
            "team_a_recent6_goals_for_avg": a["recent6_goals_for_avg"],
            "team_a_recent6_goals_against_avg": a["recent6_goals_against_avg"],
            "team_a_recent6_points_avg": a["recent6_points_avg"],
            "team_a_recent6_win_rate": a["recent6_win_rate"],
            "team_b_recent6_matches": min(int(b["matches"]), RECENT_FORM_WINDOW),
            "team_b_recent6_goals_for_avg": b["recent6_goals_for_avg"],
            "team_b_recent6_goals_against_avg": b["recent6_goals_against_avg"],
            "recent6_goals_for_sum": float(a["recent6_goals_for_avg"])
            + float(b["recent6_goals_for_avg"]),
            "recent6_goals_against_sum": float(a["recent6_goals_against_avg"])
            + float(b["recent6_goals_against_avg"]),
            "recent6_team_a_attack_vs_b_defense_avg": (
                float(a["recent6_goals_for_avg"]) + float(b["recent6_goals_against_avg"])
            )
            / 2,
            "recent6_team_b_attack_vs_a_defense_avg": (
                float(b["recent6_goals_for_avg"]) + float(a["recent6_goals_against_avg"])
            )
            / 2,
            "recent6_match_attack_defense_volume": (
                float(a["recent6_goals_for_avg"])
                + float(b["recent6_goals_for_avg"])
                + float(a["recent6_goals_against_avg"])
                + float(b["recent6_goals_against_avg"])
            )
            / 2,
            "tempo_index": math.log1p(
                max(
                    0.0,
                    0.5
                    * (
                        float(a["goals_for_avg"])
                        + float(b["goals_for_avg"])
                        + float(a["goals_against_avg"])
                        + float(b["goals_against_avg"])
                    )
                    / 2
                    + 0.5
                    * (
                        float(a["recent6_goals_for_avg"])
                        + float(b["recent6_goals_for_avg"])
                        + float(a["recent6_goals_against_avg"])
                        + float(b["recent6_goals_against_avg"])
                    )
                    / 2,
                )
            ),
            "team_b_recent6_points_avg": b["recent6_points_avg"],
            "team_b_recent6_win_rate": b["recent6_win_rate"],
            "recent6_quality_result_points_diff": float(a["recent6_quality_result_points_avg"])
            - float(b["recent6_quality_result_points_avg"]),
            "recent6_quality_goal_balance_diff": float(a["recent6_quality_goal_balance_avg"])
            - float(b["recent6_quality_goal_balance_avg"]),
            "team_a_rest_days": a["rest_days"] or 0,
            "team_b_rest_days": b["rest_days"] or 0,
            "team_a_late85_points_swing": a["recent6_late85_points_swing"],
            "team_b_late85_points_swing": b["recent6_late85_points_swing"],
            "team_a_score_state_value": a["recent6_score_state_value"],
            "team_b_score_state_value": b["recent6_score_state_value"],
            "team_a_score_control_value": a["recent6_score_control_value"],
            "team_b_score_control_value": b["recent6_score_control_value"],
            "team_a_scoring_quickness": a["recent6_scoring_quickness"],
            "team_b_scoring_quickness": b["recent6_scoring_quickness"],
            "team_a_score_control_quality": a["recent6_score_control_quality"],
            "team_b_score_control_quality": b["recent6_score_control_quality"],
            "team_a_narrow_lead_hold": a["recent6_narrow_lead_hold"],
            "team_b_narrow_lead_hold": b["recent6_narrow_lead_hold"],
            "team_a_comfortable_lead": a["recent6_comfortable_lead"],
            "team_b_comfortable_lead": b["recent6_comfortable_lead"],
            "team_a_game_state_friction": a["recent6_game_state_friction"],
            "team_b_game_state_friction": b["recent6_game_state_friction"],
            "team_a_state_change_swing": a["recent6_state_change_swing"],
            "team_b_state_change_swing": b["recent6_state_change_swing"],
            "team_a_early_state_change_swing": a["recent6_early_state_change_swing"],
            "team_b_early_state_change_swing": b["recent6_early_state_change_swing"],
            "team_a_quality_score_control_value": a[
                "recent6_quality_score_control_value"
            ],
            "team_b_quality_score_control_value": b[
                "recent6_quality_score_control_value"
            ],
            "team_a_quality_state_change_swing": a["recent6_quality_state_change_swing"],
            "team_b_quality_state_change_swing": b["recent6_quality_state_change_swing"],
            "team_a_quality_early_state_change_swing": a[
                "recent6_quality_early_state_change_swing"
            ],
            "team_b_quality_early_state_change_swing": b[
                "recent6_quality_early_state_change_swing"
            ],
            "team_a_score_timing_coverage": a[
                "recent6_score_timing_coverage"
            ],
            "team_b_score_timing_coverage": b[
                "recent6_score_timing_coverage"
            ],
            "team_a_clinical_finishing": a["clinical_summary"][
                "clinical_signal"
            ],
            "team_b_clinical_finishing": b["clinical_summary"][
                "clinical_signal"
            ],
            "team_a_clinical_coverage": a["clinical_summary"][
                "clinical_coverage"
            ],
            "team_b_clinical_coverage": b["clinical_summary"][
                "clinical_coverage"
            ],
            "team_a_low_block_profile_aligned": a["clinical_summary"][
                "low_block_profile"
            ],
            "team_b_low_block_profile_aligned": b["clinical_summary"][
                "low_block_profile"
            ],
            "team_a_low_block_coverage_aligned": a["clinical_summary"][
                "low_block_coverage"
            ],
            "team_b_low_block_coverage_aligned": b["clinical_summary"][
                "low_block_coverage"
            ],
            "team_a_club_attack_talent_signal": a["club_attack_talent_signal"],
            "team_b_club_attack_talent_signal": b["club_attack_talent_signal"],
            "team_a_club_star_finisher_signal": a["club_star_finisher_signal"],
            "team_b_club_star_finisher_signal": b["club_star_finisher_signal"],
            "team_a_club_attack_coverage": a["club_attack_coverage"],
            "team_b_club_attack_coverage": b["club_attack_coverage"],
            "club_attack_talent_edge": float(a["club_attack_talent_signal"])
            - float(b["club_attack_talent_signal"]),
            "club_star_finisher_edge": float(a["club_star_finisher_signal"])
            - float(b["club_star_finisher_signal"]),
            "club_talent_coverage_pair": min(
                float(a["club_attack_coverage"]),
                float(b["club_attack_coverage"]),
            ),
            "goals_for_diff": float(a["goals_for_avg"]) - float(b["goals_for_avg"]),
            "goals_against_diff": float(a["goals_against_avg"]) - float(b["goals_against_avg"]),
            "confederation_strength_diff": a_confederation_strength - b_confederation_strength,
            "cross_confederation_strength_diff": a_cross_strength - b_cross_strength,
            "cross_confederation_matches_diff": a_cross_stats.get("matches", 0)
            - b_cross_stats.get("matches", 0),
            "recent6_goals_for_diff": float(a["recent6_goals_for_avg"]) - float(b["recent6_goals_for_avg"]),
            "recent6_goals_against_diff": float(a["recent6_goals_against_avg"])
            - float(b["recent6_goals_against_avg"]),
            "recent6_points_diff": float(a["recent6_points_avg"]) - float(b["recent6_points_avg"]),
            "recent6_win_rate_diff": float(a["recent6_win_rate"]) - float(b["recent6_win_rate"]),
            "recent6_goal_diff_diff": float(a["recent6_goal_diff_avg"]) - float(b["recent6_goal_diff_avg"]),
            "recent6_opponent_elo_diff": float(a["recent6_opponent_elo_avg"])
            - float(b["recent6_opponent_elo_avg"]),
            "rest_days_diff": float(a["rest_days"] or 0) - float(b["rest_days"] or 0),
            "team_a_elo_pre": a_elo,
            "team_b_elo_pre": b_elo,
            "elo_diff": a_elo - b_elo,
            "team_a_counter_current_threat": a_counter_threat,
            "team_b_counter_current_threat": b_counter_threat,
            "counter_current_threat_edge": a_counter_threat - b_counter_threat,
            "team_a_recent6_opponent_elo_avg": a["recent6_opponent_elo_avg"],
            "team_b_recent6_opponent_elo_avg": b["recent6_opponent_elo_avg"],
            "team_a_worldcup_recent6_win_rate": a["worldcup_recent6_win_rate"],
            "team_b_worldcup_recent6_win_rate": b["worldcup_recent6_win_rate"],
            "worldcup_recent6_win_rate_diff": float(a["worldcup_recent6_win_rate"])
            - float(b["worldcup_recent6_win_rate"]),
            "worldcup_memory_edge": 1.2
            * (float(a["worldcup_recent6_win_rate"]) - float(b["worldcup_recent6_win_rate"])),
            "team_a_fifa_rank": fifa_a["fifa_rank"],
            "team_b_fifa_rank": fifa_b["fifa_rank"],
            "team_a_fifa_points": fifa_a["fifa_points"],
            "team_b_fifa_points": fifa_b["fifa_points"],
            "fifa_points_diff": fifa_a["fifa_points"] - fifa_b["fifa_points"],
            "team_a_historical_fifa_rank": fifa_a["fifa_rank"],
            "team_b_historical_fifa_rank": fifa_b["fifa_rank"],
            "team_a_historical_fifa_observed": fifa_a["observed"],
            "team_b_historical_fifa_observed": fifa_b["observed"],
            "historical_fifa_points_diff": fifa_a["official_fifa_points"]
            - fifa_b["official_fifa_points"],
            "team_a_live_fifa_points": fifa_a["fifa_points"],
            "team_b_live_fifa_points": fifa_b["fifa_points"],
            "live_fifa_points_diff": fifa_a["fifa_points"] - fifa_b["fifa_points"],
            "h2h_recent_2y_matches": h2h["matches"],
            "h2h_recent_2y_days_since_last": h2h["days_since_last"],
            "h2h_recent_2y_team_a_goals_avg": h2h["team_a_goals_avg"],
            "h2h_recent_2y_team_b_goals_avg": h2h["team_b_goals_avg"],
            "h2h_recent_2y_goal_diff_avg": h2h["goal_diff_avg"],
            "h2h_recent_2y_team_a_points_avg": h2h["team_a_points_avg"],
            "h2h_recent_2y_team_b_points_avg": h2h["team_b_points_avg"],
            "h2h_recent_2y_points_diff": h2h["points_diff"],
            "h2h_recent_2y_draw_rate": h2h["draw_rate"],
            "h2h_recent_2y_penalty_shootout_matches": h2h["penalty_shootout_matches"],
            "h2h_recent_2y_team_a_penalty_wins": h2h["team_a_penalty_wins"],
            "h2h_recent_2y_team_b_penalty_wins": h2h["team_b_penalty_wins"],
            "h2h_recent_2y_penalty_wins_diff": h2h["penalty_wins_diff"],
        }
        a_wc_flags = self._worldcup_last6_flags(team_a, match_date)
        b_wc_flags = self._worldcup_last6_flags(team_b, match_date)
        for index in range(1, 7):
            for outcome in ("win", "draw", "loss"):
                row[f"team_a_worldcup_last6_{index}_{outcome}"] = a_wc_flags[
                    f"worldcup_last6_{index}_{outcome}"
                ]
                row[f"team_b_worldcup_last6_{index}_{outcome}"] = b_wc_flags[
                    f"worldcup_last6_{index}_{outcome}"
                ]
        for feature in DETAIL_STAT_FEATURES:
            row[f"team_a_recent6_{feature}"] = a["detail_stats"][feature]
            row[f"team_b_recent6_{feature}"] = b["detail_stats"][feature]
            row[f"recent6_{feature}_diff"] = (
                a["detail_stats"][feature] - b["detail_stats"][feature]
            )
        for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
            row[f"team_a_worldcup_recent6_{feature}"] = a["worldcup_detail_stats"][feature]
            row[f"team_b_worldcup_recent6_{feature}"] = b["worldcup_detail_stats"][feature]
            row[f"team_a_current_worldcup_recent6_{feature}"] = a[
                "current_worldcup_detail_stats"
            ][feature]
            row[f"team_b_current_worldcup_recent6_{feature}"] = b[
                "current_worldcup_detail_stats"
            ][feature]
        row["team_a_tactical_detail_coverage"] = a["detail_coverage"]
        row["team_b_tactical_detail_coverage"] = b["detail_coverage"]
        return row

    def _cache_key(self, team_a: str, team_b: str, match_date: date, stage: str) -> tuple[str, str, str, str]:
        return (_normalise_name(team_a), _normalise_name(team_b), match_date.isoformat(), stage)

    def _lightgbm_feature_names(self, model_key: str = "result") -> list[str]:
        if self.lightgbm_model is None:
            return list(NEUTRAL_FEATURES)
        legacy_features = list(self.lightgbm_model.get("features", NEUTRAL_FEATURES))
        feature_fields = {
            "team_a_goals": "team_a_goal_features",
            "team_b_goals": "team_b_goal_features",
            "result": "result_features",
            "xg_result": "xg_result_features",
        }
        return list(self.lightgbm_model.get(feature_fields.get(model_key, "features"), legacy_features))

    def _lightgbm_frame(
        self,
        rows: list[dict[str, Any]],
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        features = features if features is not None else self._lightgbm_feature_names()
        frame = add_neutral_treated_features(pd.DataFrame(rows))
        for feature in features:
            if feature not in frame:
                frame[feature] = 0
        frame = frame[features]
        categorical = [feature for feature in CATEGORICAL_FEATURES if feature in frame]
        for column in categorical:
            frame[column] = frame[column].fillna("unknown").astype("category")
        for column in frame.columns.difference(categorical):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return frame

    def _lightgbm_prediction_frames(self, rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
        feature_sets = {
            "team_a_goals": self._lightgbm_feature_names("team_a_goals"),
            "team_b_goals": self._lightgbm_feature_names("team_b_goals"),
            "result": self._lightgbm_feature_names("result"),
            "xg_result": self._lightgbm_feature_names("xg_result"),
        }
        all_features = list(dict.fromkeys(feature for values in feature_sets.values() for feature in values))
        frame = self._lightgbm_frame(rows, all_features)
        return {name: frame[features] for name, features in feature_sets.items()}

    def _lightgbm_result_probabilities(self, frames: dict[str, pd.DataFrame]) -> Any:
        probabilities = self.lightgbm_model["result_model"].predict_proba(frames["result"])
        xg_model = self.lightgbm_model.get("xg_result_model")
        if xg_model is None:
            return probabilities
        return blend_result_probabilities(
            probabilities,
            xg_model.predict_proba(frames["xg_result"]),
            weight=float(self.lightgbm_model.get("result_probability_blend_weight", 0.50)),
        )

    def _precompute_lightgbm_cache(self) -> None:
        teams = sorted({team for group in self.groups.values() for team in group})
        requests: list[tuple[tuple[str, str, str, str], dict[str, Any], str, str, date]] = []
        knockout_stages = [
            ("ROUND_OF_32", date(2026, 6, 28)),
            ("LAST_16", date(2026, 7, 4)),
            ("QUARTER_FINALS", date(2026, 7, 9)),
            ("SEMI_FINALS", date(2026, 7, 14)),
            ("FINAL", date(2026, 7, 19)),
        ]
        for match in self.group_matches:
            team_a = match["homeTeam"]["name"]
            team_b = match["awayTeam"]["name"]
            match_date = date.fromisoformat(match["utcDate"][:10])
            key = self._cache_key(team_a, team_b, match_date, "GROUP_STAGE")
            requests.append(
                (
                    key,
                    self._lightgbm_features(team_a, team_b, match_date, "GROUP_STAGE"),
                    team_a,
                    team_b,
                    match_date,
                )
            )
        for stage, match_date in knockout_stages:
            for team_a in teams:
                for team_b in teams:
                    if team_a == team_b:
                        continue
                    key = self._cache_key(team_a, team_b, match_date, stage)
                    requests.append(
                        (
                            key,
                            self._lightgbm_features(team_a, team_b, match_date, stage),
                            team_a,
                            team_b,
                            match_date,
                        )
                    )

        frames = self._lightgbm_prediction_frames([row for _, row, _, _, _ in requests])

        team_a_goals = self.lightgbm_model["team_a_goals_model"].predict(frames["team_a_goals"])
        team_b_goals = self.lightgbm_model["team_b_goals_model"].predict(frames["team_b_goals"])
        probabilities = self._lightgbm_result_probabilities(frames)
        for index, (key, _, team_a, team_b, match_date) in enumerate(requests):
            self.prediction_cache[key] = {
                "team_a_goals": max(0.15, float(team_a_goals[index])),
                "team_b_goals": max(0.15, float(team_b_goals[index])),
                "probabilities": probabilities[index].copy(),
            }

    def lightgbm_prediction(
        self,
        team_a: str,
        team_b: str,
        match_date: date,
        stage: str,
        group_table: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(team_a, team_b, match_date, stage)
        use_cache = group_table is None
        cached = self.prediction_cache.get(cache_key) if use_cache else None
        if cached is not None:
            return cached
        if self.lightgbm_model is None:
            a_lambda, b_lambda = self.expected_goals(team_a, team_b, match_date, stage)
            prediction = {
                "team_a_goals": a_lambda,
                "team_b_goals": b_lambda,
                "probabilities": None,
            }
            if use_cache:
                self.prediction_cache[cache_key] = prediction
            return prediction
        row = self._lightgbm_features(team_a, team_b, match_date, stage, group_table)
        frames = self._lightgbm_prediction_frames([row])
        team_a_goals = max(
            0.15,
            float(self.lightgbm_model["team_a_goals_model"].predict(frames["team_a_goals"])[0]),
        )
        team_b_goals = max(
            0.15,
            float(self.lightgbm_model["team_b_goals_model"].predict(frames["team_b_goals"])[0]),
        )
        probabilities = self._lightgbm_result_probabilities(frames)[0]
        prediction = {
            "team_a_goals": team_a_goals,
            "team_b_goals": team_b_goals,
            "probabilities": probabilities.copy(),
        }
        if use_cache:
            self.prediction_cache[cache_key] = prediction
        return prediction

    def expected_goals(
        self,
        team_a: str,
        team_b: str,
        match_date: date,
        stage: str,
        group_table: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[float, float]:
        a = self._team_features(team_a, match_date)
        b = self._team_features(team_b, match_date)
        stage_text = stage.lower()
        phase_avg = self.knockout_goal_avg if stage_text != "group_stage" else self.group_goal_avg
        form_a = 1 + 0.08 * (float(a["recent6_win_rate"]) - float(b["recent6_win_rate"]))
        form_b = 1 + 0.08 * (float(b["recent6_win_rate"]) - float(a["recent6_win_rate"]))
        wc_form_a = 1 + 0.10 * (
            float(a["worldcup_recent6_win_rate"]) - float(b["worldcup_recent6_win_rate"])
        )
        wc_form_b = 1 + 0.10 * (
            float(b["worldcup_recent6_win_rate"]) - float(a["worldcup_recent6_win_rate"])
        )
        fifa_a = self._fifa_ranking_features(team_a)
        fifa_b = self._fifa_ranking_features(team_b)
        fifa_points_diff = (fifa_a["fifa_points"] - fifa_b["fifa_points"]) / 400
        fifa_rank_diff = (fifa_b["fifa_rank"] - fifa_a["fifa_rank"]) / 100
        fifa_strength_a = max(0.85, min(1.15, 1 + 0.08 * fifa_points_diff + 0.04 * fifa_rank_diff))
        fifa_strength_b = max(0.85, min(1.15, 1 - 0.08 * fifa_points_diff - 0.04 * fifa_rank_diff))
        elo_diff = (self._elo(team_a) - self._elo(team_b)) / 400
        elo_strength_a = max(0.85, min(1.15, 1 + 0.10 * elo_diff))
        elo_strength_b = max(0.85, min(1.15, 1 - 0.10 * elo_diff))
        group_context = self._group_table_context(team_a, team_b, group_table)
        points_diff = float(group_context["group_points_diff_pre"])
        position_diff = float(group_context["group_position_diff_pre"])
        incentive_a = max(0.92, min(1.08, 1 - 0.025 * points_diff - 0.015 * position_diff))
        incentive_b = max(0.92, min(1.08, 1 + 0.025 * points_diff + 0.015 * position_diff))
        a_lambda = max(
            0.15,
            (
                0.35 * self.global_goal_avg
            + 0.20 * self.major_goal_avg
            + 0.15 * phase_avg
            + 0.15 * float(a["recent6_goals_for_avg"])
                + 0.15 * float(b["recent6_goals_against_avg"])
            )
            * form_a
            * wc_form_a
            * fifa_strength_a
            * elo_strength_a
            * incentive_a,
        )
        b_lambda = max(
            0.15,
            (
                0.35 * self.global_goal_avg
            + 0.20 * self.major_goal_avg
            + 0.15 * phase_avg
            + 0.15 * float(b["recent6_goals_for_avg"])
                + 0.15 * float(a["recent6_goals_against_avg"])
            )
            * form_b
            * wc_form_b
            * fifa_strength_b
            * elo_strength_b
            * incentive_b,
        )
        return a_lambda, b_lambda

    def _sample_result(self, probabilities: Any, stage: str) -> str:
        if probabilities is None:
            return "draw"
        team_a_probability = float(probabilities[0])
        draw_probability = float(probabilities[1])
        pick = self.rng.random()
        if pick < team_a_probability:
            return "team_a"
        if pick < team_a_probability + draw_probability:
            return "draw"
        return "team_b"

    def _align_score_to_result(self, a_goals: int, b_goals: int, result: str) -> tuple[int, int]:
        if result == "team_a" and a_goals <= b_goals:
            return b_goals + 1, b_goals
        if result == "team_b" and b_goals <= a_goals:
            return a_goals, a_goals + 1
        if result == "draw" and a_goals != b_goals:
            goals = min(a_goals, b_goals)
            return goals, goals
        return a_goals, b_goals

    def _record_simulated_match(
        self,
        team_a: str,
        team_b: str,
        match_date: date,
        a_goals: int,
        b_goals: int,
    ) -> None:
        a_key = _normalise_name(team_a)
        b_key = _normalise_name(team_b)
        self.simulated_histories[a_key].append(
            {
                "date": match_date,
                "gf": a_goals,
                "ga": b_goals,
                "points": _points(a_goals, b_goals),
                "opponent_elo": self._elo(team_b),
                "quality_result_points": _quality_result_points(
                    _points(a_goals, b_goals),
                    a_goals - b_goals,
                    self._elo(team_b),
                ),
                "quality_goal_balance": _quality_goal_balance(
                    a_goals,
                    b_goals,
                    self._elo(team_b),
                ),
                "late85_points_swing": 0.0,
                "score_timing": {},
                "counter_item": {
                    "goals_for": a_goals,
                    "points": _points(a_goals, b_goals),
                    "team_elo": self._elo(team_a),
                    "opponent_elo": self._elo(team_b),
                    "ball_possession_pct": None,
                    "total_passes": None,
                    "passes_pct": None,
                    "total_shots": None,
                    "shots_on_goal": None,
                },
                "detail_stats": {feature: None for feature in DETAIL_STAT_FEATURES},
            }
        )
        self.simulated_histories[b_key].append(
            {
                "date": match_date,
                "gf": b_goals,
                "ga": a_goals,
                "points": _points(b_goals, a_goals),
                "opponent_elo": self._elo(team_a),
                "quality_result_points": _quality_result_points(
                    _points(b_goals, a_goals),
                    b_goals - a_goals,
                    self._elo(team_a),
                ),
                "quality_goal_balance": _quality_goal_balance(
                    b_goals,
                    a_goals,
                    self._elo(team_a),
                ),
                "late85_points_swing": 0.0,
                "score_timing": {},
                "counter_item": {
                    "goals_for": b_goals,
                    "points": _points(b_goals, a_goals),
                    "team_elo": self._elo(team_b),
                    "opponent_elo": self._elo(team_a),
                    "ball_possession_pct": None,
                    "total_passes": None,
                    "passes_pct": None,
                    "total_shots": None,
                    "shots_on_goal": None,
                },
                "detail_stats": {feature: None for feature in DETAIL_STAT_FEATURES},
            }
        )
        a_fifa_key = normalise_team_name(team_a)
        b_fifa_key = normalise_team_name(team_b)
        a_points = self.fifa_point_overrides.get(
            a_fifa_key,
            self.fallback_fifa_points,
        )
        b_points = self.fifa_point_overrides.get(
            b_fifa_key,
            self.fallback_fifa_points,
        )
        updated_a, updated_b = update_live_fifa_points(
            a_points,
            b_points,
            a_goals,
            b_goals,
        )
        self.fifa_point_overrides[a_fifa_key] = updated_a
        self.fifa_point_overrides[b_fifa_key] = updated_b
        # Rating-dependent predictions computed before this result are stale.
        self.prediction_cache.clear()

    def simulate_match(
        self,
        team_a: str,
        team_b: str,
        match_date: date,
        stage: str,
        match_id: int | str | None = None,
        group: str | None = None,
        group_table: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[int, int, str | None]:
        manual_result = self.manual_results.get(str(match_id)) if match_id is not None else None
        if manual_result:
            a_goals = int(manual_result["team_a_goals"])
            b_goals = int(manual_result["team_b_goals"])
            winner = manual_result["winner"]
            decided_by = "manual_result"
            penalty_winner = manual_result.get("penalty_winner")
            extra_time_winner = manual_result.get("extra_time_winner")
            penalty_probability_a = None
            if not winner:
                winner = team_a if a_goals > b_goals else team_b if b_goals > a_goals else None
            if (
                winner == "Draw"
                and stage != "GROUP_STAGE"
                and penalty_winner in {team_a, team_b}
            ):
                winner = penalty_winner
                decided_by = "manual_penalties"
                penalty_probability_a = self.penalty_model.team_a_probability(team_a, team_b, match_date)
            elif (
                winner == "Draw"
                and stage != "GROUP_STAGE"
                and extra_time_winner in {team_a, team_b}
            ):
                winner = extra_time_winner
                decided_by = "manual_extra_time"
            self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
            self._record_match_trace(
                match_id,
                stage,
                group,
                team_a,
                team_b,
                a_goals,
                b_goals,
                winner,
                decided_by=decided_by,
                penalty_winner=penalty_winner if decided_by == "manual_penalties" else None,
                penalty_team_a_probability=penalty_probability_a,
            )
            return a_goals, b_goals, winner
        if self.engine == "lightgbm" and self.lightgbm_model is not None:
            prediction = self.lightgbm_prediction(team_a, team_b, match_date, stage, group_table)
            a_lambda = prediction["team_a_goals"]
            b_lambda = prediction["team_b_goals"]
            sampled_result = self._sample_result(prediction["probabilities"], stage)
        else:
            a_lambda, b_lambda = self.expected_goals(team_a, team_b, match_date, stage, group_table)
            sampled_result = ""
        a_goals = _poisson_sample(a_lambda, self.rng)
        b_goals = _poisson_sample(b_lambda, self.rng)
        if sampled_result:
            a_goals, b_goals = self._align_score_to_result(a_goals, b_goals, sampled_result)
            winner = team_a if sampled_result == "team_a" else team_b if sampled_result == "team_b" else None
            if sampled_result == "draw" and stage != "GROUP_STAGE":
                penalty_probability_a = self.penalty_model.team_a_probability(team_a, team_b, match_date)
                winner = team_a if self.rng.random() < penalty_probability_a else team_b
                self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
                self._record_match_trace(
                    match_id,
                    stage,
                    group,
                    team_a,
                    team_b,
                    a_goals,
                    b_goals,
                    winner,
                    decided_by="penalties",
                    penalty_winner=winner,
                    penalty_team_a_probability=penalty_probability_a,
                )
                return a_goals, b_goals, winner
            self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
            self._record_match_trace(match_id, stage, group, team_a, team_b, a_goals, b_goals, winner)
            return a_goals, b_goals, winner
        if a_goals > b_goals:
            winner = team_a
            self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
            self._record_match_trace(match_id, stage, group, team_a, team_b, a_goals, b_goals, winner)
            return a_goals, b_goals, winner
        if b_goals > a_goals:
            winner = team_b
            self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
            self._record_match_trace(match_id, stage, group, team_a, team_b, a_goals, b_goals, winner)
            return a_goals, b_goals, winner
        if stage != "GROUP_STAGE":
            penalty_probability_a = self.penalty_model.team_a_probability(team_a, team_b, match_date)
            winner = team_a if self.rng.random() < penalty_probability_a else team_b
            self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
            self._record_match_trace(
                match_id,
                stage,
                group,
                team_a,
                team_b,
                a_goals,
                b_goals,
                winner,
                decided_by="penalties",
                penalty_winner=winner,
                penalty_team_a_probability=penalty_probability_a,
            )
            return a_goals, b_goals, winner
        self._record_simulated_match(team_a, team_b, match_date, a_goals, b_goals)
        self._record_match_trace(match_id, stage, group, team_a, team_b, a_goals, b_goals, None)
        return a_goals, b_goals, None

    def _record_match_trace(
        self,
        match_id: int | str | None,
        stage: str,
        group: str | None,
        team_a: str,
        team_b: str,
        a_goals: int,
        b_goals: int,
        winner: str | None,
        *,
        decided_by: str = "regular",
        penalty_winner: str | None = None,
        penalty_team_a_probability: float | None = None,
    ) -> None:
        self.current_match_records.append(
            {
                "match_id": match_id or "",
                "stage": stage,
                "group": group or "",
                "team_a": team_a,
                "team_b": team_b,
                "team_a_goals": a_goals,
                "team_b_goals": b_goals,
                "total_goals": a_goals + b_goals,
                "winner": winner or "draw",
                "decided_by": decided_by,
                "penalty_winner": penalty_winner or "",
                "penalty_team_a_probability": penalty_team_a_probability,
            }
        )

    def _rank_group(self, table: dict[str, TeamStanding]) -> list[TeamStanding]:
        return sorted(
            table.values(),
            key=lambda item: (
                item.points,
                item.goal_difference,
                item.goals_for,
                self.rng.random(),
            ),
            reverse=True,
        )

    def simulate_groups(self) -> dict[str, list[TeamStanding]]:
        standings: dict[str, list[TeamStanding]] = {}
        for group, teams in self.groups.items():
            mutable = {
                team: {"team": team, "group": group, "points": 0, "gf": 0, "ga": 0}
                for team in teams
            }
            for match in [item for item in self.group_matches if item["group"] == group]:
                team_a = match["homeTeam"]["name"]
                team_b = match["awayTeam"]["name"]
                match_date = date.fromisoformat(match["utcDate"][:10])
                a_goals, b_goals, _ = self.simulate_match(
                    team_a,
                    team_b,
                    match_date,
                    "GROUP_STAGE",
                    match_id=match.get("id"),
                    group=group,
                    group_table=mutable,
                )
                mutable[team_a]["gf"] += a_goals
                mutable[team_a]["ga"] += b_goals
                mutable[team_a]["points"] += _points(a_goals, b_goals)
                mutable[team_b]["gf"] += b_goals
                mutable[team_b]["ga"] += a_goals
                mutable[team_b]["points"] += _points(b_goals, a_goals)
            standings[group] = self._rank_group(
                {
                    team: TeamStanding(
                        team=values["team"],
                        group=values["group"],
                        points=values["points"],
                        goals_for=values["gf"],
                        goals_against=values["ga"],
                    )
                    for team, values in mutable.items()
                }
            )
        return standings

    def _best_thirds(self, standings: dict[str, list[TeamStanding]]) -> list[TeamStanding]:
        thirds = [ranking[2] for ranking in standings.values()]
        return sorted(
            thirds,
            key=lambda item: (
                item.points,
                item.goal_difference,
                item.goals_for,
                self.rng.random(),
            ),
            reverse=True,
        )[:8]

    def _third_place_match_assignments(self, thirds: list[TeamStanding]) -> dict[int, str]:
        qualified_groups = "".join(
            sorted(standing.group.replace("GROUP_", "") for standing in thirds)
        )
        assignments = self.third_place_assignment_table.get(qualified_groups)
        if assignments is None:
            raise ValueError(
                "No third-place assignment found for qualified groups "
                f"{qualified_groups}"
            )
        return assignments

    def _resolve_slot(
        self,
        match_id: int,
        slot: tuple[str, str | tuple[str, ...]],
        standings: dict[str, list[TeamStanding]],
        thirds: list[TeamStanding],
        used_thirds: set[str],
        third_place_match_assignments: dict[int, str],
    ) -> str:
        kind, value = slot
        if kind == "1":
            return standings[str(value)][0].team
        if kind == "2":
            return standings[str(value)][1].team
        exact_group = third_place_match_assignments.get(match_id)
        if exact_group:
            group = f"GROUP_{exact_group}"
            allowed = set(value)
            if group not in allowed:
                raise ValueError(
                    f"Invalid third-place table assignment: match {match_id} "
                    f"got {group}, allowed {sorted(allowed)}"
                )
            return standings[group][2].team

        allowed = set(value)
        candidates = [
            standing
            for standing in thirds
            if standing.group in allowed and standing.group not in used_thirds
        ]
        if not candidates:
            candidates = [standing for standing in thirds if standing.group not in used_thirds]
        pick = candidates[0]
        used_thirds.add(pick.group)
        return pick.team

    def simulate_tournament(
        self,
    ) -> tuple[str, dict[str, str], dict[str, list[TeamStanding]], list[TeamStanding]]:
        self._reset_tournament_state()
        standings = self.simulate_groups()
        thirds = self._best_thirds(standings)
        third_place_match_assignments = self._third_place_match_assignments(thirds)
        winners: dict[int, str] = {}
        used_thirds: set[str] = set()

        for match_id, slot_a, slot_b in R32_SLOTS:
            team_a = self._resolve_slot(
                match_id,
                slot_a,
                standings,
                thirds,
                used_thirds,
                third_place_match_assignments,
            )
            team_b = self._resolve_slot(
                match_id,
                slot_b,
                standings,
                thirds,
                used_thirds,
                third_place_match_assignments,
            )
            _, _, winner = self.simulate_match(
                team_a,
                team_b,
                date(2026, 6, 28),
                "ROUND_OF_32",
                match_id=match_id,
            )
            winners[match_id] = winner or team_a

        for match_id, previous_a, previous_b in ROUND_OF_16:
            _, _, winner = self.simulate_match(
                winners[previous_a],
                winners[previous_b],
                date(2026, 7, 4),
                "LAST_16",
                match_id=match_id,
            )
            winners[match_id] = winner or winners[previous_a]

        for match_id, previous_a, previous_b in QUARTER_FINALS:
            _, _, winner = self.simulate_match(
                winners[previous_a],
                winners[previous_b],
                date(2026, 7, 9),
                "QUARTER_FINALS",
                match_id=match_id,
            )
            winners[match_id] = winner or winners[previous_a]

        semifinal_losers: list[str] = []
        for match_id, previous_a, previous_b in SEMI_FINALS:
            team_a = winners[previous_a]
            team_b = winners[previous_b]
            _, _, winner = self.simulate_match(
                team_a,
                team_b,
                date(2026, 7, 14),
                "SEMI_FINALS",
                match_id=match_id,
            )
            winners[match_id] = winner or winners[previous_a]
            semifinal_losers.append(team_b if winners[match_id] == team_a else team_a)

        _, _, third_place_winner = self.simulate_match(
            semifinal_losers[0],
            semifinal_losers[1],
            date(2026, 7, 18),
            "THIRD_PLACE",
            match_id=104,
        )
        winners[104] = third_place_winner or semifinal_losers[0]

        _, _, champion = self.simulate_match(
            winners[101],
            winners[102],
            date(2026, 7, 19),
            "FINAL",
            match_id=103,
        )
        return (
            champion or winners[101],
            {str(key): value for key, value in winners.items()},
            standings,
            thirds,
        )


def run_worldcup_simulation(
    data_root: Path,
    *,
    simulations: int = 5000,
    seed: int = 42,
    engine: str = "lightgbm",
) -> dict[str, Any]:
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine=engine)
    champions: Counter[str] = Counter()
    runners_up: Counter[str] = Counter()
    third_place: Counter[str] = Counter()
    finalists: Counter[str] = Counter()
    semifinalists: Counter[str] = Counter()
    quarterfinalists: Counter[str] = Counter()
    round32: Counter[str] = Counter()
    group_winners: Counter[str] = Counter()
    best_third_qualifiers: Counter[str] = Counter()
    group_position_counts: dict[str, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    slot_side_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    slot_team_counts: dict[str, Counter[str]] = defaultdict(Counter)
    slot_winner_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for _ in range(simulations):
        champion, winners, standings, best_thirds = simulator.simulate_tournament()
        champions[champion] += 1
        runners_up[winners["102"] if champion == winners["101"] else winners["101"]] += 1
        third_place[winners["104"]] += 1
        finalists[winners["101"]] += 1
        finalists[winners["102"]] += 1
        for match_id in ("97", "98", "99", "100"):
            semifinalists[winners[match_id]] += 1
        for match_id in ("89", "90", "91", "92", "93", "94", "95", "96"):
            quarterfinalists[winners[match_id]] += 1
        for match_id in range(73, 89):
            round32[winners[str(match_id)]] += 1
        for group_name, ranking in standings.items():
            group_winners[ranking[0].team] += 1
            for position, standing in enumerate(ranking, start=1):
                group_position_counts[group_name][standing.team][position] += 1
        for standing in best_thirds:
            best_third_qualifiers[standing.team] += 1

        for row in simulator.current_match_records:
            if row["stage"] == "GROUP_STAGE":
                continue
            match_id = str(row["match_id"])
            winner = row["winner"]
            slot_side_counts[match_id]["equipo_a"][row["team_a"]] += 1
            slot_side_counts[match_id]["equipo_b"][row["team_b"]] += 1
            slot_team_counts[match_id][row["team_a"]] += 1
            slot_team_counts[match_id][row["team_b"]] += 1
            slot_winner_counts[match_id][winner] += 1

    group_qualification_counts: Counter[str] = Counter()
    for teams_by_position in group_position_counts.values():
        for team, position_counts in teams_by_position.items():
            group_qualification_counts[team] = (
                position_counts[1] + position_counts[2] + best_third_qualifiers[team]
            )

    teams = (
        set(champions)
        | set(runners_up)
        | set(third_place)
        | set(finalists)
        | set(semifinalists)
        | set(quarterfinalists)
        | set(round32)
        | set(group_qualification_counts)
    )
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "simulacion_mundial_2026.xlsx"

    summary_rows = [
        {
            "equipo": team,
            "probabilidad_campeon": round(champions[team] / simulations, 4),
            "probabilidad_segundo_lugar": round(runners_up[team] / simulations, 4),
            "probabilidad_tercer_lugar": round(third_place[team] / simulations, 4),
            "probabilidad_final": round(finalists[team] / simulations, 4),
            "probabilidad_semifinal": round(semifinalists[team] / simulations, 4),
            "probabilidad_cuartos": round(quarterfinalists[team] / simulations, 4),
            "probabilidad_ganar_r32": round(round32[team] / simulations, 4),
            "probabilidad_clasificar_grupos": round(
                group_qualification_counts[team] / simulations,
                4,
            ),
            "probabilidad_ganar_grupo": round(group_winners[team] / simulations, 4),
        }
        for team in sorted(teams, key=lambda item: champions[item], reverse=True)
    ]
    group_rows = []
    for group_name, teams_by_position in sorted(group_position_counts.items()):
        for team, position_counts in sorted(
            teams_by_position.items(),
            key=lambda item: (-item[1][1], -item[1][2], item[0]),
        ):
            most_common_position, most_common_position_count = position_counts.most_common(1)[0]
            top2_count = position_counts[1] + position_counts[2]
            best_third_count = best_third_qualifiers[team]
            group_rows.append(
                {
                    "grupo": group_name.replace("GROUP_", ""),
                    "equipo": team,
                    "posicion_mas_comun": most_common_position,
                    "veces_posicion_mas_comun": most_common_position_count,
                    "prob_posicion_mas_comun": round(most_common_position_count / simulations, 4),
                    "prob_1ro": round(position_counts[1] / simulations, 4),
                    "prob_2do": round(position_counts[2] / simulations, 4),
                    "prob_3ro": round(position_counts[3] / simulations, 4),
                    "prob_4to": round(position_counts[4] / simulations, 4),
                    "prob_top2": round(top2_count / simulations, 4),
                    "prob_clasifica_como_mejor_3ro": round(best_third_count / simulations, 4),
                    "prob_clasifica_grupo": round((top2_count + best_third_count) / simulations, 4),
                }
            )
    best_third_rows = [
        {
            "equipo": team,
            "veces_clasifica_como_mejor_3ro": count,
            "prob_clasifica_como_mejor_3ro": round(count / simulations, 4),
        }
        for team, count in best_third_qualifiers.most_common()
    ]
    stage_labels = {
        "GROUP_STAGE": "Fase de grupos",
        "ROUND_OF_32": "Dieciseisavos",
        "LAST_16": "Octavos",
        "QUARTER_FINALS": "Cuartos",
        "SEMI_FINALS": "Semifinales",
        "THIRD_PLACE": "Tercer lugar",
        "FINAL": "Final",
    }

    def _pick_unused(counter: Counter[str], used: set[str]) -> tuple[str, int]:
        for team, count in counter.most_common():
            if team not in used:
                used.add(team)
                return team, count
        team, count = counter.most_common(1)[0]
        return team, count

    r32_assignments: dict[tuple[str, str], tuple[str, int]] = {}
    used_r32_teams: set[str] = set()
    r32_sides = []
    for match_id, _, _ in R32_SLOTS:
        for side in ("equipo_a", "equipo_b"):
            counter = slot_side_counts[str(match_id)][side]
            top_count = counter.most_common(1)[0][1]
            r32_sides.append((top_count, str(match_id), side, counter))
    for _, match_id, side, counter in sorted(r32_sides, reverse=True):
        r32_assignments[(match_id, side)] = _pick_unused(counter, used_r32_teams)

    match_dates = {
        "ROUND_OF_32": date(2026, 6, 28),
        "LAST_16": date(2026, 7, 4),
        "QUARTER_FINALS": date(2026, 7, 9),
        "SEMI_FINALS": date(2026, 7, 14),
        "THIRD_PLACE": date(2026, 7, 18),
        "FINAL": date(2026, 7, 19),
    }

    def _prediction_row(
        match_id: int,
        stage: str,
        team_a: str,
        team_b: str,
        support_a: int,
        support_b: int,
        *,
        group: str = "",
        match_date_override: date | None = None,
    ) -> tuple[dict[str, Any], str]:
        match_date = match_date_override or match_dates[stage]
        if simulator.engine == "lightgbm" and simulator.lightgbm_model is not None:
            prediction = simulator.lightgbm_prediction(team_a, team_b, match_date, stage)
            expected_a = float(prediction["team_a_goals"])
            expected_b = float(prediction["team_b_goals"])
            probabilities = prediction["probabilities"]
            probability_a = float(probabilities[0])
            probability_draw = float(probabilities[1])
            probability_b = float(probabilities[2])
        else:
            expected_a, expected_b = simulator.expected_goals(team_a, team_b, match_date, stage)
            total = expected_a + expected_b
            base_probability_a = expected_a / total if total else 0.5
            probability_draw = 0.0 if stage == "GROUP_STAGE" else 0.25
            probability_a = base_probability_a * (1 - probability_draw)
            probability_b = (1 - base_probability_a) * (1 - probability_draw)

        penalty_probability_a = (
            0.0
            if stage == "GROUP_STAGE"
            else simulator.penalty_model.team_a_probability(team_a, team_b, match_date)
        )
        probability_advance_a = probability_a + probability_draw * penalty_probability_a
        probability_advance_b = probability_b + probability_draw * (1 - penalty_probability_a)
        winner = team_a if probability_advance_a >= probability_advance_b else team_b
        result_label = "empate"
        if probability_a >= probability_draw and probability_a >= probability_b:
            result_label = team_a
        elif probability_b >= probability_draw and probability_b >= probability_a:
            result_label = team_b
        row = {
            "fase": stage_labels[stage],
            "partido_id": match_id,
            "grupo": group,
            "equipo_a": team_a,
            "equipo_b": team_b,
            "apariciones_a_10k": support_a,
            "apariciones_b_10k": support_b,
            "goles_esperados_a": round(expected_a, 2),
            "goles_esperados_b": round(expected_b, 2),
            "prob_gana_a_en_partido": round(probability_a, 4),
            "prob_empate": round(probability_draw, 4),
            "prob_gana_b_en_partido": round(probability_b, 4),
            "prob_a_gana_penales_si_empata": round(penalty_probability_a, 4),
            "prob_b_gana_penales_si_empata": round(1 - penalty_probability_a, 4)
            if stage != "GROUP_STAGE"
            else 0.0,
            "prob_avanza_a": round(probability_advance_a, 4),
            "prob_avanza_b": round(probability_advance_b, 4),
            "resultado_mas_probable": result_label,
            "avanza_predicho": winner,
        }
        return row, winner

    group_match_rows = []
    for match in sorted(simulator.group_matches, key=lambda item: item["utcDate"]):
        team_a = match["homeTeam"]["name"]
        team_b = match["awayTeam"]["name"]
        match_date = date.fromisoformat(match["utcDate"][:10])
        row, _ = _prediction_row(
            int(match.get("id") or len(group_match_rows) + 1),
            "GROUP_STAGE",
            team_a,
            team_b,
            simulations,
            simulations,
            group=str(match.get("group", "")).replace("GROUP_", ""),
            match_date_override=match_date,
        )
        group_match_rows.append(
            {
                "grupo": row["grupo"],
                "fecha": match_date.isoformat(),
                "partido_id": row["partido_id"],
                "equipo_a": row["equipo_a"],
                "equipo_b": row["equipo_b"],
                "goles_esperados_a": row["goles_esperados_a"],
                "goles_esperados_b": row["goles_esperados_b"],
                "prob_gana_a": row["prob_gana_a_en_partido"],
                "prob_empate": row["prob_empate"],
                "prob_gana_b": row["prob_gana_b_en_partido"],
                "resultado_mas_probable": row["resultado_mas_probable"],
            }
        )

    bracket_reevaluado_rows = []
    consensus_winners: dict[int, str] = {}
    for match_id, _, _ in R32_SLOTS:
        team_a, support_a = r32_assignments[(str(match_id), "equipo_a")]
        team_b, support_b = r32_assignments[(str(match_id), "equipo_b")]
        row, winner = _prediction_row(match_id, "ROUND_OF_32", team_a, team_b, support_a, support_b)
        bracket_reevaluado_rows.append(row)
        consensus_winners[match_id] = winner

    for match_id, previous_a, previous_b in ROUND_OF_16:
        team_a = consensus_winners[previous_a]
        team_b = consensus_winners[previous_b]
        row, winner = _prediction_row(
            match_id,
            "LAST_16",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        bracket_reevaluado_rows.append(row)
        consensus_winners[match_id] = winner

    for match_id, previous_a, previous_b in QUARTER_FINALS:
        team_a = consensus_winners[previous_a]
        team_b = consensus_winners[previous_b]
        row, winner = _prediction_row(
            match_id,
            "QUARTER_FINALS",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        bracket_reevaluado_rows.append(row)
        consensus_winners[match_id] = winner

    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = consensus_winners[previous_a]
        team_b = consensus_winners[previous_b]
        row, winner = _prediction_row(
            match_id,
            "SEMI_FINALS",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        bracket_reevaluado_rows.append(row)
        consensus_winners[match_id] = winner

    semifinal_losers = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = consensus_winners[previous_a]
        team_b = consensus_winners[previous_b]
        semifinal_losers.append(team_b if consensus_winners[match_id] == team_a else team_a)
    third_row, _third = _prediction_row(
        104,
        "THIRD_PLACE",
        semifinal_losers[0],
        semifinal_losers[1],
        slot_team_counts["104"][semifinal_losers[0]],
        slot_team_counts["104"][semifinal_losers[1]],
    )
    bracket_reevaluado_rows.append(third_row)

    final_row, _champion = _prediction_row(
        103,
        "FINAL",
        consensus_winners[101],
        consensus_winners[102],
        slot_team_counts["103"][consensus_winners[101]],
        slot_team_counts["103"][consensus_winners[102]],
    )
    bracket_reevaluado_rows.append(final_row)
    metadata_rows = [
        {
            "simulaciones": simulations,
            "motor": engine if simulator.lightgbm_model is not None or engine != "lightgbm" else "poisson_fallback",
            "asignacion_mejores_terceros": "exacta_tabla_495_combinaciones",
            "nota": "Incluye top 2 de cada grupo + 8 mejores terceros. Modelo team_a/team_b con ajuste de anfitrion local cuando aplica.",
        }
    ]

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="metadata", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="probabilidades", index=False)
        pd.DataFrame(group_rows).to_excel(writer, sheet_name="fase_grupos_consenso", index=False)
        pd.DataFrame(best_third_rows).to_excel(writer, sheet_name="mejores_terceros", index=False)
        pd.DataFrame(group_match_rows).to_excel(writer, sheet_name="partidos_grupo", index=False)
        pd.DataFrame(bracket_reevaluado_rows).to_excel(
            writer,
            sheet_name="bracket_reevaluado",
            index=False,
        )

    top = [
        {
            "team": team,
            "champion_probability": round(count / simulations, 4),
            "final_probability": round(finalists[team] / simulations, 4),
        }
        for team, count in champions.most_common(15)
    ]
    return {
        "simulations": simulations,
        "engine": engine if simulator.lightgbm_model is not None or engine != "lightgbm" else "poisson_fallback",
        "top": top,
        "workbook_path": str(workbook_path),
        "third_place_assignment": "exact_495_combination_table",
    }
