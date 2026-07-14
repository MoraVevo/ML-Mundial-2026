from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from kinela.fifa_ranking import normalise_team_name

PENALTY_MODEL_FEATURES = [
    "career_rate_edge",
    "recent5_rate_edge",
    "decayed_rate_edge",
    "major_rate_edge",
    "experience_edge",
    "recent_experience_edge",
    "recency_edge",
    "last_result_edge",
    "h2h_edge",
    "elo_edge",
]

PENALTY_TEAM_ALIASES = {
    "gambia": "the gambia",
    "saint kitts and nevis": "st kitts and nevis",
    "taiwan": "chinese taipei",
    "united states virgin islands": "us virgin islands",
}


def _team_key(value: str) -> str:
    key = normalise_team_name(str(value))
    return PENALTY_TEAM_ALIASES.get(key, key)


def filter_fifa_affiliated_data(
    shootouts: pd.DataFrame,
    results: pd.DataFrame,
    fifa_team_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove non-FIFA teams before building any historical feature state."""

    eligible_keys = {_team_key(value) for value in fifa_team_keys}
    shootout_mask = (
        shootouts["home_team"].map(_team_key).isin(eligible_keys)
        & shootouts["away_team"].map(_team_key).isin(eligible_keys)
        & shootouts["winner"].map(_team_key).isin(eligible_keys)
    )
    result_mask = (
        results["home_team"].map(_team_key).isin(eligible_keys)
        & results["away_team"].map(_team_key).isin(eligible_keys)
    )
    filtered_shootouts = shootouts.loc[shootout_mask].copy()
    filtered_results = results.loc[result_mask].copy()
    audit = {
        "shootouts_input": int(len(shootouts)),
        "shootouts_eligible": int(len(filtered_shootouts)),
        "shootouts_excluded": int((~shootout_mask).sum()),
        "results_input": int(len(results)),
        "results_eligible": int(len(filtered_results)),
        "results_excluded": int((~result_mask).sum()),
    }
    return filtered_shootouts, filtered_results, audit


def _is_major_tournament(value: str) -> bool:
    name = str(value).casefold()
    return any(
        marker in name
        for marker in (
            "world cup",
            "euro",
            "copa américa",
            "copa america",
            "african cup",
            "africa cup",
            "asian cup",
            "gold cup",
            "nations league",
        )
    ) and "qualification" not in name


def _smoothed_rate(records: list[dict[str, Any]], prior: float) -> float:
    wins = sum(float(item["won"]) for item in records)
    return (wins + prior) / (len(records) + 2.0 * prior)


def _team_summary(records: list[dict[str, Any]], before: date) -> dict[str, float]:
    eligible = [item for item in records if date.fromisoformat(str(item["date"])) < before]
    recent5 = eligible[-5:]
    major = [item for item in eligible if bool(item.get("major"))]
    recent_window = [
        item
        for item in eligible
        if (before - date.fromisoformat(str(item["date"]))).days <= 8 * 365
    ]
    weighted_net = 0.0
    weight_total = 0.0
    for item in eligible:
        age_days = max(0, (before - date.fromisoformat(str(item["date"]))).days)
        weight = math.exp(-math.log(2.0) * age_days / (8.0 * 365.25))
        weighted_net += weight * (float(item["won"]) - 0.5)
        weight_total += weight
    decayed_rate = 0.5 + weighted_net / (weight_total + 4.0)
    if eligible:
        last = eligible[-1]
        days_since = max(0, (before - date.fromisoformat(str(last["date"]))).days)
        recency = math.exp(-days_since / (5.0 * 365.25))
        last_result = (float(last["won"]) - 0.5) * 2.0 * recency
    else:
        recency = 0.0
        last_result = 0.0
    return {
        "career_rate": _smoothed_rate(eligible, 3.0),
        "recent5_rate": _smoothed_rate(recent5, 2.0),
        "decayed_rate": decayed_rate,
        "major_rate": _smoothed_rate(major, 2.5),
        "experience": math.log1p(len(eligible)),
        "recent_experience": math.log1p(len(recent_window)),
        "recency": recency,
        "last_result": last_result,
    }


def penalty_feature_values(
    team_a: str,
    team_b: str,
    before: date,
    *,
    team_records: dict[str, list[dict[str, Any]]],
    h2h_records: dict[str, list[dict[str, Any]]],
    elo_ratings: dict[str, float],
) -> dict[str, float]:
    a_key = _team_key(team_a)
    b_key = _team_key(team_b)
    a = _team_summary(team_records.get(a_key, []), before)
    b = _team_summary(team_records.get(b_key, []), before)
    pair_key = "|".join(sorted((a_key, b_key)))
    pair = [
        item
        for item in h2h_records.get(pair_key, [])
        if date.fromisoformat(str(item["date"])) < before
    ]
    a_pair_wins = sum(1 for item in pair if item["winner_key"] == a_key)
    h2h_a_rate = (a_pair_wins + 1.5) / (len(pair) + 3.0)
    return {
        "career_rate_edge": 2.0 * (a["career_rate"] - b["career_rate"]),
        "recent5_rate_edge": 2.0 * (a["recent5_rate"] - b["recent5_rate"]),
        "decayed_rate_edge": 2.0 * (a["decayed_rate"] - b["decayed_rate"]),
        "major_rate_edge": 2.0 * (a["major_rate"] - b["major_rate"]),
        "experience_edge": a["experience"] - b["experience"],
        "recent_experience_edge": a["recent_experience"] - b["recent_experience"],
        "recency_edge": a["recency"] - b["recency"],
        "last_result_edge": a["last_result"] - b["last_result"],
        "h2h_edge": 2.0 * (h2h_a_rate - 0.5),
        "elo_edge": math.tanh(
            (float(elo_ratings.get(a_key, 1500.0)) - float(elo_ratings.get(b_key, 1500.0)))
            / 300.0
        ),
    }


def build_penalty_feature_frame(
    shootouts: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, float],
]:
    shootouts = shootouts.copy()
    results = results.copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"]).dt.date
    results["date"] = pd.to_datetime(results["date"]).dt.date
    shootouts["row_id"] = np.arange(len(shootouts))

    result_lookup: dict[tuple[date, str, str], dict[str, Any]] = {}
    results_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in results.to_dict("records"):
        a_key = _team_key(row["home_team"])
        b_key = _team_key(row["away_team"])
        result_lookup[(row["date"], *sorted((a_key, b_key)))] = row
        results_by_date[row["date"]].append(row)

    shootouts_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in shootouts.to_dict("records"):
        shootouts_by_date[row["date"]].append(row)

    team_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    h2h_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    elo_ratings: dict[str, float] = defaultdict(lambda: 1500.0)
    rows: list[dict[str, Any]] = []

    for match_date in sorted(set(results_by_date) | set(shootouts_by_date)):
        for shootout in shootouts_by_date.get(match_date, []):
            team_a = str(shootout["home_team"])
            team_b = str(shootout["away_team"])
            a_key = _team_key(team_a)
            b_key = _team_key(team_b)
            match = result_lookup.get((match_date, *sorted((a_key, b_key))), {})
            tournament = str(match.get("tournament") or "")
            features = penalty_feature_values(
                team_a,
                team_b,
                match_date,
                team_records=team_records,
                h2h_records=h2h_records,
                elo_ratings=elo_ratings,
            )
            first_key = _team_key(str(shootout.get("first_shooter") or ""))
            rows.append(
                {
                    "row_id": int(shootout["row_id"]),
                    "date": match_date,
                    "team_a": team_a,
                    "team_b": team_b,
                    "winner": str(shootout["winner"]),
                    "target": int(_team_key(str(shootout["winner"])) == a_key),
                    "tournament": tournament,
                    "is_world_cup": int("world cup" in tournament.casefold()),
                    "first_shooter_edge": (
                        1.0 if first_key == a_key else -1.0 if first_key == b_key else 0.0
                    ),
                    **features,
                }
            )

        for shootout in shootouts_by_date.get(match_date, []):
            a_key = _team_key(str(shootout["home_team"]))
            b_key = _team_key(str(shootout["away_team"]))
            winner_key = _team_key(str(shootout["winner"]))
            match = result_lookup.get((match_date, *sorted((a_key, b_key))), {})
            major = _is_major_tournament(str(match.get("tournament") or ""))
            team_records[a_key].append(
                {"date": match_date.isoformat(), "won": int(winner_key == a_key), "major": major}
            )
            team_records[b_key].append(
                {"date": match_date.isoformat(), "won": int(winner_key == b_key), "major": major}
            )
            pair_key = "|".join(sorted((a_key, b_key)))
            h2h_records[pair_key].append(
                {"date": match_date.isoformat(), "winner_key": winner_key}
            )

        for match in results_by_date.get(match_date, []):
            if pd.isna(match.get("home_score")) or pd.isna(match.get("away_score")):
                continue
            a_key = _team_key(str(match["home_team"]))
            b_key = _team_key(str(match["away_team"]))
            rating_a = float(elo_ratings[a_key])
            rating_b = float(elo_ratings[b_key])
            expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
            goals_a = int(match["home_score"])
            goals_b = int(match["away_score"])
            actual_a = 1.0 if goals_a > goals_b else 0.5 if goals_a == goals_b else 0.0
            k = 10.0 if str(match.get("tournament") or "").casefold() == "friendly" else 20.0
            change = k * (actual_a - expected_a)
            elo_ratings[a_key] = rating_a + change
            elo_ratings[b_key] = rating_b - change

    frame = pd.DataFrame(rows).sort_values(["date", "row_id"]).reset_index(drop=True)
    return frame, dict(team_records), dict(h2h_records), dict(elo_ratings)


class PenaltyShootoutPredictor:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.model_type = str(artifact.get("model_type", "symmetric_logistic"))
        self.features = list(artifact.get("features", []))

    def _raw_probability(self, values: dict[str, float]) -> float:
        mean = self.artifact["scaler_mean"]
        scale = self.artifact["scaler_scale"]
        coefficients = self.artifact["coefficients"]
        score = 0.0
        for feature in self.features:
            standardized = (float(values[feature]) - float(mean[feature])) / float(scale[feature])
            score += float(coefficients[feature]) * standardized
        return 1.0 / (1.0 + math.exp(-score))

    def probability(self, team_a: str, team_b: str, match_date: date) -> float:
        if self.model_type == "coin_flip":
            return 0.5
        if self.model_type != "symmetric_logistic":
            raise ValueError(f"Unsupported penalty model type: {self.model_type}")
        values = penalty_feature_values(
            team_a,
            team_b,
            match_date,
            team_records=self.artifact["team_records"],
            h2h_records=self.artifact["h2h_records"],
            elo_ratings=self.artifact["elo_ratings"],
        )
        forward = self._raw_probability(values)
        reverse_values = penalty_feature_values(
            team_b,
            team_a,
            match_date,
            team_records=self.artifact["team_records"],
            h2h_records=self.artifact["h2h_records"],
            elo_ratings=self.artifact["elo_ratings"],
        )
        reverse = self._raw_probability(reverse_values)
        probability = 0.5 * (forward + 1.0 - reverse)
        return max(0.25, min(0.75, probability))
