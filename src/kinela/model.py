from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict, deque
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from kinela.fifa_ranking import (
    fifa_ranking_for_match_date,
    load_fifa_ranking,
    load_fifa_ranking_history,
    normalise_team_name,
    update_live_fifa_points,
)

CORE_DETAIL_STAT_FEATURES = [
    "shots_on_goal",
    "shots_off_goal",
    "total_shots",
    "blocked_shots",
    "shots_inside_box",
    "shots_outside_box",
    "fouls",
    "corner_kicks",
    "offsides",
    "ball_possession_pct",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "total_passes",
    "passes_accurate",
    "passes_pct",
    "expected_goals",
    "goals_prevented",
]
ESPN_LEADER_DETAIL_FEATURES = [
    "espn_top_xg",
    "espn_keeper_xg_conceded",
    "espn_top_duels_won",
    "espn_top_big_chances_created",
    "espn_top_big_chances_missed",
]
FOTMOB_WORLD_CUP_DETAIL_FEATURES = [
    "fotmob_expected_goals",
    "fotmob_expected_goals_conceded",
    "fotmob_expected_goals_non_penalty",
    "fotmob_expected_goals_on_target",
    "fotmob_big_chances",
    "fotmob_big_chances_conceded",
    "fotmob_big_chances_missed",
    "fotmob_total_shots",
    "fotmob_shots_on_target",
    "fotmob_shots_inside_box",
    "fotmob_touches_opp_box",
    "fotmob_detail_coverage",
    "fotmob_goal_xg_delta",
    "fotmob_xgot_xg_delta",
    "fotmob_xg_per_shot",
    "fotmob_big_chance_rate",
    "fotmob_big_chance_conversion",
    "fotmob_chance_waste_rate",
    "fotmob_underlying_threat",
    "fotmob_finishing_signal",
    "fotmob_waste_signal",
    "fotmob_low_possession_punch",
    "fotmob_sterile_control_risk",
    "fotmob_defensive_resistance",
    "fotmob_chance_control_signal",
    "fotmob_unrewarded_pressure",
    "fotmob_clinical_chance_signal",
]
DETAIL_STAT_FEATURES = [
    *CORE_DETAIL_STAT_FEATURES,
    *ESPN_LEADER_DETAIL_FEATURES,
    *FOTMOB_WORLD_CUP_DETAIL_FEATURES,
]
MIN_TRAINING_DATE = date(2023, 1, 1)
BASE_ELO = 1500.0
H2H_LOOKBACK_DAYS = 730
RECENT_FORM_WINDOW = 6
ENABLE_SQUAD_CONTINUITY_WEIGHT = False
ENABLE_MATCH_ANOMALY_WEIGHT = False
ENABLE_FRIENDLY_MATCHES = True
FRIENDLY_MATCH_WEIGHT = 0.6

TEAM_CONFEDERATION_ALIASES = {
    "united states": "CONCACAF",
    "usa": "CONCACAF",
    "mexico": "CONCACAF",
    "canada": "CONCACAF",
    "costa rica": "CONCACAF",
    "panama": "CONCACAF",
    "jamaica": "CONCACAF",
    "honduras": "CONCACAF",
    "el salvador": "CONCACAF",
    "guatemala": "CONCACAF",
    "haiti": "CONCACAF",
    "trinidad and tobago": "CONCACAF",
    "curacao": "CONCACAF",
    "suriname": "CONCACAF",
    "argentina": "CONMEBOL",
    "brazil": "CONMEBOL",
    "uruguay": "CONMEBOL",
    "colombia": "CONMEBOL",
    "ecuador": "CONMEBOL",
    "peru": "CONMEBOL",
    "chile": "CONMEBOL",
    "paraguay": "CONMEBOL",
    "bolivia": "CONMEBOL",
    "venezuela": "CONMEBOL",
    "france": "UEFA",
    "spain": "UEFA",
    "england": "UEFA",
    "germany": "UEFA",
    "portugal": "UEFA",
    "netherlands": "UEFA",
    "belgium": "UEFA",
    "croatia": "UEFA",
    "switzerland": "UEFA",
    "italy": "UEFA",
    "denmark": "UEFA",
    "austria": "UEFA",
    "turkey": "UEFA",
    "turkiye": "UEFA",
    "poland": "UEFA",
    "serbia": "UEFA",
    "czech republic": "UEFA",
    "czechia": "UEFA",
    "bosnia-herzegovina": "UEFA",
    "bosnia and herzegovina": "UEFA",
    "scotland": "UEFA",
    "wales": "UEFA",
    "ukraine": "UEFA",
    "norway": "UEFA",
    "sweden": "UEFA",
    "romania": "UEFA",
    "hungary": "UEFA",
    "greece": "UEFA",
    "ireland": "UEFA",
    "northern ireland": "UEFA",
    "slovakia": "UEFA",
    "slovenia": "UEFA",
    "albania": "UEFA",
    "georgia": "UEFA",
    "finland": "UEFA",
    "russia": "UEFA",
    "morocco": "CAF",
    "senegal": "CAF",
    "tunisia": "CAF",
    "egypt": "CAF",
    "ghana": "CAF",
    "nigeria": "CAF",
    "cameroon": "CAF",
    "ivory coast": "CAF",
    "cote divoire": "CAF",
    "algeria": "CAF",
    "south africa": "CAF",
    "mali": "CAF",
    "burkina faso": "CAF",
    "dr congo": "CAF",
    "congo dr": "CAF",
    "angola": "CAF",
    "zambia": "CAF",
    "japan": "AFC",
    "south korea": "AFC",
    "korea republic": "AFC",
    "iran": "AFC",
    "australia": "AFC",
    "saudi arabia": "AFC",
    "qatar": "AFC",
    "iraq": "AFC",
    "uzbekistan": "AFC",
    "united arab emirates": "AFC",
    "china": "AFC",
    "new zealand": "OFC",
}


def _team_confederation(team: str | None) -> str:
    key = normalise_team_name(team or "")
    return TEAM_CONFEDERATION_ALIASES.get(key, "unknown")


def _poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _load_worldcup_manual_detail_stats(data_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    path = data_root / "static" / "worldcup_2026_manual_detail_stats.csv"
    if not path.exists():
        return {}
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        match_id = str(row.get("match_id") or "").strip()
        team = normalise_team_name(row.get("team") or "")
        if not match_id or not team:
            continue
        values: dict[str, float] = {}
        for feature in DETAIL_STAT_FEATURES:
            raw = row.get(feature)
            if raw not in (None, ""):
                values[feature] = float(raw)
        stats[(match_id, team)] = values
    return stats


def _load_worldcup_manual_result_rows(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return []

    manual_detail_stats = _load_worldcup_manual_detail_stats(data_root)
    fixture_lookup: dict[str, dict[str, Any]] = {}
    schedule_path = data_root / "raw" / "football_data" / "competitions" / "WC" / "matches.json"
    if schedule_path.exists():
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        for match in payload.get("matches", []):
            match_id = str(match.get("id", ""))
            if match_id:
                fixture_lookup[match_id] = match

    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        match_id = str(row["match_id"])
        fixture = fixture_lookup.get(match_id, {})
        utc_date = fixture.get("utcDate")
        match_date = utc_date[:10] if utc_date else row["date"]
        timestamp = (
            int(datetime.fromisoformat(utc_date.replace("Z", "+00:00")).timestamp())
            if utc_date
            else int(
                datetime.combine(
                    date.fromisoformat(match_date),
                    datetime.min.time(),
                    UTC,
                ).timestamp()
            )
        )
        home_goals = int(row["team_a_goals"])
        away_goals = int(row["team_b_goals"])
        if home_goals > away_goals:
            result = "home"
        elif away_goals > home_goals:
            result = "away"
        else:
            result = "draw"
        home_team = fixture.get("homeTeam", {})
        away_team = fixture.get("awayTeam", {})
        match_row = {
            "source": "manual-worldcup-2026",
            "match_id": f"fd:{match_id}",
            "timestamp": str(timestamp),
            "date": match_date,
            "home_team_id": f"fd:{home_team.get('id', row['team_a'])}",
            "away_team_id": f"fd:{away_team.get('id', row['team_b'])}",
            "home_team": row["team_a"],
            "away_team": row["team_b"],
            "competition_code": "WC",
            "competition_name": "FIFA World Cup",
            "competition_type": "major_tournament",
            "stage_or_round": row["stage"],
            "matchday": "1" if row["stage"] == "GROUP_STAGE" else "",
            "home_goals": str(home_goals),
            "away_goals": str(away_goals),
            "home_penalty_goals": _optional_int(row.get("team_a_penalty_goals")),
            "away_penalty_goals": _optional_int(row.get("team_b_penalty_goals")),
            "result": result,
            "notes": row.get("notes", ""),
        }
        for side in ("home", "away"):
            team_name = row["team_a"] if side == "home" else row["team_b"]
            stats = manual_detail_stats.get((match_id, normalise_team_name(team_name)), {})
            for feature in DETAIL_STAT_FEATURES:
                match_row[f"{side}_actual_{feature}"] = stats.get(feature)
        rows.append(match_row)
    return rows


def _load_fotmob_worldcup_detail_stats(
    data_root: Path,
) -> dict[tuple[str, str, str], dict[str, float]]:
    path = data_root / "processed" / "fotmob_world_cup" / "team_match_stats.csv"
    if not path.exists():
        return {}
    by_match: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        match_id = str(row.get("match_id") or "")
        team = normalise_team_name(row.get("team") or "")
        if not match_id or not team:
            continue
        by_match[match_id][team] = row

    stats: dict[tuple[str, str, str], dict[str, float]] = {}
    for teams in by_match.values():
        if len(teams) < 2:
            continue
        for team_key, row in teams.items():
            opponent_key = normalise_team_name(row.get("opponent") or "")
            opponent = teams.get(opponent_key, {})

            def value(source: dict[str, Any], field: str) -> float | None:
                raw = source.get(field)
                if raw in (None, ""):
                    return None
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None

            def num(source: dict[str, Any], field: str, fallback: float = 0.0) -> float:
                raw = value(source, field)
                return fallback if raw is None else raw

            def ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
                if denominator <= 0:
                    return fallback
                return numerator / denominator

            def clip01(raw: float) -> float:
                return max(0.0, min(1.0, raw))

            own_xg = value(row, "expected_goals")
            own_big = value(row, "big_chances")
            goals_for = num(row, "goals_for")
            goals_against = num(row, "goals_against")
            xg = num(row, "expected_goals")
            xgc = num(opponent, "expected_goals")
            xgot = num(row, "expected_goals_on_target")
            big = num(row, "big_chances")
            big_conceded = num(opponent, "big_chances")
            big_missed = num(row, "big_chances_missed")
            shots = num(row, "total_shots")
            shots_on_target = num(row, "shots_on_target")
            touches_box = num(row, "touches_opp_box")
            possession = num(row, "possession_pct", 50.0)
            xg_per_shot = ratio(xg, shots)
            big_rate = ratio(big, shots)
            big_conversion = ratio(max(0.0, big - big_missed), big, fallback=0.0)
            waste_rate = ratio(big_missed, big)
            goal_xg_delta = goals_for - xg
            xgot_xg_delta = xgot - xg
            creation_strength = (
                0.42 * math.tanh(xg / 1.75)
                + 0.26 * math.tanh(big / 3.2)
                + 0.14 * math.tanh(touches_box / 26.0)
                + 0.10 * math.tanh(shots_on_target / 5.2)
                + 0.08 * math.tanh(xg_per_shot / 0.14)
            )
            finishing_signal = (
                0.44 * math.tanh(goal_xg_delta / 1.15)
                + 0.34 * math.tanh(xgot_xg_delta / 0.85)
                + 0.22 * (2.0 * big_conversion - 1.0)
            )
            waste_signal = (
                0.48 * math.tanh(big_missed / 3.0)
                + 0.32 * math.tanh(max(0.0, xg - goals_for) / 1.25)
                + 0.20 * math.tanh(max(0.0, xg - xgot) / 0.85)
            )
            low_possession = clip01((48.0 - possession) / 18.0)
            high_possession = clip01((possession - 55.0) / 22.0)
            low_possession_punch = low_possession * creation_strength
            sterile_control_risk = high_possession * (1.0 - creation_strength) * (
                0.62 + 0.38 * waste_signal
            )
            defensive_resistance = (
                0.58 * (1.0 - math.tanh(xgc / 1.75))
                + 0.28 * (1.0 - math.tanh(big_conceded / 3.2))
                + 0.14 * (1.0 - math.tanh(goals_against / 2.0))
            )
            chance_control = (
                0.46 * creation_strength
                + 0.25 * defensive_resistance
                + 0.16 * finishing_signal
                - 0.08 * waste_signal
                - 0.05 * sterile_control_risk
            )
            unrewarded_pressure = float(goals_for <= goals_against) * (
                0.58 * creation_strength + 0.42 * waste_signal
            )
            clinical_chance_signal = 0.62 * finishing_signal - 0.38 * waste_signal
            values = {
                "fotmob_expected_goals": own_xg,
                "fotmob_expected_goals_conceded": value(opponent, "expected_goals"),
                "fotmob_expected_goals_non_penalty": value(row, "expected_goals_non_penalty"),
                "fotmob_expected_goals_on_target": value(row, "expected_goals_on_target"),
                "fotmob_big_chances": own_big,
                "fotmob_big_chances_conceded": value(opponent, "big_chances"),
                "fotmob_big_chances_missed": value(row, "big_chances_missed"),
                "fotmob_total_shots": value(row, "total_shots"),
                "fotmob_shots_on_target": value(row, "shots_on_target"),
                "fotmob_shots_inside_box": value(row, "shots_inside_box"),
                "fotmob_touches_opp_box": value(row, "touches_opp_box"),
                "fotmob_goal_xg_delta": goal_xg_delta,
                "fotmob_xgot_xg_delta": xgot_xg_delta,
                "fotmob_xg_per_shot": xg_per_shot,
                "fotmob_big_chance_rate": big_rate,
                "fotmob_big_chance_conversion": big_conversion,
                "fotmob_chance_waste_rate": waste_rate,
                "fotmob_underlying_threat": creation_strength,
                "fotmob_finishing_signal": finishing_signal,
                "fotmob_waste_signal": waste_signal,
                "fotmob_low_possession_punch": low_possession_punch,
                "fotmob_sterile_control_risk": sterile_control_risk,
                "fotmob_defensive_resistance": defensive_resistance,
                "fotmob_chance_control_signal": chance_control,
                "fotmob_unrewarded_pressure": unrewarded_pressure,
                "fotmob_clinical_chance_signal": clinical_chance_signal,
            }
            observed = sum(
                values[field] is not None
                for field in (
                    "fotmob_expected_goals",
                    "fotmob_expected_goals_conceded",
                    "fotmob_big_chances",
                    "fotmob_big_chances_conceded",
                    "fotmob_big_chances_missed",
                )
            )
            values["fotmob_detail_coverage"] = observed / 5.0
            clean_values = {
                key: float(val)
                for key, val in values.items()
                if val is not None
            }
            date_key = str(row.get("date") or "")
            opponent_raw = normalise_team_name(row.get("opponent") or "")
            stats[(date_key, team_key, opponent_raw)] = clean_values
    return stats


def _attach_fotmob_worldcup_detail_stats(
    data_root: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fotmob_stats = _load_fotmob_worldcup_detail_stats(data_root)
    if not fotmob_stats:
        return rows
    enriched: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        if row.get("competition_name") != "FIFA World Cup":
            enriched.append(current)
            continue
        date_key = str(row.get("date") or "")
        for side, opponent in (("home", "away"), ("away", "home")):
            team_key = normalise_team_name(str(row.get(f"{side}_team") or ""))
            opponent_key = normalise_team_name(str(row.get(f"{opponent}_team") or ""))
            stats = fotmob_stats.get((date_key, team_key, opponent_key), {})
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                if feature in stats:
                    current[f"{side}_actual_{feature}"] = stats[feature]
        enriched.append(current)
    return enriched


def _score_probs(home_lambda: float, away_lambda: float, max_goals: int = 8) -> dict[str, float]:
    home_win = draw = away_win = 0.0
    most_likely_score = "0-0"
    most_likely_probability = 0.0
    for home_goals in range(max_goals + 1):
        hp = _poisson_pmf(home_lambda, home_goals)
        for away_goals in range(max_goals + 1):
            p = hp * _poisson_pmf(away_lambda, away_goals)
            if home_goals > away_goals:
                home_win += p
            elif home_goals == away_goals:
                draw += p
            else:
                away_win += p
            if p > most_likely_probability:
                most_likely_score = f"{home_goals}-{away_goals}"
                most_likely_probability = p
    total = home_win + draw + away_win
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
        "most_likely_score": most_likely_score,
    }


def _calibrate_probs(probs: dict[str, float], temperature: float) -> dict[str, float]:
    keys = ("home_win", "draw", "away_win")
    adjusted = {key: max(probs[key], 1e-12) ** temperature for key in keys}
    total = sum(adjusted.values())
    return {key: adjusted[key] / total for key in keys}


def _score_state(goal_diff: int) -> int:
    return 1 if goal_diff > 0 else -1 if goal_diff < 0 else 0


def _points_for_score_state(score_state: int) -> int:
    if score_state > 0:
        return 3
    if score_state == 0:
        return 1
    return 0


def _opponent_quality_factor(opponent_elo: float) -> float:
    if not math.isfinite(opponent_elo) or opponent_elo <= 0:
        return 1.0
    return max(0.75, min(1.25, opponent_elo / BASE_ELO))


def _event_minute(event: dict[str, Any]) -> float | None:
    time = event.get("time") or {}
    elapsed = time.get("elapsed")
    if elapsed is None:
        return None
    return float(elapsed) + float(time.get("extra") or 0) / 100.0


def _event_elapsed_minute(event: dict[str, Any]) -> float | None:
    time = event.get("time") or {}
    elapsed = time.get("elapsed")
    if elapsed is None:
        return None
    return float(elapsed) + float(time.get("extra") or 0)


def _is_counted_goal_event(event: dict[str, Any]) -> bool:
    if event.get("type") != "Goal":
        return False
    if str(event.get("detail") or "").casefold() == "missed penalty":
        return False
    return "penalty shootout" not in str(event.get("comments") or "").casefold()


def _valid_goal_events(
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> list[dict[str, Any]] | None:
    home_key = normalise_team_name(home_team)
    away_key = normalise_team_name(away_team)
    goals = {home_key: 0, away_key: 0}
    counted: list[dict[str, Any]] = []
    for event in events:
        if not _is_counted_goal_event(event):
            continue
        team_key = normalise_team_name((event.get("team") or {}).get("name") or "")
        if team_key not in goals:
            continue
        goals[team_key] += 1
        counted.append(event)
    if goals[home_key] != int(home_goals) or goals[away_key] != int(away_goals):
        return None
    return counted


def late85_points_swing_metrics(
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, dict[str, float]]:
    home_key = normalise_team_name(home_team)
    away_key = normalise_team_name(away_team)
    score = {home_key: 0, away_key: 0}
    metrics = {
        home_key: {"late85_points_swing_edge": 0.0},
        away_key: {"late85_points_swing_edge": 0.0},
    }
    goal_events = [event for event in events if _is_counted_goal_event(event)]
    goal_events.sort(key=lambda event: _event_minute(event) if _event_minute(event) is not None else 999.0)

    for event in goal_events:
        minute = _event_minute(event)
        team_key = normalise_team_name((event.get("team") or {}).get("name") or "")
        if minute is None or team_key not in score:
            continue
        before = {
            home_key: _score_state(score[home_key] - score[away_key]),
            away_key: _score_state(score[away_key] - score[home_key]),
        }
        before_points = {key: _points_for_score_state(value) for key, value in before.items()}
        score[team_key] += 1
        after = {
            home_key: _score_state(score[home_key] - score[away_key]),
            away_key: _score_state(score[away_key] - score[home_key]),
        }
        after_points = {key: _points_for_score_state(value) for key, value in after.items()}
        if minute < 85:
            continue
        for key in (home_key, away_key):
            metrics[key]["late85_points_swing_edge"] += float(
                after_points[key] - before_points[key]
            )
    return metrics


def score_timing_metrics(
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, dict[str, float]]:
    home_key = normalise_team_name(home_team)
    away_key = normalise_team_name(away_team)
    score = {home_key: 0, away_key: 0}
    state_minutes = {
        home_key: {1: 0.0, 0: 0.0, -1: 0.0},
        away_key: {1: 0.0, 0: 0.0, -1: 0.0},
    }
    margin_minutes = {
        home_key: {"lead_1": 0.0, "lead_2plus": 0.0, "level": 0.0, "trail_1": 0.0, "trail_2plus": 0.0},
        away_key: {"lead_1": 0.0, "lead_2plus": 0.0, "level": 0.0, "trail_1": 0.0, "trail_2plus": 0.0},
    }
    late_level_minutes = {home_key: 0.0, away_key: 0.0}
    first_goal = {home_key: None, away_key: None}
    state_change_swing = {home_key: 0.0, away_key: 0.0}
    early_state_change_swing = {home_key: 0.0, away_key: 0.0}

    def add_segment(start: float, end: float) -> None:
        minutes = max(0.0, end - start)
        if minutes <= 0:
            return
        diffs = {
            home_key: score[home_key] - score[away_key],
            away_key: score[away_key] - score[home_key],
        }
        for key, diff in diffs.items():
            state = _score_state(diff)
            state_minutes[key][state] += minutes
            if diff >= 2:
                margin_minutes[key]["lead_2plus"] += minutes
            elif diff == 1:
                margin_minutes[key]["lead_1"] += minutes
            elif diff == 0:
                margin_minutes[key]["level"] += minutes
                late_level_minutes[key] += max(0.0, end - max(start, 45.0))
            elif diff == -1:
                margin_minutes[key]["trail_1"] += minutes
            else:
                margin_minutes[key]["trail_2plus"] += minutes

    last_minute = 0.0
    goal_events = [event for event in events if _is_counted_goal_event(event)]
    goal_events.sort(
        key=lambda event: (
            _event_elapsed_minute(event)
            if _event_elapsed_minute(event) is not None
            else 999.0
        )
    )

    for event in goal_events:
        minute = _event_elapsed_minute(event)
        team_key = normalise_team_name((event.get("team") or {}).get("name") or "")
        if minute is None or team_key not in score:
            continue
        minute = min(max(float(minute), last_minute), 130.0)
        add_segment(last_minute, minute)
        if first_goal[team_key] is None:
            first_goal[team_key] = minute
        before = {
            home_key: _score_state(score[home_key] - score[away_key]),
            away_key: _score_state(score[away_key] - score[home_key]),
        }
        before_points = {key: _points_for_score_state(value) for key, value in before.items()}
        score[team_key] += 1
        after = {
            home_key: _score_state(score[home_key] - score[away_key]),
            away_key: _score_state(score[away_key] - score[home_key]),
        }
        after_points = {key: _points_for_score_state(value) for key, value in after.items()}
        minute_weight = max(0.0, (100.0 - min(minute, 100.0)) / 100.0)
        for key in (home_key, away_key):
            swing = float(after_points[key] - before_points[key])
            state_change_swing[key] += swing
            early_state_change_swing[key] += swing * minute_weight
        last_minute = minute

    match_duration = max(90.0, last_minute)
    add_segment(last_minute, match_duration)

    metrics: dict[str, dict[str, float]] = {}
    for key in (home_key, away_key):
        duration = max(match_duration, 1.0)
        leading_share = state_minutes[key][1] / duration
        level_share = state_minutes[key][0] / duration
        trailing_share = state_minutes[key][-1] / duration
        lead_1_share = margin_minutes[key]["lead_1"] / duration
        lead_2plus_share = margin_minutes[key]["lead_2plus"] / duration
        trail_1_share = margin_minutes[key]["trail_1"] / duration
        trail_2plus_share = margin_minutes[key]["trail_2plus"] / duration
        late_level_share = late_level_minutes[key] / duration
        final_points = _points_for_score_state(
            _score_state(
                score[key] - score[away_key if key == home_key else home_key]
            )
        )
        goal_minute = first_goal[key]
        first_goal_minute = float(goal_minute) if goal_minute is not None else 100.0
        narrow_hold_value = lead_1_share * (final_points / 3.0)
        score_control_quality = (
            lead_2plus_share
            + 0.42 * narrow_hold_value
            - 0.15 * lead_1_share * (1.0 - final_points / 3.0)
            - 0.38 * late_level_share
            - 0.72 * trail_1_share
            - 1.05 * trail_2plus_share
        )
        game_state_friction = (
            0.52 * late_level_share
            + 0.28 * trailing_share
            + 0.20 * min(first_goal_minute, 100.0) / 100.0
        )
        metrics[key] = {
            "score_state_value": (3.0 * leading_share + level_share) / 3.0,
            "score_control_value": leading_share - trailing_share,
            "scoring_quickness": max(0.0, (100.0 - min(first_goal_minute, 100.0)) / 100.0),
            "first_goal_minute": first_goal_minute,
            "score_control_quality": score_control_quality,
            "narrow_lead_hold": narrow_hold_value,
            "comfortable_lead": lead_2plus_share,
            "game_state_friction": game_state_friction,
            "state_change_swing": state_change_swing[key],
            "early_state_change_swing": early_state_change_swing[key],
        }
    return metrics


def _load_api_late85_points_swing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    for path in (data_root / "raw" / "api_football" / "fixtures").glob("details-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for item in payload.get("response") or []:
            fixture = item.get("fixture") or {}
            fixture_id = fixture.get("id") or path.stem.removeprefix("details-")
            teams = item.get("teams") or {}
            home = (teams.get("home") or {}).get("name") or ""
            away = (teams.get("away") or {}).get("name") or ""
            if home and away:
                by_match[f"api:{fixture_id}"] = late85_points_swing_metrics(
                    item.get("events") or [],
                    home,
                    away,
                )
    return by_match


def _load_api_score_timing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    for path in (data_root / "raw" / "api_football" / "fixtures").glob("details-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for item in payload.get("response") or []:
            fixture = item.get("fixture") or {}
            fixture_id = fixture.get("id") or path.stem.removeprefix("details-")
            teams = item.get("teams") or {}
            home = (teams.get("home") or {}).get("name") or ""
            away = (teams.get("away") or {}).get("name") or ""
            if home and away:
                goals = item.get("goals") or {}
                events = _valid_goal_events(
                    item.get("events") or [],
                    home,
                    away,
                    int(goals.get("home") or 0),
                    int(goals.get("away") or 0),
                )
                if events is None:
                    continue
                by_match[f"api:{fixture_id}"] = score_timing_metrics(
                    events,
                    home,
                    away,
                )
    return by_match


def _load_statsbomb_late85_points_swing_metrics(
    data_root: Path,
    training_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    goals_path = data_root / "processed" / "statsbomb_world_cup_2022" / "goals.csv"
    if not goals_path.exists():
        return {}
    goals_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with goals_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_shootout")).lower() == "true" or row.get("minute") in (None, ""):
                continue
            goals_by_match[str(row["match_id"])].append(row)

    by_match: dict[str, dict[str, dict[str, float]]] = {}
    for row in training_rows:
        if row.get("source") != "statsbomb-open-data":
            continue
        match_id = str(row["match_id"]).removeprefix("sb:")
        events = [
            {
                "type": "Goal",
                "time": {"elapsed": int(float(goal["minute"])), "extra": None},
                "team": {"name": goal["team"]},
            }
            for goal in goals_by_match.get(match_id, [])
        ]
        by_match[f"sb:{match_id}"] = late85_points_swing_metrics(
            events,
            str(row["home_team"]),
            str(row["away_team"]),
        )
    return by_match


def _load_statsbomb_score_timing_metrics(
    data_root: Path,
    training_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    goals_path = data_root / "processed" / "statsbomb_world_cup_2022" / "goals.csv"
    if not goals_path.exists():
        return {}
    goals_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with goals_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_shootout")).lower() == "true" or row.get("minute") in (None, ""):
                continue
            goals_by_match[str(row["match_id"])].append(row)

    by_match: dict[str, dict[str, dict[str, float]]] = {}
    for row in training_rows:
        if row.get("source") != "statsbomb-open-data":
            continue
        match_id = str(row["match_id"]).removeprefix("sb:")
        events = [
            {
                "type": "Goal",
                "time": {"elapsed": int(float(goal["minute"])), "extra": None},
                "team": {"name": goal["team"]},
            }
            for goal in goals_by_match.get(match_id, [])
        ]
        events = _valid_goal_events(
            events,
            str(row["home_team"]),
            str(row["away_team"]),
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
        if events is None:
            continue
        by_match[f"sb:{match_id}"] = score_timing_metrics(
            events,
            str(row["home_team"]),
            str(row["away_team"]),
        )
    return by_match


def _load_manual_late85_points_swing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return {}
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    minute_pattern = re.compile(r"(\d{1,3})(?:\+(\d{1,2}))?'(?:[^.;]*?)for ([^.;]+)")
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            home = row["team_a"]
            away = row["team_b"]
            allowed = {normalise_team_name(home): home, normalise_team_name(away): away}
            events = []
            for match in minute_pattern.finditer(row.get("notes") or ""):
                minute = int(match.group(1))
                extra = int(match.group(2) or 0) or None
                raw_team = match.group(3).strip()
                raw_team = re.split(r"\s+(?:and|No red|Yellow|Red)\b", raw_team)[0].strip()
                team = allowed.get(normalise_team_name(raw_team))
                if team is None:
                    if normalise_team_name(home) in normalise_team_name(raw_team):
                        team = home
                    elif normalise_team_name(away) in normalise_team_name(raw_team):
                        team = away
                if team:
                    events.append(
                        {
                            "type": "Goal",
                            "time": {"elapsed": minute, "extra": extra},
                            "team": {"name": team},
                        }
                    )
            events = _valid_goal_events(
                events,
                home,
                away,
                int(row["team_a_goals"]),
                int(row["team_b_goals"]),
            )
            if events is None:
                continue
            by_match[f"fd:{row['match_id']}"] = late85_points_swing_metrics(events, home, away)
    return by_match


def _load_manual_score_timing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return {}
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    minute_pattern = re.compile(r"(\d{1,3})(?:\+(\d{1,2}))?'(?:[^.;]*?)for ([^.;]+)")
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            home = row["team_a"]
            away = row["team_b"]
            allowed = {normalise_team_name(home): home, normalise_team_name(away): away}
            events = []
            for match in minute_pattern.finditer(row.get("notes") or ""):
                minute = int(match.group(1))
                extra = int(match.group(2) or 0) or None
                raw_team = match.group(3).strip()
                raw_team = re.split(r"\s+(?:and|No red|Yellow|Red)\b", raw_team)[0].strip()
                team = allowed.get(normalise_team_name(raw_team))
                if team is None:
                    if normalise_team_name(home) in normalise_team_name(raw_team):
                        team = home
                    elif normalise_team_name(away) in normalise_team_name(raw_team):
                        team = away
                if team:
                    events.append(
                        {
                            "type": "Goal",
                            "time": {"elapsed": minute, "extra": extra},
                            "team": {"name": team},
                        }
                    )
            events = _valid_goal_events(
                events,
                home,
                away,
                int(row["team_a_goals"]),
                int(row["team_b_goals"]),
            )
            if events is None:
                continue
            by_match[f"fd:{row['match_id']}"] = score_timing_metrics(events, home, away)
    return by_match


def _espn_event_id(row: dict[str, Any]) -> str | None:
    text = f"{row.get('source', '')} {row.get('notes', '')}"
    match = re.search(r"\bESPN event (\d+)\b", text)
    return match.group(1) if match else None


def _espn_goal_minute(play: dict[str, Any]) -> tuple[int, int | None] | None:
    display = str((play.get("clock") or {}).get("displayValue") or "")
    match = re.search(r"(\d{1,3})'(?:\+(\d{1,2})')?", display)
    if match:
        return int(match.group(1)), int(match.group(2) or 0) or None
    value = (play.get("clock") or {}).get("value")
    if value in (None, ""):
        return None
    try:
        minute = int(float(value) // 60)
    except (TypeError, ValueError):
        return None
    return minute, None


def _espn_goal_events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for play in summary.get("keyEvents") or []:
        if play.get("shootout") is True:
            continue
        play_type = (play.get("type") or {}).get("type")
        text = f"{(play.get('type') or {}).get('text', '')} {play.get('text', '')}".casefold()
        if not (play.get("scoringPlay") is True or play_type == "goal" or "goal!" in text):
            continue
        team = (play.get("team") or {}).get("displayName") or ""
        minute = _espn_goal_minute(play)
        if not team or minute is None:
            continue
        elapsed, extra = minute
        events.append(
            {
                "type": "Goal",
                "detail": (play.get("type") or {}).get("text"),
                "comments": "",
                "time": {"elapsed": elapsed, "extra": extra},
                "team": {"name": team},
            }
        )
    return events


def _load_espn_score_timing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return {}
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    summary_dir = data_root / "raw" / "espn" / "worldcup_2026"
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = _espn_event_id(row)
            if not event_id:
                continue
            summary_path = summary_dir / f"summary_{event_id}.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            home = row["team_a"]
            away = row["team_b"]
            events = _espn_goal_events(summary)
            valid = _valid_goal_events(
                events,
                home,
                away,
                int(row["team_a_goals"]),
                int(row["team_b_goals"]),
            )
            if valid is None:
                continue
            by_match[f"fd:{row['match_id']}"] = score_timing_metrics(valid, home, away)
    return by_match


def _load_espn_late85_points_swing_metrics(data_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return {}
    by_match: dict[str, dict[str, dict[str, float]]] = {}
    summary_dir = data_root / "raw" / "espn" / "worldcup_2026"
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = _espn_event_id(row)
            if not event_id:
                continue
            summary_path = summary_dir / f"summary_{event_id}.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            home = row["team_a"]
            away = row["team_b"]
            events = _espn_goal_events(summary)
            valid = _valid_goal_events(
                events,
                home,
                away,
                int(row["team_a_goals"]),
                int(row["team_b_goals"]),
            )
            if valid is None:
                continue
            by_match[f"fd:{row['match_id']}"] = late85_points_swing_metrics(valid, home, away)
    return by_match


def load_late85_points_swing_metrics(
    data_root: Path,
    training_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metrics.update(_load_api_late85_points_swing_metrics(data_root))
    metrics.update(_load_statsbomb_late85_points_swing_metrics(data_root, training_rows))
    metrics.update(_load_espn_late85_points_swing_metrics(data_root))
    metrics.update(_load_manual_late85_points_swing_metrics(data_root))
    return metrics


def load_score_timing_metrics(
    data_root: Path,
    training_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metrics.update(_load_api_score_timing_metrics(data_root))
    metrics.update(_load_statsbomb_score_timing_metrics(data_root, training_rows))
    metrics.update(_load_espn_score_timing_metrics(data_root))
    metrics.update(_load_manual_score_timing_metrics(data_root))
    return metrics


def _load_match_rows(data_root: Path, *, combined: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    api_detail_stats = _load_api_detail_stats(data_root)
    api_red_cards = _load_api_red_cards(data_root)
    statsbomb_shootout_scores = _load_statsbomb_shootout_scores(data_root)
    api_path = data_root / "processed" / "api_football" / "matches.csv"
    if api_path.exists():
        for row in csv.DictReader(api_path.open(encoding="utf-8")):
            fixture_id = row["fixture_id"]
            match_row = {
                "source": "api-football",
                "match_id": f"api:{fixture_id}",
                "timestamp": row["timestamp"],
                "date": row["date"],
                "home_team_id": f"api:{row['home_team_id']}",
                "away_team_id": f"api:{row['away_team_id']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "competition_code": f"api:{row['league_id']}",
                "competition_name": row["league_name"],
                "competition_type": row["competition_type"],
                "stage_or_round": row["round"],
                "matchday": "",
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "home_penalty_goals": _optional_int(row.get("home_penalty_goals")),
                "away_penalty_goals": _optional_int(row.get("away_penalty_goals")),
                "result": row["result"],
                "home_red_cards": api_red_cards.get((fixture_id, normalise_team_name(row["home_team"])), 0),
                "away_red_cards": api_red_cards.get((fixture_id, normalise_team_name(row["away_team"])), 0),
            }
            for side in ("home", "away"):
                stats = api_detail_stats.get((fixture_id, normalise_team_name(row[f"{side}_team"])), {})
                for feature in DETAIL_STAT_FEATURES:
                    match_row[f"{side}_actual_{feature}"] = stats.get(feature)
            rows.append(match_row)
    if combined:
        fd_path = data_root / "processed" / "football_data" / "matches.csv"
        if fd_path.exists():
            for row in csv.DictReader(fd_path.open(encoding="utf-8")):
                rows.append(
                    {
                        "source": "football-data.org",
                        "match_id": f"fd:{row['match_id']}",
                        "timestamp": row["timestamp"],
                        "date": row["date"],
                        "home_team_id": f"fd:{row['home_team_id']}",
                        "away_team_id": f"fd:{row['away_team_id']}",
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "competition_code": row["competition_code"],
                        "competition_name": row["competition_name"],
                        "competition_type": row["competition_type"],
                        "stage_or_round": row["stage"],
                        "matchday": row["matchday"],
                        "home_goals": row["home_goals"],
                        "away_goals": row["away_goals"],
                        "result": row["result"],
                    }
                )
        rows.extend(_load_worldcup_manual_result_rows(data_root))
    statsbomb_path = data_root / "processed" / "statsbomb_world_cup_2022" / "matches.csv"
    if combined and statsbomb_path.exists():
        for row in csv.DictReader(statsbomb_path.open(encoding="utf-8")):
            home_goals = int(row["home_score"])
            away_goals = int(row["away_score"])
            shootout = statsbomb_shootout_scores.get(row["match_id"], {})
            home_penalty_goals = (
                shootout.get(normalise_team_name(row["home_team"]), 0) if shootout else None
            )
            away_penalty_goals = (
                shootout.get(normalise_team_name(row["away_team"]), 0) if shootout else None
            )
            if home_goals > away_goals:
                result = "home"
            elif away_goals > home_goals:
                result = "away"
            else:
                result = "draw"
            rows.append(
                {
                    "source": "statsbomb-open-data",
                    "match_id": f"sb:{row['match_id']}",
                    "timestamp": str(
                        int(datetime.combine(date.fromisoformat(row["date"]), datetime.min.time(), UTC).timestamp())
                    ),
                    "date": row["date"],
                    "home_team_id": f"name:{row['home_team']}",
                    "away_team_id": f"name:{row['away_team']}",
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "competition_code": "statsbomb:wc",
                    "competition_name": "FIFA World Cup",
                    "competition_type": "major_tournament",
                    "stage_or_round": row["stage"],
                    "matchday": "",
                    "home_goals": row["home_score"],
                    "away_goals": row["away_score"],
                    "home_penalty_goals": home_penalty_goals,
                    "away_penalty_goals": away_penalty_goals,
                    "result": result,
                }
            )
    rows = _attach_fotmob_worldcup_detail_stats(data_root, rows)
    rows.sort(key=lambda row: int(row["timestamp"]))
    return _deduplicate_real_matches(rows)


def _real_match_key(row: dict[str, Any]) -> tuple[str, tuple[str, str]]:
    teams = sorted(
        (
            normalise_team_name(str(row["home_team"])),
            normalise_team_name(str(row["away_team"])),
        )
    )
    return str(row["date"]), (teams[0], teams[1])


def _source_priority(row: dict[str, Any]) -> int:
    return {
        "manual-worldcup-2026": 4,
        "statsbomb-open-data": 3,
        "api-football": 2,
        "football-data.org": 1,
    }.get(str(row.get("source") or ""), 0)


def _has_observed_value(value: Any) -> bool:
    return value not in (None, "")


def _merge_duplicate_match_rows(
    preferred: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(preferred)
    for key, value in fallback.items():
        if not _has_observed_value(merged.get(key)) and _has_observed_value(value):
            merged[key] = value
    return merged


def _align_match_orientation(
    row: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    row_home = normalise_team_name(str(row["home_team"]))
    row_away = normalise_team_name(str(row["away_team"]))
    reference_home = normalise_team_name(str(reference["home_team"]))
    reference_away = normalise_team_name(str(reference["away_team"]))
    if (row_home, row_away) != (reference_away, reference_home):
        return dict(row)

    aligned = dict(row)
    suffixes = {
        key.removeprefix("home_")
        for key in row
        if key.startswith("home_")
    } | {
        key.removeprefix("away_")
        for key in row
        if key.startswith("away_")
    }
    for suffix in suffixes:
        aligned[f"home_{suffix}"] = row.get(f"away_{suffix}")
        aligned[f"away_{suffix}"] = row.get(f"home_{suffix}")
    result = str(row.get("result") or "")
    if result == "home":
        aligned["result"] = "away"
    elif result == "away":
        aligned["result"] = "home"
    return aligned


def _deduplicate_real_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse provider copies before any rolling history is constructed."""

    by_match_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_match_id[str(row["match_id"])] = row

    by_real_match: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for row in sorted(by_match_id.values(), key=lambda item: int(item["timestamp"])):
        key = _real_match_key(row)
        current = by_real_match.get(key)
        if current is None:
            by_real_match[key] = dict(row)
            continue
        if _source_priority(row) > _source_priority(current):
            aligned_current = _align_match_orientation(current, row)
            by_real_match[key] = _merge_duplicate_match_rows(row, aligned_current)
        else:
            aligned_row = _align_match_orientation(row, current)
            by_real_match[key] = _merge_duplicate_match_rows(current, aligned_row)
    return sorted(by_real_match.values(), key=lambda row: int(row["timestamp"]))


def _filter_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (
            date.fromisoformat(row["date"]) >= MIN_TRAINING_DATE
            or row.get("competition_name") == "FIFA World Cup"
        )
        and (ENABLE_FRIENDLY_MATCHES or row.get("competition_type") != "friendly")
    ]


def _is_national_row(row: dict[str, Any]) -> bool:
    competition_name = (row.get("competition_name") or "").lower()
    competition_code = (row.get("competition_code") or "").lower()
    competition_type = (row.get("competition_type") or "").lower()
    national_markers = [
        "world cup",
        "euro",
        "european championship",
        "copa america",
        "africa cup",
        "asian cup",
        "gold cup",
        "nations league",
        "gulf cup",
        "cafa",
        "qualification",
        "qualifier",
        "qualifying",
    ]
    if competition_code in {"wc", "ec"}:
        return True
    if ENABLE_FRIENDLY_MATCHES and competition_type == "friendly" and row.get("source") == "api-football":
        return True
    if competition_type in {"qualifier", "major_tournament"} and any(
        marker in competition_name for marker in national_markers
    ):
        return True
    return any(marker in competition_name for marker in national_markers)


def _filter_national_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_national_row(row)]


def _training_recency_weight(match_date: str, latest_date: date) -> float:
    age_days = max(0, (latest_date - date.fromisoformat(match_date)).days)
    weight = 0.55 + 0.75 * math.exp(-age_days / 730)
    return round(weight, 4)


def _match_anomaly_weight(row: dict[str, Any]) -> float:
    if not ENABLE_MATCH_ANOMALY_WEIGHT:
        return 1.0
    red_cards = int(row.get("home_red_cards") or 0) + int(row.get("away_red_cards") or 0)
    if red_cards:
        return round(max(0.65, 1.0 - 0.12 * red_cards), 4)
    notes = (row.get("notes") or "").lower()
    if re.search(r"\bno\s+red\s+cards?\b", notes):
        return 1.0
    if "red card" not in notes and "red cards" not in notes:
        return 1.0
    card_notes = notes.split("red cards:", 1)[-1].split("red card:", 1)[-1]
    count = len(re.findall(r"\d+(?:\+\d+)?'", card_notes))
    if count == 0:
        count = notes.count("red card")
        if "red cards" in notes and count == 1:
            count = 2
    return round(max(0.65, 1.0 - 0.12 * count), 4)


def _normalise_player_name(name: str | None) -> str:
    return normalise_team_name(name or "")


def _load_current_worldcup_squads(data_root: Path) -> dict[str, set[str]]:
    path = data_root / "processed" / "football_data" / "squad_players.csv"
    if not path.exists():
        return {}
    squads: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("competition_code") != "WC":
            continue
        team_key = normalise_team_name(row.get("team") or "")
        player_key = _normalise_player_name(row.get("player"))
        if team_key and player_key:
            squads[team_key].add(player_key)
    return {team: players for team, players in squads.items() if players}


def _load_match_starting_lineups(data_root: Path) -> dict[tuple[str, str], set[str]]:
    lineups: dict[tuple[str, str], set[str]] = {}

    api_dir = data_root / "raw" / "api_football" / "fixtures"
    if api_dir.exists():
        for path in api_dir.glob("details-*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in payload.get("response") or []:
                fixture_id = (item.get("fixture") or {}).get("id")
                if not fixture_id:
                    continue
                match_id = f"api:{fixture_id}"
                for lineup in item.get("lineups") or []:
                    team_key = normalise_team_name((lineup.get("team") or {}).get("name") or "")
                    starters = {
                        _normalise_player_name((entry.get("player") or {}).get("name"))
                        for entry in lineup.get("startXI") or []
                    }
                    starters.discard("")
                    if team_key and len(starters) >= 7:
                        lineups[(match_id, team_key)] = starters

    statsbomb_path = data_root / "processed" / "statsbomb_world_cup_2022" / "lineups.csv"
    if statsbomb_path.exists():
        grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in csv.DictReader(statsbomb_path.open(encoding="utf-8")):
            if str(row.get("starter")).casefold() != "true":
                continue
            match_id = f"sb:{row.get('match_id')}"
            team_key = normalise_team_name(row.get("team") or "")
            player_key = _normalise_player_name(row.get("player"))
            if team_key and player_key:
                grouped[(match_id, team_key)].add(player_key)
        for key, starters in grouped.items():
            if len(starters) >= 7:
                lineups[key] = starters

    return lineups


def _squad_continuity_share(
    row: dict[str, Any],
    current_squads: dict[str, set[str]],
    starting_lineups: dict[tuple[str, str], set[str]],
) -> float | None:
    shares: list[float] = []
    for side in ("home", "away"):
        team_key = normalise_team_name(row.get(f"{side}_team") or "")
        current_squad = current_squads.get(team_key)
        starters = starting_lineups.get((row.get("match_id") or "", team_key))
        if current_squad and starters:
            shares.append(len(starters & current_squad) / len(starters))
    if not shares:
        return None
    return sum(shares) / len(shares)


def _squad_continuity_weight(
    row: dict[str, Any],
    latest_date: date,
    current_squads: dict[str, set[str]],
    starting_lineups: dict[tuple[str, str], set[str]],
) -> tuple[float | None, float]:
    if not ENABLE_SQUAD_CONTINUITY_WEIGHT:
        return None, 1.0
    age_days = max(0, (latest_date - date.fromisoformat(row["date"])).days)
    if age_days < 730:
        return None, 1.0
    share = _squad_continuity_share(row, current_squads, starting_lineups)
    if share is None:
        return None, 1.0
    return round(share, 4), round(0.85 + 0.15 * share, 4)


def _quality_result_points(points: int, goal_diff: int, opponent_elo: float) -> float:
    if points == 0:
        base = 0.35 if goal_diff == -1 else 0.15 if goal_diff == -2 else 0.0
    else:
        base = float(points)
    margin_bonus = 0.25 * max(0, min(2, goal_diff))
    opponent_factor = _opponent_quality_factor(opponent_elo)
    return (base + margin_bonus) * opponent_factor


def _quality_goal_balance(goals_for: int, goals_against: int, opponent_elo: float) -> float:
    opponent_factor = _opponent_quality_factor(opponent_elo)
    conceded_factor = max(0.75, min(1.25, BASE_ELO / opponent_elo))
    return math.log2(1 + goals_for) * opponent_factor - math.log2(
        1 + goals_against
    ) * conceded_factor


def _tournament_hosts(row: dict[str, Any]) -> set[str]:
    competition_name = (row.get("competition_name") or "").lower()
    match_date = row.get("date") or ""
    if "world cup" in competition_name:
        if match_date.startswith("2022"):
            return {"qatar"}
        if match_date.startswith("2026"):
            return {"canada", "mexico", "united states", "usa"}
    return set()


def _is_tournament_host(row: dict[str, Any], side: str) -> bool:
    hosts = _tournament_hosts(row)
    return normalise_team_name(row.get(f"{side}_team", "")) in hosts


def _competition_family(row: dict[str, Any]) -> str:
    name = (row.get("competition_name") or "").lower()
    code = (row.get("competition_code") or "").lower()
    competition_type = (row.get("competition_type") or "").lower()
    if competition_type == "qualifier" or any(
        marker in name for marker in ("qualification", "qualifier", "qualifying")
    ):
        return "national_qualifier"
    if "world cup" in name or code == "wc":
        return "national_world_cup"
    if any(
        marker in name
        for marker in (
            "euro",
            "european championship",
            "copa america",
            "africa cup",
            "asian cup",
            "gold cup",
            "gulf cup",
            "cafa",
        )
    ):
        return "national_continental_tournament"
    if "nations league" in name:
        return "national_nations_league"
    if "champions league" in name or "libertadores" in name:
        return "club_continental"
    if row.get("source") == "football-data.org":
        return "club_league"
    return competition_type or "unknown"


def _normalise_stage_or_round(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    canonical = text.casefold().replace("_", " ")
    if "group" in canonical:
        return "GROUP_STAGE"
    if "round of 32" in canonical or "last 32" in canonical:
        return "ROUND_OF_32"
    if "round of 16" in canonical or "last 16" in canonical:
        return "LAST_16"
    if "quarter" in canonical:
        return "QUARTER_FINALS"
    if "semi" in canonical:
        return "SEMI_FINALS"
    if "3rd" in canonical or "third" in canonical:
        return "THIRD_PLACE"
    if canonical == "final":
        return "FINAL"
    return text


def _load_api_detail_stats(data_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    path = data_root / "processed" / "api_football" / "team_detailed_stats.csv"
    if not path.exists():
        return {}
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        values: dict[str, float] = {}
        for feature in DETAIL_STAT_FEATURES:
            raw = row.get(feature)
            if raw not in (None, ""):
                values[feature] = float(raw)
        stats[(row["fixture_id"], normalise_team_name(row["team"]))] = values
    return stats


def _load_api_red_cards(data_root: Path) -> dict[tuple[str, str], int]:
    raw_dir = data_root / "raw" / "api_football" / "fixtures"
    if not raw_dir.exists():
        return {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for path in raw_dir.glob("details-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for fixture in payload.get("response", []):
            fixture_id = str(
                fixture.get("fixture", {}).get("id") or path.stem.removeprefix("details-")
            )
            for event in fixture.get("events", []) or []:
                if event.get("type") != "Card":
                    continue
                detail = (event.get("detail") or "").lower()
                if "red card" not in detail and "second yellow" not in detail:
                    continue
                team_key = normalise_team_name(event.get("team", {}).get("name") or "")
                if team_key:
                    counts[(fixture_id, team_key)] += 1
    return counts


def _load_statsbomb_shootout_scores(data_root: Path) -> dict[str, dict[str, int]]:
    path = data_root / "processed" / "statsbomb_world_cup_2022" / "goals.csv"
    if not path.exists():
        return {}
    scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("is_shootout") != "True":
            continue
        scores[row["match_id"]][normalise_team_name(row["team"])] += 1
    return {match_id: dict(team_scores) for match_id, team_scores in scores.items()}


def _empty_stats() -> dict[str, float]:
    return {"matches": 0, "home_gf": 0.0, "home_ga": 0.0, "away_gf": 0.0, "away_ga": 0.0}


def _side_average(stats: dict[str, float], side: str, fallback: float) -> float:
    matches = stats.get("matches", 0)
    if not matches:
        return fallback
    return stats[f"{side}_gf"] / matches


def _rolling_team_averages(stats: dict[str, float]) -> tuple[float, float, int]:
    matches = int(stats.get("matches", 0))
    if not matches:
        return 0.0, 0.0, 0
    return (
        float(stats.get("gf", 0.0)) / matches,
        float(stats.get("ga", 0.0)) / matches,
        matches,
    )


def _update_rolling_team_stats(
    stats: dict[str, dict[str, float]],
    team_key: str,
    goals_for: int,
    goals_against: int,
) -> None:
    current = stats.setdefault(
        team_key,
        {"matches": 0.0, "gf": 0.0, "ga": 0.0},
    )
    current["matches"] += 1.0
    current["gf"] += float(goals_for)
    current["ga"] += float(goals_against)


def _empty_confederation_stats() -> dict[str, float]:
    return {
        "matches": 0,
        "points": 0.0,
        "adjusted_points": 0.0,
        "gf": 0.0,
        "ga": 0.0,
    }


def _safe_float(value: Any, fallback: float = BASE_ELO) -> float:
    if value in (None, ""):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _cross_confederation_adjusted_points(
    points: int,
    goal_diff: int,
    team_elo: float,
    opponent_elo: float,
) -> float:
    """Credit underdog cross-confederation performances without overreacting."""
    underdog_gap = max(0.0, min(1.35, (opponent_elo - team_elo) / 350.0))
    if points == 3:
        bonus = 0.35 * underdog_gap
    elif points == 1:
        bonus = 0.55 * underdog_gap
    elif goal_diff == -1:
        bonus = 0.30 * underdog_gap
    elif goal_diff == -2:
        bonus = 0.12 * underdog_gap
    else:
        bonus = 0.0
    return min(3.35, float(points) + bonus)


def _confederation_strength(stats: dict[str, float], fallback: float = 1.0) -> float:
    matches = stats.get("matches", 0)
    if not matches:
        return fallback
    points_per_match = stats.get("adjusted_points", stats["points"]) / matches
    goal_diff_per_match = (stats["gf"] - stats["ga"]) / matches
    return points_per_match + 0.35 * goal_diff_per_match


def _phase_key(row: dict[str, Any]) -> str:
    competition_type = (row.get("competition_type") or "unknown").lower()
    stage = _normalise_stage_or_round(row.get("stage_or_round")).lower()
    return f"{competition_type}::{stage}"


def _competition_key(row: dict[str, Any]) -> str:
    return (row.get("competition_name") or row.get("competition_code") or "unknown").lower()


def _tournament_key(row: dict[str, Any]) -> tuple[str, str]:
    played_at = date.fromisoformat(row["date"])
    competition = row.get("competition_code") or row.get("competition_name") or "unknown"
    return (str(competition).lower(), str(played_at.year))


def _explicit_group_key(row: dict[str, Any]) -> str | None:
    stage = row.get("stage_or_round") or ""
    match = re.search(r"\bgroup\s+([a-z0-9]+)\b", stage, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    if value == "stage":
        return None
    return value


def _group_stage_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    stage = (row.get("stage_or_round") or "").lower()
    if "group" not in stage:
        return None
    group = row.get("_derived_group_key") or _explicit_group_key(row)
    if not group:
        return None
    competition, season = _tournament_key(row)
    return (competition, season, group)


def _infer_group_keys(rows: list[dict[str, Any]]) -> None:
    by_tournament: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "group" in (row.get("stage_or_round") or "").lower():
            by_tournament[_tournament_key(row)].append(row)
    for tournament, group_rows in by_tournament.items():
        explicit = [row for row in group_rows if _explicit_group_key(row)]
        if explicit:
            for row in group_rows:
                row["_derived_group_key"] = _explicit_group_key(row)
            continue
        graph: dict[str, set[str]] = defaultdict(set)
        for row in group_rows:
            home = row["home_team_id"]
            away = row["away_team_id"]
            graph[home].add(away)
            graph[away].add(home)
        component_by_team: dict[str, str] = {}
        seen: set[str] = set()
        component_index = 0
        for team in sorted(graph):
            if team in seen:
                continue
            component_index += 1
            stack = [team]
            component = []
            seen.add(team)
            while stack:
                current = stack.pop()
                component.append(current)
                for opponent in graph[current]:
                    if opponent not in seen:
                        seen.add(opponent)
                        stack.append(opponent)
            component_name = f"inferred_{component_index}"
            for item in component:
                component_by_team[item] = component_name
        if len(set(component_by_team.values())) <= 1:
            continue
        for row in group_rows:
            home_group = component_by_team.get(row["home_team_id"])
            away_group = component_by_team.get(row["away_team_id"])
            if home_group and home_group == away_group:
                row["_derived_group_key"] = home_group


def _empty_group_table_row() -> dict[str, int]:
    return {"played": 0, "points": 0, "gf": 0, "ga": 0, "position": 0}


def _rank_group_table(table: dict[str, dict[str, int]]) -> dict[str, int]:
    ordered = sorted(
        table.items(),
        key=lambda item: (
            item[1]["points"],
            item[1]["gf"] - item[1]["ga"],
            item[1]["gf"],
            item[0],
        ),
        reverse=True,
    )
    return {team_id: index + 1 for index, (team_id, _) in enumerate(ordered)}


def _group_table_features(
    group_tables: dict[tuple[str, str, str], dict[str, dict[str, int]]],
    row: dict[str, Any],
) -> dict[str, int]:
    key = _group_stage_key(row)
    table = group_tables.get(key, {}) if key else {}
    positions = _rank_group_table(table) if table else {}
    home = {**_empty_group_table_row(), **table.get(row["home_team_id"], {})}
    away = {**_empty_group_table_row(), **table.get(row["away_team_id"], {})}
    home["position"] = positions.get(row["home_team_id"], 0)
    away["position"] = positions.get(row["away_team_id"], 0)
    return {
        "home_group_matches_pre": home["played"],
        "home_group_points_pre": home["points"],
        "home_group_goal_diff_pre": home["gf"] - home["ga"],
        "home_group_goals_for_pre": home["gf"],
        "home_group_goals_against_pre": home["ga"],
        "home_group_position_pre": home["position"],
        "away_group_matches_pre": away["played"],
        "away_group_points_pre": away["points"],
        "away_group_goal_diff_pre": away["gf"] - away["ga"],
        "away_group_goals_for_pre": away["gf"],
        "away_group_goals_against_pre": away["ga"],
        "away_group_position_pre": away["position"],
        "group_points_diff_pre": home["points"] - away["points"],
        "group_goal_diff_diff_pre": (home["gf"] - home["ga"]) - (away["gf"] - away["ga"]),
        "group_position_diff_pre": home["position"] - away["position"],
    }


def _update_group_table(
    group_tables: dict[tuple[str, str, str], dict[str, dict[str, int]]],
    row: dict[str, Any],
) -> None:
    key = _group_stage_key(row)
    if key is None:
        return
    table = group_tables.setdefault(key, {})
    home = table.setdefault(row["home_team_id"], _empty_group_table_row())
    away = table.setdefault(row["away_team_id"], _empty_group_table_row())
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])
    home["played"] += 1
    home["points"] += 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
    home["gf"] += home_goals
    home["ga"] += away_goals
    away["played"] += 1
    away["points"] += 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0
    away["gf"] += away_goals
    away["ga"] += home_goals


def _team_history_key(row: dict[str, Any], side: str) -> str:
    name = (row.get(f"{side}_team") or "").strip().casefold()
    return name or row[f"{side}_team_id"]


def _is_knockout(stage: str) -> bool:
    text = (stage or "").lower()
    markers = ["round of 16", "quarter", "semi", "final", "knockout", "last 16"]
    return any(marker in text for marker in markers)


def _head_to_head_key(home_team: str, away_team: str) -> tuple[str, str]:
    return tuple(sorted((normalise_team_name(home_team), normalise_team_name(away_team))))


def _head_to_head_features(
    history: list[dict[str, Any]],
    *,
    team_a: str,
    team_b: str,
    before: date,
) -> dict[str, float | int | None]:
    team_a_key = normalise_team_name(team_a)
    recent = [
        match
        for match in history
        if 0 < (before - match["date"]).days <= H2H_LOOKBACK_DAYS
    ]
    if not recent:
        return {
            "matches": 0,
            "days_since_last": None,
            "team_a_goals_avg": None,
            "team_b_goals_avg": None,
            "goal_diff_avg": None,
            "team_a_points_avg": None,
            "team_b_points_avg": None,
            "draw_rate": None,
            "penalty_shootout_matches": 0,
            "team_a_penalty_wins": 0,
            "team_b_penalty_wins": 0,
        }
    team_a_goals: list[int] = []
    team_b_goals: list[int] = []
    team_a_points: list[int] = []
    team_b_points: list[int] = []
    draws = 0
    penalty_matches = 0
    team_a_penalty_wins = 0
    team_b_penalty_wins = 0
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
        team_a_goals.append(goals_a)
        team_b_goals.append(goals_b)
        team_a_points.append(3 if goals_a > goals_b else 1 if goals_a == goals_b else 0)
        team_b_points.append(3 if goals_b > goals_a else 1 if goals_a == goals_b else 0)
        draws += int(goals_a == goals_b)
        if penalties_a is not None and penalties_b is not None:
            penalty_matches += 1
            team_a_penalty_wins += int(penalties_a > penalties_b)
            team_b_penalty_wins += int(penalties_b > penalties_a)
    return {
        "matches": len(recent),
        "days_since_last": (before - recent[-1]["date"]).days,
        "team_a_goals_avg": sum(team_a_goals) / len(team_a_goals),
        "team_b_goals_avg": sum(team_b_goals) / len(team_b_goals),
        "goal_diff_avg": sum(a - b for a, b in zip(team_a_goals, team_b_goals, strict=True))
        / len(team_a_goals),
        "team_a_points_avg": sum(team_a_points) / len(team_a_points),
        "team_b_points_avg": sum(team_b_points) / len(team_b_points),
        "draw_rate": draws / len(recent),
        "penalty_shootout_matches": penalty_matches,
        "team_a_penalty_wins": team_a_penalty_wins,
        "team_b_penalty_wins": team_b_penalty_wins,
    }


def _with_recent_features(
    rows: list[dict[str, Any]],
    window: int = RECENT_FORM_WINDOW,
) -> list[dict[str, Any]]:
    histories: dict[str, deque[dict[str, int]]] = defaultdict(lambda: deque(maxlen=window))
    worldcup_histories: dict[str, deque[dict[str, int]]] = defaultdict(lambda: deque(maxlen=window))
    worldcup_detail_histories: dict[str, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=window)
    )
    current_worldcup_detail_histories: dict[
        tuple[str, str],
        deque[dict[str, Any]],
    ] = defaultdict(lambda: deque(maxlen=window))
    head_to_head_histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    last_played: dict[str, date] = {}
    elo_ratings: dict[str, float] = defaultdict(lambda: BASE_ELO)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        match_date = date.fromisoformat(row["date"])
        h2h_key = _head_to_head_key(row["home_team"], row["away_team"])
        h2h = _head_to_head_features(
            head_to_head_histories[h2h_key],
            team_a=row["home_team"],
            team_b=row["away_team"],
            before=match_date,
        )
        current["h2h_recent_2y_matches"] = h2h["matches"]
        current["h2h_recent_2y_days_since_last"] = h2h["days_since_last"]
        current["h2h_recent_2y_home_goals_avg"] = h2h["team_a_goals_avg"]
        current["h2h_recent_2y_away_goals_avg"] = h2h["team_b_goals_avg"]
        current["h2h_recent_2y_goal_diff_avg"] = h2h["goal_diff_avg"]
        current["h2h_recent_2y_home_points_avg"] = h2h["team_a_points_avg"]
        current["h2h_recent_2y_away_points_avg"] = h2h["team_b_points_avg"]
        current["h2h_recent_2y_draw_rate"] = h2h["draw_rate"]
        current["h2h_recent_2y_penalty_shootout_matches"] = h2h["penalty_shootout_matches"]
        current["h2h_recent_2y_home_penalty_wins"] = h2h["team_a_penalty_wins"]
        current["h2h_recent_2y_away_penalty_wins"] = h2h["team_b_penalty_wins"]
        for side, opp in (("home", "away"), ("away", "home")):
            team_key = _team_history_key(row, side)
            opponent_key = _team_history_key(row, opp)
            history = list(histories[team_key])
            # Keep the historical recent6 column names for model compatibility;
            # the actual rolling window is controlled by RECENT_FORM_WINDOW.
            prefix = f"{side}_recent6"
            current[f"{prefix}_matches"] = len(history)
            current[f"{prefix}_goals_for_avg"] = (
                sum(item["gf"] for item in history) / len(history) if history else None
            )
            current[f"{prefix}_goals_against_avg"] = (
                sum(item["ga"] for item in history) / len(history) if history else None
            )
            current[f"{prefix}_points_avg"] = (
                sum(item["points"] for item in history) / len(history) if history else None
            )
            current[f"{prefix}_win_rate"] = (
                sum(1 for item in history if item["points"] == 3) / len(history)
                if history
                else None
            )
            current[f"{prefix}_goal_diff_avg"] = (
                sum(item["gf"] - item["ga"] for item in history) / len(history)
                if history
                else None
            )
            current[f"{prefix}_opponent_elo_avg"] = (
                sum(item["opponent_elo"] for item in history) / len(history)
                if history
                else None
            )
            current[f"{prefix}_quality_result_points_avg"] = (
                sum(item["quality_result_points"] for item in history) / len(history)
                if history
                else None
            )
            current[f"{prefix}_quality_goal_balance_avg"] = (
                sum(item["quality_goal_balance"] for item in history) / len(history)
                if history
                else None
            )
            current[f"{side}_rest_days"] = (
                (match_date - last_played[team_key]).days if team_key in last_played else None
            )
            current[f"{side}_elo_pre"] = elo_ratings[team_key]
            current[f"{side}_opponent_elo_pre"] = elo_ratings[opponent_key]
            wc_history = list(worldcup_histories[team_key])
            for index in range(6):
                outcome = wc_history[-(index + 1)]["outcome"] if index < len(wc_history) else None
                current[f"{side}_worldcup_last6_{index + 1}_win"] = outcome == 1
                current[f"{side}_worldcup_last6_{index + 1}_draw"] = outcome == 0
                current[f"{side}_worldcup_last6_{index + 1}_loss"] = outcome == -1
            current[f"{side}_worldcup_recent6_win_rate"] = (
                sum(1 for item in wc_history if item["outcome"] == 1) / len(wc_history)
                if wc_history
                else None
            )
            wc_detail_history = list(worldcup_detail_histories[team_key])
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                values = [
                    item[feature]
                    for item in wc_detail_history
                    if item.get(feature) is not None
                ]
                current[f"{side}_worldcup_recent6_{feature}_avg"] = (
                    sum(values) / len(values) if values else None
                )
            current_wc_detail_history = list(
                current_worldcup_detail_histories[(row["date"][:4], team_key)]
            )
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                values = [
                    item[feature]
                    for item in current_wc_detail_history
                    if item.get(feature) is not None
                ]
                current[f"{side}_current_worldcup_recent6_{feature}_avg"] = (
                    sum(values) / len(values) if values else None
                )
            for feature in DETAIL_STAT_FEATURES:
                values = [item[feature] for item in history if item.get(feature) is not None]
                current[f"{side}_recent6_{feature}_avg"] = (
                    sum(values) / len(values) if values else None
                )
        enriched.append(current)
        for side, opp in (("home", "away"), ("away", "home")):
            team_key = _team_history_key(row, side)
            opponent_key = _team_history_key(row, opp)
            gf = int(row[f"{side}_goals"])
            ga = int(row[f"{opp}_goals"])
            opponent_elo = elo_ratings[opponent_key]
            points = 3 if gf > ga else 1 if gf == ga else 0
            history_item: dict[str, Any] = {
                "gf": gf,
                "ga": ga,
                "points": points,
                "opponent_elo": opponent_elo,
                "quality_result_points": _quality_result_points(points, gf - ga, opponent_elo),
                "quality_goal_balance": _quality_goal_balance(gf, ga, opponent_elo),
            }
            for feature in DETAIL_STAT_FEATURES:
                history_item[feature] = row.get(f"{side}_actual_{feature}")
            histories[team_key].append(history_item)
            if row["competition_name"] == "FIFA World Cup":
                outcome = 1 if gf > ga else -1 if ga > gf else 0
                worldcup_histories[team_key].append({"outcome": outcome})
                detail_coverage = row.get(f"{side}_actual_fotmob_detail_coverage")
                if detail_coverage not in (None, ""):
                    detail_item = {
                        feature: row.get(f"{side}_actual_{feature}")
                        for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
                    }
                    worldcup_detail_histories[team_key].append(detail_item)
                    current_worldcup_detail_histories[(row["date"][:4], team_key)].append(
                        detail_item
                    )
            last_played[team_key] = match_date
        home_key = _team_history_key(row, "home")
        away_key = _team_history_key(row, "away")
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        home_elo = elo_ratings[home_key]
        away_elo = elo_ratings[away_key]
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        goal_margin = min(abs(home_goals - away_goals), 3)
        k_factor = 24 * (1 + 0.15 * goal_margin)
        change = k_factor * (actual_home - expected_home)
        elo_ratings[home_key] = home_elo + change
        elo_ratings[away_key] = away_elo - change
        head_to_head_histories[h2h_key].append(
            {
                "date": match_date,
                "home_key": normalise_team_name(row["home_team"]),
                "away_key": normalise_team_name(row["away_team"]),
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "home_penalty_goals": row.get("home_penalty_goals"),
                "away_penalty_goals": row.get("away_penalty_goals"),
            }
        )
    return enriched


def train_baseline(
    data_root: Path,
    *,
    combined: bool = False,
    use_competition: bool = False,
    use_stage: bool = False,
    use_recent: bool = False,
    national_only: bool = False,
) -> dict[str, Any]:
    rows = _filter_model_rows(_load_match_rows(data_root, combined=combined))
    if national_only:
        rows = _filter_national_rows(rows)
    rows = _with_recent_features(rows, window=RECENT_FORM_WINDOW)
    rows.sort(key=lambda row: row["timestamp"])
    split = max(1, int(len(rows) * 0.8))
    train = rows[:split]
    test = rows[split:]

    global_home_goals = sum(int(row["home_goals"]) for row in train) / len(train)
    global_away_goals = sum(int(row["away_goals"]) for row in train) / len(train)

    team_stats: dict[str, dict[str, float]] = {}
    competition_stats: dict[str, dict[str, float]] = {}
    phase_stats: dict[str, dict[str, float]] = {}
    for row in train:
        competition = competition_stats.setdefault(_competition_key(row), _empty_stats())
        competition["matches"] += 1
        competition["home_gf"] += int(row["home_goals"])
        competition["home_ga"] += int(row["away_goals"])
        competition["away_gf"] += int(row["away_goals"])
        competition["away_ga"] += int(row["home_goals"])
        phase = phase_stats.setdefault(_phase_key(row), _empty_stats())
        phase["matches"] += 1
        phase["home_gf"] += int(row["home_goals"])
        phase["home_ga"] += int(row["away_goals"])
        phase["away_gf"] += int(row["away_goals"])
        phase["away_ga"] += int(row["home_goals"])
        for side, opp in (("home", "away"), ("away", "home")):
            team_id = row[f"{side}_team_id"]
            stats = team_stats.setdefault(
                team_id,
                {"matches": 0, "gf": 0.0, "ga": 0.0},
            )
            stats["matches"] += 1
            stats["gf"] += int(row[f"{side}_goals"])
            stats["ga"] += int(row[f"{opp}_goals"])

    def predict_row(row: dict[str, Any]) -> dict[str, Any]:
        home_stats = team_stats.get(row["home_team_id"], {})
        away_stats = team_stats.get(row["away_team_id"], {})
        home_attack = home_stats.get("gf", 0.0) / home_stats.get("matches", 1.0)
        home_defense = home_stats.get("ga", 0.0) / home_stats.get("matches", 1.0)
        away_attack = away_stats.get("gf", 0.0) / away_stats.get("matches", 1.0)
        away_defense = away_stats.get("ga", 0.0) / away_stats.get("matches", 1.0)
        if use_recent:
            home_attack = row["home_recent6_goals_for_avg"] or home_attack
            home_defense = row["home_recent6_goals_against_avg"] or home_defense
            away_attack = row["away_recent6_goals_for_avg"] or away_attack
            away_defense = row["away_recent6_goals_against_avg"] or away_defense
        competition = competition_stats.get(_competition_key(row), _empty_stats())
        competition_home_goals = _side_average(competition, "home", global_home_goals)
        competition_away_goals = _side_average(competition, "away", global_away_goals)
        phase = phase_stats.get(_phase_key(row), _empty_stats())
        phase_home_goals = _side_average(phase, "home", competition_home_goals)
        phase_away_goals = _side_average(phase, "away", competition_away_goals)
        if use_competition and use_stage:
            home_lambda = max(
                0.15,
                0.3 * global_home_goals
                + 0.2 * competition_home_goals
                + 0.15 * phase_home_goals
                + 0.175 * home_attack
                + 0.175 * away_defense,
            )
            away_lambda = max(
                0.15,
                0.3 * global_away_goals
                + 0.2 * competition_away_goals
                + 0.15 * phase_away_goals
                + 0.175 * away_attack
                + 0.175 * home_defense,
            )
        elif use_competition:
            home_lambda = max(
                0.15,
                0.35 * global_home_goals
                + 0.25 * competition_home_goals
                + 0.2 * home_attack
                + 0.2 * away_defense,
            )
            away_lambda = max(
                0.15,
                0.35 * global_away_goals
                + 0.25 * competition_away_goals
                + 0.2 * away_attack
                + 0.2 * home_defense,
            )
        else:
            home_lambda = max(
                0.15,
                0.5 * global_home_goals + 0.25 * home_attack + 0.25 * away_defense,
            )
            away_lambda = max(
                0.15,
                0.5 * global_away_goals + 0.25 * away_attack + 0.25 * home_defense,
            )
        probs = _score_probs(home_lambda, away_lambda)
        return {"home_lambda": home_lambda, "away_lambda": away_lambda, "probs": probs}

    calibration = train[max(1, int(len(train) * 0.8)) :]
    best_temperature = 1.0
    if calibration:
        candidates = [round(0.35 + index * 0.05, 2) for index in range(34)]
        losses: list[tuple[float, float]] = []
        for candidate in candidates:
            candidate_loss = 0.0
            for row in calibration:
                predicted = predict_row(row)
                calibrated = _calibrate_probs(predicted["probs"], candidate)
                result_key = {"home": "home_win", "draw": "draw", "away": "away_win"}[
                    row["result"]
                ]
                candidate_loss -= math.log(max(calibrated[result_key], 1e-12))
            losses.append((candidate_loss / len(calibration), candidate))
        best_temperature = min(losses)[1]

    predictions: list[dict[str, Any]] = []
    mae_home = mae_away = log_loss = correct = 0.0
    calibration_bins: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "p": 0.0, "hit": 0.0})
    for row in test:
        predicted = predict_row(row)
        home_lambda = predicted["home_lambda"]
        away_lambda = predicted["away_lambda"]
        raw_probs = predicted["probs"]
        calibrated_probs = _calibrate_probs(raw_probs, best_temperature)
        probs = {**raw_probs, **calibrated_probs}
        predicted_result = max(
            (("home", probs["home_win"]), ("draw", probs["draw"]), ("away", probs["away_win"])),
            key=lambda item: item[1],
        )[0]
        actual_result = row["result"]
        actual_home_goals = int(row["home_goals"])
        actual_away_goals = int(row["away_goals"])
        mae_home += abs(home_lambda - actual_home_goals)
        mae_away += abs(away_lambda - actual_away_goals)
        result_key = {"home": "home_win", "draw": "draw", "away": "away_win"}[actual_result]
        log_loss -= math.log(max(probs[result_key], 1e-12))
        correct += int(predicted_result == actual_result)
        predicted_probability = probs[f"{predicted_result}_win"] if predicted_result != "draw" else probs["draw"]
        bucket = min(9, int(predicted_probability * 10))
        calibration_bins[bucket]["n"] += 1
        calibration_bins[bucket]["p"] += predicted_probability
        calibration_bins[bucket]["hit"] += int(predicted_result == actual_result)
        predictions.append(
            {
                "match_id": row["match_id"],
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "competition_code": row["competition_code"],
                "competition_name": row["competition_name"],
                "competition_type": row["competition_type"],
                "stage_or_round": row["stage_or_round"],
                "matchday": row["matchday"],
                "actual_score": f"{actual_home_goals}-{actual_away_goals}",
                "actual_result": actual_result,
                "expected_home_goals": round(home_lambda, 3),
                "expected_away_goals": round(away_lambda, 3),
                "home_win": round(probs["home_win"], 4),
                "draw": round(probs["draw"], 4),
                "away_win": round(probs["away_win"], 4),
                "raw_home_win": round(raw_probs["home_win"], 4),
                "raw_draw": round(raw_probs["draw"], 4),
                "raw_away_win": round(raw_probs["away_win"], 4),
                "predicted_result": predicted_result,
                "most_likely_score": probs["most_likely_score"],
            }
        )

    models_dir = data_root / "models"
    reports_dir = data_root / "processed" / ("combined" if combined else "api_football")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "baseline_poisson_team_averages",
        "dataset": "combined" if combined else "api_football",
        "uses_competition_feature": use_competition,
        "uses_stage_feature": use_stage,
        "uses_recent6_feature": use_recent,
        "probability_temperature": best_temperature,
        "matches": len(rows),
        "train_matches": len(train),
        "calibration_matches": len(calibration),
        "test_matches": len(test),
        "global_home_goals": round(global_home_goals, 4),
        "global_away_goals": round(global_away_goals, 4),
        "mae_home_goals": round(mae_home / len(test), 4) if test else None,
        "mae_away_goals": round(mae_away / len(test), 4) if test else None,
        "result_accuracy": round(correct / len(test), 4) if test else None,
        "log_loss": round(log_loss / len(test), 4) if test else None,
    }
    if combined and use_competition and use_stage and use_recent:
        model_name = "baseline_poisson_combined_recent6.json"
    elif combined and use_competition and use_stage:
        model_name = "baseline_poisson_combined_stage.json"
    elif combined and use_competition:
        model_name = "baseline_poisson_combined_competition.json"
    else:
        model_name = "baseline_poisson_combined.json" if combined else "baseline_poisson.json"
    (models_dir / model_name).write_text(
        json.dumps(
            {
                "metrics": metrics,
                "team_stats": team_stats,
                "competition_stats": competition_stats,
                "phase_stats": phase_stats,
                "calibration_bins": {
                    str(bucket): {
                        "n": values["n"],
                        "avg_predicted_probability": round(values["p"] / values["n"], 4)
                        if values["n"]
                        else None,
                        "empirical_accuracy": round(values["hit"] / values["n"], 4)
                        if values["n"]
                        else None,
                    }
                    for bucket, values in sorted(calibration_bins.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (reports_dir / "baseline_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(predictions[0].keys()) if predictions else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    return metrics


def export_training_frame(
    data_root: Path,
    *,
    combined: bool = True,
    use_competition: bool = True,
    use_stage: bool = True,
    use_recent: bool = True,
    national_only: bool = False,
    live_fifa_friendly_weight: float = FRIENDLY_MATCH_WEIGHT,
) -> list[dict[str, Any]]:
    rows = _filter_model_rows(_load_match_rows(data_root, combined=combined))
    if national_only:
        rows = _filter_national_rows(rows)
    _infer_group_keys(rows)
    rows = _with_recent_features(rows, window=RECENT_FORM_WINDOW)
    latest_date = max(date.fromisoformat(row["date"]) for row in rows)
    split = max(1, int(len(rows) * 0.8))
    train = rows[:split]
    current_worldcup_squads = (
        _load_current_worldcup_squads(data_root) if ENABLE_SQUAD_CONTINUITY_WEIGHT else {}
    )
    starting_lineups = _load_match_starting_lineups(data_root) if ENABLE_SQUAD_CONTINUITY_WEIGHT else {}

    global_home_goals = sum(int(row["home_goals"]) for row in train) / len(train)
    global_away_goals = sum(int(row["away_goals"]) for row in train) / len(train)
    team_cross_confederation_stats: dict[str, dict[str, float]] = {}
    confederation_stats: dict[str, dict[str, float]] = {}
    competition_stats: dict[str, dict[str, float]] = {}
    phase_stats: dict[str, dict[str, float]] = {}
    fifa_rankings = load_fifa_ranking(data_root)
    fifa_ranking_history = load_fifa_ranking_history(data_root)
    fallback_fifa_rank = max((int(item["rank"]) for item in fifa_rankings.values()), default=211) + 25
    fallback_fifa_points = min(
        (float(item["points"]) for item in fifa_rankings.values()),
        default=800.0,
    )

    def historical_ranking_features(team: str, match_date: date) -> dict[str, Any]:
        if fifa_ranking_history:
            historical = fifa_ranking_for_match_date(fifa_ranking_history, team, match_date)
            if historical:
                return {
                    "rank": int(historical.get("rank") or fallback_fifa_rank),
                    "points": float(historical.get("points") or fallback_fifa_points),
                    "schedule_id": historical.get("schedule_id", ""),
                    "effective_date": historical.get("effective_date", ""),
                    "observed": True,
                }
            return {
                "rank": fallback_fifa_rank,
                "points": fallback_fifa_points,
                "schedule_id": "",
                "effective_date": "",
                "observed": False,
            }
        latest = latest_ranking_features(team)
        return {
            "rank": latest["rank"],
            "points": latest["points"],
            "schedule_id": "",
            "effective_date": "",
            "observed": False,
        }

    def latest_ranking_features(team: str) -> dict[str, float | int]:
        latest = fifa_rankings.get(normalise_team_name(team), {})
        return {
            "rank": int(latest.get("rank", fallback_fifa_rank)),
            "points": float(latest.get("points", fallback_fifa_points)),
        }

    live_fifa_points: dict[str, float] = {}
    live_fifa_schedule: dict[str, str] = {}

    def live_ranking_points(team: str, official: dict[str, Any]) -> float:
        key = normalise_team_name(team)
        schedule_id = str(official.get("schedule_id", ""))
        if key not in live_fifa_points or live_fifa_schedule.get(key) != schedule_id:
            live_fifa_points[key] = float(official["points"])
            live_fifa_schedule[key] = schedule_id
        return live_fifa_points[key]

    for row in train:
        competition = competition_stats.setdefault(_competition_key(row), _empty_stats())
        competition["matches"] += 1
        competition["home_gf"] += int(row["home_goals"])
        competition["home_ga"] += int(row["away_goals"])
        competition["away_gf"] += int(row["away_goals"])
        competition["away_ga"] += int(row["home_goals"])
        phase = phase_stats.setdefault(_phase_key(row), _empty_stats())
        phase["matches"] += 1
        phase["home_gf"] += int(row["home_goals"])
        phase["home_ga"] += int(row["away_goals"])
        phase["away_gf"] += int(row["away_goals"])
        phase["away_ga"] += int(row["home_goals"])
        for side, opp in (("home", "away"), ("away", "home")):
            team_id = row[f"{side}_team_id"]
            team_confederation = _team_confederation(row[f"{side}_team"])
            opponent_confederation = _team_confederation(row[f"{opp}_team"])
            goals_for = int(row[f"{side}_goals"])
            goals_against = int(row[f"{opp}_goals"])
            points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            goal_diff = goals_for - goals_against
            adjusted_points = float(points)
            confed = confederation_stats.setdefault(
                team_confederation,
                _empty_confederation_stats(),
            )
            confed["matches"] += 1
            confed["points"] += points
            confed["adjusted_points"] += adjusted_points
            confed["gf"] += goals_for
            confed["ga"] += goals_against
            if (
                team_confederation != "unknown"
                and opponent_confederation != "unknown"
                and team_confederation != opponent_confederation
            ):
                adjusted_points = _cross_confederation_adjusted_points(
                    points,
                    goal_diff,
                    _safe_float(row.get(f"{side}_elo_pre")),
                    _safe_float(row.get(f"{opp}_elo_pre")),
                )
                confed["adjusted_points"] += adjusted_points - points
                cross = team_cross_confederation_stats.setdefault(
                    team_id,
                    _empty_confederation_stats(),
                )
                cross["matches"] += 1
                cross["points"] += points
                cross["adjusted_points"] += adjusted_points
                cross["gf"] += goals_for
                cross["ga"] += goals_against

    exported: list[dict[str, Any]] = []
    group_tables: dict[tuple[str, str, str], dict[str, dict[str, int]]] = {}
    rolling_team_stats: dict[str, dict[str, float]] = {}
    for index, row in enumerate(rows):
        match_date = date.fromisoformat(row["date"])
        group_table = _group_table_features(group_tables, row)
        squad_continuity_share, squad_continuity_weight = _squad_continuity_weight(
            row,
            latest_date,
            current_worldcup_squads,
            starting_lineups,
        )
        home_history_key = normalise_team_name(row["home_team"])
        away_history_key = normalise_team_name(row["away_team"])
        home_stats = rolling_team_stats.get(home_history_key, {})
        away_stats = rolling_team_stats.get(away_history_key, {})
        home_cross_stats = team_cross_confederation_stats.get(row["home_team_id"], {})
        away_cross_stats = team_cross_confederation_stats.get(row["away_team_id"], {})
        home_confederation = _team_confederation(row["home_team"])
        away_confederation = _team_confederation(row["away_team"])
        home_confederation_strength = _confederation_strength(
            confederation_stats.get(home_confederation, {}),
        )
        away_confederation_strength = _confederation_strength(
            confederation_stats.get(away_confederation, {}),
        )
        home_cross_strength = _confederation_strength(home_cross_stats)
        away_cross_strength = _confederation_strength(away_cross_stats)
        home_fifa = latest_ranking_features(row["home_team"])
        away_fifa = latest_ranking_features(row["away_team"])
        home_historical_fifa = historical_ranking_features(row["home_team"], match_date)
        away_historical_fifa = historical_ranking_features(row["away_team"], match_date)
        home_live_fifa_points = live_ranking_points(
            row["home_team"],
            home_historical_fifa,
        )
        away_live_fifa_points = live_ranking_points(
            row["away_team"],
            away_historical_fifa,
        )
        home_attack, home_defense, home_matches = _rolling_team_averages(home_stats)
        away_attack, away_defense, away_matches = _rolling_team_averages(away_stats)
        home_recent_attack = row["home_recent6_goals_for_avg"]
        home_recent_defense = row["home_recent6_goals_against_avg"]
        away_recent_attack = row["away_recent6_goals_for_avg"]
        away_recent_defense = row["away_recent6_goals_against_avg"]
        model_home_attack = home_recent_attack if use_recent and home_recent_attack is not None else home_attack
        model_home_defense = (
            home_recent_defense if use_recent and home_recent_defense is not None else home_defense
        )
        model_away_attack = away_recent_attack if use_recent and away_recent_attack is not None else away_attack
        model_away_defense = (
            away_recent_defense if use_recent and away_recent_defense is not None else away_defense
        )
        competition = competition_stats.get(_competition_key(row), _empty_stats())
        competition_home_goals = _side_average(competition, "home", global_home_goals)
        competition_away_goals = _side_average(competition, "away", global_away_goals)
        phase = phase_stats.get(_phase_key(row), _empty_stats())
        phase_home_goals = _side_average(phase, "home", competition_home_goals)
        phase_away_goals = _side_average(phase, "away", competition_away_goals)
        if use_competition and use_stage:
            home_lambda = max(
                0.15,
                0.3 * global_home_goals
                + 0.2 * competition_home_goals
                + 0.15 * phase_home_goals
                + 0.175 * model_home_attack
                + 0.175 * model_away_defense,
            )
            away_lambda = max(
                0.15,
                0.3 * global_away_goals
                + 0.2 * competition_away_goals
                + 0.15 * phase_away_goals
                + 0.175 * model_away_attack
                + 0.175 * model_home_defense,
            )
        elif use_competition:
            home_lambda = max(
                0.15,
                0.35 * global_home_goals
                + 0.25 * competition_home_goals
                + 0.2 * model_home_attack
                + 0.2 * model_away_defense,
            )
            away_lambda = max(
                0.15,
                0.35 * global_away_goals
                + 0.25 * competition_away_goals
                + 0.2 * model_away_attack
                + 0.2 * model_home_defense,
            )
        else:
            home_lambda = max(
                0.15,
                0.5 * global_home_goals + 0.25 * home_attack + 0.25 * away_defense,
            )
            away_lambda = max(
                0.15,
                0.5 * global_away_goals + 0.25 * away_attack + 0.25 * home_defense,
            )
        probs = _score_probs(home_lambda, away_lambda)
        predicted_result = max(
            (("home", probs["home_win"]), ("draw", probs["draw"]), ("away", probs["away_win"])),
            key=lambda item: item[1],
        )[0]
        export_row = {
            "row_index": index + 1,
            "split": "train" if index < split else "test",
            "source": row["source"],
            "match_id": row["match_id"],
            "date": row["date"],
            "match_recency_weight": round(
                _training_recency_weight(row["date"], latest_date)
                * _match_anomaly_weight(row)
                * squad_continuity_weight
                * (FRIENDLY_MATCH_WEIGHT if row["competition_type"] == "friendly" else 1.0),
                4,
            ),
            "squad_continuity_share": squad_continuity_share,
            "squad_continuity_weight": squad_continuity_weight,
            "competition_code": row["competition_code"],
            "competition_name": row["competition_name"],
            "competition_family": _competition_family(row),
            "competition_type": row["competition_type"],
            "stage_or_round": _normalise_stage_or_round(row["stage_or_round"]),
            "matchday": row["matchday"],
            "is_friendly": row["competition_type"] == "friendly",
            "is_qualifier": row["competition_type"] == "qualifier",
            "is_knockout": _is_knockout(row["stage_or_round"]),
            "home_is_tournament_host": _is_tournament_host(row, "home"),
            "away_is_tournament_host": _is_tournament_host(row, "away"),
            "home_team_id": row["home_team_id"],
            "home_team": row["home_team"],
            "home_confederation": home_confederation,
            "away_team_id": row["away_team_id"],
            "away_team": row["away_team"],
            "away_confederation": away_confederation,
            "same_confederation": home_confederation == away_confederation,
            **group_table,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "home_penalty_goals": _optional_int(row.get("home_penalty_goals")),
            "away_penalty_goals": _optional_int(row.get("away_penalty_goals")),
            "total_goals": int(row["home_goals"]) + int(row["away_goals"]),
            "result": row["result"],
            "global_home_goals_train": round(global_home_goals, 4),
            "global_away_goals_train": round(global_away_goals, 4),
            "competition_train_matches": int(competition.get("matches", 0)),
            "competition_home_goals_avg": round(competition_home_goals, 4),
            "competition_away_goals_avg": round(competition_away_goals, 4),
            "phase_train_matches": int(phase.get("matches", 0)),
            "phase_home_goals_avg": round(phase_home_goals, 4),
            "phase_away_goals_avg": round(phase_away_goals, 4),
            "home_team_train_matches": int(home_matches),
            "home_team_goals_for_avg": round(home_attack, 4),
            "home_team_goals_against_avg": round(home_defense, 4),
            "home_confederation_strength": round(home_confederation_strength, 4),
            "home_cross_confederation_matches": int(home_cross_stats.get("matches", 0)),
            "home_cross_confederation_strength": round(home_cross_strength, 4),
            "home_recent6_matches": row["home_recent6_matches"],
            "home_recent6_goals_for_avg": round(home_recent_attack, 4)
            if home_recent_attack is not None
            else None,
            "home_recent6_goals_against_avg": round(home_recent_defense, 4)
            if home_recent_defense is not None
            else None,
            "home_recent6_points_avg": round(row["home_recent6_points_avg"], 4)
            if row["home_recent6_points_avg"] is not None
            else None,
            "home_recent6_win_rate": round(row["home_recent6_win_rate"], 4)
            if row["home_recent6_win_rate"] is not None
            else None,
            "home_recent6_goal_diff_avg": round(row["home_recent6_goal_diff_avg"], 4)
            if row["home_recent6_goal_diff_avg"] is not None
            else None,
            "home_recent6_opponent_elo_avg": round(row["home_recent6_opponent_elo_avg"], 4)
            if row["home_recent6_opponent_elo_avg"] is not None
            else None,
            "home_recent6_quality_result_points_avg": round(
                row["home_recent6_quality_result_points_avg"], 4
            )
            if row["home_recent6_quality_result_points_avg"] is not None
            else None,
            "home_recent6_quality_goal_balance_avg": round(
                row["home_recent6_quality_goal_balance_avg"], 4
            )
            if row["home_recent6_quality_goal_balance_avg"] is not None
            else None,
            "home_rest_days": row["home_rest_days"],
            "home_elo_pre": round(row["home_elo_pre"], 4),
            "home_opponent_elo_pre": round(row["home_opponent_elo_pre"], 4),
            "home_worldcup_recent6_win_rate": round(row["home_worldcup_recent6_win_rate"], 4)
            if row["home_worldcup_recent6_win_rate"] is not None
            else None,
            "home_fifa_rank": int(home_fifa["rank"]),
            "home_fifa_points": round(float(home_fifa["points"]), 2),
            "home_historical_fifa_rank": int(home_historical_fifa["rank"]),
            "home_historical_fifa_points": round(float(home_historical_fifa["points"]), 2),
            "home_live_fifa_points": round(home_live_fifa_points, 4),
            "home_historical_fifa_schedule_id": home_historical_fifa["schedule_id"],
            "home_historical_fifa_effective_date": str(home_historical_fifa["effective_date"]),
            "home_historical_fifa_observed": int(home_historical_fifa["observed"]),
            "away_team_train_matches": int(away_matches),
            "away_team_goals_for_avg": round(away_attack, 4),
            "away_team_goals_against_avg": round(away_defense, 4),
            "away_confederation_strength": round(away_confederation_strength, 4),
            "away_cross_confederation_matches": int(away_cross_stats.get("matches", 0)),
            "away_cross_confederation_strength": round(away_cross_strength, 4),
            "away_recent6_matches": row["away_recent6_matches"],
            "away_recent6_goals_for_avg": round(away_recent_attack, 4)
            if away_recent_attack is not None
            else None,
            "away_recent6_goals_against_avg": round(away_recent_defense, 4)
            if away_recent_defense is not None
            else None,
            "away_recent6_points_avg": round(row["away_recent6_points_avg"], 4)
            if row["away_recent6_points_avg"] is not None
            else None,
            "away_recent6_win_rate": round(row["away_recent6_win_rate"], 4)
            if row["away_recent6_win_rate"] is not None
            else None,
            "away_recent6_goal_diff_avg": round(row["away_recent6_goal_diff_avg"], 4)
            if row["away_recent6_goal_diff_avg"] is not None
            else None,
            "away_recent6_opponent_elo_avg": round(row["away_recent6_opponent_elo_avg"], 4)
            if row["away_recent6_opponent_elo_avg"] is not None
            else None,
            "away_recent6_quality_result_points_avg": round(
                row["away_recent6_quality_result_points_avg"], 4
            )
            if row["away_recent6_quality_result_points_avg"] is not None
            else None,
            "away_recent6_quality_goal_balance_avg": round(
                row["away_recent6_quality_goal_balance_avg"], 4
            )
            if row["away_recent6_quality_goal_balance_avg"] is not None
            else None,
            "away_rest_days": row["away_rest_days"],
            "away_elo_pre": round(row["away_elo_pre"], 4),
            "away_opponent_elo_pre": round(row["away_opponent_elo_pre"], 4),
            "away_worldcup_recent6_win_rate": round(row["away_worldcup_recent6_win_rate"], 4)
            if row["away_worldcup_recent6_win_rate"] is not None
            else None,
            "away_fifa_rank": int(away_fifa["rank"]),
            "away_fifa_points": round(float(away_fifa["points"]), 2),
            "away_historical_fifa_rank": int(away_historical_fifa["rank"]),
            "away_historical_fifa_points": round(float(away_historical_fifa["points"]), 2),
            "away_live_fifa_points": round(away_live_fifa_points, 4),
            "away_historical_fifa_schedule_id": away_historical_fifa["schedule_id"],
            "away_historical_fifa_effective_date": str(away_historical_fifa["effective_date"]),
            "away_historical_fifa_observed": int(away_historical_fifa["observed"]),
            "h2h_recent_2y_matches": row["h2h_recent_2y_matches"],
            "h2h_recent_2y_days_since_last": row["h2h_recent_2y_days_since_last"],
            "h2h_recent_2y_home_goals_avg": round(row["h2h_recent_2y_home_goals_avg"], 4)
            if row["h2h_recent_2y_home_goals_avg"] is not None
            else None,
            "h2h_recent_2y_away_goals_avg": round(row["h2h_recent_2y_away_goals_avg"], 4)
            if row["h2h_recent_2y_away_goals_avg"] is not None
            else None,
            "h2h_recent_2y_goal_diff_avg": round(row["h2h_recent_2y_goal_diff_avg"], 4)
            if row["h2h_recent_2y_goal_diff_avg"] is not None
            else None,
            "h2h_recent_2y_home_points_avg": round(row["h2h_recent_2y_home_points_avg"], 4)
            if row["h2h_recent_2y_home_points_avg"] is not None
            else None,
            "h2h_recent_2y_away_points_avg": round(row["h2h_recent_2y_away_points_avg"], 4)
            if row["h2h_recent_2y_away_points_avg"] is not None
            else None,
            "h2h_recent_2y_draw_rate": round(row["h2h_recent_2y_draw_rate"], 4)
            if row["h2h_recent_2y_draw_rate"] is not None
            else None,
            "h2h_recent_2y_penalty_shootout_matches": row[
                "h2h_recent_2y_penalty_shootout_matches"
            ],
            "h2h_recent_2y_home_penalty_wins": row["h2h_recent_2y_home_penalty_wins"],
            "h2h_recent_2y_away_penalty_wins": row["h2h_recent_2y_away_penalty_wins"],
            "expected_home_goals": round(home_lambda, 4),
            "expected_away_goals": round(away_lambda, 4),
            "home_win_probability": round(probs["home_win"], 6),
            "draw_probability": round(probs["draw"], 6),
            "away_win_probability": round(probs["away_win"], 6),
            "predicted_result": predicted_result,
            "most_likely_score": probs["most_likely_score"],
        }
        for side in ("home", "away"):
            for feature in DETAIL_STAT_FEATURES:
                actual_value = row.get(f"{side}_actual_{feature}")
                export_row[f"{side}_actual_{feature}"] = (
                    round(float(actual_value), 4) if actual_value is not None else None
                )
                value = row[f"{side}_recent6_{feature}_avg"]
                export_row[f"{side}_recent6_{feature}_avg"] = (
                    round(value, 4) if value is not None else None
                )
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES:
                value = row[f"{side}_worldcup_recent6_{feature}_avg"]
                export_row[f"{side}_worldcup_recent6_{feature}_avg"] = (
                    round(value, 4) if value is not None else None
                )
                value = row[f"{side}_current_worldcup_recent6_{feature}_avg"]
                export_row[f"{side}_current_worldcup_recent6_{feature}_avg"] = (
                    round(value, 4) if value is not None else None
                )
            for worldcup_index in range(1, 7):
                for outcome in ("win", "draw", "loss"):
                    export_row[f"{side}_worldcup_last6_{worldcup_index}_{outcome}"] = row[
                        f"{side}_worldcup_last6_{worldcup_index}_{outcome}"
                    ]
        exported.append(export_row)
        home_key = normalise_team_name(row["home_team"])
        away_key = normalise_team_name(row["away_team"])
        updated_home_points, updated_away_points = update_live_fifa_points(
            home_live_fifa_points,
            away_live_fifa_points,
            int(row["home_goals"]),
            int(row["away_goals"]),
            importance_weight=(
                live_fifa_friendly_weight
                if row.get("competition_type") == "friendly"
                else 1.0
            ),
        )
        live_fifa_points[home_key] = updated_home_points
        live_fifa_points[away_key] = updated_away_points
        for team_key, goals_for, goals_against in (
            (home_history_key, int(row["home_goals"]), int(row["away_goals"])),
            (away_history_key, int(row["away_goals"]), int(row["home_goals"])),
        ):
            _update_rolling_team_stats(
                rolling_team_stats,
                team_key,
                goals_for,
                goals_against,
            )
        _update_group_table(group_tables, row)
    return exported


def export_clean_training_matrix(
    data_root: Path,
    *,
    combined: bool = True,
    use_competition: bool = True,
    use_stage: bool = True,
    national_only: bool = False,
    live_fifa_friendly_weight: float = FRIENDLY_MATCH_WEIGHT,
) -> list[dict[str, Any]]:
    full = export_training_frame(
        data_root,
        combined=combined,
        use_competition=use_competition,
        use_stage=use_stage,
        national_only=national_only,
        live_fifa_friendly_weight=live_fifa_friendly_weight,
    )
    clean_columns = [
        "split",
        "match_recency_weight",
        "competition_name",
        "competition_family",
        "competition_type",
        "stage_or_round",
        "home_confederation",
        "away_confederation",
        "same_confederation",
        "is_friendly",
        "is_qualifier",
        "is_knockout",
        "home_is_tournament_host",
        "away_is_tournament_host",
        "home_group_matches_pre",
        "home_group_points_pre",
        "home_group_goal_diff_pre",
        "home_group_goals_for_pre",
        "home_group_goals_against_pre",
        "home_group_position_pre",
        "away_group_matches_pre",
        "away_group_points_pre",
        "away_group_goal_diff_pre",
        "away_group_goals_for_pre",
        "away_group_goals_against_pre",
        "away_group_position_pre",
        "group_points_diff_pre",
        "group_goal_diff_diff_pre",
        "group_position_diff_pre",
        "global_home_goals_train",
        "global_away_goals_train",
        "competition_train_matches",
        "competition_home_goals_avg",
        "competition_away_goals_avg",
        "phase_train_matches",
        "phase_home_goals_avg",
        "phase_away_goals_avg",
        "home_team_train_matches",
        "home_team_goals_for_avg",
        "home_team_goals_against_avg",
        "home_confederation_strength",
        "home_cross_confederation_matches",
        "home_cross_confederation_strength",
        "away_team_train_matches",
        "away_team_goals_for_avg",
        "away_team_goals_against_avg",
        "away_confederation_strength",
        "away_cross_confederation_matches",
        "away_cross_confederation_strength",
        "home_recent6_matches",
        "home_recent6_goals_for_avg",
        "home_recent6_goals_against_avg",
        "home_recent6_points_avg",
        "home_recent6_win_rate",
        "home_recent6_goal_diff_avg",
        "home_recent6_opponent_elo_avg",
        "home_recent6_quality_result_points_avg",
        "home_recent6_quality_goal_balance_avg",
        "away_recent6_matches",
        "away_recent6_goals_for_avg",
        "away_recent6_goals_against_avg",
        "away_recent6_points_avg",
        "away_recent6_win_rate",
        "away_recent6_goal_diff_avg",
        "away_recent6_opponent_elo_avg",
        "away_recent6_quality_result_points_avg",
        "away_recent6_quality_goal_balance_avg",
        "home_rest_days",
        "away_rest_days",
        "home_elo_pre",
        "home_opponent_elo_pre",
        "away_elo_pre",
        "away_opponent_elo_pre",
        "home_worldcup_recent6_win_rate",
        "away_worldcup_recent6_win_rate",
        "home_fifa_rank",
        "home_fifa_points",
        "home_historical_fifa_rank",
        "home_historical_fifa_points",
        "home_historical_fifa_observed",
        "home_live_fifa_points",
        "away_fifa_rank",
        "away_fifa_points",
        "away_historical_fifa_rank",
        "away_historical_fifa_points",
        "away_historical_fifa_observed",
        "away_live_fifa_points",
        "h2h_recent_2y_matches",
        "h2h_recent_2y_days_since_last",
        "h2h_recent_2y_home_goals_avg",
        "h2h_recent_2y_away_goals_avg",
        "h2h_recent_2y_goal_diff_avg",
        "h2h_recent_2y_home_points_avg",
        "h2h_recent_2y_away_points_avg",
        "h2h_recent_2y_draw_rate",
        "h2h_recent_2y_penalty_shootout_matches",
        "h2h_recent_2y_home_penalty_wins",
        "h2h_recent_2y_away_penalty_wins",
        "home_goals",
        "away_goals",
        "total_goals",
        "result",
        "expected_home_goals",
        "expected_away_goals",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "predicted_result",
        "most_likely_score",
    ]
    for side in ("home", "away"):
        insert_at = clean_columns.index(f"{side}_rest_days") + 1
        detail_columns = [f"{side}_recent6_{feature}_avg" for feature in DETAIL_STAT_FEATURES]
        clean_columns[insert_at:insert_at] = detail_columns
    for side in ("home", "away"):
        insert_at = clean_columns.index(f"{side}_worldcup_recent6_win_rate") + 1
        worldcup_detail_columns = [
            f"{side}_worldcup_recent6_{feature}_avg"
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
        ]
        worldcup_detail_columns.extend(
            f"{side}_current_worldcup_recent6_{feature}_avg"
            for feature in FOTMOB_WORLD_CUP_DETAIL_FEATURES
        )
        clean_columns[insert_at:insert_at] = worldcup_detail_columns
    for side in ("home", "away"):
        insert_at = clean_columns.index(f"{side}_worldcup_recent6_win_rate")
        boolean_columns: list[str] = []
        for index in range(1, 7):
            boolean_columns.extend(
                [
                    f"{side}_worldcup_last6_{index}_win",
                    f"{side}_worldcup_last6_{index}_draw",
                    f"{side}_worldcup_last6_{index}_loss",
                ]
            )
        clean_columns[insert_at:insert_at] = boolean_columns
    return [{column: row[column] for column in clean_columns} for row in full]
