from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


FINISHED_STATUSES = {"FT", "AET", "PEN"}
API_TEAM_STAT_TYPES = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession_pct",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "passes_pct",
    "expected_goals": "expected_goals",
    "goals_prevented": "goals_prevented",
}
SQUAD_REFERENCE_DATE = date(2026, 6, 11)
COMPETITION_STRENGTH = {
    "CL": 10.0,
    "PL": 9.3,
    "PD": 9.1,
    "SA": 8.8,
    "BL1": 8.7,
    "FL1": 8.2,
    "CLI": 8.0,
    "DED": 7.4,
    "PPL": 7.2,
    "ELC": 7.0,
    "BSA": 7.0,
    "MLS": 6.8,
    "EL": 6.8,
    "WC": 6.5,
    "EC": 6.5,
}


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _competition_type(league_id: int | None, league_name: str, round_name: str) -> str:
    text = f"{league_name} {round_name}".lower()
    if league_id == 10 or "friendlies" in text or "friendly" in text:
        return "friendly"
    if "qualifi" in text:
        return "qualifier"
    if "world cup" in text or "euro championship" in text or "copa america" in text:
        return "major_tournament"
    return "competition"


def _competition_stage(round_name: str) -> str:
    text = round_name.lower()
    if "qualification" in text or "qualifying" in text or "qualifiers" in text:
        return round_name
    if "group" in text:
        return "group_stage"
    if "round of 16" in text:
        return "round_of_16"
    if "quarter" in text:
        return "quarter_final"
    if "semi" in text:
        return "semi_final"
    if "third" in text:
        return "third_place"
    if "final" in text:
        return "final"
    if "friendlies" in text or "friendly" in text:
        return "friendly"
    return round_name or "unknown"


def _read_api_football_fixtures(raw_dir: Path) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for path in (raw_dir / "api_football" / "fixtures").glob("team-*-season-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("response", []):
            fixture_id = int(item["fixture"]["id"])
            by_id[fixture_id] = item

    for path in (raw_dir / "api_football" / "fixtures").glob("details-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("response"):
            item = payload["response"][0]
            by_id[int(item["fixture"]["id"])] = item

    fixtures = [
        item
        for item in by_id.values()
        if item.get("fixture", {}).get("status", {}).get("short") in FINISHED_STATUSES
        and item.get("goals", {}).get("home") is not None
        and item.get("goals", {}).get("away") is not None
    ]
    fixtures.sort(key=lambda item: item["fixture"]["date"])
    return fixtures


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)
    return len(rows)


def _stat_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            value = value[:-1]
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_on(date_of_birth: str | None, reference_date: date = SQUAD_REFERENCE_DATE) -> float | None:
    if not date_of_birth:
        return None
    try:
        born = date.fromisoformat(date_of_birth)
    except ValueError:
        return None
    return round((reference_date - born).days / 365.25, 2)


def _competition_strength(code: str | None, name: str | None) -> float:
    if code in COMPETITION_STRENGTH:
        return COMPETITION_STRENGTH[code]
    text = (name or "").lower()
    if "champions league" in text:
        return 10.0
    if "premier league" in text:
        return 9.3
    if "laliga" in text or "primera division" in text:
        return 9.1
    if "serie a" in text:
        return 8.8
    if "bundesliga" in text:
        return 8.7
    if "ligue 1" in text:
        return 8.2
    if "libertadores" in text:
        return 8.0
    if "major league soccer" in text or "mls" in text:
        return 6.8
    return 4.5


def _load_player_competition_strengths(raw_dir: Path) -> dict[int, dict[str, float]]:
    strengths: dict[int, dict[str, float]] = {}
    persons_dir = raw_dir / "football_data" / "persons"
    for path in persons_dir.glob("*/matches-last-*.json"):
        try:
            player_id = int(path.parent.name)
        except ValueError:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = [
            _competition_strength(
                (match.get("competition") or {}).get("code"),
                (match.get("competition") or {}).get("name"),
            )
            for match in payload.get("matches", [])
            if match.get("status") == "FINISHED"
        ]
        strengths[player_id] = {
            "matches": float(len(values)),
            "avg_strength": round(sum(values) / len(values), 4) if values else 0.0,
            "max_strength": round(max(values), 4) if values else 0.0,
            "top_competition_share": round(sum(1 for value in values if value >= 8.5) / len(values), 4)
            if values
            else 0.0,
            "elite_competition_share": round(sum(1 for value in values if value >= 9.5) / len(values), 4)
            if values
            else 0.0,
        }
    return strengths


def build_api_football_etl(data_root: Path) -> dict[str, int]:
    fixtures = _read_api_football_fixtures(data_root / "raw")
    processed = data_root / "processed" / "api_football"

    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    detailed_team_rows: list[dict[str, Any]] = []
    histories: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))
    last_played: dict[int, datetime] = {}

    for item in fixtures:
        fixture = item["fixture"]
        league = item["league"]
        teams = item["teams"]
        goals = item["goals"]
        score = item.get("score", {})
        penalty_score = score.get("penalty") or {}
        fixture_id = int(fixture["id"])
        played_at = _parse_dt(fixture["date"])
        round_name = league.get("round") or ""
        league_id = league.get("id")
        competition_type = _competition_type(league_id, league.get("name") or "", round_name)
        competition_stage = _competition_stage(round_name)
        home = teams["home"]
        away = teams["away"]
        home_goals = int(goals["home"])
        away_goals = int(goals["away"])
        total_goals = home_goals + away_goals
        result = "draw"
        if home_goals > away_goals:
            result = "home"
        elif away_goals > home_goals:
            result = "away"

        match_rows.append(
            {
                "fixture_id": fixture_id,
                "date": played_at.date().isoformat(),
                "timestamp": int(fixture["timestamp"]),
                "league_id": league_id,
                "league_name": league.get("name"),
                "league_season": league.get("season"),
                "round": round_name,
                "competition_type": competition_type,
                "competition_stage": competition_stage,
                "home_team_id": home["id"],
                "home_team": home["name"],
                "away_team_id": away["id"],
                "away_team": away["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_penalty_goals": penalty_score.get("home"),
                "away_penalty_goals": penalty_score.get("away"),
                "total_goals": total_goals,
                "result": result,
            }
        )

        sides = [
            (home, away, True, home_goals, away_goals),
            (away, home, False, away_goals, home_goals),
        ]
        current_match_team_rows: list[dict[str, Any]] = []
        for team, opponent, is_home, goals_for, goals_against in sides:
            team_id = int(team["id"])
            history = list(histories[team_id])
            last5 = history[-5:]
            rest_days = None
            if team_id in last_played:
                rest_days = (played_at.date() - last_played[team_id].date()).days
            row = {
                "fixture_id": fixture_id,
                "date": played_at.date().isoformat(),
                "team_id": team_id,
                "team": team["name"],
                "opponent_id": opponent["id"],
                "opponent": opponent["name"],
                "is_home": is_home,
                "league_id": league_id,
                "league_name": league.get("name"),
                "round": round_name,
                "competition_type": competition_type,
                "competition_stage": competition_stage,
                "rest_days": rest_days,
                "prior_matches": len(history),
                "prior5_goals_for_avg": round(sum(h["gf"] for h in last5) / len(last5), 4)
                if last5
                else None,
                "prior5_goals_against_avg": round(sum(h["ga"] for h in last5) / len(last5), 4)
                if last5
                else None,
                "prior5_points_avg": round(sum(h["points"] for h in last5) / len(last5), 4)
                if last5
                else None,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "points": 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0,
            }
            team_rows.append(row)
            current_match_team_rows.append(row)

        for team_stats in item.get("statistics") or []:
            team = team_stats.get("team") or {}
            detail_row: dict[str, Any] = {
                "fixture_id": fixture_id,
                "date": played_at.date().isoformat(),
                "team_id": team.get("id"),
                "team": team.get("name"),
            }
            values = {
                API_TEAM_STAT_TYPES[stat["type"]]: _stat_value(stat.get("value"))
                for stat in team_stats.get("statistics") or []
                if stat.get("type") in API_TEAM_STAT_TYPES
            }
            for column in API_TEAM_STAT_TYPES.values():
                detail_row[column] = values.get(column)
            detailed_team_rows.append(detail_row)

        for row in current_match_team_rows:
            team_id = int(row["team_id"])
            histories[team_id].append(
                {
                    "fixture_id": fixture_id,
                    "gf": row["goals_for"],
                    "ga": row["goals_against"],
                    "points": row["points"],
                }
            )
            last_played[team_id] = played_at

    counts = {
        "matches": _write_csv(
            processed / "matches.csv",
            match_rows,
            [
                "fixture_id",
                "date",
                "timestamp",
                "league_id",
                "league_name",
                "league_season",
                "round",
                "competition_type",
                "competition_stage",
                "home_team_id",
                "home_team",
                "away_team_id",
                "away_team",
                "home_goals",
                "away_goals",
                "home_penalty_goals",
                "away_penalty_goals",
                "total_goals",
                "result",
            ],
        ),
        "team_match_features": _write_csv(
            processed / "team_match_features.csv",
            team_rows,
            [
                "fixture_id",
                "date",
                "team_id",
                "team",
                "opponent_id",
                "opponent",
                "is_home",
                "league_id",
                "league_name",
                "round",
                "competition_type",
                "competition_stage",
                "rest_days",
                "prior_matches",
                "prior5_goals_for_avg",
                "prior5_goals_against_avg",
                "prior5_points_avg",
                "goals_for",
                "goals_against",
                "points",
            ],
        ),
        "team_detailed_stats": _write_csv(
            processed / "team_detailed_stats.csv",
            detailed_team_rows,
            [
                "fixture_id",
                "date",
                "team_id",
                "team",
                *API_TEAM_STAT_TYPES.values(),
            ],
        ),
    }
    return counts


def _read_football_data_competitions(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    competitions_dir = raw_dir / "football_data" / "competitions"
    for path in competitions_dir.glob("*/matches*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for match in payload.get("matches", []):
            score = match.get("score", {}).get("fullTime", {})
            if match.get("status") != "FINISHED":
                continue
            if score.get("home") is None or score.get("away") is None:
                continue
            rows.append(match)
    rows.sort(key=lambda row: row["utcDate"])
    return rows


def _fd_result(winner: str | None, home_goals: int, away_goals: int) -> str:
    if winner == "HOME_TEAM" or home_goals > away_goals:
        return "home"
    if winner == "AWAY_TEAM" or away_goals > home_goals:
        return "away"
    return "draw"


def _fd_competition_type(competition: dict[str, Any], stage: str | None) -> str:
    code = competition.get("code")
    comp_type = competition.get("type")
    if code in {"WC", "EC", "CL", "CLI"} or comp_type == "CUP":
        return "major_tournament"
    if stage and "QUAL" in stage.upper():
        return "qualifier"
    return "competition"


def normalize_football_data(data_root: Path) -> dict[str, int]:
    matches = _read_football_data_competitions(data_root / "raw")
    processed = data_root / "processed" / "football_data"
    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    scorer_rows_by_key: dict[
        tuple[str, int, int, int],
        dict[str, Any],
    ] = {}
    team_catalog_rows: list[dict[str, Any]] = []
    squad_player_rows: list[dict[str, Any]] = []
    squad_quality_rows: list[dict[str, Any]] = []
    player_strengths = _load_player_competition_strengths(data_root / "raw")

    histories: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))
    last_played: dict[str, datetime] = {}

    for match in matches:
        played_at = _parse_dt(match["utcDate"])
        competition = match["competition"]
        home = match["homeTeam"]
        away = match["awayTeam"]
        score = match["score"]["fullTime"]
        home_goals = int(score["home"])
        away_goals = int(score["away"])
        stage = match.get("stage")
        competition_type = _fd_competition_type(competition, stage)
        result = _fd_result(match.get("score", {}).get("winner"), home_goals, away_goals)
        match_rows.append(
            {
                "source": "football-data.org",
                "match_id": match["id"],
                "date": played_at.date().isoformat(),
                "timestamp": int(played_at.timestamp()),
                "competition_code": competition.get("code"),
                "competition_name": competition.get("name"),
                "competition_type": competition_type,
                "stage": stage,
                "matchday": match.get("matchday"),
                "home_team_id": home["id"],
                "home_team": home["name"],
                "away_team_id": away["id"],
                "away_team": away["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": home_goals + away_goals,
                "result": result,
            }
        )

        for team, opponent, is_home, goals_for, goals_against in (
            (home, away, True, home_goals, away_goals),
            (away, home, False, away_goals, home_goals),
        ):
            team_id = f"fd:{team['id']}"
            history = list(histories[team_id])
            last5 = history[-5:]
            rest_days = None
            if team_id in last_played:
                rest_days = (played_at.date() - last_played[team_id].date()).days
            team_rows.append(
                {
                    "source": "football-data.org",
                    "match_id": match["id"],
                    "date": played_at.date().isoformat(),
                    "team_id": team_id,
                    "team": team["name"],
                    "opponent_id": f"fd:{opponent['id']}",
                    "opponent": opponent["name"],
                    "is_home": is_home,
                    "competition_code": competition.get("code"),
                    "competition_name": competition.get("name"),
                    "competition_type": competition_type,
                    "stage": stage,
                    "rest_days": rest_days,
                    "prior_matches": len(history),
                    "prior5_goals_for_avg": round(sum(h["gf"] for h in last5) / len(last5), 4)
                    if last5
                    else None,
                    "prior5_goals_against_avg": round(sum(h["ga"] for h in last5) / len(last5), 4)
                    if last5
                    else None,
                    "prior5_points_avg": round(sum(h["points"] for h in last5) / len(last5), 4)
                    if last5
                    else None,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "points": 3
                    if goals_for > goals_against
                    else 1
                    if goals_for == goals_against
                    else 0,
                }
            )
        for team, goals_for, goals_against in (
            (home, home_goals, away_goals),
            (away, away_goals, home_goals),
        ):
            team_id = f"fd:{team['id']}"
            points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            histories[team_id].append({"gf": goals_for, "ga": goals_against, "points": points})
            last_played[team_id] = played_at

    competitions_dir = data_root / "raw" / "football_data" / "competitions"
    for path in competitions_dir.glob("*/scorers*-limit-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        competition = payload.get("competition", {})
        season = payload.get("season") or {}
        filters = payload.get("filters") or {}
        season_start = int(
            filters.get("season")
            or str(season.get("startDate") or "0")[:4]
            or 0
        )
        for scorer in payload.get("scorers", []):
            player = scorer.get("player", {})
            team = scorer.get("team", {})
            competition_code = str(competition.get("code") or "")
            player_id = int(player.get("id") or 0)
            team_id = int(team.get("id") or 0)
            scorer_rows_by_key[
                (competition_code, season_start, player_id, team_id)
            ] = {
                "competition_code": competition_code,
                "competition_name": competition.get("name"),
                "season_start": season_start,
                "season_start_date": season.get("startDate"),
                "season_end_date": season.get("endDate"),
                "player_id": player.get("id"),
                "player": player.get("name"),
                "nationality": player.get("nationality"),
                "team_id": team.get("id"),
                "team": team.get("name"),
                "played_matches": scorer.get("playedMatches"),
                "goals": scorer.get("goals"),
                "assists": scorer.get("assists"),
                "penalties": scorer.get("penalties"),
            }
    scorer_rows = sorted(
        scorer_rows_by_key.values(),
        key=lambda row: (
            int(row["season_start"]),
            str(row["competition_code"]),
            -int(row.get("goals") or 0),
            str(row.get("player") or ""),
        ),
    )

    for path in competitions_dir.glob("*/teams.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        competition = payload.get("competition", {})
        for team in payload.get("teams", []):
            squad = team.get("squad") or []
            ages: list[float] = []
            player_scores: list[float] = []
            player_max_scores: list[float] = []
            top_shares: list[float] = []
            elite_shares: list[float] = []
            known_strength_players = 0
            position_counts = defaultdict(int)
            for player in squad:
                player_id = player.get("id")
                age = _age_on(player.get("dateOfBirth"))
                if age is not None:
                    ages.append(age)
                position = player.get("position") or "Unknown"
                position_counts[position] += 1
                strength = player_strengths.get(int(player_id), {}) if player_id else {}
                if strength:
                    known_strength_players += 1
                avg_strength = float(strength.get("avg_strength", 0.0))
                max_strength = float(strength.get("max_strength", 0.0))
                top_share = float(strength.get("top_competition_share", 0.0))
                elite_share = float(strength.get("elite_competition_share", 0.0))
                player_scores.append(avg_strength)
                player_max_scores.append(max_strength)
                top_shares.append(top_share)
                elite_shares.append(elite_share)
                squad_player_rows.append(
                    {
                        "competition_code": competition.get("code"),
                        "competition_name": competition.get("name"),
                        "team_id": team.get("id"),
                        "team": team.get("name"),
                        "player_id": player_id,
                        "player": player.get("name"),
                        "position": position,
                        "date_of_birth": player.get("dateOfBirth"),
                        "age": age,
                        "nationality": player.get("nationality"),
                        "recent_match_count": int(strength.get("matches", 0)),
                        "avg_competition_strength": avg_strength,
                        "max_competition_strength": max_strength,
                        "top_competition_share": top_share,
                        "elite_competition_share": elite_share,
                    }
                )
            sorted_scores = sorted(player_max_scores, reverse=True)
            sorted_avg_scores = sorted(player_scores, reverse=True)
            squad_size = len(squad)
            squad_quality_rows.append(
                {
                    "competition_code": competition.get("code"),
                    "competition_name": competition.get("name"),
                    "team_id": team.get("id"),
                    "team": team.get("name"),
                    "squad_size": squad_size,
                    "avg_age": round(sum(ages) / len(ages), 4) if ages else None,
                    "min_age": round(min(ages), 4) if ages else None,
                    "max_age": round(max(ages), 4) if ages else None,
                    "goalkeepers": position_counts["Goalkeeper"],
                    "defenders": position_counts["Defence"],
                    "midfielders": position_counts["Midfield"],
                    "attackers": position_counts["Offence"],
                    "known_strength_players": known_strength_players,
                    "known_strength_share": round(known_strength_players / squad_size, 4)
                    if squad_size
                    else 0.0,
                    "squad_avg_competition_strength": round(sum(player_scores) / squad_size, 4)
                    if squad_size
                    else 0.0,
                    "squad_top11_competition_strength": round(sum(sorted_scores[:11]) / min(11, squad_size), 4)
                    if squad_size
                    else 0.0,
                    "squad_top5_competition_strength": round(sum(sorted_scores[:5]) / min(5, squad_size), 4)
                    if squad_size
                    else 0.0,
                    "squad_depth_competition_strength": round(sum(sorted_avg_scores[:18]) / min(18, squad_size), 4)
                    if squad_size
                    else 0.0,
                    "squad_top_competition_share": round(sum(top_shares) / squad_size, 4)
                    if squad_size
                    else 0.0,
                    "squad_elite_competition_share": round(sum(elite_shares) / squad_size, 4)
                    if squad_size
                    else 0.0,
                }
            )
            team_catalog_rows.append(
                {
                    "competition_code": competition.get("code"),
                    "competition_name": competition.get("name"),
                    "team_id": team.get("id"),
                    "team": team.get("name"),
                    "short_name": team.get("shortName"),
                    "area": (team.get("area") or {}).get("name"),
                    "founded": team.get("founded"),
                    "venue": team.get("venue"),
                    "coach": (team.get("coach") or {}).get("name"),
                    "squad_size": squad_size,
                }
            )

    return {
        "matches": _write_csv(
            processed / "matches.csv",
            match_rows,
            [
                "source",
                "match_id",
                "date",
                "timestamp",
                "competition_code",
                "competition_name",
                "competition_type",
                "stage",
                "matchday",
                "home_team_id",
                "home_team",
                "away_team_id",
                "away_team",
                "home_goals",
                "away_goals",
                "total_goals",
                "result",
            ],
        ),
        "team_match_features": _write_csv(
            processed / "team_match_features.csv",
            team_rows,
            [
                "source",
                "match_id",
                "date",
                "team_id",
                "team",
                "opponent_id",
                "opponent",
                "is_home",
                "competition_code",
                "competition_name",
                "competition_type",
                "stage",
                "rest_days",
                "prior_matches",
                "prior5_goals_for_avg",
                "prior5_goals_against_avg",
                "prior5_points_avg",
                "goals_for",
                "goals_against",
                "points",
            ],
        ),
        "scorers": _write_csv(
            processed / "scorers.csv",
            scorer_rows,
            [
                "competition_code",
                "competition_name",
                "season_start",
                "season_start_date",
                "season_end_date",
                "player_id",
                "player",
                "nationality",
                "team_id",
                "team",
                "played_matches",
                "goals",
                "assists",
                "penalties",
            ],
        ),
        "teams": _write_csv(
            processed / "teams.csv",
            team_catalog_rows,
            [
                "competition_code",
                "competition_name",
                "team_id",
                "team",
                "short_name",
                "area",
                "founded",
                "venue",
                "coach",
                "squad_size",
            ],
        ),
        "squad_players": _write_csv(
            processed / "squad_players.csv",
            squad_player_rows,
            [
                "competition_code",
                "competition_name",
                "team_id",
                "team",
                "player_id",
                "player",
                "position",
                "date_of_birth",
                "age",
                "nationality",
                "recent_match_count",
                "avg_competition_strength",
                "max_competition_strength",
                "top_competition_share",
                "elite_competition_share",
            ],
        ),
        "squad_quality": _write_csv(
            processed / "squad_quality.csv",
            squad_quality_rows,
            [
                "competition_code",
                "competition_name",
                "team_id",
                "team",
                "squad_size",
                "avg_age",
                "min_age",
                "max_age",
                "goalkeepers",
                "defenders",
                "midfielders",
                "attackers",
                "known_strength_players",
                "known_strength_share",
                "squad_avg_competition_strength",
                "squad_top11_competition_strength",
                "squad_top5_competition_strength",
                "squad_depth_competition_strength",
                "squad_top_competition_share",
                "squad_elite_competition_share",
            ],
        ),
    }
