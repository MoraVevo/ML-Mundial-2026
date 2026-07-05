from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
            count += 1
    return count


def _goal_type(event: dict[str, Any]) -> str:
    if event.get("period") == 5:
        return "penalty_shootout"
    shot = event.get("shot", {})
    shot_type = shot.get("type", {}).get("name")
    if shot_type == "Penalty":
        return "penalty"
    if shot_type == "Free Kick":
        return "free_kick"
    if shot_type == "Corner":
        return "corner"
    location = event.get("location") or []
    if location and float(location[0]) <= 60:
        return "own_half_or_halfway"
    return "open_play"


def normalize_statsbomb_world_cup(data_root: Path) -> dict[str, int]:
    raw = data_root / "raw" / "statsbomb"
    processed = data_root / "processed" / "statsbomb_world_cup_2022"
    match_path = raw / "matches" / "43" / "106.json"
    matches = json.loads(match_path.read_text(encoding="utf-8"))

    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    goal_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []

    for match in matches:
        match_id = int(match["match_id"])
        events = json.loads((raw / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
        lineups = json.loads((raw / "lineups" / f"{match_id}.json").read_text(encoding="utf-8"))
        home = match["home_team"]["home_team_name"]
        away = match["away_team"]["away_team_name"]
        match_rows.append(
            {
                "match_id": match_id,
                "date": match["match_date"],
                "kick_off": match.get("kick_off"),
                "stage": match.get("competition_stage", {}).get("name"),
                "home_team": home,
                "away_team": away,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "stadium": match.get("stadium", {}).get("name"),
            }
        )

        starters: set[int] = set()
        for event in events:
            if event.get("type", {}).get("name") != "Starting XI":
                continue
            for lineup in event.get("tactics", {}).get("lineup", []):
                player_id = lineup.get("player", {}).get("id")
                if player_id:
                    starters.add(int(player_id))

        for team_lineup in lineups:
            team = team_lineup.get("team_name")
            for player in team_lineup.get("lineup", []):
                player_id = int(player["player_id"])
                positions = player.get("positions", [])
                lineup_rows.append(
                    {
                        "match_id": match_id,
                        "team": team,
                        "player_id": player_id,
                        "player": player.get("player_name"),
                        "nickname": player.get("player_nickname"),
                        "jersey_number": player.get("jersey_number"),
                        "country": player.get("country", {}).get("name"),
                        "starter": player_id in starters,
                        "positions_json": json.dumps(positions, ensure_ascii=False),
                    }
                )

        team_stats: dict[str, Counter[str]] = defaultdict(Counter)
        player_stats: dict[tuple[int, str, int, str], Counter[str]] = defaultdict(Counter)
        possession_events: Counter[str] = Counter()

        for event in events:
            team = event.get("team", {}).get("name")
            player = event.get("player", {})
            player_id = player.get("id")
            player_name = player.get("name")
            event_type = event.get("type", {}).get("name")
            if team:
                possession_events[team] += 1
                team_stats[team]["events"] += 1

            key = (match_id, team or "", int(player_id or 0), player_name or "")
            if event_type == "Pass" and team:
                completed = not event.get("pass", {}).get("outcome")
                team_stats[team]["passes"] += 1
                team_stats[team]["passes_completed"] += int(completed)
                if player_id:
                    player_stats[key]["passes"] += 1
                    player_stats[key]["passes_completed"] += int(completed)
                if event.get("pass", {}).get("type", {}).get("name") == "Corner":
                    team_stats[team]["corners"] += 1
            elif event_type == "Shot" and team:
                team_stats[team]["shots"] += 1
                if player_id:
                    player_stats[key]["shots"] += 1
                outcome = event.get("shot", {}).get("outcome", {}).get("name")
                if outcome == "Goal":
                    is_shootout = event.get("period") == 5
                    if not is_shootout:
                        team_stats[team]["goals"] += 1
                        if player_id:
                            player_stats[key]["goals"] += 1
                    goal_rows.append(
                        {
                            "match_id": match_id,
                            "team": team,
                            "player_id": player_id,
                            "player": player_name,
                            "minute": event.get("minute"),
                            "second": event.get("second"),
                            "goal_type": _goal_type(event),
                            "period": event.get("period"),
                            "is_shootout": is_shootout,
                            "x": (event.get("location") or [None])[0],
                            "y": (event.get("location") or [None, None])[1],
                            "xg": event.get("shot", {}).get("statsbomb_xg"),
                        }
                    )
            elif event_type == "Own Goal For" and team:
                team_stats[team]["goals"] += 1
                goal_rows.append(
                    {
                        "match_id": match_id,
                        "team": team,
                        "player_id": None,
                        "player": None,
                        "minute": event.get("minute"),
                        "second": event.get("second"),
                        "goal_type": "own_goal",
                        "period": event.get("period"),
                        "is_shootout": False,
                        "x": (event.get("location") or [None])[0],
                        "y": (event.get("location") or [None, None])[1],
                        "xg": None,
                    }
                )
            elif event_type == "Own Goal Against" and player_id:
                player_stats[key]["own_goals"] += 1
            elif event_type == "Foul Committed" and team:
                team_stats[team]["fouls"] += 1
                if player_id:
                    player_stats[key]["fouls"] += 1

        total_possession_events = sum(possession_events.values())
        for team in (home, away):
            stats = team_stats[team]
            passes = stats["passes"]
            team_rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "opponent": away if team == home else home,
                    "is_home": team == home,
                    "goals": stats["goals"],
                    "own_goals": stats["own_goals"],
                    "shots": stats["shots"],
                    "passes": passes,
                    "passes_completed": stats["passes_completed"],
                    "pass_accuracy_pct": round(100 * stats["passes_completed"] / passes, 2)
                    if passes
                    else None,
                    "possession_event_share_pct": round(
                        100 * possession_events[team] / total_possession_events, 2
                    )
                    if total_possession_events
                    else None,
                    "corners": stats["corners"],
                    "fouls": stats["fouls"],
                }
            )

        for (row_match_id, team, player_id, player_name), stats in player_stats.items():
            passes = stats["passes"]
            player_rows.append(
                {
                    "match_id": row_match_id,
                    "team": team,
                    "player_id": player_id,
                    "player": player_name,
                    "goals": stats["goals"],
                    "shots": stats["shots"],
                    "passes": passes,
                    "passes_completed": stats["passes_completed"],
                    "pass_accuracy_pct": round(100 * stats["passes_completed"] / passes, 2)
                    if passes
                    else None,
                    "fouls": stats["fouls"],
                }
            )

    counts = {
        "matches": _write_csv(
            processed / "matches.csv",
            match_rows,
            [
                "match_id",
                "date",
                "kick_off",
                "stage",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "stadium",
            ],
        ),
        "team_match_stats": _write_csv(
            processed / "team_match_stats.csv",
            team_rows,
            [
                "match_id",
                "team",
                "opponent",
                "is_home",
                "goals",
                "own_goals",
                "shots",
                "passes",
                "passes_completed",
                "pass_accuracy_pct",
                "possession_event_share_pct",
                "corners",
                "fouls",
            ],
        ),
        "player_match_stats": _write_csv(
            processed / "player_match_stats.csv",
            player_rows,
            [
                "match_id",
                "team",
                "player_id",
                "player",
                "goals",
                "shots",
                "passes",
                "passes_completed",
                "pass_accuracy_pct",
                "fouls",
            ],
        ),
        "goals": _write_csv(
            processed / "goals.csv",
            goal_rows,
            [
                "match_id",
                "team",
                "player_id",
                "player",
                "minute",
                "second",
                "goal_type",
                "period",
                "is_shootout",
                "x",
                "y",
                "xg",
            ],
        ),
        "lineups": _write_csv(
            processed / "lineups.csv",
            lineup_rows,
            [
                "match_id",
                "team",
                "player_id",
                "player",
                "nickname",
                "jersey_number",
                "country",
                "starter",
                "positions_json",
            ],
        ),
    }
    return counts


def _find_fotmob_match_content(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, dict) and ("stats" in content or "shotmap" in content):
            return content
        if "stats" in value and ("general" in value or "shotmap" in value):
            return value
        for child in value.values():
            found = _find_fotmob_match_content(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_fotmob_match_content(child)
            if found is not None:
                return found
    return None


def _fotmob_shots(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("shots", "shotmap", "events"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _fotmob_stat_lookup(content: dict[str, Any]) -> dict[str, list[Any]]:
    lookup: dict[str, list[Any]] = {}
    stats = content.get("stats") or {}
    groups = stats.get("Periods", {}).get("All", {}).get("stats", [])
    for group in groups:
        for item in group.get("stats", []) if isinstance(group, dict) else []:
            key = item.get("key")
            values = item.get("stats")
            if key and isinstance(values, list) and len(values) >= 2:
                lookup[str(key)] = values[:2]
    return lookup


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fotmob_side_value(values: dict[str, list[Any]], key: str, index: int) -> Any:
    item = values.get(key)
    if not item or len(item) <= index:
        return None
    return item[index]


def normalize_fotmob_world_cups(data_root: Path) -> dict[str, int]:
    raw = data_root / "raw" / "fotmob" / "world_cup"
    processed = data_root / "processed" / "fotmob_world_cup"
    match_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []

    for fixture_path in sorted(raw.glob("*/fixtures.json")):
        season = fixture_path.parent.name
        fixtures_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        fixtures = (
            fixtures_payload.get("props", {})
            .get("pageProps", {})
            .get("fixtures", {})
            .get("allMatches", [])
        )
        for fixture in fixtures:
            if not fixture.get("status", {}).get("finished"):
                continue
            match_id = str(fixture.get("id") or "")
            match_path = fixture_path.parent / "matches" / f"{match_id}.json"
            if not match_path.exists():
                continue
            home = fixture.get("home", {}) or {}
            away = fixture.get("away", {}) or {}
            score = str(fixture.get("status", {}).get("scoreStr") or "")
            home_score, away_score = None, None
            if " - " in score:
                left, right = score.split(" - ", 1)
                home_score = _float_or_none(left)
                away_score = _float_or_none(right)
            content = _find_fotmob_match_content(
                json.loads(match_path.read_text(encoding="utf-8"))
            ) or {}
            stats = _fotmob_stat_lookup(content)
            match_rows.append(
                {
                    "season": season,
                    "match_id": match_id,
                    "date": str(fixture.get("status", {}).get("utcTime") or "")[:10],
                    "round": fixture.get("round"),
                    "round_name": fixture.get("roundName"),
                    "group": fixture.get("group"),
                    "home_team": home.get("name"),
                    "away_team": away.get("name"),
                    "home_team_id": home.get("id"),
                    "away_team_id": away.get("id"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": fixture.get("status", {}).get("reason", {}).get("short"),
                    "page_url": fixture.get("pageUrl"),
                }
            )
            teams = [(0, "home", home), (1, "away", away)]
            for index, side, team in teams:
                team_rows.append(
                    {
                        "season": season,
                        "match_id": match_id,
                        "date": str(fixture.get("status", {}).get("utcTime") or "")[:10],
                        "side": side,
                        "team": team.get("name"),
                        "team_id": team.get("id"),
                        "opponent": away.get("name") if side == "home" else home.get("name"),
                        "goals_for": home_score if side == "home" else away_score,
                        "goals_against": away_score if side == "home" else home_score,
                        "expected_goals": _fotmob_side_value(stats, "expected_goals", index),
                        "expected_goals_non_penalty": _fotmob_side_value(
                            stats,
                            "expected_goals_non_penalty",
                            index,
                        ),
                        "expected_goals_on_target": _fotmob_side_value(
                            stats,
                            "expected_goals_on_target",
                            index,
                        ),
                        "big_chances": _fotmob_side_value(stats, "big_chance", index),
                        "big_chances_missed": _fotmob_side_value(
                            stats,
                            "big_chance_missed_title",
                            index,
                        ),
                        "total_shots": _fotmob_side_value(stats, "total_shots", index),
                        "shots_on_target": _fotmob_side_value(stats, "ShotsOnTarget", index),
                        "shots_inside_box": _fotmob_side_value(stats, "shots_inside_box", index),
                        "shots_outside_box": _fotmob_side_value(stats, "shots_outside_box", index),
                        "touches_opp_box": _fotmob_side_value(stats, "touches_opp_box", index),
                        "corners": _fotmob_side_value(stats, "corners", index),
                        "possession_pct": _fotmob_side_value(stats, "BallPossesion", index),
                    }
                )
            for shot in _fotmob_shots(content.get("shotmap")):
                shot_rows.append(
                    {
                        "season": season,
                        "match_id": match_id,
                        "team_id": shot.get("teamId"),
                        "player_id": shot.get("playerId"),
                        "player": shot.get("playerName"),
                        "minute": shot.get("min"),
                        "minute_added": shot.get("minAdded"),
                        "period": shot.get("period"),
                        "event_type": shot.get("eventType"),
                        "situation": shot.get("situation"),
                        "shot_type": shot.get("shotType"),
                        "x": shot.get("x"),
                        "y": shot.get("y"),
                        "expected_goals": shot.get("expectedGoals"),
                        "expected_goals_on_target": shot.get("expectedGoalsOnTarget"),
                        "is_on_target": shot.get("isOnTarget"),
                        "is_blocked": shot.get("isBlocked"),
                        "is_own_goal": shot.get("isOwnGoal"),
                        "is_from_inside_box": shot.get("isFromInsideBox"),
                    }
                )

    counts = {
        "matches": _write_csv(
            processed / "matches.csv",
            match_rows,
            [
                "season",
                "match_id",
                "date",
                "round",
                "round_name",
                "group",
                "home_team",
                "away_team",
                "home_team_id",
                "away_team_id",
                "home_score",
                "away_score",
                "status",
                "page_url",
            ],
        ),
        "team_match_stats": _write_csv(
            processed / "team_match_stats.csv",
            team_rows,
            [
                "season",
                "match_id",
                "date",
                "side",
                "team",
                "team_id",
                "opponent",
                "goals_for",
                "goals_against",
                "expected_goals",
                "expected_goals_non_penalty",
                "expected_goals_on_target",
                "big_chances",
                "big_chances_missed",
                "total_shots",
                "shots_on_target",
                "shots_inside_box",
                "shots_outside_box",
                "touches_opp_box",
                "corners",
                "possession_pct",
            ],
        ),
        "shots": _write_csv(
            processed / "shots.csv",
            shot_rows,
            [
                "season",
                "match_id",
                "team_id",
                "player_id",
                "player",
                "minute",
                "minute_added",
                "period",
                "event_type",
                "situation",
                "shot_type",
                "x",
                "y",
                "expected_goals",
                "expected_goals_on_target",
                "is_on_target",
                "is_blocked",
                "is_own_goal",
                "is_from_inside_box",
            ],
        ),
    }
    return counts
