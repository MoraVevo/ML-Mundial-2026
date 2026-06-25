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
