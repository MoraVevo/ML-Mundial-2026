from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from kinela.providers.espn import EspnWorldCupClient, extract_strategic_team_rows


DETAIL_FIELDS = [
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

ALIASES = {
    "bosnia and herzegovina": "bosnia-herzegovina",
    "cape verde": "cape verde islands",
    "cabo verde": "cape verde islands",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "tÃ¼rkiye": "turkey",
    "czech republic": "czechia",
    "cÃ´te d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "korea republic": "south korea",
    "ir iran": "iran",
}


def _norm(value: str) -> str:
    text = value.casefold().replace(".", "").strip()
    return ALIASES.get(text, text)


def _team_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((_norm(team_a), _norm(team_b))))


def _football_data_schedule(data_root: Path) -> list[dict]:
    path = (
        data_root
        / "raw"
        / "football_data"
        / "competitions"
        / "WC"
        / "matches-season-2026.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing football-data World Cup schedule. Run the WC competition collector first."
        )
    return json.loads(path.read_text(encoding="utf-8")).get("matches", [])


def _scoreboard_index(
    client: EspnWorldCupClient,
    days: set[date],
    refresh_days: set[date],
) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for day in sorted(days):
        payload = client.scoreboard(day, refresh=day in refresh_days)
        for event in payload.get("events", []):
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors", [])
            if len(competitors) != 2:
                continue
            teams = [item.get("team", {}).get("displayName", "") for item in competitors]
            if all(teams):
                index[_team_key(*teams)] = event
    return index


def _event_score(event: dict) -> dict[str, int]:
    competition = (event.get("competitions") or [{}])[0]
    return {
        _norm(item.get("team", {}).get("displayName", "")): int(item.get("score", 0))
        for item in competition.get("competitors", [])
        if item.get("team", {}).get("displayName")
    }


def _event_is_complete(event: dict) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    return bool(
        competition.get("status", {}).get("type", {}).get("completed")
        or event.get("status", {}).get("type", {}).get("completed")
    )


def _sync_finished_results(
    manual_rows: list[dict[str, str]],
    schedule: list[dict],
    events: dict[tuple[str, str], dict],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    known_ids = {row["match_id"] for row in manual_rows}
    added: list[dict[str, str]] = []

    for match in schedule:
        match_id = str(match.get("id", ""))
        score = match.get("score", {}).get("fullTime", {})
        team_a = match.get("homeTeam", {}).get("name", "")
        team_b = match.get("awayTeam", {}).get("name", "")
        goals_a = score.get("home")
        goals_b = score.get("away")
        if match_id in known_ids or not team_a or not team_b:
            continue
        event = events.get(_team_key(team_a, team_b))
        espn_finished = bool(event and _event_is_complete(event))
        if (
            match.get("status") != "FINISHED" and not espn_finished
        ):
            continue

        if not espn_finished:
            continue
        espn_score = _event_score(event)
        espn_goals_a = espn_score.get(_norm(team_a))
        espn_goals_b = espn_score.get(_norm(team_b))
        if espn_goals_a is None or espn_goals_b is None:
            continue
        football_data_finished = (
            match.get("status") == "FINISHED"
            and goals_a is not None
            and goals_b is not None
        )
        if football_data_finished and (
            espn_goals_a != int(goals_a) or espn_goals_b != int(goals_b)
        ):
            continue
        if not football_data_finished:
            goals_a = espn_goals_a
            goals_b = espn_goals_b

        winner = "Draw"
        if int(goals_a) > int(goals_b):
            winner = team_a
        elif int(goals_b) > int(goals_a):
            winner = team_b
        event_id = str(event.get("id", ""))
        result_source = (
            f"football-data.org; ESPN event {event_id}"
            if football_data_finished
            else f"ESPN event {event_id}; football-data.org schedule metadata"
        )
        verification_note = (
            "Result cross-checked between football-data.org and ESPN."
            if football_data_finished
            else (
                "ESPN marked the event Full Time; football-data.org supplied "
                "the official schedule ID, stage, and group but had not yet refreshed the score."
            )
        )
        row = {
            "match_id": match_id,
            "date": str(match.get("utcDate", ""))[:10],
            "stage": match.get("stage", ""),
            "group": match.get("group", ""),
            "team_a": team_a,
            "team_b": team_b,
            "team_a_goals": str(goals_a),
            "team_b_goals": str(goals_b),
            "winner": winner,
            "source": result_source,
            "notes": (
                f"Final: {team_a} {goals_a}-{goals_b} {team_b}. "
                f"{verification_note}"
            ),
        }
        manual_rows.append(row)
        added.append(row)
        known_ids.add(match_id)

    manual_rows.sort(key=lambda row: (row["date"], int(row["match_id"])))
    return manual_rows, added


def _summary_stats(
    client: EspnWorldCupClient,
    event_id: str,
    match_id: str,
    *,
    refresh: bool = False,
) -> dict[str, dict[str, str]]:
    payload = client.summary(event_id, refresh=refresh)
    stats_by_team: dict[str, dict[str, str]] = {}
    for row in extract_strategic_team_rows(
        payload,
        match_id=match_id,
        event_id=event_id,
    ):
        values = {field: row.get(field, "") for field in DETAIL_FIELDS}
        passes_pct = values.get("passes_pct", "")
        try:
            numeric_pct = float(passes_pct)
        except (TypeError, ValueError):
            numeric_pct = 0.0
        if passes_pct and 0 < numeric_pct <= 1:
            values["passes_pct"] = f"{numeric_pct * 100:.1f}".rstrip("0").rstrip(".")
        stats_by_team[_norm(row["team"])] = values
    return stats_by_team


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_root = Path("data")
    manual_path = data_root / "static" / "worldcup_2026_manual_results.csv"
    detail_path = data_root / "static" / "worldcup_2026_manual_detail_stats.csv"
    manual_rows = list(csv.DictReader(manual_path.open(encoding="utf-8")))
    detail_rows = list(csv.DictReader(detail_path.open(encoding="utf-8")))
    schedule = _football_data_schedule(data_root)

    finished_days = {
        datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date()
        for match in schedule
        if match.get("status") == "FINISHED" and match.get("utcDate")
    }
    scoreboard_days = {
        candidate + timedelta(days=offset)
        for candidate in finished_days
        for offset in (-1, 0, 1)
    }
    utc_today = datetime.now(timezone.utc).date()
    scoreboard_days.update(
        utc_today - timedelta(days=offset)
        for offset in range(3)
    )
    latest_finished_day = max(finished_days)
    refresh_days = {
        day for day in scoreboard_days if day >= latest_finished_day - timedelta(days=1)
    }
    client = EspnWorldCupClient(data_root)
    events = _scoreboard_index(client, scoreboard_days, refresh_days)
    manual_rows, added_results = _sync_finished_results(manual_rows, schedule, events)

    manual_fields = [
        "match_id",
        "date",
        "stage",
        "group",
        "team_a",
        "team_b",
        "team_a_goals",
        "team_b_goals",
        "winner",
        "source",
        "notes",
    ]
    _write_csv(manual_path, manual_rows, manual_fields)

    rows_by_key: dict[tuple[str, str], dict[str, str]] = {
        (row["match_id"], _norm(row["team"])): row for row in detail_rows
    }
    updated_fields: dict[str, int] = defaultdict(int)
    missing_events: list[dict[str, str]] = []

    for manual in manual_rows:
        event = events.get(_team_key(manual["team_a"], manual["team_b"]))
        if not event:
            missing_events.append(manual)
            continue
        event_id = str(event["id"])
        match_day = date.fromisoformat(manual["date"])
        stats_by_team = _summary_stats(
            client,
            event_id,
            manual["match_id"],
            refresh=match_day >= latest_finished_day - timedelta(days=1),
        )
        for team in (manual["team_a"], manual["team_b"]):
            key = (manual["match_id"], _norm(team))
            row = rows_by_key.get(key)
            if row is None:
                row = {"match_id": manual["match_id"], "team": team}
                for field in DETAIL_FIELDS:
                    row[field] = ""
                row["source"] = ""
                row["notes"] = ""
                rows_by_key[key] = row
            values = stats_by_team.get(_norm(team), {})
            for field, value in values.items():
                if field in DETAIL_FIELDS and value and not row.get(field):
                    row[field] = value
                    updated_fields[field] += 1
            source = row.get("source", "")
            espn_source = f"ESPN event {event_id}"
            if espn_source not in source:
                row["source"] = f"{source}; {espn_source}".strip("; ")
            note = row.get("notes", "")
            if "ESPN summary API used to fill blank fields" not in note:
                row["notes"] = f"{note} ESPN summary API used to fill blank fields.".strip()

    detail_fields = ["match_id", "team", *DETAIL_FIELDS, "source", "notes"]
    rows = sorted(rows_by_key.values(), key=lambda item: (int(item["match_id"]), item["team"]))
    _write_csv(detail_path, rows, detail_fields)
    print(
        json.dumps(
            {
                "manual_results": len(manual_rows),
                "added_results": added_results,
                "detail_rows": len(rows),
                "updated_fields": dict(updated_fields),
                "missing_events": missing_events,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
