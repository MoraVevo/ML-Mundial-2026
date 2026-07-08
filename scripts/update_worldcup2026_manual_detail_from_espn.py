from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinela.model import DETAIL_STAT_FEATURES
from kinela.providers.espn import EspnWorldCupClient, extract_strategic_team_rows


DETAIL_FIELDS = list(DETAIL_STAT_FEATURES)

KNOCKOUT_MATCHES = [
    {"match_id": "73", "date": "2026-06-28", "stage": "ROUND_OF_32", "team_a": "South Africa", "team_b": "Canada"},
    {"match_id": "74", "date": "2026-06-29", "stage": "ROUND_OF_32", "team_a": "Germany", "team_b": "Paraguay"},
    {"match_id": "75", "date": "2026-06-29", "stage": "ROUND_OF_32", "team_a": "Netherlands", "team_b": "Morocco"},
    {"match_id": "76", "date": "2026-06-29", "stage": "ROUND_OF_32", "team_a": "Brazil", "team_b": "Japan"},
    {"match_id": "77", "date": "2026-06-30", "stage": "ROUND_OF_32", "team_a": "France", "team_b": "Sweden"},
    {"match_id": "78", "date": "2026-06-30", "stage": "ROUND_OF_32", "team_a": "Ivory Coast", "team_b": "Norway"},
    {"match_id": "79", "date": "2026-06-30", "stage": "ROUND_OF_32", "team_a": "Mexico", "team_b": "Ecuador"},
    {"match_id": "80", "date": "2026-07-01", "stage": "ROUND_OF_32", "team_a": "England", "team_b": "Congo DR"},
    {"match_id": "81", "date": "2026-07-01", "stage": "ROUND_OF_32", "team_a": "United States", "team_b": "Bosnia-Herzegovina"},
    {"match_id": "82", "date": "2026-07-01", "stage": "ROUND_OF_32", "team_a": "Belgium", "team_b": "Senegal"},
    {"match_id": "83", "date": "2026-07-02", "stage": "ROUND_OF_32", "team_a": "Portugal", "team_b": "Croatia"},
    {"match_id": "84", "date": "2026-07-02", "stage": "ROUND_OF_32", "team_a": "Spain", "team_b": "Austria"},
    {"match_id": "85", "date": "2026-07-02", "stage": "ROUND_OF_32", "team_a": "Switzerland", "team_b": "Algeria"},
    {"match_id": "86", "date": "2026-07-03", "stage": "ROUND_OF_32", "team_a": "Argentina", "team_b": "Cape Verde Islands"},
    {"match_id": "87", "date": "2026-07-03", "stage": "ROUND_OF_32", "team_a": "Colombia", "team_b": "Ghana"},
    {"match_id": "88", "date": "2026-07-03", "stage": "ROUND_OF_32", "team_a": "Australia", "team_b": "Egypt"},
    {"match_id": "89", "date": "2026-07-04", "stage": "LAST_16", "team_a": "Paraguay", "team_b": "France"},
    {"match_id": "90", "date": "2026-07-04", "stage": "LAST_16", "team_a": "Canada", "team_b": "Morocco"},
    {"match_id": "91", "date": "2026-07-05", "stage": "LAST_16", "team_a": "Brazil", "team_b": "Norway"},
    {"match_id": "92", "date": "2026-07-05", "stage": "LAST_16", "team_a": "Mexico", "team_b": "England"},
    {"match_id": "93", "date": "2026-07-06", "stage": "LAST_16", "team_a": "Portugal", "team_b": "Spain"},
    {"match_id": "94", "date": "2026-07-06", "stage": "LAST_16", "team_a": "United States", "team_b": "Belgium"},
    {"match_id": "95", "date": "2026-07-07", "stage": "LAST_16", "team_a": "Argentina", "team_b": "Egypt"},
    {"match_id": "96", "date": "2026-07-07", "stage": "LAST_16", "team_a": "Switzerland", "team_b": "Colombia"},
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


def _event_penalty_score(event: dict) -> dict[str, int]:
    competition = (event.get("competitions") or [{}])[0]
    scores: dict[str, int] = {}
    for item in competition.get("competitors", []):
        team = item.get("team", {}).get("displayName", "")
        if not team:
            continue
        value = item.get("shootoutScore")
        if value in (None, ""):
            value = item.get("penaltyScore")
        if value in (None, ""):
            continue
        scores[_norm(team)] = int(value)
    return scores


def _event_winner(event: dict) -> str:
    competition = (event.get("competitions") or [{}])[0]
    for item in competition.get("competitors", []):
        if item.get("winner") is True:
            return item.get("team", {}).get("displayName", "")
    return ""


def _is_truthy_completed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _event_is_complete(event: dict) -> bool:
    competition = (event.get("competitions") or [{}])[0]
    return any(
        _is_truthy_completed(value)
        for value in (
            competition.get("status", {}).get("type", {}).get("completed"),
            event.get("status", {}).get("type", {}).get("completed"),
        )
    )


def _manual_espn_event_id(row: dict[str, str]) -> str:
    text = f"{row.get('source', '')} {row.get('notes', '')}"
    match = re.search(r"\bESPN event (\d+)\b", text)
    return match.group(1) if match else ""


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


def _sync_finished_knockout_results(
    manual_rows: list[dict[str, str]],
    events: dict[tuple[str, str], dict],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows_by_id = {row["match_id"]: row for row in manual_rows}
    added: list[dict[str, str]] = []
    for match in KNOCKOUT_MATCHES:
        match_id = match["match_id"]
        event = events.get(_team_key(match["team_a"], match["team_b"]))
        if not event or not _event_is_complete(event):
            continue
        espn_score = _event_score(event)
        goals_a = espn_score.get(_norm(match["team_a"]))
        goals_b = espn_score.get(_norm(match["team_b"]))
        if goals_a is None or goals_b is None:
            continue
        penalty_score = _event_penalty_score(event)
        penalty_a = penalty_score.get(_norm(match["team_a"]))
        penalty_b = penalty_score.get(_norm(match["team_b"]))
        penalty_winner = _event_winner(event) if penalty_a is not None and penalty_b is not None else ""
        winner = "Draw"
        if goals_a > goals_b:
            winner = match["team_a"]
        elif goals_b > goals_a:
            winner = match["team_b"]
        event_id = str(event.get("id", ""))
        row = rows_by_id.get(match_id)
        is_new = row is None
        if row is None:
            row = {"match_id": match_id}
            manual_rows.append(row)
            rows_by_id[match_id] = row
        row.update(
            {
                "date": match["date"],
                "stage": match["stage"],
                "group": "",
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "team_a_goals": str(goals_a),
                "team_b_goals": str(goals_b),
                "winner": winner,
                "team_a_penalty_goals": "" if penalty_a is None else str(penalty_a),
                "team_b_penalty_goals": "" if penalty_b is None else str(penalty_b),
                "penalty_winner": penalty_winner,
                "source": f"ESPN event {event_id}; elimination bracket metadata",
                "notes": (
                    f"Final: {match['team_a']} {goals_a}-{goals_b} {match['team_b']}. "
                    + (
                        f"Penalties: {match['team_a']} {penalty_a}-{penalty_b} "
                        f"{match['team_b']}; {penalty_winner} advanced. "
                        if penalty_a is not None and penalty_b is not None and penalty_winner
                        else ""
                    )
                    + "ESPN marked the event Full Time; knockout match_id comes from the local World Cup bracket."
                ),
            }
        )
        if is_new:
            added.append(row)

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
    manual_rows, added_knockouts = _sync_finished_knockout_results(manual_rows, events)
    added_results.extend(added_knockouts)

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
        "team_a_penalty_goals",
        "team_b_penalty_goals",
        "penalty_winner",
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
        event_id = str(event["id"]) if event else _manual_espn_event_id(manual)
        if not event_id:
            missing_events.append(manual)
            continue
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
