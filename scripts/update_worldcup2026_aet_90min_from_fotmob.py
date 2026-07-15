from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _team_key(value: str) -> str:
    aliases = {
        "cape verde": "cape verde islands",
        "dr congo": "congo dr",
        "usa": "united states",
        "bosnia and herzegovina": "bosnia-herzegovina",
    }
    return aliases.get(value.casefold(), value.casefold())


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((_team_key(a), _team_key(b))))


def _repair_text(value: str) -> str:
    if "Ã" not in value:
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value


def _aet_note(data: Path, result: dict[str, str]) -> str:
    match = re.search(r"ESPN event (\d+)", result.get("source", ""))
    if not match:
        return result.get("notes", "")
    summary_path = data / "raw" / "espn" / "worldcup_2026" / f"summary_{match.group(1)}.json"
    if not summary_path.exists():
        return result.get("notes", "")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    competition = (payload.get("header", {}).get("competitions") or [{}])[0]
    final_scores = {
        item.get("team", {}).get("displayName", ""): int(item.get("score", 0))
        for item in competition.get("competitors", [])
    }
    regulation_goals: list[str] = []
    extra_goals: list[str] = []
    for event in payload.get("keyEvents", []):
        if not event.get("scoringPlay"):
            continue
        team = event.get("team", {}).get("displayName", "")
        player = _repair_text(str(event.get("shortText") or "").split(" Goal", 1)[0])
        minute = event.get("clock", {}).get("displayValue", "")
        text = f"{team}: {player} {minute}".strip()
        bucket = regulation_goals if float(event.get("clock", {}).get("value") or 0) <= 5400 else extra_goals
        bucket.append(text)
    resolution = (
        f"Penalty shootout: {result.get('team_a_penalty_goals', '')}-"
        f"{result.get('team_b_penalty_goals', '')}; {result.get('penalty_winner', '')} advanced."
        if result.get("penalty_winner")
        else f"{result.get('extra_time_winner', '')} advanced."
    )
    return (
        f"90 minutes: {result['team_a']} {result['team_a_goals']}-{result['team_b_goals']} "
        f"{result['team_b']}. Goals: {'; '.join(regulation_goals)}. "
        f"After extra time: {result['team_a']} {final_scores.get(result['team_a'], '')}-"
        f"{final_scores.get(result['team_b'], '')} {result['team_b']}. {resolution} Extra-time goals: "
        f"{'; '.join(extra_goals)}."
    )


def _extra_time_winner(data: Path, result: dict[str, str]) -> str:
    match = re.search(r"ESPN event (\d+)", result.get("source", ""))
    if not match:
        return result.get("extra_time_winner", "")
    path = data / "raw" / "espn" / "worldcup_2026" / f"summary_{match.group(1)}.json"
    if not path.exists():
        return result.get("extra_time_winner", "")
    payload = json.loads(path.read_text(encoding="utf-8"))
    competitors = (payload.get("header", {}).get("competitions") or [{}])[0].get(
        "competitors", []
    )
    scores = {
        item.get("team", {}).get("displayName", ""): int(item.get("score", 0))
        for item in competitors
    }
    if scores.get(result["team_a"], 0) > scores.get(result["team_b"], 0):
        return result["team_a"]
    if scores.get(result["team_b"], 0) > scores.get(result["team_a"], 0):
        return result["team_b"]
    return result.get("extra_time_winner", "")


def main() -> None:
    data = Path("data")
    manual_path = data / "static" / "worldcup_2026_manual_results.csv"
    detail_path = data / "static" / "worldcup_2026_manual_detail_stats.csv"
    matches_path = data / "processed" / "fotmob_world_cup" / "matches.csv"
    stats_path = data / "processed" / "fotmob_world_cup" / "team_match_stats.csv"

    manual = _read(manual_path)
    details = _read(detail_path)
    matches = _read(matches_path)
    stats = _read(stats_path)
    manual_fields = list(manual[0])
    detail_fields = list(details[0])

    shot_rows = _read(data / "processed" / "fotmob_world_cup" / "shots.csv")
    extended_ids = {
        row["match_id"]
        for row in shot_rows
        if "Extra" in str(row.get("period") or "")
    }
    aet_matches = {
        _pair(row["home_team"], row["away_team"]): row
        for row in matches
        if row.get("season") == "2026" and row.get("match_id") in extended_ids
    }
    stats_by_match_team = {(row["match_id"], _team_key(row["team"])): row for row in stats}
    manual_by_pair = {_pair(row["team_a"], row["team_b"]): row for row in manual}
    detail_by_key = {(row["match_id"], _team_key(row["team"])): row for row in details}

    updated: list[str] = []
    for pair, match in aet_matches.items():
        result = manual_by_pair.get(pair)
        if result is None:
            continue
        home = stats_by_match_team[(match["match_id"], _team_key(match["home_team"]))]
        away = stats_by_match_team[(match["match_id"], _team_key(match["away_team"]))]
        score_by_team = {
            _team_key(match["home_team"]): home["goals_for"],
            _team_key(match["away_team"]): away["goals_for"],
        }
        result["team_a_goals"] = str(int(float(score_by_team[_team_key(result["team_a"])])))
        result["team_b_goals"] = str(int(float(score_by_team[_team_key(result["team_b"])])))
        result["winner"] = "Draw"
        if not result.get("penalty_winner"):
            result["extra_time_winner"] = _extra_time_winner(data, result)
        result["notes"] = _aet_note(data, result)

        for team in (result["team_a"], result["team_b"]):
            source = stats_by_match_team[(match["match_id"], _team_key(team))]
            opponent = result["team_b"] if team == result["team_a"] else result["team_a"]
            opponent_source = stats_by_match_team[(match["match_id"], _team_key(opponent))]
            row = detail_by_key[(result["match_id"], _team_key(team))]
            for field in detail_fields:
                if field not in {"match_id", "team", "source", "notes"}:
                    row[field] = ""
            mapping = {
                "total_shots": "total_shots",
                "shots_on_goal": "shots_on_target",
                "shots_inside_box": "shots_inside_box",
                "shots_outside_box": "shots_outside_box",
                "expected_goals": "expected_goals",
                "fotmob_expected_goals": "expected_goals",
                "fotmob_expected_goals_non_penalty": "expected_goals_non_penalty",
                "fotmob_expected_goals_on_target": "expected_goals_on_target",
                "fotmob_total_shots": "total_shots",
                "fotmob_shots_on_target": "shots_on_target",
                "fotmob_shots_inside_box": "shots_inside_box",
            }
            for target, source_field in mapping.items():
                row[target] = source.get(source_field, "")
            row["fotmob_expected_goals_conceded"] = opponent_source.get("expected_goals", "")
            row["source"] = f"{row.get('source', '')}; FotMob 90-minute shotmap".strip("; ")
            row["notes"] = (
                "Only regulation-time shotmap fields are populated. Provider aggregates "
                "without a 90-minute split remain blank; extra-time events are excluded."
            )
        updated.append(result["match_id"])

    _write(manual_path, manual, manual_fields)
    _write(detail_path, details, detail_fields)
    print({"updated_aet_matches": updated})


if __name__ == "__main__":
    main()
