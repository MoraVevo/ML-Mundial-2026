from __future__ import annotations

import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


EVENTS = {
    "537327": 93089,
    "537328": 93101,
    "537333": 93100,
    "537345": 93088,
    "537340": 93085,
    "537346": 93107,
    "537351": 93083,
    "537357": 93084,
    "537352": 93082,
    "537334": 93087,
    "537339": 93086,
    "537358": 93099,
    "537363": 93079,
    "537364": 93078,
    "537391": 93075,
    "537392": 93106,
    "537397": 93077,
    "537398": 93076,
}

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
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde Islands",
    "Côte d'Ivoire": "Ivory Coast",
}


def _num(value: str) -> str:
    return value.replace(",", "").strip()


def _title_teams(text: str) -> tuple[str, str]:
    match = re.search(r"<title>(.*?) \|", text, re.S)
    if not match:
        raise ValueError("Missing title")
    title = BeautifulSoup(match.group(1), "html.parser").get_text()
    left, right = [part.strip() for part in title.split(" vs ", 1)]
    return ALIASES.get(left, left), ALIASES.get(right, right)


def _team_comparison(text: str) -> dict[str, tuple[str, str]]:
    soup = BeautifulSoup(text, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    start = lines.index("Team Comparison")
    end = lines.index("Game Details", start)
    block = lines[start + 1 : end]
    metrics: dict[str, tuple[str, str]] = {}
    for index, line in enumerate(block):
        if line in {
            "Possession",
            "Shots (on goal)",
            "Corner Kicks",
            "Fouls",
            "Yellow Cards",
            "Red Cards",
            "Offsides",
        }:
            metrics[line] = (_num(block[index - 1]), _num(block[index + 1]))
    return metrics


def _row(match_id: str, team: str, values: dict[str, str], source: str) -> dict[str, str]:
    row = {"match_id": match_id, "team": team}
    for field in DETAIL_FIELDS:
        row[field] = values.get(field, "")
    row["source"] = source
    row["notes"] = (
        "Imported from theScore Team Comparison. "
        "theScore labels shots as shots on goal; total shots are left blank unless another source provides them."
    )
    return row


def main() -> None:
    path = Path("data/static/worldcup_2026_manual_detail_stats.csv")
    existing = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    covered_match_ids = {row["match_id"] for row in existing}
    rows = existing[:]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for match_id, event_id in EVENTS.items():
        if match_id in covered_match_ids:
            continue
        url = f"https://www.thescore.com/worldcup/event/{event_id}"
        text = session.get(url, timeout=20).text
        team_left, team_right = _title_teams(text)
        metrics = _team_comparison(text)
        left_values = {
            "ball_possession_pct": metrics.get("Possession", ("", ""))[0],
            "shots_on_goal": metrics.get("Shots (on goal)", ("", ""))[0],
            "corner_kicks": metrics.get("Corner Kicks", ("", ""))[0],
            "fouls": metrics.get("Fouls", ("", ""))[0],
            "yellow_cards": metrics.get("Yellow Cards", ("", ""))[0],
            "red_cards": metrics.get("Red Cards", ("", ""))[0],
            "offsides": metrics.get("Offsides", ("", ""))[0],
        }
        right_values = {
            "ball_possession_pct": metrics.get("Possession", ("", ""))[1],
            "shots_on_goal": metrics.get("Shots (on goal)", ("", ""))[1],
            "corner_kicks": metrics.get("Corner Kicks", ("", ""))[1],
            "fouls": metrics.get("Fouls", ("", ""))[1],
            "yellow_cards": metrics.get("Yellow Cards", ("", ""))[1],
            "red_cards": metrics.get("Red Cards", ("", ""))[1],
            "offsides": metrics.get("Offsides", ("", ""))[1],
        }
        rows.append(_row(match_id, team_left, left_values, f"theScore event {event_id}"))
        rows.append(_row(match_id, team_right, right_values, f"theScore event {event_id}"))

    fieldnames = ["match_id", "team", *DETAIL_FIELDS, "source", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "path": str(path)})


if __name__ == "__main__":
    main()
