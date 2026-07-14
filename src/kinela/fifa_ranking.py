from __future__ import annotations

import csv
import json
import unicodedata
from datetime import date
from html import unescape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
RANKING_API_URL = "https://api.fifa.com/api/v3/rankings/?gender=1&count=300&language=en"
RANKING_SCHEDULES_API_URL = (
    "https://api.fifa.com/api/v3/rankingschedules/all?type=0&gender=1&language=en"
)
RANKING_BY_SCHEDULE_API_URL = (
    "https://api.fifa.com/api/v3/rankingsbyschedule?rankingScheduleId={schedule_id}&language=en"
)

ALIASES = {
    "czechia": "czech republic",
    "turkiye": "turkey",
    "ir iran": "iran",
    "korea republic": "south korea",
    "usa": "united states",
    "cote d'ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "curaçao": "curacao",
    "cape verde": "cabo verde",
    "cape verde islands": "cabo verde",
    "dr congo": "congo dr",
    "democratic republic of congo": "congo dr",
    "congo democratic republic": "congo dr",
    "china pr": "china",
    "hong kong, china": "hong kong",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "rep of ireland": "republic of ireland",
    "fyr macedonia": "north macedonia",
    "kyrgyzstan": "kyrgyz republic",
    "st. kitts and nevis": "saint kitts and nevis",
    "st. lucia": "saint lucia",
    "st. vincent and the grenadines": "saint vincent and the grenadines",
    "trinidad and tobago": "trinidad & tobago",
}


def normalise_team_name(name: str) -> str:
    key = (
        unicodedata.normalize("NFKD", unescape(name or ""))
        .encode("ascii", "ignore")
        .decode()
        .strip()
        .casefold()
    )
    key = key.replace("’", "'").replace(".", "").replace("-", " ")
    key = " ".join(key.split())
    return ALIASES.get(key, key)


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _date_part(value: str | None) -> str:
    return (value or "")[:10]


def _date_from_iso(value: str | None) -> date | None:
    raw = _date_part(value)
    if not raw:
        return None
    return date.fromisoformat(raw)


def _ranking_record(
    item: dict[str, Any],
    *,
    schedule_id: str,
    official_date: str,
    visibility_date: str,
    match_window_end_date: str,
) -> dict[str, Any]:
    name = item.get("TeamName", [{}])[0].get("Description", "")
    return {
        "schedule_id": schedule_id,
        "official_date": official_date,
        "visibility_date": visibility_date,
        "match_window_end_date": match_window_end_date,
        "rank": item.get("Rank"),
        "previous_rank": item.get("PrevRank"),
        "ranking_movement": item.get("RankingMovement"),
        "country_code": item.get("IdCountry"),
        "team_name": name,
        "team_key": normalise_team_name(name),
        "confederation": item.get("ConfederationName"),
        "total_points": item.get("DecimalTotalPoints"),
        "previous_points": item.get("DecimalPrevPoints"),
        "matches": item.get("Matches"),
        "published_at": item.get("PubDate"),
        "previous_published_at": item.get("PrePubDate"),
    }


def collect_fifa_ranking(data_root: Path, *, refresh: bool = False) -> dict[str, Any]:
    raw_dir = data_root / "raw" / "fifa"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_path = raw_dir / "mens_ranking_page.html"
    ranking_path = raw_dir / "mens_ranking_latest.json"

    if refresh or not page_path.exists():
        page_path.write_text(_fetch_text(RANKING_PAGE_URL), encoding="utf-8")
    if refresh or not ranking_path.exists():
        ranking_path.write_text(
            json.dumps(_fetch_json(RANKING_API_URL), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    rows = payload.get("Results", [])
    return {
        "provider": "fifa",
        "source_url": RANKING_PAGE_URL,
        "api_url": RANKING_API_URL,
        "ranking_rows": len(rows),
        "raw_path": str(ranking_path),
        "page_path": str(page_path),
    }


def collect_fifa_ranking_history(
    data_root: Path,
    *,
    refresh: bool = False,
    detail_limit: int = 0,
    from_date: date | None = date(2022, 1, 1),
) -> dict[str, Any]:
    raw_dir = data_root / "raw" / "fifa"
    schedule_dir = raw_dir / "ranking_schedules"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)
    schedules_path = raw_dir / "mens_ranking_schedules.json"

    if refresh or not schedules_path.exists():
        schedules_path.write_text(
            json.dumps(_fetch_json(RANKING_SCHEDULES_API_URL), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = json.loads(schedules_path.read_text(encoding="utf-8"))
    schedules = payload.get("Results", [])
    pending = []
    eligible_schedules = []
    for item in schedules:
        schedule_id = item.get("IdRankingSchedule")
        if not schedule_id:
            continue
        schedule_date = _date_from_iso(item.get("MatchWindowEndDate")) or _date_from_iso(
            item.get("OfficialDate")
        )
        if from_date is not None and schedule_date is not None and schedule_date < from_date:
            continue
        eligible_schedules.append(schedule_id)
        output = schedule_dir / f"{schedule_id}.json"
        if refresh or not output.exists():
            pending.append((schedule_id, output))

    limit = detail_limit or len(pending)
    downloaded = 0
    errors: list[dict[str, str]] = []
    for schedule_id, output in pending[:limit]:
        try:
            output.write_text(
                json.dumps(
                    _fetch_json(RANKING_BY_SCHEDULE_API_URL.format(schedule_id=schedule_id)),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"schedule_id": schedule_id, "error": str(exc)[:300]})
            if any(marker in str(exc).lower() for marker in ("429", "rate", "limit", "quota")):
                break

    return {
        "provider": "fifa",
        "scope": "mens-ranking-history",
        "schedules": len(schedules),
        "eligible_schedules": len(eligible_schedules),
        "from_date": str(from_date) if from_date else None,
        "schedule_raw_path": str(schedules_path),
        "ranking_schedule_dir": str(schedule_dir),
        "missing_before": len(pending),
        "downloaded_now": downloaded,
        "remaining": max(0, len(pending) - downloaded),
        "errors": errors,
    }


def normalize_fifa_ranking(data_root: Path) -> dict[str, Any]:
    raw_path = data_root / "raw" / "fifa" / "mens_ranking_latest.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    output_dir = data_root / "processed" / "fifa"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mens_ranking_latest.csv"

    records: list[dict[str, Any]] = []
    for item in payload.get("Results", []):
        name = item.get("TeamName", [{}])[0].get("Description", "")
        records.append(
            {
                "rank": item.get("Rank"),
                "previous_rank": item.get("PrevRank"),
                "ranking_movement": item.get("RankingMovement"),
                "country_code": item.get("IdCountry"),
                "team_name": name,
                "team_key": normalise_team_name(name),
                "confederation": item.get("ConfederationName"),
                "total_points": item.get("DecimalTotalPoints"),
                "previous_points": item.get("DecimalPrevPoints"),
                "matches": item.get("Matches"),
                "published_at": item.get("PubDate"),
                "previous_published_at": item.get("PrePubDate"),
                "schedule_id": item.get("IdSchedule"),
            }
        )

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return {"rows": len(records), "path": str(output)}


def normalize_fifa_ranking_history(data_root: Path) -> dict[str, Any]:
    schedules_path = data_root / "raw" / "fifa" / "mens_ranking_schedules.json"
    schedule_dir = data_root / "raw" / "fifa" / "ranking_schedules"
    if not schedules_path.exists():
        raise FileNotFoundError("Run `kinela collect fifa-ranking-history` first")

    schedules_payload = json.loads(schedules_path.read_text(encoding="utf-8"))
    schedule_meta: dict[str, dict[str, str]] = {}
    for item in schedules_payload.get("Results", []):
        schedule_id = item.get("IdRankingSchedule")
        if not schedule_id:
            continue
        schedule_meta[schedule_id] = {
            "official_date": _date_part(item.get("OfficialDate")),
            "visibility_date": _date_part(item.get("VisibilityDate")),
            "match_window_end_date": _date_part(item.get("MatchWindowEndDate")),
        }

    records: list[dict[str, Any]] = []
    missing_files = 0
    for schedule_id, meta in schedule_meta.items():
        path = schedule_dir / f"{schedule_id}.json"
        if not path.exists():
            missing_files += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("Results", []):
            records.append(_ranking_record(item, schedule_id=schedule_id, **meta))

    records.sort(key=lambda row: (row["match_window_end_date"], int(row["rank"] or 9999)))
    output_dir = data_root / "processed" / "fifa"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mens_ranking_history.csv"
    if not records:
        raise FileNotFoundError("No cached ranking schedule payloads to normalize")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return {
        "rows": len(records),
        "schedules_with_rows": len({row["schedule_id"] for row in records}),
        "missing_schedule_files": missing_files,
        "path": str(output),
    }


def load_fifa_ranking(data_root: Path) -> dict[str, dict[str, float | int | str]]:
    path = data_root / "processed" / "fifa" / "mens_ranking_latest.csv"
    if not path.exists():
        return {}
    rankings: dict[str, dict[str, float | int | str]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = row["team_key"]
        rankings[key] = {
            "rank": int(row["rank"]) if row.get("rank") else 0,
            "points": float(row["total_points"]) if row.get("total_points") else 0.0,
            "confederation": row.get("confederation", ""),
        }
    return rankings


def load_fifa_ranking_history(data_root: Path) -> dict[str, list[dict[str, float | int | str | date]]]:
    path = data_root / "processed" / "fifa" / "mens_ranking_history.csv"
    if not path.exists():
        return {}
    rankings: dict[str, list[dict[str, float | int | str | date]]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = row["team_key"]
        # Rankings are pre-match information only after FIFA publishes them.
        # MatchWindowEndDate precedes publication and would leak the new table
        # into matches played during that gap.
        effective_date = (
            _date_from_iso(row.get("published_at"))
            or _date_from_iso(row.get("visibility_date"))
            or _date_from_iso(row.get("official_date"))
        )
        if effective_date is None:
            continue
        rankings.setdefault(key, []).append(
            {
                "rank": int(row["rank"]) if row.get("rank") else 0,
                "points": float(row["total_points"]) if row.get("total_points") else 0.0,
                "confederation": row.get("confederation", ""),
                "schedule_id": row.get("schedule_id", ""),
                "effective_date": effective_date,
            }
        )
    for rows in rankings.values():
        rows.sort(key=lambda item: item["effective_date"])
    return rankings


def fifa_ranking_for_match_date(
    history: dict[str, list[dict[str, float | int | str | date]]],
    team: str,
    match_date: date,
) -> dict[str, float | int | str | date] | None:
    rows = history.get(normalise_team_name(team), [])
    selected = None
    for item in rows:
        effective_date = item["effective_date"]
        if isinstance(effective_date, date) and effective_date < match_date:
            selected = item
        else:
            break
    return selected


def update_live_fifa_points(
    team_a_points: float,
    team_b_points: float,
    team_a_goals: int,
    team_b_goals: int,
    *,
    importance_weight: float = 1.0,
) -> tuple[float, float]:
    """Apply a small causal Elo-style update between official FIFA releases."""
    expected_a = 1.0 / (1.0 + 10 ** ((team_b_points - team_a_points) / 600.0))
    actual_a = (
        1.0
        if team_a_goals > team_b_goals
        else 0.5
        if team_a_goals == team_b_goals
        else 0.0
    )
    margin = min(abs(team_a_goals - team_b_goals), 3)
    change = (
        max(0.0, float(importance_weight))
        * 5.0
        * (1.0 + 0.10 * margin)
        * (actual_a - expected_a)
    )
    return team_a_points + change, team_b_points - change


def update_fifa_sum_points(
    team_a_points: float,
    team_b_points: float,
    *,
    team_a_result: float,
    team_b_result: float,
    importance: float,
    protect_negative: bool = False,
) -> tuple[float, float]:
    """Apply FIFA's published SUM formula to one completed match.

    Results are 1/0 for a normal win/loss, 0.5/0.5 for a draw and
    0.75/0.5 for a penalty-shootout winner/loser.  Final-competition
    knockout rounds protect each team independently from negative points.
    """
    expected_a = 1.0 / (10.0 ** (-(team_a_points - team_b_points) / 600.0) + 1.0)
    expected_b = 1.0 - expected_a
    change_a = max(0.0, float(importance)) * (float(team_a_result) - expected_a)
    change_b = max(0.0, float(importance)) * (float(team_b_result) - expected_b)
    if protect_negative:
        change_a = max(0.0, change_a)
        change_b = max(0.0, change_b)
    return team_a_points + change_a, team_b_points + change_b


def fifa_sum_match_importance(
    competition_name: str,
    competition_type: str,
    stage_or_round: str,
) -> float:
    """Return FIFA SUM's published importance bucket for a national match."""
    competition = (competition_name or "").casefold()
    competition_kind = (competition_type or "").casefold()
    stage = (stage_or_round or "").upper()
    if competition == "fifa world cup":
        return (
            60.0
            if stage in {"QUARTER_FINALS", "SEMI_FINALS", "THIRD_PLACE", "FINAL"}
            else 50.0
        )
    if competition_kind == "qualifier":
        return 25.0
    if competition_kind == "friendly":
        # The source data does not identify out-of-window friendlies reliably;
        # use the official in-window value and retain this limitation in audits.
        return 10.0
    if "nations league" in competition:
        return 15.0 if "GROUP" in stage else 25.0
    knockout = any(marker in stage for marker in ("QUARTER", "SEMI", "FINAL"))
    return 40.0 if knockout else 35.0
