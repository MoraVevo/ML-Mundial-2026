from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from kinela.http import CachedJsonClient


ESPN_SOCCER_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WORLD_CUP_SOCCER_BASE_URL = f"{ESPN_SOCCER_BASE_URL}/fifa.world"

TEAM_ALIASES = {
    "bosnia and herzegovina": "bosnia-herzegovina",
    "cabo verde": "cape verde islands",
    "cape verde": "cape verde islands",
    "cote d'ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "czech republic": "czechia",
    "ir iran": "iran",
    "korea republic": "south korea",
    "türkiye": "turkey",
    "turkiye": "turkey",
}

ESPN_STRATEGIC_STATS = {
    "foulsCommitted": "fouls",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "offsides": "offsides",
    "wonCorners": "corner_kicks",
    "saves": "goalkeeper_saves",
    "possessionPct": "ball_possession_pct",
    "totalShots": "total_shots",
    "shotsOnTarget": "shots_on_goal",
    "shotPct": "shot_pct",
    "penaltyKickGoals": "penalty_kick_goals",
    "penaltyKickShots": "penalty_kick_shots",
    "accuratePasses": "passes_accurate",
    "totalPasses": "total_passes",
    "passPct": "passes_pct",
    "accurateCrosses": "crosses_accurate",
    "totalCrosses": "total_crosses",
    "crossPct": "crosses_pct",
    "accurateLongBalls": "long_balls_accurate",
    "totalLongBalls": "total_long_balls",
    "longballPct": "long_balls_pct",
    "blockedShots": "blocked_shots",
    "effectiveTackles": "tackles_effective",
    "totalTackles": "total_tackles",
    "tacklePct": "tackles_pct",
    "interceptions": "interceptions",
    "effectiveClearance": "clearances_effective",
    "totalClearance": "total_clearances",
}

ESPN_LEADER_STRATEGIC_STATS = {
    "expectedGoals": "espn_top_xg",
    "expectedGoalsConceded": "espn_keeper_xg_conceded",
    "duelsWon": "espn_top_duels_won",
    "bigChanceCreated": "espn_top_big_chances_created",
    "bigChanceMissed": "espn_top_big_chances_missed",
}

STRATEGIC_FIELDS = list(
    dict.fromkeys([*ESPN_STRATEGIC_STATS.values(), *ESPN_LEADER_STRATEGIC_STATS.values()])
)


def normalize_team_name(value: str) -> str:
    text = value.casefold().replace(".", "").strip()
    text = re.sub(r"\s+", " ", text)
    return TEAM_ALIASES.get(text, text)


def _date_key(day: date) -> str:
    return day.strftime("%Y%m%d")


def _display_value(stat: dict[str, Any]) -> str:
    value = stat.get("displayValue")
    if value not in (None, ""):
        return str(value).replace("%", "").strip()
    value = stat.get("value")
    if value in (None, ""):
        return ""
    return str(value).replace("%", "").strip()


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_strategic_team_rows(
    summary_payload: dict[str, Any],
    *,
    match_id: str = "",
    event_id: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    resolved_event_id = event_id or str(summary_payload.get("header", {}).get("id", ""))
    leader_stats = extract_leader_team_stats(summary_payload)
    for team_block in summary_payload.get("boxscore", {}).get("teams", []):
        team = team_block.get("team", {}).get("displayName", "")
        if not team:
            continue
        row = {
            "match_id": match_id,
            "espn_event_id": resolved_event_id,
            "team": team,
            **{field: "" for field in STRATEGIC_FIELDS},
        }
        for stat in team_block.get("statistics", []):
            field = ESPN_STRATEGIC_STATS.get(stat.get("name", ""))
            if field:
                row[field] = _display_value(stat)
        row.update(leader_stats.get(normalize_team_name(team), {}))
        row.update(derived_strategic_signals(row))
        rows.append(row)
    return rows


def extract_leader_team_stats(summary_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for group in summary_payload.get("leaders") or []:
        team = (group.get("team") or {}).get("displayName", "")
        team_key = normalize_team_name(team)
        if not team_key:
            continue
        row = rows.setdefault(
            team_key,
            {field: "" for field in ESPN_LEADER_STRATEGIC_STATS.values()},
        )
        for leader_category in group.get("leaders") or []:
            for entry in leader_category.get("leaders") or []:
                for stat in entry.get("statistics") or []:
                    field = ESPN_LEADER_STRATEGIC_STATS.get(stat.get("name", ""))
                    if not field:
                        continue
                    value = _float_or_none(_display_value(stat))
                    current = _float_or_none(row.get(field, ""))
                    if value is not None and (current is None or value > current):
                        row[field] = _display_value(stat)
    return rows


def derived_strategic_signals(row: dict[str, str]) -> dict[str, str]:
    possession = _float_or_none(row.get("ball_possession_pct", ""))
    total_passes = _float_or_none(row.get("total_passes", ""))
    total_clearances = _float_or_none(row.get("total_clearances", ""))
    total_long_balls = _float_or_none(row.get("total_long_balls", ""))
    total_shots = _float_or_none(row.get("total_shots", ""))
    shots_on_goal = _float_or_none(row.get("shots_on_goal", ""))

    signals: dict[str, str] = {
        "low_block_profile_score": "",
        "direct_play_share": "",
        "clearances_per_30pct_opp_possession": "",
        "shot_on_goal_rate": "",
    }

    if possession is not None and total_passes is not None and total_clearances is not None:
        low_block_score = 0.0
        low_block_score += max(0.0, min(1.0, (42.0 - possession) / 22.0)) * 0.45
        low_block_score += max(0.0, min(1.0, (380.0 - total_passes) / 230.0)) * 0.25
        low_block_score += max(0.0, min(1.0, total_clearances / 30.0)) * 0.30
        signals["low_block_profile_score"] = f"{low_block_score:.3f}"

    if total_long_balls is not None and total_passes:
        signals["direct_play_share"] = f"{total_long_balls / total_passes:.3f}"

    if possession is not None and total_clearances is not None:
        opponent_possession = max(1.0, 100.0 - possession)
        signals["clearances_per_30pct_opp_possession"] = (
            f"{total_clearances / opponent_possession * 30.0:.3f}"
        )

    if total_shots and shots_on_goal is not None:
        signals["shot_on_goal_rate"] = f"{shots_on_goal / total_shots:.3f}"

    return signals


def coverage_by_field(rows: list[dict[str, str]]) -> dict[str, int]:
    fields = [field for field in rows[0].keys() if field not in {"match_id", "espn_event_id", "team"}] if rows else []
    return {
        field: sum(1 for row in rows if row.get(field) not in (None, ""))
        for field in fields
    }


class EspnWorldCupClient:
    def __init__(
        self,
        data_root: Path,
        *,
        league_slug: str = "fifa.world",
        min_interval_seconds: float = 1.25,
    ) -> None:
        self.league_slug = league_slug
        cache_name = "worldcup_2026" if league_slug == "fifa.world" else league_slug
        self.client = CachedJsonClient(
            f"{ESPN_SOCCER_BASE_URL}/{league_slug}",
            data_root / "raw" / "espn" / cache_name,
            min_interval_seconds=min_interval_seconds,
            headers={"Accept": "application/json"},
        )

    def scoreboard(self, day: date, *, refresh: bool = False) -> dict[str, Any]:
        key = _date_key(day)
        return self.client.get(
            "scoreboard",
            {"dates": key},
            cache_name=f"scoreboard_{key}.json",
            refresh=refresh,
        )

    def summary(self, event_id: str, *, refresh: bool = False) -> dict[str, Any]:
        return self.client.get(
            "summary",
            {"event": event_id},
            cache_name=f"summary_{event_id}.json",
            refresh=refresh,
        )
