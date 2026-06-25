from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any

from kinela.fifa_ranking import normalise_team_name


PERSONNEL_LINEUP_WINDOW = 6
PLAYER_PRODUCTION_WINDOW = 12
_LINEUP_RECENCY_WEIGHTS = (1.0, 0.82, 0.67, 0.55, 0.45, 0.37)
_POSITION_PRIORS = {
    "G": 0.01,
    "D": 0.07,
    "M": 0.16,
    "F": 0.32,
}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fixture_date(item: dict[str, Any]) -> date | None:
    value = str((item.get("fixture") or {}).get("date") or "")[:10]
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _canonical_match_key(item: dict[str, Any]) -> tuple[date, tuple[str, str]] | None:
    match_date = _fixture_date(item)
    teams = item.get("teams") or {}
    home = normalise_team_name(str((teams.get("home") or {}).get("name") or ""))
    away = normalise_team_name(str((teams.get("away") or {}).get("name") or ""))
    if match_date is None or not home or not away:
        return None
    return match_date, tuple(sorted((home, away)))


def load_finished_api_details(data_root: Path) -> list[dict[str, Any]]:
    by_fixture: dict[int, dict[str, Any]] = {}
    details_dir = data_root / "raw" / "api_football" / "fixtures"
    for path in details_dir.glob("details-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("response") or []:
            fixture = item.get("fixture") or {}
            fixture_id = fixture.get("id")
            status = (fixture.get("status") or {}).get("short")
            goals = item.get("goals") or {}
            if (
                fixture_id is None
                or status not in {"FT", "AET", "PEN"}
                or goals.get("home") is None
                or goals.get("away") is None
            ):
                continue
            by_fixture[int(fixture_id)] = item

    # Some provider copies reverse the nominal home/away order. Collapse them
    # before building player histories so one real match cannot count twice.
    by_real_match: dict[tuple[date, tuple[str, str]], dict[str, Any]] = {}
    for item in sorted(
        by_fixture.values(),
        key=lambda row: (
            str((row.get("fixture") or {}).get("date") or ""),
            int((row.get("fixture") or {}).get("id") or 0),
        ),
    ):
        key = _canonical_match_key(item)
        if key is None:
            continue
        current = by_real_match.get(key)
        if current is None:
            by_real_match[key] = item
            continue
        current_score = (
            int(bool(current.get("lineups"))) + int(bool(current.get("events")))
            + int(bool(current.get("players")))
        )
        item_score = (
            int(bool(item.get("lineups"))) + int(bool(item.get("events")))
            + int(bool(item.get("players")))
        )
        if item_score > current_score:
            by_real_match[key] = item
    return sorted(
        by_real_match.values(),
        key=lambda row: (
            str((row.get("fixture") or {}).get("date") or ""),
            int((row.get("fixture") or {}).get("id") or 0),
        ),
    )


def _lineup_players(
    item: dict[str, Any],
) -> dict[str, dict[int, dict[str, float | str]]]:
    teams: dict[str, dict[int, dict[str, float | str]]] = {}
    for lineup in item.get("lineups") or []:
        team_key = normalise_team_name(
            str((lineup.get("team") or {}).get("name") or "")
        )
        if not team_key:
            continue
        players: dict[int, dict[str, float | str]] = {}
        for starter, entries in (
            (True, lineup.get("startXI") or []),
            (False, lineup.get("substitutes") or []),
        ):
            for entry in entries:
                player = entry.get("player") or {}
                player_id = player.get("id")
                if player_id is None:
                    continue
                players[int(player_id)] = {
                    "position": str(player.get("pos") or "M").upper(),
                    "selection_weight": 1.0 if starter else 0.25,
                }
        if players:
            teams[team_key] = players
    return teams


def _player_stat_blocks(
    item: dict[str, Any],
) -> dict[str, dict[int, dict[str, float | str]]]:
    result: dict[str, dict[int, dict[str, float | str]]] = {}
    for team_block in item.get("players") or []:
        team_key = normalise_team_name(
            str((team_block.get("team") or {}).get("name") or "")
        )
        if not team_key:
            continue
        players: dict[int, dict[str, float | str]] = {}
        for player_block in team_block.get("players") or []:
            player = player_block.get("player") or {}
            player_id = player.get("id")
            if player_id is None:
                continue
            stats = (player_block.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            players[int(player_id)] = {
                "position": str(games.get("position") or "M").upper(),
                "minutes": _optional_float(games.get("minutes")) or 0.0,
                "goals": _optional_float((stats.get("goals") or {}).get("total"))
                or 0.0,
                "assists": _optional_float(
                    (stats.get("goals") or {}).get("assists")
                )
                or 0.0,
                "shots": _optional_float((stats.get("shots") or {}).get("total"))
                or 0.0,
                "shots_on_target": _optional_float(
                    (stats.get("shots") or {}).get("on")
                )
                or 0.0,
                "key_passes": _optional_float(
                    (stats.get("passes") or {}).get("key")
                )
                or 0.0,
            }
        if players:
            result[team_key] = players
    return result


def _goal_event_contributions(
    item: dict[str, Any],
) -> dict[str, dict[int, dict[str, float]]]:
    contributions: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"goals": 0.0, "assists": 0.0})
    )
    for event in item.get("events") or []:
        if event.get("type") != "Goal":
            continue
        detail = str(event.get("detail") or "").casefold()
        if detail in {"missed penalty", "own goal"} or "penalty shootout" in str(
            event.get("comments") or ""
        ).casefold():
            continue
        team_key = normalise_team_name(
            str((event.get("team") or {}).get("name") or "")
        )
        if not team_key:
            continue
        player_id = (event.get("player") or {}).get("id")
        assist_id = (event.get("assist") or {}).get("id")
        if player_id is not None:
            contributions[team_key][int(player_id)]["goals"] += (
                0.75 if detail == "penalty" else 1.0
            )
        if assist_id is not None:
            contributions[team_key][int(assist_id)]["assists"] += 1.0
    return {
        team: {player: dict(values) for player, values in players.items()}
        for team, players in contributions.items()
    }


def _player_signal(
    history: deque[dict[str, float | str]],
    position: str,
) -> tuple[float, float]:
    rows = list(history)[-PLAYER_PRODUCTION_WINDOW:]
    if not rows:
        return 0.0, 0.0
    exposure = sum(float(row["exposure"]) for row in rows)
    if exposure <= 0.0:
        return 0.0, 0.0
    prior = _POSITION_PRIORS.get(position[:1].upper(), _POSITION_PRIORS["M"])
    goals = sum(float(row["goals"]) for row in rows)
    assists = sum(float(row["assists"]) for row in rows)
    shots = sum(float(row["shots"]) for row in rows)
    shots_on_target = sum(float(row["shots_on_target"]) for row in rows)
    key_passes = sum(float(row["key_passes"]) for row in rows)
    covered_minutes = sum(float(row["covered_minutes"]) for row in rows)
    posterior_contribution = (
        goals + 0.55 * assists + 4.0 * prior
    ) / (exposure + 4.0)
    contribution_signal = math.tanh(
        (posterior_contribution - prior) / 0.24
    )
    detail_exposure = min(exposure, covered_minutes / 75.0)
    if detail_exposure > 0.0:
        shot_pressure = (
            shots_on_target + 0.20 * shots + 0.18 * key_passes
        ) / detail_exposure
        shot_prior = {"F": 1.25, "M": 0.75, "D": 0.30, "G": 0.02}.get(
            position[:1].upper(),
            0.75,
        )
        shot_signal = math.tanh((shot_pressure - shot_prior) / 1.35)
        detail_confidence = min(1.0, detail_exposure / 4.0)
    else:
        shot_signal = 0.0
        detail_confidence = 0.0
    confidence = min(1.0, exposure / 6.0)
    signal = math.sqrt(confidence) * (
        0.78 * contribution_signal
        + 0.22 * detail_confidence * shot_signal
    )
    return signal, confidence


def _team_summary(
    team_key: str,
    lineup_history: dict[str, deque[dict[int, dict[str, float | str]]]],
    player_history: dict[int, deque[dict[str, float | str]]],
) -> dict[str, float]:
    recent = list(lineup_history.get(team_key, []))[-PERSONNEL_LINEUP_WINDOW:]
    if not recent:
        return {
            "attacking_personnel_signal": 0.0,
            "star_finisher_signal": 0.0,
            "attack_core_signal": 0.0,
            "personnel_coverage": 0.0,
        }
    recency = _LINEUP_RECENCY_WEIGHTS[: len(recent)]
    recency = tuple(reversed(recency))
    denominator = sum(recency)
    selection: dict[int, float] = defaultdict(float)
    positions: dict[int, str] = {}
    for weight, lineup in zip(recency, recent, strict=True):
        for player_id, values in lineup.items():
            selection[player_id] += weight * float(values["selection_weight"])
            positions[player_id] = str(values["position"])
    candidates: list[tuple[float, float, float]] = []
    for player_id, selected_weight in selection.items():
        selection_share = selected_weight / denominator
        position = positions.get(player_id, "M")
        signal, confidence = _player_signal(
            player_history.get(player_id, deque()),
            position,
        )
        if position[:1].upper() not in {"F", "M"} and signal <= 0.0:
            continue
        candidates.append(
            (
                selection_share * signal,
                selection_share,
                confidence,
            )
        )
    candidates.sort(reverse=True)
    top = candidates[:3]
    star = top[0][0] if top else 0.0
    selected_total = sum(item[1] for item in top)
    core = (
        sum(item[0] for item in top) / selected_total
        if selected_total > 0.0
        else 0.0
    )
    lineup_coverage = len(recent) / PERSONNEL_LINEUP_WINDOW
    player_coverage = (
        sum(item[1] * item[2] for item in top) / selected_total
        if selected_total > 0.0
        else 0.0
    )
    coverage = math.sqrt(lineup_coverage * player_coverage)
    return {
        "attacking_personnel_signal": coverage * (0.62 * star + 0.38 * core),
        "star_finisher_signal": coverage * star,
        "attack_core_signal": coverage * core,
        "personnel_coverage": coverage,
    }


def build_personnel_state(
    data_root: Path,
) -> tuple[
    dict[tuple[date, tuple[str, str]], dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, list[tuple[date, dict[str, float]]]],
]:
    fixtures = load_finished_api_details(data_root)
    lineup_history: dict[
        str,
        deque[dict[int, dict[str, float | str]]],
    ] = defaultdict(lambda: deque(maxlen=PERSONNEL_LINEUP_WINDOW))
    player_history: dict[
        int,
        deque[dict[str, float | str]],
    ] = defaultdict(lambda: deque(maxlen=PLAYER_PRODUCTION_WINDOW))
    snapshots: dict[
        tuple[date, tuple[str, str]],
        dict[str, dict[str, float]],
    ] = {}
    timelines: dict[str, list[tuple[date, dict[str, float]]]] = defaultdict(
        list
    )

    for item in fixtures:
        key = _canonical_match_key(item)
        if key is None:
            continue
        teams = item.get("teams") or {}
        team_keys = [
            normalise_team_name(
                str((teams.get(side) or {}).get("name") or "")
            )
            for side in ("home", "away")
        ]
        snapshots[key] = {
            team_key: _team_summary(
                team_key,
                lineup_history,
                player_history,
            )
            for team_key in team_keys
            if team_key
        }

        lineups = _lineup_players(item)
        stats = _player_stat_blocks(item)
        event_contributions = _goal_event_contributions(item)
        for team_key, players in lineups.items():
            lineup_history[team_key].append(players)
            team_stats = stats.get(team_key, {})
            team_events = event_contributions.get(team_key, {})
            for player_id, lineup_values in players.items():
                stat_values = team_stats.get(player_id, {})
                event_values = team_events.get(player_id, {})
                goals = max(
                    float(stat_values.get("goals") or 0.0),
                    float(event_values.get("goals") or 0.0),
                )
                assists = max(
                    float(stat_values.get("assists") or 0.0),
                    float(event_values.get("assists") or 0.0),
                )
                minutes = float(stat_values.get("minutes") or 0.0)
                player_history[player_id].append(
                    {
                        "position": str(
                            stat_values.get("position")
                            or lineup_values.get("position")
                            or "M"
                        ),
                        "exposure": float(lineup_values["selection_weight"]),
                        "covered_minutes": minutes,
                        "goals": goals,
                        "assists": assists,
                        "shots": float(stat_values.get("shots") or 0.0),
                        "shots_on_target": float(
                            stat_values.get("shots_on_target") or 0.0
                        ),
                        "key_passes": float(
                            stat_values.get("key_passes") or 0.0
                        ),
                    }
                )
            timelines[team_key].append(
                (
                    key[0],
                    _team_summary(
                        team_key,
                        lineup_history,
                        player_history,
                    ),
                )
            )

    current = {
        team_key: _team_summary(team_key, lineup_history, player_history)
        for team_key in lineup_history
    }
    return snapshots, current, dict(timelines)


def build_pre_match_personnel_snapshots(
    data_root: Path,
) -> tuple[
    dict[tuple[date, tuple[str, str]], dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
]:
    snapshots, current, _ = build_personnel_state(data_root)
    return snapshots, current


def build_personnel_timelines(
    data_root: Path,
) -> dict[str, list[tuple[date, dict[str, float]]]]:
    _, _, timelines = build_personnel_state(data_root)
    return timelines


def personnel_summary_before(
    timelines: dict[str, list[tuple[date, dict[str, float]]]],
    team: str,
    before: date,
) -> dict[str, float]:
    team_key = normalise_team_name(team)
    for observed_date, summary in reversed(timelines.get(team_key, [])):
        if observed_date < before:
            return summary
    return {
        "attacking_personnel_signal": 0.0,
        "star_finisher_signal": 0.0,
        "attack_core_signal": 0.0,
        "personnel_coverage": 0.0,
    }
