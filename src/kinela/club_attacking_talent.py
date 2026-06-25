from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any

from kinela.fifa_ranking import normalise_team_name
from kinela.player_attacking_personnel import load_finished_api_details


CLUB_TALENT_SEASON = 2024
CLUB_TALENT_AVAILABLE_FROM = date(2025, 7, 1)
FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM = date(2026, 6, 1)
CLUB_LINEUP_WINDOW = 6
_RECENCY_WEIGHTS = (0.37, 0.45, 0.55, 0.67, 0.82, 1.0)
_CLUB_COMPETITION_STRENGTH = {
    "uefa champions league": 10.0,
    "premier league": 9.3,
    "la liga": 9.1,
    "primera division": 9.1,
    "serie a": 8.8,
    "bundesliga": 8.7,
    "ligue 1": 8.2,
    "primeira liga": 7.2,
    "eredivisie": 7.4,
    "championship": 7.0,
    "uefa europa league": 6.8,
    "copa libertadores": 8.0,
    "major league soccer": 6.8,
    "pro league": 6.2,
    "saudi pro league": 6.2,
    "afc champions league elite": 6.0,
    "serie a brazil": 7.0,
}
_EXCLUDED_FOOTBALL_DATA_SCORER_COMPETITIONS = {
    "ec",
    "fifa world cup",
    "wc",
    "european championship",
}


def _optional_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _optional_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _competition_strength(name: str) -> float:
    text = normalise_team_name(name)
    if text in _CLUB_COMPETITION_STRENGTH:
        return _CLUB_COMPETITION_STRENGTH[text]
    for marker, strength in _CLUB_COMPETITION_STRENGTH.items():
        if marker in text:
            return strength
    return 0.0


def load_club_player_attack_profiles(
    data_root: Path,
    *,
    season: int = CLUB_TALENT_SEASON,
) -> dict[int, dict[str, float]]:
    profiles: dict[int, dict[str, float]] = {}
    directory = data_root / "raw" / "api_football" / "players"
    for path in directory.glob(f"player-*-season-{season}.json"):
        try:
            player_id = int(path.name.split("-")[1])
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        response = payload.get("response") or []
        if not response:
            continue
        weighted_minutes = 0.0
        weighted_contributions = 0.0
        raw_minutes = 0.0
        for statistics in response[0].get("statistics") or []:
            league_name = str((statistics.get("league") or {}).get("name") or "")
            strength = _competition_strength(league_name)
            if strength <= 0.0:
                continue
            games = statistics.get("games") or {}
            goals = statistics.get("goals") or {}
            shots = statistics.get("shots") or {}
            penalties = statistics.get("penalty") or {}
            minutes = _optional_float(games.get("minutes"))
            if minutes <= 0.0:
                appearances = _optional_float(games.get("appearences"))
                minutes = 65.0 * appearances
            if minutes <= 0.0:
                continue
            total_goals = _optional_float(goals.get("total"))
            penalty_goals = min(
                total_goals,
                _optional_float(penalties.get("scored")),
            )
            non_penalty_goals = max(0.0, total_goals - penalty_goals)
            assists = _optional_float(goals.get("assists"))
            shots_on_target = _optional_float(shots.get("on"))
            quality_weight = strength / 10.0
            weighted_minutes += minutes * quality_weight
            raw_minutes += minutes
            weighted_contributions += quality_weight * (
                non_penalty_goals
                + 0.60 * assists
                + 0.08 * shots_on_target
                + 0.75 * penalty_goals
            )
        if raw_minutes <= 0.0:
            continue
        profiles[player_id] = {
            "weighted_minutes": weighted_minutes,
            "raw_minutes": raw_minutes,
            "weighted_contributions": weighted_contributions,
        }
    return profiles


def _profile_rate_key(name: str) -> str:
    return normalise_team_name(name)


def _abbreviated_name_key(name: str) -> str:
    key = _profile_rate_key(name)
    parts = key.split()
    if len(parts) < 2:
        return key
    return f"{parts[0][0]} {parts[-1]}"


def _add_profile_values(
    profile: dict[str, float],
    *,
    matches: float,
    goals: float,
    assists: float,
    penalties: float,
    strength: float,
) -> None:
    if matches <= 0.0 or strength <= 0.0:
        return
    minutes = matches * 70.0
    quality_weight = strength / 10.0
    total_goals = max(0.0, goals)
    penalty_goals = min(total_goals, max(0.0, penalties))
    non_penalty_goals = max(0.0, total_goals - penalty_goals)
    profile["weighted_minutes"] += minutes * quality_weight
    profile["raw_minutes"] += minutes
    profile["weighted_contributions"] += quality_weight * (
        non_penalty_goals + 0.60 * max(0.0, assists) + 0.75 * penalty_goals
    )


def load_football_data_scorer_attack_profiles(
    data_root: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Fallback club scorer profiles keyed by (nationality, normalized name).

    API-Football player ids are not always stable across national-team lineups
    and club season endpoints. This fallback uses local football-data scorer
    rows, but only for club competitions with a known strength. National-team
    competitions such as the World Cup or Euros are intentionally excluded.
    """
    path = data_root / "processed" / "football_data" / "scorers.csv"
    if not path.exists():
        return {}
    by_exact_name: dict[
        tuple[str, str],
        dict[str, float | set[str]],
    ] = {}
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        return {}
    for row in rows:
        competition_code = normalise_team_name(str(row.get("competition_code") or ""))
        competition_name = str(row.get("competition_name") or "")
        if (
            competition_code in _EXCLUDED_FOOTBALL_DATA_SCORER_COMPETITIONS
            or normalise_team_name(competition_name)
            in _EXCLUDED_FOOTBALL_DATA_SCORER_COMPETITIONS
        ):
            continue
        season_end = _optional_date(row.get("season_end_date"))
        if (
            season_end is not None
            and season_end > FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM
        ):
            continue
        strength = _competition_strength(competition_name)
        if strength <= 0.0:
            continue
        player = str(row.get("player") or "")
        nationality = normalise_team_name(str(row.get("nationality") or ""))
        if not player or not nationality:
            continue
        key = nationality, _profile_rate_key(player)
        profile = by_exact_name.setdefault(
            key,
            {
                "weighted_minutes": 0.0,
                "raw_minutes": 0.0,
                "weighted_contributions": 0.0,
                "player_ids": set(),
            },
        )
        player_id = str(row.get("player_id") or "")
        if player_id:
            player_ids = profile["player_ids"]
            assert isinstance(player_ids, set)
            player_ids.add(player_id)
        _add_profile_values(
            profile,  # type: ignore[arg-type]
            matches=_optional_float(row.get("played_matches")),
            goals=_optional_float(row.get("goals")),
            assists=_optional_float(row.get("assists")),
            penalties=_optional_float(row.get("penalties")),
            strength=strength,
        )
    profiles: dict[tuple[str, str], dict[str, float]] = {}
    abbreviation_candidates: dict[
        tuple[str, str],
        list[tuple[str, dict[str, float]]],
    ] = defaultdict(list)
    for (nationality, player_key), profile in by_exact_name.items():
        player_ids = profile["player_ids"]
        assert isinstance(player_ids, set)
        raw_minutes = float(profile["raw_minutes"])
        if raw_minutes <= 0.0 or len(player_ids) != 1:
            continue
        clean_profile = {
            "weighted_minutes": float(profile["weighted_minutes"]),
            "raw_minutes": raw_minutes,
            "weighted_contributions": float(profile["weighted_contributions"]),
        }
        profiles[nationality, player_key] = clean_profile
        abbreviation_candidates[nationality, _abbreviated_name_key(player_key)].append(
            (player_key, clean_profile)
        )
    for alias_key, candidates in abbreviation_candidates.items():
        exact_names = {player_key for player_key, _ in candidates}
        if len(exact_names) == 1:
            profiles[alias_key] = candidates[0][1]
    return profiles


def _lineup_players(
    item: dict[str, Any],
) -> dict[str, dict[int, dict[str, float | str]]]:
    result: dict[str, dict[int, dict[str, float | str]]] = {}
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
                position = str(player.get("pos") or "").upper()
                if player_id is None or position not in {"F", "M"}:
                    continue
                players[int(player_id)] = {
                    "name": str(player.get("name") or ""),
                    "position": position,
                    "selection_weight": 1.0 if starter else 0.25,
                }
        if players:
            result[team_key] = players
    return result


def _player_rate(profile: dict[str, float], position: str) -> float:
    prior_rate = 0.28 if position == "F" else 0.14
    prior_minutes = 450.0
    effective_matches = profile["weighted_minutes"] / 90.0
    prior_matches = prior_minutes / 90.0
    return (
        profile["weighted_contributions"] + prior_matches * prior_rate
    ) / (effective_matches + prior_matches)


def _team_summary(
    recent_lineups: deque[dict[int, dict[str, float | str]]],
    profiles: dict[int, dict[str, float]],
    global_rate: float,
    fallback_profiles: dict[tuple[str, str], dict[str, float]] | None = None,
    team_key: str = "",
) -> dict[str, float]:
    recent = list(recent_lineups)[-CLUB_LINEUP_WINDOW:]
    if not recent:
        return {
            "club_attack_talent_signal": 0.0,
            "club_star_finisher_signal": 0.0,
            "club_attack_coverage": 0.0,
        }
    weights = _RECENCY_WEIGHTS[-len(recent) :]
    selection: dict[int, float] = defaultdict(float)
    positions: dict[int, str] = {}
    names: dict[int, str] = {}
    for recency, lineup in zip(weights, recent, strict=True):
        for player_id, values in lineup.items():
            selection[player_id] += recency * float(
                values["selection_weight"]
            )
            positions[player_id] = str(values["position"])
            names[player_id] = str(values.get("name") or "")
    ranked = sorted(selection.items(), key=lambda item: item[1], reverse=True)[:5]
    total_selection = sum(value for _, value in ranked)
    observed: list[tuple[float, float]] = []
    for player_id, selected_weight in ranked:
        profile = profiles.get(player_id)
        if profile is None and fallback_profiles is not None:
            name = names.get(player_id, "")
            profile = fallback_profiles.get(
                (team_key, _profile_rate_key(name)),
            ) or fallback_profiles.get((team_key, _abbreviated_name_key(name)))
        if profile is None:
            continue
        observed.append(
            (
                selected_weight,
                _player_rate(profile, positions.get(player_id, "M")),
            )
        )
    observed_selection = sum(weight for weight, _ in observed)
    coverage = (
        observed_selection / total_selection if total_selection > 0.0 else 0.0
    )
    lineup_coverage = len(recent) / CLUB_LINEUP_WINDOW
    coverage *= lineup_coverage
    if observed_selection <= 0.0:
        return {
            "club_attack_talent_signal": 0.0,
            "club_star_finisher_signal": 0.0,
            "club_attack_coverage": 0.0,
        }
    average_rate = sum(weight * rate for weight, rate in observed) / observed_selection
    star_rate = max(rate for _, rate in observed)
    confidence = math.sqrt(coverage)
    return {
        "club_attack_talent_signal": confidence
        * math.tanh((average_rate - global_rate) / 0.28),
        "club_star_finisher_signal": confidence
        * math.tanh((star_rate - global_rate) / 0.35),
        "club_attack_coverage": coverage,
    }


def build_club_talent_state(
    data_root: Path,
) -> tuple[
    dict[tuple[date, tuple[str, str]], dict[str, dict[str, float]]],
    dict[str, list[tuple[date, dict[str, float]]]],
]:
    profiles = load_club_player_attack_profiles(data_root)
    fallback_profiles = load_football_data_scorer_attack_profiles(data_root)
    fallback_unique_profiles = []
    fallback_seen: set[int] = set()
    for profile in fallback_profiles.values():
        profile_id = id(profile)
        if profile_id in fallback_seen:
            continue
        fallback_seen.add(profile_id)
        fallback_unique_profiles.append(profile)
    api_rates = [
        _player_rate(profile, "F")
        for profile in profiles.values()
    ]
    combined_rates = [
        _player_rate(profile, "F")
        for profile in [*profiles.values(), *fallback_unique_profiles]
    ]
    api_global_rate = (
        sorted(api_rates)[len(api_rates) // 2] if api_rates else 0.25
    )
    combined_global_rate = (
        sorted(combined_rates)[len(combined_rates) // 2]
        if combined_rates
        else api_global_rate
    )
    histories: dict[
        str,
        deque[dict[int, dict[str, float | str]]],
    ] = defaultdict(lambda: deque(maxlen=CLUB_LINEUP_WINDOW))
    snapshots: dict[
        tuple[date, tuple[str, str]],
        dict[str, dict[str, float]],
    ] = {}
    timelines: dict[str, list[tuple[date, dict[str, float]]]] = defaultdict(
        list
    )
    availability_initialized = False
    fallback_initialized = False
    for item in load_finished_api_details(data_root):
        fixture = item.get("fixture") or {}
        try:
            match_date = date.fromisoformat(str(fixture.get("date") or "")[:10])
        except ValueError:
            continue
        teams = item.get("teams") or {}
        team_keys = [
            normalise_team_name(
                str((teams.get(side) or {}).get("name") or "")
            )
            for side in ("home", "away")
        ]
        key = match_date, tuple(sorted(team_keys))
        if (
            match_date >= CLUB_TALENT_AVAILABLE_FROM
            and not availability_initialized
        ):
            for team_key, history in histories.items():
                if history:
                    timelines[team_key].append(
                        (
                            CLUB_TALENT_AVAILABLE_FROM,
                            _team_summary(
                                history,
                                profiles,
                                api_global_rate,
                                None,
                                team_key,
                            ),
                        )
                    )
            availability_initialized = True
        if (
            match_date >= FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM
            and not fallback_initialized
        ):
            for team_key, history in histories.items():
                if history:
                    timelines[team_key].append(
                        (
                            FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM,
                            _team_summary(
                                history,
                                profiles,
                                combined_global_rate,
                                fallback_profiles,
                                team_key,
                            ),
                        )
                    )
            fallback_initialized = True
        active_fallback_profiles = (
            fallback_profiles
            if match_date >= FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM
            else None
        )
        active_global_rate = (
            combined_global_rate
            if match_date >= FOOTBALL_DATA_SCORER_FALLBACK_AVAILABLE_FROM
            else api_global_rate
        )
        if match_date >= CLUB_TALENT_AVAILABLE_FROM:
            snapshots[key] = {
                team_key: _team_summary(
                    histories[team_key],
                    profiles,
                    active_global_rate,
                    active_fallback_profiles,
                    team_key,
                )
                for team_key in team_keys
                if team_key
            }
        for team_key, lineup in _lineup_players(item).items():
            histories[team_key].append(lineup)
            if match_date >= CLUB_TALENT_AVAILABLE_FROM:
                timelines[team_key].append(
                    (
                        match_date,
                        _team_summary(
                            histories[team_key],
                            profiles,
                            active_global_rate,
                            active_fallback_profiles,
                            team_key,
                        ),
                    )
                )
    return snapshots, dict(timelines)


def club_talent_summary_before(
    timelines: dict[str, list[tuple[date, dict[str, float]]]],
    team: str,
    before: date,
) -> dict[str, float]:
    team_key = normalise_team_name(team)
    for observed_date, summary in reversed(timelines.get(team_key, [])):
        if observed_date < before:
            return summary
    return {
        "club_attack_talent_signal": 0.0,
        "club_star_finisher_signal": 0.0,
        "club_attack_coverage": 0.0,
    }
