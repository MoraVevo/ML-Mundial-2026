from collections import defaultdict, deque
from datetime import date

from kinela.player_attacking_personnel import (
    _goal_event_contributions,
    _team_summary,
    personnel_summary_before,
)


def test_goal_events_exclude_own_goals_and_discount_penalties() -> None:
    item = {
        "events": [
            {
                "type": "Goal",
                "detail": "Normal Goal",
                "team": {"name": "Norway"},
                "player": {"id": 9},
                "assist": {"id": 10},
            },
            {
                "type": "Goal",
                "detail": "Penalty",
                "team": {"name": "Norway"},
                "player": {"id": 9},
                "assist": {},
            },
            {
                "type": "Goal",
                "detail": "Own Goal",
                "team": {"name": "Norway"},
                "player": {"id": 4},
                "assist": {},
            },
        ]
    }

    values = _goal_event_contributions(item)["norway"]

    assert values[9]["goals"] == 1.75
    assert values[10]["assists"] == 1.0
    assert 4 not in values


def test_missing_personnel_history_is_neutral() -> None:
    summary = _team_summary("norway", {}, {})

    assert summary["attacking_personnel_signal"] == 0.0
    assert summary["personnel_coverage"] == 0.0


def test_team_summary_uses_only_existing_player_history() -> None:
    lineups = defaultdict(lambda: deque(maxlen=6))
    histories = defaultdict(lambda: deque(maxlen=12))
    for _ in range(6):
        lineups["norway"].append(
            {
                9: {"position": "F", "selection_weight": 1.0},
                10: {"position": "M", "selection_weight": 1.0},
            }
        )
    for _ in range(6):
        histories[9].append(
            {
                "position": "F",
                "exposure": 1.0,
                "covered_minutes": 90.0,
                "goals": 1.0,
                "assists": 0.0,
                "shots": 4.0,
                "shots_on_target": 2.0,
                "key_passes": 0.0,
            }
        )
        histories[10].append(
            {
                "position": "M",
                "exposure": 1.0,
                "covered_minutes": 90.0,
                "goals": 0.0,
                "assists": 0.0,
                "shots": 0.0,
                "shots_on_target": 0.0,
                "key_passes": 0.0,
            }
        )

    summary = _team_summary("norway", lineups, histories)

    assert summary["attacking_personnel_signal"] > 0.0
    assert summary["star_finisher_signal"] > summary["attack_core_signal"]
    assert summary["personnel_coverage"] == 1.0


def test_personnel_timeline_uses_only_strictly_earlier_state() -> None:
    timelines = {
        "norway": [
            (
                date(2026, 6, 1),
                {
                    "attacking_personnel_signal": 0.2,
                    "star_finisher_signal": 0.3,
                    "attack_core_signal": 0.1,
                    "personnel_coverage": 0.8,
                },
            ),
            (
                date(2026, 6, 10),
                {
                    "attacking_personnel_signal": 0.5,
                    "star_finisher_signal": 0.6,
                    "attack_core_signal": 0.4,
                    "personnel_coverage": 1.0,
                },
            ),
        ]
    }

    summary = personnel_summary_before(
        timelines,
        "Norway",
        date(2026, 6, 10),
    )

    assert summary["attacking_personnel_signal"] == 0.2
