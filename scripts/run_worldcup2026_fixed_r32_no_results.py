from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.worldcup_2026 import (  # noqa: E402
    QUARTER_FINALS,
    R32_SLOTS,
    ROUND_OF_16,
    SEMI_FINALS,
    WorldCup2026Simulator,
)


KNOCKOUT_DATES = {
    "ROUND_OF_32": date(2026, 6, 28),
    "LAST_16": date(2026, 7, 4),
    "QUARTER_FINALS": date(2026, 7, 9),
    "SEMI_FINALS": date(2026, 7, 14),
    "THIRD_PLACE": date(2026, 7, 18),
    "FINAL": date(2026, 7, 19),
}


def _default_model_path(data_root: Path) -> Path:
    # This is the no-16avos-leakage default. Pass --model-path explicitly if you
    # want to compare against an all-played/future artifact.
    return data_root / "models" / "lightgbm_neutral_worldcup_holdout.joblib"


def _drop_manual_knockout_results(simulator: WorldCup2026Simulator) -> None:
    simulator.manual_results = {
        match_id: row
        for match_id, row in simulator.manual_results.items()
        if row.get("stage") == "GROUP_STAGE"
    }


def _is_round_of_32_history_row(row: dict[str, Any]) -> bool:
    return (
        row.get("source") == "manual-worldcup-2026"
        and str(row.get("stage_or_round", "")).upper() == "ROUND_OF_32"
    )


def _drop_round_of_32_from_feature_context(simulator: WorldCup2026Simulator) -> int:
    original_count = len(simulator.history)
    simulator.history = [
        row for row in simulator.history if not _is_round_of_32_history_row(row)
    ]
    removed = original_count - len(simulator.history)
    simulator.team_histories = simulator._index_team_histories()
    simulator.head_to_head_histories = simulator._index_head_to_head_histories()
    simulator.elo_ratings = simulator._build_elo_ratings()
    simulator.confederation_stats, simulator.team_cross_confederation_stats = (
        simulator._build_confederation_contexts()
    )
    contexts = simulator._goal_contexts()
    simulator.global_goal_avg = contexts["global_avg"]
    simulator.major_goal_avg = contexts["major_avg"]
    simulator.group_goal_avg = contexts["group_avg"]
    simulator.knockout_goal_avg = contexts["knockout_avg"]
    simulator.major_match_count = contexts["major_matches"]
    simulator.group_match_count = contexts["group_matches"]
    simulator.knockout_match_count = contexts["knockout_matches"]
    simulator.fifa_point_overrides = simulator._build_fifa_point_overrides()
    simulator.prediction_cache.clear()
    return removed


def _top_items(counter: Counter[str], runs: int, limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "team": team,
            "count": count,
            "probability": round(count / runs, 6),
        }
        for team, count in counter.most_common(limit)
    ]


def _top_tuple_items(
    counter: Counter[tuple[str, str]],
    runs: int,
    limit: int = 6,
) -> list[dict[str, Any]]:
    return [
        {
            "team_a": team_a,
            "team_b": team_b,
            "count": count,
            "probability": round(count / runs, 6),
        }
        for (team_a, team_b), count in counter.most_common(limit)
    ]


def _knockout_signature(
    records: list[dict[str, Any]],
) -> tuple[tuple[str, str, str, str, str], ...]:
    rows = [
        (
            str(row["match_id"]),
            row["stage"],
            row["team_a"],
            row["team_b"],
            row["winner"],
        )
        for row in records
        if row["stage"] != "GROUP_STAGE"
    ]
    return tuple(
        sorted(rows, key=lambda item: int(item[0]) if item[0].isdigit() else 999)
    )


def _signature_to_rows(
    signature: tuple[tuple[str, str, str, str, str], ...],
    support: int,
    runs: int,
) -> dict[str, Any]:
    return {
        "support_count": support,
        "support_probability": round(support / runs, 6),
        "winner": next(
            (
                winner
                for match_id, _stage, _team_a, _team_b, winner in signature
                if match_id == "103"
            ),
            "",
        ),
        "matches": [
            {
                "match_id": match_id,
                "stage": stage,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
            }
            for match_id, stage, team_a, team_b, winner in signature
        ],
    }


def _resolve_r32_matchups(
    simulator: WorldCup2026Simulator,
) -> dict[int, tuple[str, str]]:
    standings = simulator.simulate_groups()
    thirds = simulator._best_thirds(standings)
    assignments = simulator._third_place_match_assignments(thirds)
    used_thirds: set[str] = set()
    matchups: dict[int, tuple[str, str]] = {}
    for match_id, slot_a, slot_b in R32_SLOTS:
        team_a = simulator._resolve_slot(
            match_id,
            slot_a,
            standings,
            thirds,
            used_thirds,
            assignments,
        )
        team_b = simulator._resolve_slot(
            match_id,
            slot_b,
            standings,
            thirds,
            used_thirds,
            assignments,
        )
        matchups[match_id] = (team_a, team_b)
    return matchups


def _simulate_once(
    simulator: WorldCup2026Simulator,
    r32_matchups: dict[int, tuple[str, str]],
) -> tuple[dict[int, str], list[str], str, str, str, str]:
    winners: dict[int, str] = {}
    for match_id, (team_a, team_b) in r32_matchups.items():
        _a_goals, _b_goals, winner = simulator.simulate_match(
            team_a,
            team_b,
            KNOCKOUT_DATES["ROUND_OF_32"],
            "ROUND_OF_32",
            match_id=match_id,
        )
        winners[match_id] = winner or team_a

    for match_id, previous_a, previous_b in ROUND_OF_16:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        _a_goals, _b_goals, winner = simulator.simulate_match(
            team_a,
            team_b,
            KNOCKOUT_DATES["LAST_16"],
            "LAST_16",
            match_id=match_id,
        )
        winners[match_id] = winner or team_a

    for match_id, previous_a, previous_b in QUARTER_FINALS:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        _a_goals, _b_goals, winner = simulator.simulate_match(
            team_a,
            team_b,
            KNOCKOUT_DATES["QUARTER_FINALS"],
            "QUARTER_FINALS",
            match_id=match_id,
        )
        winners[match_id] = winner or team_a

    semifinal_losers: list[str] = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        _a_goals, _b_goals, winner = simulator.simulate_match(
            team_a,
            team_b,
            KNOCKOUT_DATES["SEMI_FINALS"],
            "SEMI_FINALS",
            match_id=match_id,
        )
        winners[match_id] = winner or team_a
        semifinal_losers.append(team_b if winners[match_id] == team_a else team_a)

    _a_goals, _b_goals, third_place = simulator.simulate_match(
        semifinal_losers[0],
        semifinal_losers[1],
        KNOCKOUT_DATES["THIRD_PLACE"],
        "THIRD_PLACE",
        match_id=104,
    )
    third_place = third_place or semifinal_losers[0]
    fourth_place = semifinal_losers[1] if third_place == semifinal_losers[0] else semifinal_losers[0]

    finalist_a = winners[101]
    finalist_b = winners[102]
    _a_goals, _b_goals, champion = simulator.simulate_match(
        finalist_a,
        finalist_b,
        KNOCKOUT_DATES["FINAL"],
        "FINAL",
        match_id=103,
    )
    champion = champion or finalist_a
    runner_up = finalist_b if champion == finalist_a else finalist_a
    return winners, semifinal_losers, champion, runner_up, third_place, fourth_place


def run_fixed_r32_no_results(
    data_root: Path,
    *,
    runs: int,
    seed: int,
    model_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine="lightgbm")
    simulator.lightgbm_model = joblib.load(model_path)
    _drop_manual_knockout_results(simulator)
    removed_history_rows = _drop_round_of_32_from_feature_context(simulator)
    base_fifa_overrides = dict(simulator.fifa_point_overrides)

    started = datetime.now()
    r32_matchups = _resolve_r32_matchups(simulator)
    simulator.simulated_histories = defaultdict(list)
    simulator.current_match_records = []
    simulator.fifa_point_overrides = dict(base_fifa_overrides)
    simulator.prediction_cache.clear()

    champions: Counter[str] = Counter()
    runners_up: Counter[str] = Counter()
    third_place: Counter[str] = Counter()
    fourth_place: Counter[str] = Counter()
    finalists: Counter[str] = Counter()
    semifinalists: Counter[str] = Counter()
    slot_matchups: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    slot_winners: dict[str, Counter[str]] = defaultdict(Counter)
    full_brackets: Counter[tuple[tuple[str, str, str, str, str], ...]] = Counter()

    for index in range(1, runs + 1):
        simulator.simulated_histories = defaultdict(list)
        simulator.current_match_records = []
        simulator.fifa_point_overrides = dict(base_fifa_overrides)
        simulator.prediction_cache.clear()
        # Group-stage results are already in the filtered base history and live
        # FIFA overrides. Replaying groups here would double-count them.

        winners, semi_losers, champion, runner_up, third, fourth = _simulate_once(
            simulator,
            r32_matchups,
        )
        champions[champion] += 1
        runners_up[runner_up] += 1
        third_place[third] += 1
        fourth_place[fourth] += 1
        finalists[winners[101]] += 1
        finalists[winners[102]] += 1
        semifinalists[winners[97]] += 1
        semifinalists[winners[98]] += 1
        semifinalists[winners[99]] += 1
        semifinalists[winners[100]] += 1
        for loser in semi_losers:
            semifinalists[loser] += 0

        full_brackets[_knockout_signature(simulator.current_match_records)] += 1
        for row in simulator.current_match_records:
            if row["stage"] == "GROUP_STAGE":
                continue
            match_id = str(row["match_id"])
            slot_matchups[match_id][(row["team_a"], row["team_b"])] += 1
            slot_winners[match_id][row["winner"]] += 1

        if progress_every and index % progress_every == 0:
            elapsed = (datetime.now() - started).total_seconds()
            leader, count = champions.most_common(1)[0]
            print(
                f"{index}/{runs} runs | lider campeon={leader} "
                f"({count / index:.1%}) | {elapsed:.1f}s",
                flush=True,
            )

    teams = sorted(
        set(champions)
        | set(runners_up)
        | set(third_place)
        | set(fourth_place)
        | set(finalists)
        | set(semifinalists),
        key=lambda team: (-champions[team], -finalists[team], team),
    )
    team_summary = [
        {
            "equipo": team,
            "veces_campeon": champions[team],
            "prob_campeon": round(champions[team] / runs, 6),
            "veces_segundo": runners_up[team],
            "prob_segundo": round(runners_up[team] / runs, 6),
            "veces_tercero": third_place[team],
            "prob_tercero": round(third_place[team] / runs, 6),
            "veces_cuarto": fourth_place[team],
            "prob_cuarto": round(fourth_place[team] / runs, 6),
            "prob_final": round(finalists[team] / runs, 6),
            "prob_semifinal": round(semifinalists[team] / runs, 6),
        }
        for team in teams
    ]
    slot_summary = [
        {
            "match_id": match_id,
            "top_matchups": _top_tuple_items(slot_matchups[match_id], runs),
            "top_winners": _top_items(slot_winners[match_id], runs),
        }
        for match_id in sorted(slot_matchups, key=lambda value: int(value))
    ]
    return {
        "metadata": {
            "runs": runs,
            "seed": seed,
            "engine": "lightgbm",
            "model_path": str(model_path),
            "simulation_mode": "fixed_r32_full_context_no_round_of_32_results",
            "simulation_mode_note": (
                "Completed group-stage results are replayed to build the bracket. "
                "All Round of 32 matches are simulated as unknown, including matches "
                "that may already be present in manual results."
            ),
            "round_of_32_history_rows_removed": removed_history_rows,
            "third_place_assignment": "exact_495_combination_table",
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        "round_of_32_matchups": [
            {"match_id": match_id, "team_a": team_a, "team_b": team_b}
            for match_id, (team_a, team_b) in sorted(r32_matchups.items())
        ],
        "team_summary": team_summary,
        "top_champion": _top_items(champions, runs),
        "top_runner_up": _top_items(runners_up, runs),
        "top_third_place": _top_items(third_place, runs),
        "top_fourth_place": _top_items(fourth_place, runs),
        "slot_summary": slot_summary,
        "most_frequent_full_bracket": _signature_to_rows(
            *full_brackets.most_common(1)[0],
            runs,
        )
        if full_brackets
        else {},
    }


def _write_outputs(result: dict[str, Any], output_path: Path) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    team_csv = output_path.with_name(output_path.stem + "_teams.csv")
    with team_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["team_summary"][0]))
        writer.writeheader()
        writer.writerows(result["team_summary"])

    matchup_csv = output_path.with_name(output_path.stem + "_matchups.csv")
    with matchup_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "match_id",
            "rank",
            "team_a",
            "team_b",
            "matchup_count",
            "matchup_probability",
            "winner",
            "winner_count",
            "winner_probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for slot in result["slot_summary"]:
            winners = slot["top_winners"]
            for rank, matchup in enumerate(slot["top_matchups"], start=1):
                winner = winners[rank - 1] if rank <= len(winners) else {}
                writer.writerow(
                    {
                        "match_id": slot["match_id"],
                        "rank": rank,
                        "team_a": matchup["team_a"],
                        "team_b": matchup["team_b"],
                        "matchup_count": matchup["count"],
                        "matchup_probability": matchup["probability"],
                        "winner": winner.get("team", ""),
                        "winner_count": winner.get("count", ""),
                        "winner_probability": winner.get("probability", ""),
                    }
                )

    return {
        "json_path": str(output_path),
        "team_csv_path": str(team_csv),
        "matchup_csv_path": str(matchup_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = args.model_path or _default_model_path(args.data_root)
    output = args.output or Path("outputs") / (
        f"worldcup2026_fixed_r32_no_results_{args.runs}_{date.today().isoformat()}.json"
    )
    result = run_fixed_r32_no_results(
        args.data_root,
        runs=args.runs,
        seed=args.seed,
        model_path=model_path,
        progress_every=args.progress_every,
    )
    paths = _write_outputs(result, output)
    print(json.dumps({**paths, **result["metadata"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
