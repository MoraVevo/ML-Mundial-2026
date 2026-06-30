from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.worldcup_2026 import QUARTER_FINALS, SEMI_FINALS, WorldCup2026Simulator

KNOCKOUT_STAGES = {
    "ROUND_OF_32",
    "LAST_16",
    "QUARTER_FINALS",
    "SEMI_FINALS",
    "THIRD_PLACE",
    "FINAL",
}


def _selected_model_path(data_root: Path, engine: str) -> str:
    if engine != "lightgbm":
        return ""
    for path in (
        data_root / "models" / "lightgbm_neutral_all_played_wc2026.joblib",
        data_root / "models" / "lightgbm_neutral_model.joblib",
    ):
        if path.exists():
            return str(path)
    return ""


def _top_items(counter: Counter[str], runs: int, limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"item": item, "count": count, "probability": round(count / runs, 6)}
        for item, count in counter.most_common(limit)
    ]


def _top_tuple_items(
    counter: Counter[tuple[str, str]],
    runs: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return [
        {"team_a": team_a, "team_b": team_b, "count": count, "probability": round(count / runs, 6)}
        for (team_a, team_b), count in counter.most_common(limit)
    ]


def _semifinal_losers(winners: dict[str, str]) -> list[str]:
    losers: list[str] = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[str(previous_a)]
        team_b = winners[str(previous_b)]
        losers.append(team_b if winners[str(match_id)] == team_a else team_a)
    return losers


def _knockout_signature(records: list[dict[str, Any]]) -> tuple[tuple[str, str, str, str, str], ...]:
    bracket_rows = []
    for row in records:
        if row["stage"] not in KNOCKOUT_STAGES:
            continue
        bracket_rows.append(
            (
                str(row["match_id"]),
                row["stage"],
                row["team_a"],
                row["team_b"],
                row["winner"],
            )
        )
    return tuple(sorted(bracket_rows, key=lambda item: int(item[0]) if item[0].isdigit() else 999))


def _signature_to_rows(
    signature: tuple[tuple[str, str, str, str, str], ...],
    support: int,
    runs: int,
) -> dict[str, Any]:
    return {
        "support_count": support,
        "support_probability": round(support / runs, 6),
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


def run_consensus_bracket_simulation(
    data_root: Path,
    *,
    runs: int,
    seed: int,
    progress_every: int,
    frozen_context_cache: bool,
    model_path: Path | None,
    model_label: str,
) -> dict[str, Any]:
    engine = "lightgbm"
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine=engine)
    selected_model_path = str(model_path) if model_path else _selected_model_path(data_root, engine)
    if model_path:
        simulator.lightgbm_model = joblib.load(model_path)
        simulator.prediction_cache.clear()
    if simulator.lightgbm_model is None:
        raise FileNotFoundError(
            "Missing trained LightGBM model. Run "
            "`python scripts\\predict_next4_with_all_played_worldcup.py --limit 1` first."
        )
    if frozen_context_cache:
        class NoClearPredictionCache(dict):
            def clear(self) -> None:
                return None

        simulator.prediction_cache = NoClearPredictionCache(simulator.prediction_cache)
        original_lightgbm_prediction = simulator.lightgbm_prediction

        def cached_lightgbm_prediction(
            team_a: str,
            team_b: str,
            match_date,
            stage: str,
            group_table=None,
        ) -> dict[str, Any]:
            return original_lightgbm_prediction(team_a, team_b, match_date, stage, None)

        simulator.lightgbm_prediction = cached_lightgbm_prediction  # type: ignore[method-assign]

    champions: Counter[str] = Counter()
    finalists: Counter[str] = Counter()
    semifinalists: Counter[str] = Counter()
    first: Counter[str] = Counter()
    second: Counter[str] = Counter()
    third: Counter[str] = Counter()
    fourth: Counter[str] = Counter()
    best_thirds: Counter[str] = Counter()
    group_positions: dict[str, Counter[str]] = defaultdict(Counter)
    slot_matchups: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    slot_winners: dict[str, Counter[str]] = defaultdict(Counter)
    full_brackets: Counter[tuple[tuple[str, str, str, str, str], ...]] = Counter()

    started = datetime.now()
    for index in range(1, runs + 1):
        champion, winners, standings, best_third_rows = simulator.simulate_tournament()

        finalist_a = winners["101"]
        finalist_b = winners["102"]
        runner_up = finalist_b if champion == finalist_a else finalist_a
        semi_losers = _semifinal_losers(winners)
        third_place = winners["104"]
        fourth_place = semi_losers[1] if third_place == semi_losers[0] else semi_losers[0]

        champions[champion] += 1
        finalists[finalist_a] += 1
        finalists[finalist_b] += 1
        first[champion] += 1
        second[runner_up] += 1
        third[third_place] += 1
        fourth[fourth_place] += 1
        for match_id, _previous_a, _previous_b in QUARTER_FINALS:
            semifinalists[winners[str(match_id)]] += 1
        for standing in best_third_rows:
            best_thirds[standing.team] += 1
        for ranking in standings.values():
            for position, standing in enumerate(ranking, start=1):
                group_positions[standing.team][str(position)] += 1

        full_brackets[_knockout_signature(simulator.current_match_records)] += 1
        for row in simulator.current_match_records:
            if row["stage"] not in KNOCKOUT_STAGES:
                continue
            match_id = str(row["match_id"])
            slot_matchups[match_id][(row["team_a"], row["team_b"])] += 1
            slot_winners[match_id][row["winner"]] += 1

        if progress_every and index % progress_every == 0:
            elapsed = (datetime.now() - started).total_seconds()
            print(
                f"{index}/{runs} simulaciones | campeon={champion} | "
                f"lider={champions.most_common(1)[0][0]} "
                f"({champions.most_common(1)[0][1]}/{index}) | {elapsed:.1f}s",
                flush=True,
            )

    teams = sorted(
        set(first) | set(second) | set(third) | set(fourth) | set(finalists) | set(semifinalists),
        key=lambda team: (-first[team], -finalists[team], team),
    )
    team_summary = [
        {
            "equipo": team,
            "veces_campeon": first[team],
            "prob_campeon": round(first[team] / runs, 6),
            "veces_segundo": second[team],
            "prob_segundo": round(second[team] / runs, 6),
            "veces_tercero": third[team],
            "prob_tercero": round(third[team] / runs, 6),
            "veces_cuarto": fourth[team],
            "prob_cuarto": round(fourth[team] / runs, 6),
            "prob_final": round(finalists[team] / runs, 6),
            "prob_semifinal": round(semifinalists[team] / runs, 6),
        }
        for team in teams
    ]

    most_common_signature, support = full_brackets.most_common(1)[0]
    slot_summary = []
    for match_id in sorted(slot_matchups, key=lambda value: int(value) if value.isdigit() else 999):
        slot_summary.append(
            {
                "match_id": match_id,
                "top_matchups": _top_tuple_items(slot_matchups[match_id], runs, limit=8),
                "top_winners": _top_items(slot_winners[match_id], runs, limit=8),
            }
        )

    return {
        "metadata": {
            "runs": runs,
            "seed": seed,
            "engine": engine,
            "model_path": selected_model_path,
            "model_label": model_label,
            "model_note": "Neutral LightGBM; World Cup simulation uses advancement after penalties in knockouts.",
            "accuracy_reference": "Played World Cup 2026 held-out evaluation used by the project.",
            "third_place_assignment": "exact_495_combination_table",
            "frozen_context_cache": frozen_context_cache,
            "frozen_context_cache_note": (
                "Fast mode: LightGBM matchup/stage predictions are cached without live group-table "
                "context recomputation inside the simulated tournament."
                if frozen_context_cache
                else ""
            ),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        "team_summary": team_summary,
        "champion_ranking": _top_items(champions, runs, limit=20),
        "best_thirds": _top_items(best_thirds, runs, limit=20),
        "group_positions": {
            team: {
                position: {
                    "count": count,
                    "probability": round(count / runs, 6),
                }
                for position, count in sorted(counter.items(), key=lambda item: int(item[0]))
            }
            for team, counter in sorted(group_positions.items())
        },
        "slot_summary": slot_summary,
        "most_frequent_full_bracket": _signature_to_rows(most_common_signature, support, runs),
    }


def _write_outputs(result: dict[str, Any], output_path: Path) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    team_csv_path = output_path.with_name(output_path.stem + "_teams.csv")
    with team_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["team_summary"][0]))
        writer.writeheader()
        writer.writerows(result["team_summary"])

    matchup_csv_path = output_path.with_name(output_path.stem + "_matchups.csv")
    with matchup_csv_path.open("w", newline="", encoding="utf-8") as handle:
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
                        "winner": winner.get("item", ""),
                        "winner_count": winner.get("count", ""),
                        "winner_probability": winner.get("probability", ""),
                    }
                )

    return {
        "json_path": str(output_path),
        "team_csv_path": str(team_csv_path),
        "matchup_csv_path": str(matchup_csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--frozen-context-cache", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    output = (
        Path(args.output)
        if args.output
        else Path("outputs")
        / f"worldcup2026_consensus_bracket_{args.runs}_{datetime.now().date().isoformat()}.json"
    )
    result = run_consensus_bracket_simulation(
        Path(args.data_root),
        runs=args.runs,
        seed=args.seed,
        progress_every=args.progress_every,
        frozen_context_cache=args.frozen_context_cache,
        model_path=args.model_path,
        model_label=args.model_label or ("explicit_model_path" if args.model_path else "default_model_priority"),
    )
    paths = _write_outputs(result, output)
    print(json.dumps({**paths, **result["metadata"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
