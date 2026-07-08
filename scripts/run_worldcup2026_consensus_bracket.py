from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        data_root / "models" / "lightgbm_neutral_worldcup_holdout.joblib",
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


def _top_matchup_items(
    counter: Counter[tuple[str, str]],
    matchup_winners: dict[tuple[str, str], Counter[str]],
    runs: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for (team_a, team_b), count in counter.most_common(limit):
        winner_counter = matchup_winners.get((team_a, team_b), Counter())
        winner, winner_count = (
            winner_counter.most_common(1)[0] if winner_counter else ("", 0)
        )
        items.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "count": count,
                "probability": round(count / runs, 6),
                "top_winner": winner,
                "top_winner_count": winner_count,
                "top_winner_probability_given_matchup": (
                    round(winner_count / count, 6) if count else 0.0
                ),
            }
        )
    return items


def _semifinal_losers(winners: dict[str, str]) -> list[str]:
    losers: list[str] = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[str(previous_a)]
        team_b = winners[str(previous_b)]
        losers.append(team_b if winners[str(match_id)] == team_a else team_a)
    return losers


def _knockout_signature(
    records: list[dict[str, Any]],
) -> tuple[tuple[str, str, str, str, str], ...]:
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


def _run_simulation_counts(
    data_root: Path,
    *,
    runs: int,
    seed: int,
    progress_every: int,
    fast_mode: bool,
    model_path: Path | None,
    worker_label: str = "",
) -> dict[str, Any]:
    engine = "lightgbm"
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine=engine)
    if model_path:
        simulator.lightgbm_model = joblib.load(model_path)
        simulator.prediction_cache.clear()
    if simulator.lightgbm_model is None:
        raise FileNotFoundError(
            "Missing trained LightGBM model. Run "
            "`python scripts\\train_worldcup2026_holdout_model.py` first."
        )
    if fast_mode:
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
    slot_matchup_winners: dict[str, dict[tuple[str, str], Counter[str]]] = defaultdict(dict)
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
            matchup = (row["team_a"], row["team_b"])
            slot_matchups[match_id][matchup] += 1
            slot_winners[match_id][row["winner"]] += 1
            slot_matchup_winners[match_id].setdefault(matchup, Counter())[row["winner"]] += 1

        if progress_every and index % progress_every == 0:
            elapsed = (datetime.now() - started).total_seconds()
            top3 = " | ".join(
                f"{team}: {count / index:.1%}"
                for team, count in champions.most_common(3)
            )
            prefix = f"worker {worker_label} | " if worker_label else ""
            print(
                f"{prefix}{index}/{runs} simulaciones | top campeones: {top3} | "
                f"ultima={champion} | {elapsed:.1f}s",
                flush=True,
            )

    return {
        "champions": champions,
        "finalists": finalists,
        "semifinalists": semifinalists,
        "first": first,
        "second": second,
        "third": third,
        "fourth": fourth,
        "best_thirds": best_thirds,
        "group_positions": group_positions,
        "slot_matchups": slot_matchups,
        "slot_winners": slot_winners,
        "slot_matchup_winners": slot_matchup_winners,
        "full_brackets": full_brackets,
    }


def _merge_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "champions",
        "finalists",
        "semifinalists",
        "first",
        "second",
        "third",
        "fourth",
        "best_thirds",
        "full_brackets",
    ):
        target[key].update(source[key])
    for team, counter in source["group_positions"].items():
        target["group_positions"][team].update(counter)
    for match_id, counter in source["slot_matchups"].items():
        target["slot_matchups"][match_id].update(counter)
    for match_id, counter in source["slot_winners"].items():
        target["slot_winners"][match_id].update(counter)
    for match_id, matchup_winners in source["slot_matchup_winners"].items():
        target_matchup_winners = target["slot_matchup_winners"][match_id]
        for matchup, counter in matchup_winners.items():
            target_matchup_winners.setdefault(matchup, Counter()).update(counter)


def _empty_counts() -> dict[str, Any]:
    return {
        "champions": Counter(),
        "finalists": Counter(),
        "semifinalists": Counter(),
        "first": Counter(),
        "second": Counter(),
        "third": Counter(),
        "fourth": Counter(),
        "best_thirds": Counter(),
        "group_positions": defaultdict(Counter),
        "slot_matchups": defaultdict(Counter),
        "slot_winners": defaultdict(Counter),
        "slot_matchup_winners": defaultdict(dict),
        "full_brackets": Counter(),
    }


def _run_worker(args: dict[str, Any]) -> dict[str, Any]:
    return _run_simulation_counts(**args)


def _worker_run_counts(runs: int, workers: int) -> list[int]:
    workers = max(1, min(workers, runs))
    base = runs // workers
    remainder = runs % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _summarize_counts(
    counts: dict[str, Any],
    *,
    runs: int,
    seed: int,
    engine: str,
    selected_model_path: str,
    model_label: str,
    fast_mode: bool,
    workers: int,
    started: datetime,
) -> dict[str, Any]:
    champions = counts["champions"]
    finalists = counts["finalists"]
    semifinalists = counts["semifinalists"]
    first = counts["first"]
    second = counts["second"]
    third = counts["third"]
    fourth = counts["fourth"]
    best_thirds = counts["best_thirds"]
    group_positions = counts["group_positions"]
    slot_matchups = counts["slot_matchups"]
    slot_winners = counts["slot_winners"]
    slot_matchup_winners = counts["slot_matchup_winners"]
    full_brackets = counts["full_brackets"]

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
                "top_matchups": _top_matchup_items(
                    slot_matchups[match_id],
                    slot_matchup_winners[match_id],
                    runs,
                    limit=8,
                ),
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
            "model_note": (
                "Neutral LightGBM; World Cup simulation uses advancement "
                "after penalties in knockouts."
            ),
            "accuracy_reference": "Played World Cup 2026 held-out evaluation used by the project.",
            "third_place_assignment": "exact_495_combination_table",
            "simulation_mode": "fast_fixed_matchup_probabilities" if fast_mode else "full_context",
            "simulation_mode_note": (
                "Fast mode keeps each team-vs-team stage probability fixed across runs."
                if fast_mode
                else (
                    "Full mode recomputes group-table context and live FIFA point "
                    "updates inside each simulated tournament; recent-form features "
                    "stay anchored to real completed matches to avoid fake-result feedback."
                )
            ),
            "workers": workers,
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


def run_consensus_bracket_simulation(
    data_root: Path,
    *,
    runs: int,
    seed: int,
    progress_every: int,
    fast_mode: bool,
    model_path: Path | None,
    model_label: str,
    workers: int = 1,
) -> dict[str, Any]:
    engine = "lightgbm"
    selected_model_path = str(model_path) if model_path else _selected_model_path(data_root, engine)
    started = datetime.now()
    if workers <= 1:
        counts = _run_simulation_counts(
            data_root,
            runs=runs,
            seed=seed,
            progress_every=progress_every,
            fast_mode=fast_mode,
            model_path=model_path,
        )
    else:
        counts = _empty_counts()
        run_counts = _worker_run_counts(runs, workers)
        with ProcessPoolExecutor(max_workers=len(run_counts)) as executor:
            futures = []
            for index, worker_runs in enumerate(run_counts, start=1):
                futures.append(
                    executor.submit(
                        _run_worker,
                        {
                            "data_root": data_root,
                            "runs": worker_runs,
                            "seed": seed + index * 100_003,
                            "progress_every": progress_every,
                            "fast_mode": fast_mode,
                            "model_path": model_path,
                            "worker_label": f"{index}/{len(run_counts)}",
                        },
                    )
                )
            for future in as_completed(futures):
                _merge_counts(counts, future.result())

    return _summarize_counts(
        counts,
        runs=runs,
        seed=seed,
        engine=engine,
        selected_model_path=selected_model_path,
        model_label=model_label,
        fast_mode=fast_mode,
        workers=max(1, min(workers, runs)),
        started=started,
    )


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
            "winner_probability_given_matchup",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for slot in result["slot_summary"]:
            for rank, matchup in enumerate(slot["top_matchups"], start=1):
                writer.writerow(
                    {
                        "match_id": slot["match_id"],
                        "rank": rank,
                        "team_a": matchup["team_a"],
                        "team_b": matchup["team_b"],
                        "matchup_count": matchup["count"],
                        "matchup_probability": matchup["probability"],
                        "winner": matchup.get("top_winner", ""),
                        "winner_count": matchup.get("top_winner_count", ""),
                        "winner_probability_given_matchup": matchup.get(
                            "top_winner_probability_given_matchup",
                            "",
                        ),
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fast", dest="fast_mode", action="store_true")
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
        fast_mode=args.fast_mode,
        model_path=args.model_path,
        model_label=args.model_label
        or ("explicit_model_path" if args.model_path else "default_model_priority"),
        workers=args.workers,
    )
    paths = _write_outputs(result, output)
    print(json.dumps({**paths, **result["metadata"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
