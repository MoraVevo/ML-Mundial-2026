from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from kinela.worldcup_2026 import QUARTER_FINALS, SEMI_FINALS, WorldCup2026Simulator


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


def _semifinal_losers(winners: dict[str, str]) -> list[str]:
    losers: list[str] = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[str(previous_a)]
        team_b = winners[str(previous_b)]
        losers.append(team_b if winners[str(match_id)] == team_a else team_a)
    return losers


def run_top4_simulation(
    data_root: Path,
    *,
    simulations: int,
    seed: int,
    engine: str,
    progress_every: int,
    frozen_context_cache: bool,
) -> dict[str, Any]:
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine=engine)
    model_path = _selected_model_path(data_root, engine)
    if frozen_context_cache and simulator.lightgbm_model is not None:
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
    first: Counter[str] = Counter()
    second: Counter[str] = Counter()
    third: Counter[str] = Counter()
    fourth: Counter[str] = Counter()
    top4: Counter[str] = Counter()
    finalists: Counter[str] = Counter()
    semifinalists: Counter[str] = Counter()

    started = datetime.now()
    for index in range(1, simulations + 1):
        champion, winners, _standings, _best_thirds = simulator.simulate_tournament()
        finalist_a = winners["101"]
        finalist_b = winners["102"]
        runner_up = finalist_b if champion == finalist_a else finalist_a
        semi_losers = _semifinal_losers(winners)
        third_place = winners["104"]
        fourth_place = semi_losers[1] if third_place == semi_losers[0] else semi_losers[0]

        first[champion] += 1
        second[runner_up] += 1
        third[third_place] += 1
        fourth[fourth_place] += 1
        finalists[finalist_a] += 1
        finalists[finalist_b] += 1
        for match_id, _previous_a, _previous_b in QUARTER_FINALS:
            semifinalists[winners[str(match_id)]] += 1
        for team in (champion, runner_up, third_place, fourth_place):
            top4[team] += 1

        if progress_every and index % progress_every == 0:
            elapsed = (datetime.now() - started).total_seconds()
            print(f"{index}/{simulations} simulations complete in {elapsed:.1f}s", flush=True)

    teams = sorted(
        set(first) | set(second) | set(third) | set(fourth) | set(top4),
        key=lambda team: (-top4[team], -first[team], team),
    )
    rows = [
        {
            "equipo": team,
            "veces_top4": top4[team],
            "prob_top4": round(top4[team] / simulations, 6),
            "veces_campeon": first[team],
            "prob_campeon": round(first[team] / simulations, 6),
            "veces_segundo": second[team],
            "prob_segundo": round(second[team] / simulations, 6),
            "veces_tercero": third[team],
            "prob_tercero": round(third[team] / simulations, 6),
            "veces_cuarto": fourth[team],
            "prob_cuarto": round(fourth[team] / simulations, 6),
            "prob_final": round(finalists[team] / simulations, 6),
            "prob_semifinal": round(semifinalists[team] / simulations, 6),
        }
        for team in teams
    ]
    return {
        "metadata": {
            "simulations": simulations,
            "seed": seed,
            "engine_requested": engine,
            "engine_used": engine if simulator.lightgbm_model is not None or engine != "lightgbm" else "poisson_fallback",
            "model_path": model_path,
            "third_place_assignment": "exact_495_combination_table",
            "frozen_context_cache": frozen_context_cache,
            "frozen_context_cache_note": (
                "LightGBM matchup/stage predictions are cached without live FIFA/group-table "
                "recomputation after simulated matches."
                if frozen_context_cache
                else ""
            ),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", choices=["lightgbm", "poisson"], default="lightgbm")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--frozen-context-cache", action="store_true")
    args = parser.parse_args()

    result = run_top4_simulation(
        Path(args.data_root),
        simulations=args.runs,
        seed=args.seed,
        engine=args.engine,
        progress_every=args.progress_every,
        frozen_context_cache=args.frozen_context_cache,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().date().isoformat()
    base_name = f"worldcup_2026_top4_{args.runs}_{date_tag}"
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])

    print(json.dumps({"json_path": str(json_path), "csv_path": str(csv_path), **result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
