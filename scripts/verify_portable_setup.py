from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT") or "1"
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
)


REQUIRED_FILES = {
    "data/models/lightgbm_neutral_all_played_wc2026.joblib": "future simulation model",
    "data/processed/combined/training_frame.csv": "match history/features",
    "data/processed/combined/training_frame_national.csv": "national match history/features",
    "data/processed/combined/neutral_training_matrix_national.csv": "neutral training matrix",
    "data/processed/fifa/mens_ranking_latest.csv": "latest FIFA ranking",
    "data/processed/football_data/squad_quality.csv": "squad quality features",
    "data/processed/football_data/squad_players.csv": "squad player context",
    "data/static/worldcup_2026_manual_results.csv": "played World Cup 2026 results",
    "data/static/worldcup_2026_manual_detail_stats.csv": "played World Cup 2026 match detail",
    "data/static/worldcup_2026_third_place_assignments.csv": "exact best-third allocation table",
    "data/raw/football_data/competitions/WC/matches.json": "World Cup fixture source",
}

REQUIRED_IMPORTS = [
    "joblib",
    "lightgbm",
    "numpy",
    "pandas",
    "sklearn",
]


def _data_path(data_root: Path, relative: str) -> Path:
    return data_root / Path(relative).relative_to("data")


def _module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def verify(data_root: Path, *, quick_sim: bool) -> dict[str, Any]:
    missing = []
    present = []
    for relative, description in REQUIRED_FILES.items():
        path = _data_path(data_root, relative)
        if path.exists():
            present.append(
                {
                    "path": relative,
                    "description": description,
                    "bytes": path.stat().st_size,
                }
            )
        else:
            missing.append({"path": relative, "description": description})
    if missing:
        return {
            "ok": False,
            "stage": "files",
            "missing": missing,
            "present_count": len(present),
        }

    imports = {name: _module_version(name) for name in REQUIRED_IMPORTS}

    from kinela.worldcup_2026 import WorldCup2026Simulator

    simulator = WorldCup2026Simulator(data_root, seed=42, engine="lightgbm")
    if simulator.lightgbm_model is None:
        return {
            "ok": False,
            "stage": "model",
            "message": "WorldCup2026Simulator did not load a LightGBM model.",
        }
    prediction = simulator.lightgbm_prediction(
        "France",
        "Brazil",
        date(2026, 7, 19),
        "FINAL",
    )
    result: dict[str, Any] = {
        "ok": True,
        "imports": imports,
        "present_count": len(present),
        "model_loaded": True,
        "sample_prediction": {
            "team_a": "France",
            "team_b": "Brazil",
            "stage": "FINAL",
            "team_a_goals": round(float(prediction["team_a_goals"]), 4),
            "team_b_goals": round(float(prediction["team_b_goals"]), 4),
            "result_probabilities": [
                round(float(value), 4)
                for value in prediction["probabilities"]
            ],
        },
    }
    if quick_sim:
        champion, winners, _standings, _best_thirds = simulator.simulate_tournament()
        result["quick_simulation"] = {
            "champion": champion,
            "final_winner": winners.get("103"),
            "last16_slots": [winners.get(str(match_id)) for match_id in range(89, 97)],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--quick-sim", action="store_true")
    args = parser.parse_args()

    result = verify(args.data_root, quick_sim=args.quick_sim)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
