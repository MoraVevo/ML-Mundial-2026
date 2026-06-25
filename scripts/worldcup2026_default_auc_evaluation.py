from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_model_evaluation_report import (  # noqa: E402
    _evaluate,
    _load_frames,
    _split_worldcup_2026,
)


def _latest_manual_result_date(data_root: Path) -> str:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return date.today().isoformat()
    with path.open(encoding="utf-8") as handle:
        dates = [row["date"] for row in csv.DictReader(handle) if row.get("date")]
    return max(dates, default=date.today().isoformat())


def main() -> None:
    data_root = Path("data")
    training, clean = _load_frames(data_root)
    split_clean, split_info = _split_worldcup_2026(training, clean)
    result = _evaluate(
        "worldcup_2026",
        "Holdout Mundial 2026",
        split_info["policy"],
        data_root,
        training,
        split_clean,
    )
    payload = {
        "accuracy_policy": result.policy,
        "model": "lightgbm_neutral_parsimonious",
        "test_matches": result.test_matches,
        "test_start": result.test_start,
        "test_end": result.test_end,
        "class_counts": result.result_distribution,
        "accuracy": result.metrics["accuracy"],
        "correct": result.metrics["correct"],
        "log_loss": result.metrics["log_loss"],
        "mae_team_a_goals": result.metrics["mae_team_a_goals"],
        "mae_team_b_goals": result.metrics["mae_team_b_goals"],
        "mae_goals_avg": result.metrics["mae_goals_avg"],
        "confusion": result.confusion,
    }
    output = Path(
        f"outputs/worldcup2026_default_auc_evaluation_{_latest_manual_result_date(data_root)}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
