from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from kinela.lightgbm_model import CATEGORICAL_FEATURES, NEUTRAL_FEATURES
from worldcup2026_signal_pruning_evaluation import _prepare


CLF_PARAMS = {
    "objective": "multiclass",
    "n_estimators": 300,
    "learning_rate": 0.035,
    "num_leaves": 23,
    "min_child_samples": 35,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "verbosity": -1,
}


def _latest_manual_result_date(data_root: Path) -> str:
    path = data_root / "static" / "worldcup_2026_manual_results.csv"
    if not path.exists():
        return date.today().isoformat()
    with path.open(encoding="utf-8") as handle:
        dates = [
            row["date"]
            for row in csv.DictReader(handle)
            if row.get("date")
        ]
    return max(dates, default=date.today().isoformat())


def main() -> None:
    data_root = Path("data")
    output = Path(
        f"outputs/worldcup2026_default_auc_evaluation_{_latest_manual_result_date(data_root)}.json"
    )
    neutral = _prepare()
    train = neutral[neutral["split"].eq("train")].copy()
    test = neutral[neutral["split"].eq("test")].copy()
    features = list(NEUTRAL_FEATURES)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    weights = train["match_recency_weight"].astype(float).to_numpy(copy=True)

    calibrated = CalibratedClassifierCV(
        lgb.LGBMClassifier(**CLF_PARAMS),
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(
        train[features],
        train["result_label"],
        sample_weight=weights,
        categorical_feature=categorical,
    )
    probabilities = calibrated.predict_proba(test[features])
    labels = test["result_label"].astype(int).to_numpy()
    one_hot = np.eye(3)[labels]
    class_names = ["team_a", "draw", "team_b"]
    class_auc = {
        class_names[index]: round(
            float(roc_auc_score(one_hot[:, index], probabilities[:, index])),
            4,
        )
        for index in range(3)
    }
    payload = {
        "accuracy_policy": (
            f"All {len(test)} played World Cup 2026 matches are forced to test "
            "and excluded from training."
        ),
        "model": "lightgbm_neutral_current_default",
        "features": features,
        "test_matches": int(len(test)),
        "class_counts": {
            class_names[index]: int((labels == index).sum())
            for index in range(3)
        },
        "accuracy": round(
            float(accuracy_score(labels, probabilities.argmax(axis=1))),
            4,
        ),
        "correct": int((labels == probabilities.argmax(axis=1)).sum()),
        "log_loss": round(
            float(log_loss(labels, probabilities, labels=[0, 1, 2])),
            4,
        ),
        "auc_macro_ovr": round(
            float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
            4,
        ),
        "auc_weighted_ovr": round(
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            ),
            4,
        ),
        "auc_by_class": class_auc,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
