from pathlib import Path

import pandas as pd

import scripts.predict_next4_with_all_played_worldcup as all_played
import scripts.worldcup2026_default_auc_evaluation as auc_evaluation
import scripts.worldcup2026_signal_pruning_evaluation as signal_pruning


def test_default_evaluation_frames_are_national_only(monkeypatch, tmp_path) -> None:
    calls = []

    def export_training_frame(data_root: Path, **kwargs):
        calls.append(("training", data_root, kwargs))
        return [{"match_id": "m1"}]

    def export_clean_training_matrix(data_root: Path, **kwargs):
        calls.append(("clean", data_root, kwargs))
        return [{"split": "train"}]

    monkeypatch.setattr(signal_pruning.base_model, "export_training_frame", export_training_frame)
    monkeypatch.setattr(
        signal_pruning.base_model,
        "export_clean_training_matrix",
        export_clean_training_matrix,
    )

    training, clean = signal_pruning._default_evaluation_frames(tmp_path)

    assert len(training) == 1
    assert len(clean) == 1
    assert [name for name, _, _ in calls] == ["training", "clean"]
    assert all(kwargs["national_only"] is True for _, _, kwargs in calls)


def test_latest_manual_result_date_uses_manual_file(tmp_path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "worldcup_2026_manual_results.csv").write_text(
        "\n".join(
            [
                "match_id,date,team_a,team_b,team_a_goals,team_b_goals,winner",
                "1,2026-06-22,A,B,1,0,A",
                "2,2026-06-24,C,D,0,0,Draw",
                "3,2026-06-23,E,F,2,1,E",
            ]
        ),
        encoding="utf-8",
    )

    assert auc_evaluation._latest_manual_result_date(tmp_path) == "2026-06-24"


def test_all_played_dedup_preserves_training_order_after_priority_choice() -> None:
    metadata = pd.DataFrame(
        [
            {"canonical_key": ("2026-06-11", ("a", "b")), "priority": 1},
            {"canonical_key": ("2026-06-12", ("c", "d")), "priority": 0},
            {"canonical_key": ("2026-06-11", ("a", "b")), "priority": 2},
            {"canonical_key": ("2026-06-12", ("e", "f")), "priority": 0},
        ]
    )

    assert all_played._deduplicated_training_indices(metadata) == [1, 2, 3]
