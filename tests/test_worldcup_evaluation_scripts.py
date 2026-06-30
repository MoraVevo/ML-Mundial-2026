import pandas as pd

import scripts.generate_model_evaluation_report as report
import scripts.worldcup2026_model_metrics as model_metrics


def test_worldcup_split_uses_only_prior_training_rows() -> None:
    training = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "source": "provider",
                "competition_name": "Friendly",
                "is_friendly": True,
            },
            {
                "date": "2026-06-10",
                "source": "provider",
                "competition_name": "World Cup - Qualification Europe",
                "is_friendly": False,
            },
            {
                "date": "2026-06-11",
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
                "is_friendly": False,
            },
            {
                "date": "2026-06-12",
                "source": "provider",
                "competition_name": "World Cup - Qualification Europe",
                "is_friendly": False,
            },
        ]
    )
    clean = pd.DataFrame({"split": ["train", "train", "train", "train"]})

    split, info = report._split_worldcup_2026(training, clean)

    assert split["split"].tolist() == ["train", "train", "test", "excluded"]
    assert "before the first World Cup 2026 match" in info["policy"]


def test_external_random_split_excludes_same_and_later_nonselected_rows() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "row_index": index,
                "date": f"2025-01-{index + 1:02d}",
                "source": "provider",
                "competition_name": "UEFA Nations League",
                "is_friendly": False,
            }
        )
    training = pd.DataFrame(rows)
    clean = pd.DataFrame({"split": ["train"] * len(training)})

    split, _ = report._split_external_random_temporal(
        training,
        clean,
        test_matches=4,
        seed=7,
    )
    test_dates = pd.to_datetime(training.loc[split["split"].eq("test"), "date"])
    cutoff = test_dates.min()
    later_non_test = pd.to_datetime(training["date"]).ge(cutoff) & ~split["split"].eq("test")

    assert set(split["split"].unique()) == {"train", "test", "excluded"}
    assert split.loc[later_non_test, "split"].eq("excluded").all()


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

    assert model_metrics._latest_manual_result_date(tmp_path) == "2026-06-24"
